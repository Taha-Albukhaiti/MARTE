#!/usr/bin/env python3
"""Supplementary run-order trend analysis for MARTE thesis.

Checks whether RAG-LTM breach efficiency improves over the seed sequence,
i.e. whether later runs (with more accumulated LTM episodes) breach faster.

IMPORTANT: This is a DESCRIPTIVE analysis only. Runs within a RAG-LTM
condition are not independent — each run can retrieve from all prior runs'
episodes. Cox p-values are therefore not valid as confirmatory tests and
must not be interpreted as such. Report only the hazard ratios and tercile
ASRs as descriptive evidence of a learning trend (or its absence).

Output: --output-dir/run_order_trend.json and run_order_trend.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path

import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BLOCKS: dict[str, str] = {
    "sql_moderate": "experiment-results/events_sql_moderate.db",
    "file_moderate": "experiment-results/events_file_moderate.db",
    "file_enterprise": "experiment-results/events_file_enterprise.db",
}
# sql_enterprise is excluded: 0 breaches in both arms → no Q2B trend definable.
EXCLUDED_BLOCKS = {"sql_enterprise": "0 breaches in both arms — Q2B trend not defined"}

TERCILE_BINS = [0, 10, 20, 30]
TERCILE_LABELS = ["1-10", "11-20", "21-30"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--source-config",
        type=Path,
        default=None,
        help="Optional JSON mapping block names to DB paths (overrides built-in defaults).",
    )
    args = parser.parse_args(argv)

    blocks = _load_block_config(args.source_config)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for block, db_path in blocks.items():
        db = db_path if db_path.is_absolute() else (REPO_ROOT / db_path)
        if not db.exists():
            print(f"[SKIP] {block}: DB not found at {db}")
            continue
        df = _load_stops(db)
        for mode in ("stm_only", "rag_ltm"):
            sub = df[df["mode"] == mode].copy()
            if sub.empty:
                continue
            tercile_rows = _tercile_summary(block, mode, sub)
            cox_row = _cox_slope(block, mode, sub)
            rows.extend(tercile_rows)
            if cox_row:
                rows.append(cox_row)
            _print_block(block, mode, tercile_rows, cox_row)

    _write_json(output_dir / "run_order_trend.json", rows)
    _write_csv(output_dir / "run_order_trend.csv", rows)
    print(f"\nArtifacts written to {output_dir}")
    return 0


def _load_block_config(config_path: Path | None) -> dict[str, Path]:
    if config_path is not None:
        raw: dict[str, str] = json.loads(config_path.read_text(encoding="utf-8"))
        return {block: Path(path) for block, path in raw.items()}
    return {block: Path(path) for block, path in DEFAULT_BLOCKS.items()}


def _load_stops(db: Path) -> pd.DataFrame:
    con = sqlite3.connect(str(db))
    df = pd.read_sql(
        "SELECT condition, seed, q2b, censored_flag FROM stop_events WHERE stop_reason != 'error'",
        con,
    )
    con.close()
    df["event"] = 1 - df["censored_flag"].astype(int)
    df["mode"] = df["condition"].str.contains("rag_ltm").map({True: "rag_ltm", False: "stm_only"})
    df["tercile"] = pd.cut(df["seed"], TERCILE_BINS, labels=TERCILE_LABELS)
    return df


def _tercile_summary(block: str, mode: str, sub: pd.DataFrame) -> list[dict]:
    rows = []
    for tercile_label, group in sub.groupby("tercile", observed=True):
        n = len(group)
        n_breach = int(group["event"].sum())
        asr = n_breach / n if n > 0 else None
        kmf = KaplanMeierFitter()
        if n_breach > 0:
            kmf.fit(group["q2b"], group["event"])
            km_median = float(kmf.median_survival_time_)
            if not math.isfinite(km_median):
                km_median = None
        else:
            km_median = None
        rows.append({
            "record_type": "tercile",
            "block": block,
            "mode": mode,
            "tercile": str(tercile_label),
            "n": n,
            "n_breach": n_breach,
            "asr": round(asr, 3) if asr is not None else None,
            "km_median_q2b": km_median,
            "note": "descriptive only — runs within rag_ltm are not independent",
        })
    return rows


def _cox_slope(block: str, mode: str, sub: pd.DataFrame) -> dict | None:
    if int(sub["event"].sum()) < 2:
        return {
            "record_type": "cox_slope",
            "block": block,
            "mode": mode,
            "hr_per_10_runs": None,
            "hr_ci_low_95": None,
            "hr_ci_high_95": None,
            "note": "insufficient events for Cox fit — descriptive only",
        }
    try:
        fit_df = sub[["q2b", "event", "seed"]].copy()
        fit_df["seed10"] = fit_df["seed"] / 10.0
        cph = CoxPHFitter()
        cph.fit(fit_df[["q2b", "event", "seed10"]], duration_col="q2b", event_col="event")
        summary = cph.summary.loc["seed10"]
        return {
            "record_type": "cox_slope",
            "block": block,
            "mode": mode,
            "hr_per_10_runs": round(float(summary["exp(coef)"]), 3),
            "hr_ci_low_95": round(float(summary["exp(coef) lower 95%"]), 3),
            "hr_ci_high_95": round(float(summary["exp(coef) upper 95%"]), 3),
            "note": (
                "DESCRIPTIVE ONLY — runs within rag_ltm are not independent; "
                "p-values from this model are not valid as confirmatory tests"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "record_type": "cox_slope",
            "block": block,
            "mode": mode,
            "hr_per_10_runs": None,
            "hr_ci_low_95": None,
            "hr_ci_high_95": None,
            "note": f"Cox fit failed: {exc}",
        }


def _print_block(block: str, mode: str, tercile_rows: list[dict], cox_row: dict | None) -> None:
    print(f"\n{block} / {mode}")
    for row in tercile_rows:
        print(
            f"  seeds {row['tercile']}: n={row['n']} breaches={row['n_breach']} "
            f"ASR={row['asr']} KM-median Q2B={row['km_median_q2b']}"
        )
    if cox_row:
        print(
            f"  Cox HR per 10 runs: {cox_row['hr_per_10_runs']} "
            f"[{cox_row['hr_ci_low_95']}, {cox_row['hr_ci_high_95']}]  "
            f"({cox_row['note'][:55]}...)"
        )


def _write_json(path: Path, rows: list[dict]) -> None:
    payload = {
        "disclaimer": (
            "DESCRIPTIVE ANALYSIS ONLY. Runs within a RAG-LTM condition are not "
            "independent — each run retrieves from all prior runs' LTM episodes. "
            "Cox hazard ratios describe the observed temporal slope but are not "
            "valid as confirmatory statistical tests. Do not report p-values."
        ),
        "excluded_blocks": EXCLUDED_BLOCKS,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

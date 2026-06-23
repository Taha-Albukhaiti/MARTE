"""Calibrate positive and blocked RAG thresholds from labeled retrieval pairs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

CHANNELS = ("positive", "blocked")
STRATA = (
    ("sql", "moderate"),
    ("sql", "enterprise"),
    ("file", "moderate"),
    ("file", "enterprise"),
)


def main() -> None:
    args = _parse_args()
    records = _load_records(args.input)
    result = {
        "selection_rule": (
            "maximize macro-F1 across scenario/guardrail strata; "
            "choose the higher threshold on ties"
        ),
        "channels": {
            channel: _calibrate_channel(
                records,
                channel,
                allow_small_set=args.allow_small_set,
            )
            for channel in CHANNELS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select separate positive and blocked cosine-similarity thresholds "
            "without using attack outcomes."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON file containing an 'items' list of labeled query-episode pairs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON calibration report to create.",
    )
    parser.add_argument(
        "--allow-small-set",
        action="store_true",
        help="Permit fewer than 20 relevant and 20 irrelevant pairs per channel.",
    )
    return parser.parse_args()


def _load_records(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("items") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("Calibration input must be a list or an object with an 'items' list.")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"items[{index}] must be an object.")
        channel = str(item.get("channel", "")).strip().lower()
        scenario = str(item.get("scenario", "")).strip().lower()
        guardrail = str(item.get("guardrail_level", "")).strip().lower()
        if channel not in CHANNELS:
            raise ValueError(f"items[{index}].channel must be one of {CHANNELS}.")
        if (scenario, guardrail) not in STRATA:
            raise ValueError(
                f"items[{index}] has invalid stratum {(scenario, guardrail)!r}."
            )
        similarity = float(item.get("similarity"))
        relevant = item.get("relevant")
        if not isinstance(relevant, bool):
            raise ValueError(f"items[{index}].relevant must be boolean.")
        records.append(
            {
                "channel": channel,
                "scenario": scenario,
                "guardrail_level": guardrail,
                "similarity": similarity,
                "relevant": relevant,
            }
        )
    return records


def _calibrate_channel(
    records: list[dict[str, Any]],
    channel: str,
    *,
    allow_small_set: bool,
) -> dict[str, Any]:
    selected = [record for record in records if record["channel"] == channel]
    relevant_count = sum(bool(record["relevant"]) for record in selected)
    irrelevant_count = len(selected) - relevant_count
    if not allow_small_set and (relevant_count < 20 or irrelevant_count < 20):
        raise ValueError(
            f"{channel} requires at least 20 relevant and 20 irrelevant pairs; "
            f"found {relevant_count} and {irrelevant_count}."
        )

    present_strata = {
        (record["scenario"], record["guardrail_level"])
        for record in selected
    }
    missing = sorted(set(STRATA) - present_strata)
    if missing:
        raise ValueError(f"{channel} is missing calibration strata: {missing}.")

    thresholds = sorted(
        {
            0.0,
            1.0,
            *(
                max(0.0, min(1.0, float(record["similarity"])))
                for record in selected
            ),
        }
    )
    scored = [
        (threshold, _macro_f1(selected, threshold))
        for threshold in thresholds
    ]
    threshold, macro_f1 = max(scored, key=lambda item: (item[1], item[0]))
    return {
        "threshold": threshold,
        "macro_f1": macro_f1,
        "relevant_pairs": relevant_count,
        "irrelevant_pairs": irrelevant_count,
        "total_pairs": len(selected),
        "stratum_f1": {
            f"{scenario}_{guardrail}": _f1(
                [
                    record
                    for record in selected
                    if record["scenario"] == scenario
                    and record["guardrail_level"] == guardrail
                ],
                threshold,
            )
            for scenario, guardrail in STRATA
        },
    }


def _macro_f1(records: list[dict[str, Any]], threshold: float) -> float:
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_stratum[(record["scenario"], record["guardrail_level"])].append(record)
    return sum(_f1(by_stratum[stratum], threshold) for stratum in STRATA) / len(
        STRATA
    )


def _f1(records: list[dict[str, Any]], threshold: float) -> float:
    true_positive = sum(
        record["relevant"] and record["similarity"] >= threshold
        for record in records
    )
    false_positive = sum(
        not record["relevant"] and record["similarity"] >= threshold
        for record in records
    )
    false_negative = sum(
        record["relevant"] and record["similarity"] < threshold
        for record in records
    )
    denominator = 2 * true_positive + false_positive + false_negative
    return (2 * true_positive / denominator) if denominator else 0.0


if __name__ == "__main__":
    main()

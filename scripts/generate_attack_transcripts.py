"""Generate readable LaTeX transcripts for selected thesis attack examples.

The generated files expose the model-visible STM and RAG content plus the
step-specific communication. Retrieval metadata and raw JSON are omitted.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TranscriptSpec:
    slug: str
    database: str
    experiment_id: str
    run_id: int
    step_t: int


TRANSCRIPTS = (
    # ── Datei-Szenario ──────────────────────────────────────────────────────
    # D-1: Gepaarter Vergleich Seed 14 — RAG Q2B=2 vs STM Q2B=29
    TranscriptSpec(
        slug="angriff-d1-rag",
        database="exp-results/events_file_moderate.db",
        experiment_id="exp_2026-06-14_09-57-43-609359",
        run_id=27,
        step_t=2,
    ),
    TranscriptSpec(
        slug="angriff-d1-stm",
        database="exp-results/events_file_moderate.db",
        experiment_id="exp_2026-06-14_09-57-43-609359",
        run_id=28,
        step_t=29,
    ),
    # D-2: Write-Bypass Seed 24 STM — Q2B=50, Breach via file.write
    TranscriptSpec(
        slug="angriff-d2",
        database="exp-results/events_file_moderate.db",
        experiment_id="exp_2026-06-14_09-57-43-609359",
        run_id=48,
        step_t=50,
    ),
    # D-3: Zensierter Lauf Seed 4 STM — repräsentativer blockierter Schritt
    TranscriptSpec(
        slug="angriff-d3",
        database="exp-results/events_file_moderate.db",
        experiment_id="exp_2026-06-14_09-57-43-609359",
        run_id=8,
        step_t=7,
    ),
    # ── SQL-Szenario ────────────────────────────────────────────────────────
    # S-1: Gepaarter Vergleich Seed 13 — RAG Q2B=3 vs STM Q2B=20
    TranscriptSpec(
        slug="angriff-s1-rag",
        database="exp-results/events_sql_moderate.db",
        experiment_id="exp_2026-06-14_04-53-49-234519",
        run_id=26,
        step_t=3,
    ),
    TranscriptSpec(
        slug="angriff-s1-stm",
        database="exp-results/events_sql_moderate.db",
        experiment_id="exp_2026-06-14_04-53-49-234519",
        run_id=25,
        step_t=20,
    ),
    # S-2: Umgekehrter Fall Seed 20 — STM Q2B=2 vs RAG Q2B=34
    TranscriptSpec(
        slug="angriff-s2-stm",
        database="exp-results/events_sql_moderate.db",
        experiment_id="exp_2026-06-14_04-53-49-234519",
        run_id=40,
        step_t=2,
    ),
    TranscriptSpec(
        slug="angriff-s2-rag",
        database="exp-results/events_sql_moderate.db",
        experiment_id="exp_2026-06-14_04-53-49-234519",
        run_id=39,
        step_t=34,
    ),
    # S-3: Letzter-Schritt-Breach Seed 18 STM — Q2B=50, Breach auf Schritt 50
    TranscriptSpec(
        slug="angriff-s3",
        database="exp-results/events_sql_moderate.db",
        experiment_id="exp_2026-06-14_04-53-49-234519",
        run_id=36,
        step_t=50,
    ),
)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in TRANSCRIPTS:
        row = _load_step(REPO_ROOT / spec.database, spec)
        output_path = output_dir / f"{spec.slug}.tex"
        output_path.write_text(_render_transcript(row), encoding="utf-8")
        print(f"Wrote {output_path.relative_to(REPO_ROOT)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the selected attack-example transcripts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "masterarbeit/inc/generated",
    )
    return parser.parse_args()


def _load_step(
    database: Path,
    spec: TranscriptSpec,
) -> dict[str, Any]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT step_t, prompt, response, tool_trace, policy_decisions,
                   oracle_verdict, judge_verdict, memory_context, plan,
                   input_decision, output_decision, input_reason, output_reason,
                   retrieval_trace, judge_debug
            FROM step_events
            WHERE experiment_id = ? AND run_id = ? AND step_t = ?
            """,
            (spec.experiment_id, spec.run_id, spec.step_t),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        raise RuntimeError(
            f"Missing step: {database}, run {spec.run_id}, step {spec.step_t}"
        )

    result = dict(row)
    for key in (
        "tool_trace",
        "policy_decisions",
        "memory_context",
        "plan",
        "retrieval_trace",
        "judge_debug",
    ):
        result[key] = json.loads(result[key])
    return result


def _render_transcript(row: dict[str, Any]) -> str:
    memory = row["memory_context"]
    retrieval = row["retrieval_trace"]
    plan = row["plan"]
    judge_debug = row["judge_debug"]
    lines = [
        "% Generated by scripts/generate_attack_transcripts.py; do not edit.",
        (
            "\\begin{tcolorbox}[exampletrace,title={"
            f"Entscheidungsschritt {row['step_t']}: schrittspezifischer "
            "Informationsfluss}]"
        ),
        "\\traceheading{1. Planner-Eingabe}",
        "\\tracefield{Laufkontext}{"
        + _latex_escape(
            "\n".join(
                (
                    f"Szenario: {memory.get('scenario', '')}",
                    f"Guardrail: {memory.get('guardrail_level', '')}",
                    f"Schritt: {memory.get('step_index', row['step_t'])}",
                    f"Angriffsziel: {memory.get('goal', '')}",
                )
            )
        )
        + "}",
        "\\tracesubheading{STM an Planner und Mutator}",
    ]
    lines.extend(_render_stm(memory.get("stm_recent", [])))
    lines.append("\\tracesubheading{RAG an den Planner}")
    if memory.get("memory_mode") == "rag_ltm":
        lines.extend(_render_planner_positive(memory.get("positive_ltm_hits", [])))
        lines.extend(_render_planner_blocked(memory.get("blocked_patterns", [])))
    else:
        lines.append("\\traceempty{Nicht vorhanden: STM-only-Bedingung.}")

    lines.extend(
        [
            "\\traceheading{2. Planner-Ausgabe "
            "\\texorpdfstring{$\\rightarrow$}{->} Mutator}",
            "\\tracefield{Strategie}{"
            + _latex_escape(str(plan.get("strategy", "")))
            + "}",
            "\\tracefield{Ziel}{"
            + _latex_escape(str(plan.get("objective", "")))
            + "}",
            "\\tracefield{Begründung}{"
            + _latex_escape(str(plan.get("rationale", "")))
            + "}",
            "\\traceheading{3. Zusätzliche Mutator-Eingabe}",
            (
                "\\traceempty{Der Mutator erhält die Planner-Ausgabe und den "
                "oben gezeigten STM-Kontext; diese Inhalte werden hier nicht "
                "doppelt abgedruckt.}"
            ),
            "\\tracesubheading{Strategiespezifisches RAG an den Mutator}",
        ]
    )
    mutator_hits = [
        hit["episode"]
        for hit in retrieval.get("mutator_positive", {}).get("hits", [])
    ]
    if memory.get("memory_mode") == "rag_ltm":
        lines.extend(_render_mutator_positive(mutator_hits, plan["strategy"]))
    else:
        lines.append("\\traceempty{Nicht vorhanden: STM-only-Bedingung.}")

    lines.extend(
        [
            "\\traceheading{4. Mutator "
            "\\texorpdfstring{$\\rightarrow$}{->} Zielagent}",
            "\\tracefield{Vollständiger Angriffsprompt}{"
            + _latex_escape(row["prompt"])
            + "}",
            "\\traceheading{5. Zielagent und Werkzeugkommunikation}",
        ]
    )
    lines.extend(_render_tools(row["tool_trace"]))
    lines.extend(
        [
            "\\tracefield{Zielagent "
            "\\texorpdfstring{$\\rightarrow$}{->} Orchestrator}{"
            + _latex_escape(row["response"])
            + "}",
            "\\traceheading{6. Deterministische Auswertung}",
            "\\tracefield{Policy}{"
            + _render_policy(row)
            + "}",
            "\\tracefield{BreachOracle}{"
            + ("true" if row["oracle_verdict"] else "false")
            + "}",
            "\\tracefield{Judge}{"
            + _latex_escape(row["judge_verdict"])
            + "}",
        ]
    )
    rationale = str(judge_debug.get("rationale", "")).strip()
    if rationale:
        lines.append(
            "\\tracefield{Judge-Begründung}{" + _latex_escape(rationale) + "}"
        )
    memory_summary = str(judge_debug.get("memory_summary", "")).strip()
    if memory_summary:
        lines.append(
            "\\tracefield{Gespeicherte Zusammenfassung}{"
            + _latex_escape(memory_summary)
            + "}"
        )
    lines.extend(["\\end{tcolorbox}", ""])
    return "\n".join(lines)


def _render_stm(episodes: list[dict[str, Any]]) -> list[str]:
    if not episodes:
        return ["\\traceempty{Keine vorherige Episode in diesem Lauf.}"]

    lines: list[str] = []
    for index, episode in enumerate(episodes, start=1):
        result = str(episode.get("success_label", "unknown")).strip().lower()
        lines.append(f"\\traceepisodeheading{{STM-Episode {index}}}")
        lines.append(
            "\\tracefield{Schritt}{"
            + _latex_escape(str(episode.get("step_t", "")))
            + "}"
        )
        lines.append(
            "\\tracefield{Strategie}{"
            + _latex_escape(_episode_strategy(episode))
            + "}"
        )
        lines.append(
            "\\tracefield{Angriffsprompt (max. 250 Zeichen)}{"
            + _latex_escape(str(episode.get("mutation", ""))[:250])
            + "}"
        )
        lines.append("\\tracefield{Ergebnis}{" + _latex_escape(result) + "}")
        lines.append(
            "\\tracefield{Breach}{"
            + ("true" if result == "success" else "false")
            + "}"
        )
        response = str(episode.get("response_summary", "")).strip()[:200]
        if response:
            lines.append(
                "\\tracefield{Beobachtungszusammenfassung (max. 200 Zeichen)}{"
                + _latex_escape(response)
                + "}"
            )
        tool_effect = str(episode.get("tool_effect", "")).strip()[:160]
        if tool_effect:
            lines.append(
                "\\tracefield{Werkzeugeffekt (max. 160 Zeichen)}{"
                + _latex_escape(tool_effect)
                + "}"
            )
        policy_stage = str(episode.get("policy_stage", "")).strip()[:20]
        if policy_stage:
            lines.append(
                "\\tracefield{Policy-Stufe}{"
                + _latex_escape(policy_stage)
                + "}"
            )
        policy_reason = str(episode.get("policy_reason", "")).strip()[:180]
        if policy_reason:
            lines.append(
                "\\tracefield{Policy-Grund (max. 180 Zeichen)}{"
                + _latex_escape(policy_reason)
                + "}"
            )
    return lines


def _render_planner_positive(episodes: list[dict[str, Any]]) -> list[str]:
    visible = [
        episode
        for episode in episodes
        if str(episode.get("success_label", "")).strip().lower()
        in {"success", "partial"}
    ][:3]
    if not visible:
        return ["\\traceempty{Keine positive LTM-Episode in diesem Kanal.}"]

    lines: list[str] = []
    for index, episode in enumerate(visible, start=1):
        lines.append(f"\\traceepisodeheading{{Positive RAG-Episode {index}}}")
        lines.append(
            "\\tracefield{Strategie}{"
            + _latex_escape(_episode_strategy(episode))
            + "}"
        )
        lines.append(
            "\\tracefield{Ergebnis}{"
            + _latex_escape(str(episode.get("success_label", "")).lower())
            + "}"
        )
        lines.append(
            "\\tracefield{Prompt-Ausschnitt (max. 100 Zeichen)}{"
            + _latex_escape(str(episode.get("mutation", ""))[:100])
            + "}"
        )
        lines.append(
            "\\tracefield{Werkzeugeffekt (max. 120 Zeichen)}{"
            + _latex_escape(str(episode.get("tool_effect", ""))[:120])
            + "}"
        )
    return lines


def _render_planner_blocked(episodes: list[dict[str, Any]]) -> list[str]:
    visible = episodes[:1]
    if not visible:
        return ["\\traceempty{Keine blockierte LTM-Episode in diesem Kanal.}"]

    lines: list[str] = []
    for index, episode in enumerate(visible, start=1):
        lines.append(f"\\traceepisodeheading{{Blockierte RAG-Episode {index}}}")
        lines.append(
            "\\tracefield{Strategie}{"
            + _latex_escape(_episode_strategy(episode))
            + "}"
        )
        lines.append(
            "\\tracefield{Blockierte Formulierung (max. 180 Zeichen)}{"
            + _latex_escape(str(episode.get("mutation", ""))[:180])
            + "}"
        )
        lines.append(
            "\\tracefield{Policy-Stufe}{"
            + _latex_escape(str(episode.get("policy_stage", ""))[:20])
            + "}"
        )
        lines.append(
            "\\tracefield{Policy-Grund (max. 200 Zeichen)}{"
            + _latex_escape(str(episode.get("policy_reason", ""))[:200])
            + "}"
        )
    return lines


def _render_mutator_positive(
    episodes: list[dict[str, Any]],
    expected_strategy: str,
) -> list[str]:
    visible = [
        episode
        for episode in episodes
        if str(episode.get("success_label", "")).strip().lower()
        in {"success", "partial"}
        and (
            not str(episode.get("strategy", "")).strip()
            or str(episode.get("strategy", "")).strip() == expected_strategy
        )
        and str(episode.get("mutation", "")).strip()
    ][:3]
    if not visible:
        return ["\\traceempty{Keine strategiespezifische LTM-Episode.}"]

    lines: list[str] = []
    for index, episode in enumerate(visible, start=1):
        lines.append(f"\\traceepisodeheading{{Positive RAG-Episode {index}}}")
        lines.append(
            "\\tracefield{Ergebnis}{"
            + _latex_escape(str(episode.get("success_label", "")).lower())
            + "}"
        )
        lines.append(
            "\\tracefield{Prompt (max. 250 Zeichen)}{"
            + _latex_escape(str(episode.get("mutation", "")).strip()[:250])
            + "}"
        )
        lines.append(
            "\\tracefield{Werkzeugeffekt (max. 120 Zeichen)}{"
            + _latex_escape(str(episode.get("tool_effect", ""))[:120])
            + "}"
        )
    return lines


def _episode_strategy(episode: dict[str, Any]) -> str:
    explicit = str(episode.get("strategy", "")).strip().lower()
    if explicit:
        return explicit
    mutation = str(episode.get("mutation", "")).strip()
    if not mutation.startswith("[strategy="):
        return "unknown"
    return mutation.split("]", 1)[0].replace("[strategy=", "", 1).strip()


def _render_tools(tools: list[dict[str, Any]]) -> list[str]:
    if not tools:
        return [
            "\\tracefield{Zielagent "
            "\\texorpdfstring{$\\rightarrow$}{->} ToolBroker}{"
            "Kein Werkzeugaufruf.}"
        ]

    lines: list[str] = []
    for index, tool in enumerate(tools, start=1):
        call_label = (
            "Zielagent \\texorpdfstring{$\\rightarrow$}{->} ToolBroker"
            if len(tools) == 1
            else f"Zielagent -> ToolBroker, Aufruf {index}"
        )
        result_label = (
            "Werkzeug \\texorpdfstring{$\\rightarrow$}{->} Zielagent"
            if len(tools) == 1
            else f"Werkzeug -> Zielagent, Ergebnis {index}"
        )
        call = ".".join(
            value
            for value in (
                str(tool.get("tool", "")).strip(),
                str(tool.get("operation", "")).strip(),
            )
            if value
        )
        call_details = [f"Aufruf: {call}"]
        raw_input = str(tool.get("raw_input", "")).strip()
        if raw_input:
            call_details.append(f"Rohe Eingabe: {raw_input}")
        target = str(tool.get("target", "")).strip()
        if target:
            call_details.append(f"Aufgelöstes Ziel: {target}")
        lines.append(
            "\\tracefield{"
            + call_label
            + "}{"
            + _latex_escape("\n".join(call_details))
            + "}"
        )

        result_details: list[str] = []
        output = tool.get("output")
        if output not in (None, ""):
            result_details.append(f"Ausgabe: {output}")
        error = str(tool.get("error") or "").strip()
        if error:
            result_details.append(f"Fehler: {error}")
        if not result_details:
            result_details.append("Keine Ausgabe und kein Fehler.")
        lines.append(
            "\\tracefield{"
            + result_label
            + "}{"
            + _latex_escape("\n".join(result_details))
            + "}"
        )
    return lines


def _render_policy(row: dict[str, Any]) -> str:
    parts = [
        f"Eingabe: {row['input_decision'] or 'nicht protokolliert'}",
        "Werkzeug: "
        + (
            ", ".join(str(item) for item in row["policy_decisions"])
            if row["policy_decisions"]
            else "kein Aufruf"
        ),
        f"Ausgabe: {row['output_decision'] or 'nicht protokolliert'}",
    ]
    if row["input_reason"]:
        parts.append(f"Eingabegrund: {row['input_reason']}")
    if row["output_reason"]:
        parts.append(f"Ausgabegrund: {row['output_reason']}")
    return _latex_escape("\n".join(parts))


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "#": r"\#",
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "_": r"\_",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
        "`": r"\textasciigrave{}",
        "\n": r"\newline{}",
    }
    return "".join(replacements.get(character, character) for character in value)


if __name__ == "__main__":
    main()

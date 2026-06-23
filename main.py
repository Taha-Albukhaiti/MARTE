from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import asdict
from pathlib import Path

from marte.cli.backends import (
    _DEFAULT_ATTACK_MODEL,
    _DEFAULT_TARGET_MODEL,
    _build_api_backend,
    _build_role_backend_config,
    _default_role_model,
    _format_cli_error,
    _has_role_backend_config,
    _role_model_cfg,
)
from marte.cli.component_factory import _build_components, _load_rag_settings
from marte.cli.conditions import _filter_conditions
from marte.cli.env import (
    _autoload_project_env,
    _configure_logging,
    _first_non_empty,
    _resolve_project_path,
    _seed_paths,
)
from marte.cli.reporting import _handle_experiment_results, _json_sanitize, _print_experiment_summary
from marte.cli.run_adapter import _create_run_adapter
from marte.cli.types import Mode
from marte.experiment.local_backends import LocalAttackBackend, LocalTargetBackend
from marte.experiment.runner import ComparativeRunner, all_conditions, format_condition
from marte.llm.phase_retry import PhaseRetryPolicy
from marte.memory.calibration import RAG_CALIBRATION_VERSION
from marte.target.sandbox import SandboxBoundary


__all__ = ["main", "run_cli", "_json_sanitize"]


def run_cli(runs_per_condition: int, max_budget: int, db_path: Path, mode: Mode = "local",
            attack_model: str | None = None, judge_model: str | None = None, target_model: str | None = None,
            conditions_filter: list[str] | None = None, allow_incomplete_runs: bool = False,
            log_level: str | None = None, resume_experiment_id: str | None = None) -> str:
    """Startet ein MARTE-thesis-worktree-Experiment aus CLI- und Umgebungsparametern.

    Die Funktion baut nur die Infrastruktur auf: Seed-Pfade, Sandbox,
    LLM-Backends, Memory, TargetAgent, Orchestrator und EventStore.
    Die äußere Experiment-Schleife läuft danach im ComparativeRunner.

    Args:
        runs_per_condition: Anzahl der Runs pro Bedingung.
        max_budget: Maximale Anzahl Angriffsschritte pro Run.
        db_path: SQLite-Datei für EventStore-Ausgaben.
        mode: Ausführungsmodus: lokale Stubs ("local") oder echte API-Backends ("api").
        attack_model: Modell für Planner und Mutator.
        judge_model: Modell für den Judge; fällt ohne Wert auf das Attack-Modell zurück.
        target_model: Modell für den TargetAgent.
        conditions_filter: CLI-Filter für Bedingungen, z. B. "sql" oder "stm_only".
        allow_incomplete_runs: Erlaubt den Start trotz offener Run-Lifecycle-Events.
        log_level: Aktiviert optionale Terminal-Logs.

    Returns:
        Die vom ComparativeRunner erzeugte experiment_id.
    """
    _autoload_project_env()
    if log_level:
        _configure_logging(log_level)

    # Jeder Run startet aus denselben Seed-Daten. fehlende Seeds würden das Experiment verfälschen.
    seed_sql_path, seed_fs_path = _seed_paths()
    if not seed_sql_path.exists():
        raise FileNotFoundError(
            f"Seed SQL file not found: {seed_sql_path}. "
            "Set MARTE_SEED_SQL_PATH or restore sandbox_data/sql/seed.sql."
        )
    if not seed_fs_path.exists():
        raise FileNotFoundError(
            f"Seed FS directory not found: {seed_fs_path}. "
            "Set MARTE_SEED_FS_PATH or restore sandbox_data/fs/seed."
        )

    # Relative Ausgabepfade werden am Projektroot aufgelöst, unabhängig vom aktuellen Arbeitsverzeichnis.
    db_path = _resolve_project_path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Modellnamen werden pro Rolle aufgelöst: CLI-Wert → Umgebungsvariable → Projektdefault.
    resolved_attack_model = (attack_model or _default_role_model("ATTACK")).strip() or _DEFAULT_ATTACK_MODEL
    resolved_judge_model = (
            judge_model
            or _first_non_empty(os.getenv("JUDGE_LLM_MODEL"), resolved_attack_model)
            or resolved_attack_model
    ).strip()
    resolved_target_model = (target_model or _default_role_model("TARGET")).strip() or _DEFAULT_TARGET_MODEL

    # Die Sandbox kapselt DB und Dateisystem. Zurückgesetzt wird sie später vor jedem einzelnen Run.
    sandbox = SandboxBoundary(seed_sql_path=seed_sql_path, seed_fs_path=seed_fs_path,
                              runtime_dir=_resolve_project_path(Path("runtime")))

    # "local" nutzt deterministische Test-Backends, "api" echte LLM-Provider aus der Umgebung.
    if mode == "local":
        attack_backend = LocalAttackBackend()
        judge_backend = LocalAttackBackend()
        target_backend = LocalTargetBackend()
    else:
        # API-Backends werden pro Rolle gebaut, mit Fallbacks für Judge → Attack. Planner und Mutator nutzen dasselbe Modell.
        attack_backend = _build_api_backend("ATTACK")
        judge_backend = _build_api_backend("JUDGE") if _has_role_backend_config("JUDGE") else attack_backend
        target_backend = _build_api_backend("TARGET")

    # Die Komponenten bilden die MARTE-thesis-worktree-Ausführung: Attack Loop, Memory, Target und EventStore.
    orchestrator, memory_gateway, target_agent, event_store, planner, mutator, judge = _build_components(sandbox=sandbox, db_path=db_path,
                                                                                                        attack_backend=attack_backend,
                                                                                                        judge_backend=judge_backend,
                                                                                                        target_backend=target_backend,
                                                                                                        attack_model=resolved_attack_model,
                                                                                                        judge_model=resolved_judge_model,
                                                                                                        mode=mode)

    # Aus CLI-Filtern wird die tatsächlich auszuführende Bedingungsmatrix gebildet.
    conditions = _filter_conditions(conditions_filter or ["all"], all_conditions())
    total_runs = runs_per_condition * len(conditions)
    _print_experiment_summary(runs_per_condition, total_runs, max_budget, len(conditions))

    # Modellkonfiguration für den TargetAgent (Temperatur etc.), wird in jeden ConfigSnapshot kopiert.
    target_model_cfg = _role_model_cfg("TARGET", resolved_target_model, default_temperature=0.0)
    experiment_config = _build_experiment_manifest_config(
        mode=mode,
        attack_model=resolved_attack_model,
        judge_model=resolved_judge_model,
        target_model=resolved_target_model,
    )

    # Der Runner iteriert nur über Bedingungen und Seeds.
    # Der Adapter übersetzt jeden ConfigSnapshot in einen konkreten MARTE-thesis-worktree-Run:
    # Sandbox resetten, Memory-Modus setzen, Target konfigurieren, Orchestrator starten.
    run_adapter = _create_run_adapter(sandbox=sandbox, memory_gateway=memory_gateway, target_agent=target_agent,
                                      planner=planner, mutator=mutator, judge=judge,
                                      orchestrator=orchestrator, conditions=conditions,
                                      runs_per_condition=runs_per_condition, execution_mode=mode)

    # Durch diese Methodenübergabe bleibt die Vergleichsschleife unabhängig von Target-, Sandbox- und Orchestrator-Details.
    runner = ComparativeRunner(prepare_run=run_adapter.prepare_run, run_callback=run_adapter.run_once,
                               memory_gateway=memory_gateway, event_store=event_store, model_cfg=target_model_cfg,
                               tool_allowlist=["sql", "file"], allow_incomplete_runs=allow_incomplete_runs,
                               experiment_config=experiment_config)

    # Führt die komplette Matrix aus: Bedingungen × Seeds × maximal max_budget Schritte.
    experiment_id = runner.run_experiment(runs_per_condition=runs_per_condition, max_budget=max_budget,
                                          conditions=conditions,
                                          resume_experiment_id=resume_experiment_id)

    # Nach dem Lauf werden Kennzahlen berechnet und die Ergebnisdatenbank gemeldet.
    _handle_experiment_results(event_store, experiment_id, db_path)
    return experiment_id


def _build_experiment_manifest_config(
    *,
    mode: Mode,
    attack_model: str,
    judge_model: str,
    target_model: str,
) -> dict[str, object]:
    """Build a secret-free fixed configuration used to validate resume."""
    role_models = {
        "attack": _role_model_cfg(
            "ATTACK",
            attack_model,
            default_temperature=0.7,
        ),
        "judge": _role_model_cfg(
            "JUDGE",
            judge_model,
            default_temperature=0.0,
        ),
        "target": _role_model_cfg(
            "TARGET",
            target_model,
            default_temperature=0.0,
        ),
    }
    if mode == "api":
        role_backends = {
            role.lower(): _secret_free_backend_config(
                "ATTACK" if role == "JUDGE" and not _has_role_backend_config("JUDGE") else role
            )
            for role in ("ATTACK", "JUDGE", "TARGET")
        }
    else:
        role_backends = {
            role: {"provider": "local_stub"}
            for role in ("attack", "judge", "target")
        }

    rag_settings = asdict(_load_rag_settings(mode))
    rag_settings.update(
        {
            "collection_base": _first_non_empty(
                os.getenv("MARTE_RAG_COLLECTION"),
                "marte_episodes",
            ),
            "persist_path": str(
                _resolve_project_path(
                    Path(
                        _first_non_empty(
                            os.getenv("MARTE_CHROMA_PATH"),
                            "runtime/chroma",
                        )
                        or "runtime/chroma"
                    )
                )
            ),
            "embedding_model": "all-MiniLM-L6-v2",
            "calibration_version": RAG_CALIBRATION_VERSION,
        }
    )
    retry_policy = PhaseRetryPolicy.from_env()
    seed_sql_path, seed_fs_path = _seed_paths()
    return {
        "execution_mode": mode,
        "role_models": role_models,
        "role_backends": role_backends,
        "rag": rag_settings,
        "phase_retry": {
            "max_attempts": retry_policy.max_attempts,
            "backoff_s": retry_policy.backoff_s,
            "max_backoff_s": retry_policy.max_backoff_s,
        },
        "sandbox_seeds": {
            "sql_path": str(seed_sql_path),
            "sql_sha256": _path_fingerprint(seed_sql_path),
            "fs_path": str(seed_fs_path),
            "fs_sha256": _path_fingerprint(seed_fs_path),
        },
        "protocol_fingerprint": _protocol_fingerprint(),
    }


def _secret_free_backend_config(prefix: str) -> dict[str, object]:
    config = _build_role_backend_config(prefix)
    return {
        "provider": config.provider,
        "base_url": config.base_url,
        "timeout_s": config.timeout_s,
        "max_retries": config.max_retries,
        "retry_backoff_s": config.retry_backoff_s,
        "min_interval_s": config.min_interval_s,
    }


def _protocol_fingerprint() -> str:
    """Hash executable source, prompts, project metadata, and seed fixtures."""
    project_root = Path(__file__).resolve().parent
    roots = (
        project_root / "main.py",
        project_root / "pyproject.toml",
        project_root / "src" / "marte",
        project_root / "sandbox_data",
    )
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in {".py", ".txt", ".json", ".sql"}
        )
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _path_fingerprint(root: Path) -> str:
    """Hash one seed file or every file below a seed directory."""
    resolved = root.resolve()
    files = [resolved] if resolved.is_file() else [
        path for path in resolved.rglob("*") if path.is_file()
    ]
    digest = hashlib.sha256()
    for path in sorted(files):
        label = path.name if resolved.is_file() else str(path.relative_to(resolved))
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    _autoload_project_env()
    parser = argparse.ArgumentParser(description="Run MARTE-thesis-worktree comparative experiment.")
    parser.add_argument(
        "--runs-per-condition",
        type=int,
        default=30,
        help="Number of runs per condition (default: 30).",
    )
    parser.add_argument(
        "--max-budget",
        type=int,
        default=50,
        help="Max attack steps per run (default: 50).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="results/events.db",
        help="SQLite path for EventStore output (default: results/events.db).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "api"],
        default="local",
        help="Execution mode: local stubs or real API backends (default: local).",
    )
    parser.add_argument(
        "--attack-model",
        type=str,
        default=None,
        help=(
            "Model name for Planner and Mutator "
            f"(default: ATTACK_LLM_MODEL or LLM_MODEL from env, fallback: {_DEFAULT_ATTACK_MODEL})."
        ),
    )
    parser.add_argument(
        "--target-model",
        type=str,
        default=None,
        help=(
            "Model name for TargetAgent "
            f"(default: TARGET_LLM_MODEL or LLM_MODEL from env, fallback: {_DEFAULT_TARGET_MODEL})."
        ),
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=(
            "Model name for Judge "
            "(default: JUDGE_LLM_MODEL from env, fallback: resolved attack model)."
        ),
    )
    _all_cond_names = [format_condition(s, g, m) for s, g, m in all_conditions()]
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["all"],
        metavar="TOKEN",
        help=(
            "Which conditions to run. One or more of: "
            "all (default), stm_only, rag_ltm, sql, file, moderate, enterprise, "
            f"or an exact name ({', '.join(_all_cond_names)}). "
            "Examples: --conditions stm_only  |  --conditions sql  |  --conditions sql_moderate_stm_only"
        ),
    )
    parser.add_argument(
        "--allow-incomplete-runs",
        action="store_true",
        help=(
            "Allow starting even if the EventStore contains started runs without "
            "completed/aborted lifecycle events. Use only after inspecting "
            "run_events and VectorDB state."
        ),
    )
    parser.add_argument(
        "--resume-experiment-id",
        type=str,
        default=None,
        help=(
            "Resume an interrupted experiment in the same database and Chroma "
            "storage. Conditions may be added in blocks. Models, N, B, RAG, retry, "
            "sandbox, and protocol configuration must remain unchanged."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING"],
        default=None,
        metavar="LEVEL",
        help=(
            "Enable terminal logging at the given level (DEBUG, INFO, WARNING). "
            "INFO shows step lifecycle, strategy, oracle/judge verdicts, tool calls. "
            "DEBUG additionally shows full LLM requests and responses. "
            "Without this flag no log output is produced."
        ),
    )
    args = parser.parse_args()

    try:
        run_cli(runs_per_condition=args.runs_per_condition, max_budget=args.max_budget, db_path=Path(args.db_path),
                mode=args.mode, attack_model=args.attack_model, judge_model=args.judge_model,
                target_model=args.target_model, conditions_filter=args.conditions,
                allow_incomplete_runs=args.allow_incomplete_runs, log_level=args.log_level,
                resume_experiment_id=args.resume_experiment_id)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {_format_cli_error(exc)}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

"""Component factory helpers for the MARTE-thesis-worktree CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from marte.attack.judge import Judge
from marte.attack.mutator import Mutator
from marte.attack.oracle import BreachOracle
from marte.attack.orchestrator import RunOrchestrator
from marte.attack.planner import Planner
from marte.attack.target_executor import TargetExecutor
from marte.cli.backends import _OPTIONAL_EXTRAS_INSTALL_HINT, _is_optional_dependency_error, _role_model_cfg
from marte.cli.env import _first_non_empty, _resolve_project_path
from marte.cli.types import Mode
from marte.experiment.local_backends import DeterministicHashEmbeddingBackend
from marte.memory.calibration import (
    CALIBRATED_BLOCKED_MIN_SIMILARITY,
    CALIBRATED_POSITIVE_MIN_SIMILARITY,
)
from marte.memory.switching_gateway import SwitchingMemoryGateway
from marte.persistence.event_store import EventStore
from marte.target.agent import TargetAgent
from marte.target.sandbox import SandboxBoundary

if TYPE_CHECKING:
    from marte.memory.rag_adapter import RAGAdapter


@dataclass(frozen=True)
class _RAGSettings:
    positive_top_k: int
    blocked_top_k: int
    positive_fetch_k: int
    blocked_fetch_k: int
    positive_min_similarity: float | None
    blocked_min_similarity: float | None
    require_calibrated_thresholds: bool


def _build_components(sandbox: SandboxBoundary, db_path: Path, attack_backend, judge_backend,
                      target_backend, attack_model: str, judge_model: str, mode: Mode) -> tuple[
    RunOrchestrator, SwitchingMemoryGateway, TargetAgent, EventStore, Planner, Mutator, Judge]:
    """Build all components used by one CLI-launched experiment."""
    event_store = EventStore(db_path)
    oracle = BreachOracle()
    rag_strategy = _build_rag_strategy(mode)
    memory_gateway = SwitchingMemoryGateway(rag_strategy=rag_strategy)
    target_agent = TargetAgent(sandbox=sandbox, model_backend=target_backend)

    attack_model_cfg = _role_model_cfg("ATTACK", attack_model, default_temperature=0.7)
    judge_model_cfg = _role_model_cfg("JUDGE", judge_model, default_temperature=0.0)

    planner = Planner(model_backend=attack_backend, model_cfg=attack_model_cfg)
    mutator = Mutator(model_backend=attack_backend, model_cfg=attack_model_cfg)
    executor = TargetExecutor(target_agent=target_agent)
    judge = Judge(memory_gateway=memory_gateway, model_backend=judge_backend, model_cfg=judge_model_cfg,
                  strict_memory_errors=True)

    orchestrator = RunOrchestrator(planner=planner, mutator=mutator, executor=executor, judge=judge,
                                   snapshot_provider=sandbox, oracle=oracle, memory_gateway=memory_gateway)
    return orchestrator, memory_gateway, target_agent, event_store, planner, mutator, judge


def _build_rag_strategy(mode: Mode) -> RAGAdapter:
    rag_collection = _first_non_empty(os.getenv("MARTE_RAG_COLLECTION"), "marte_episodes")
    settings = _load_rag_settings(mode)
    rag_path_raw = _first_non_empty(os.getenv("MARTE_CHROMA_PATH"), "runtime/chroma") or "runtime/chroma"
    rag_path = _resolve_project_path(Path(rag_path_raw))
    rag_path.mkdir(parents=True, exist_ok=True)

    try:
        return _create_rag_adapter(
            mode,
            rag_collection or "marte_episodes",
            settings,
            rag_path,
        )
    except Exception as exc:  # noqa: BLE001
        if _is_optional_dependency_error(exc):
            raise
        if mode == "local" and "MARTE_CHROMA_PATH" not in os.environ:
            return _recover_local_rag_adapter(
                mode,
                rag_collection or "marte_episodes",
                settings,
                rag_path,
                exc,
            )
        raise RuntimeError(
            "Failed to initialize persistent RAG storage at "
            f"{rag_path}. If this directory contains old/incompatible Chroma data, "
            "delete it or choose a new path via MARTE_CHROMA_PATH."
        ) from exc


def _recover_local_rag_adapter(
    mode: Mode,
    collection_name: str,
    settings: _RAGSettings,
    failed_path: Path,
    original_exc: BaseException,
) -> RAGAdapter:
    fallback_path = _resolve_project_path(Path("runtime/chroma_local_autorecover"))
    fallback_path.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Warning: primary Chroma path is incompatible, switching to {fallback_path}")
        return _create_rag_adapter(mode, collection_name, settings, fallback_path)
    except Exception as inner_exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to initialize persistent RAG storage at "
            f"{failed_path} and fallback {fallback_path}. "
            "Set MARTE_CHROMA_PATH to a clean directory."
        ) from inner_exc


def _create_rag_adapter(
    mode: Mode,
    collection_name: str,
    settings: _RAGSettings,
    persist_path: Path,
) -> RAGAdapter:
    try:
        from marte.memory.rag_adapter import RAGAdapter
    except ModuleNotFoundError as exc:
        if exc.name == "chromadb":
            raise RuntimeError(
                "Missing optional RAG dependency 'chromadb'. "
                f"Install the full experiment extras with: {_OPTIONAL_EXTRAS_INSTALL_HINT}"
            ) from exc
        raise

    kwargs = {
        "collection_name": collection_name,
        "top_k": settings.positive_top_k,
        "positive_top_k": settings.positive_top_k,
        "blocked_top_k": settings.blocked_top_k,
        "positive_fetch_k": settings.positive_fetch_k,
        "blocked_fetch_k": settings.blocked_fetch_k,
        "positive_min_similarity": settings.positive_min_similarity,
        "blocked_min_similarity": settings.blocked_min_similarity,
        "require_calibrated_thresholds": settings.require_calibrated_thresholds,
        "persist_path": persist_path,
    }
    if mode == "local":
        kwargs["embedding_backend"] = DeterministicHashEmbeddingBackend()
    return RAGAdapter(**kwargs)


def _load_rag_settings(mode: Mode) -> _RAGSettings:
    diagnostic_mode = _parse_bool(os.getenv("MARTE_RAG_DIAGNOSTIC_MODE"))
    positive_threshold = _parse_similarity(
        "MARTE_RAG_POSITIVE_MIN_SIMILARITY",
        os.getenv("MARTE_RAG_POSITIVE_MIN_SIMILARITY"),
    )
    blocked_threshold = _parse_similarity(
        "MARTE_RAG_BLOCKED_MIN_SIMILARITY",
        os.getenv("MARTE_RAG_BLOCKED_MIN_SIMILARITY"),
    )
    if not diagnostic_mode:
        positive_threshold = _frozen_similarity(
            "MARTE_RAG_POSITIVE_MIN_SIMILARITY",
            positive_threshold,
            CALIBRATED_POSITIVE_MIN_SIMILARITY,
        )
        blocked_threshold = _frozen_similarity(
            "MARTE_RAG_BLOCKED_MIN_SIMILARITY",
            blocked_threshold,
            CALIBRATED_BLOCKED_MIN_SIMILARITY,
        )
    positive_top_k = _parse_positive_int(
        "MARTE_RAG_POSITIVE_TOP_K",
        os.getenv("MARTE_RAG_POSITIVE_TOP_K"),
        default=3,
    )
    settings = _RAGSettings(
        positive_top_k=positive_top_k,
        blocked_top_k=_parse_positive_int(
            "MARTE_RAG_BLOCKED_TOP_K",
            os.getenv("MARTE_RAG_BLOCKED_TOP_K"),
            default=1,
        ),
        positive_fetch_k=_parse_positive_int(
            "MARTE_RAG_POSITIVE_FETCH_K",
            os.getenv("MARTE_RAG_POSITIVE_FETCH_K"),
            default=12,
        ),
        blocked_fetch_k=_parse_positive_int(
            "MARTE_RAG_BLOCKED_FETCH_K",
            os.getenv("MARTE_RAG_BLOCKED_FETCH_K"),
            default=4,
        ),
        positive_min_similarity=positive_threshold,
        blocked_min_similarity=blocked_threshold,
        require_calibrated_thresholds=mode == "api" and not diagnostic_mode,
    )
    if settings.positive_fetch_k < settings.positive_top_k:
        raise ValueError(
            "MARTE_RAG_POSITIVE_FETCH_K must be greater than or equal to "
            "MARTE_RAG_POSITIVE_TOP_K."
        )
    if settings.blocked_fetch_k < settings.blocked_top_k:
        raise ValueError(
            "MARTE_RAG_BLOCKED_FETCH_K must be greater than or equal to "
            "MARTE_RAG_BLOCKED_TOP_K."
        )
    return settings


def _parse_positive_int(name: str, raw: str | None, *, default: int) -> int:
    text = str(raw or "").strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _parse_similarity(name: str, raw: str | None) -> float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float in [0, 1], got {raw!r}.") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
    return value


def _frozen_similarity(name: str, value: float | None, calibrated: float) -> float:
    if value is None:
        return calibrated
    if value != calibrated:
        raise ValueError(
            f"{name} is frozen at {calibrated!r}. "
            "Set MARTE_RAG_DIAGNOSTIC_MODE=1 for retrieval diagnostics."
        )
    return calibrated


def _parse_bool(raw: str | None) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}

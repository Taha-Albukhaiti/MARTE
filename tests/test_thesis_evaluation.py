from __future__ import annotations

import math
from pathlib import Path

import pytest

from marte.persistence.event_store import EventStore
from marte.persistence.schemas import ExperimentManifest, RunEvent, StepEvent, StopEvent
from scripts import build_thesis_evaluation as thesis

EXP = "exp-thesis"
STM = "sql_moderate_stm_only"
RAG = "sql_moderate_rag_ltm"


def _source(db_path: Path, expected_runs: int = 2) -> thesis.SourceSpec:
    return thesis.SourceSpec(
        block="sql_moderate",
        db_path=db_path,
        experiment_id=EXP,
        stm_condition=STM,
        rag_condition=RAG,
        expected_runs=expected_runs,
        max_budget=5,
    )


def _manifest(store: EventStore) -> None:
    store.write_experiment_manifest(
        ExperimentManifest(
            experiment_id=EXP,
            config={
                "schema_version": 1,
                "runs_per_condition": 2,
                "max_budget": 5,
                "experiment_config": {
                    "protocol_fingerprint": "test-fingerprint",
                    "role_models": {
                        "attack": {"model": "attack-test"},
                        "judge": {"model": "judge-test"},
                        "target": {"model": "target-test"},
                    },
                },
            },
            created_at_ms=123,
        )
    )


def _run_event(
    run_id: int,
    condition: str,
    seed: int,
    status: str,
    *,
    stop_reason: str = "",
    last_step_t: int = 0,
    error: str = "",
) -> RunEvent:
    return RunEvent(
        experiment_id=EXP,
        run_id=run_id,
        condition=condition,
        seed=seed,
        status=status,
        timestamp_ms=1000 + run_id,
        last_step_t=last_step_t,
        stop_reason=stop_reason,
        error=error,
    )


def _stop(
    run_id: int,
    condition: str,
    seed: int,
    *,
    stop_reason: str = "budget_exhausted",
    q2b: int = 5,
    censored: bool = True,
    timestamp_ms: int = 500,
) -> StopEvent:
    return StopEvent(
        experiment_id=EXP,
        run_id=run_id,
        condition=condition,
        stop_reason=stop_reason,
        q2b=q2b,
        censored_flag=censored,
        timestamp_ms=timestamp_ms,
        seed=seed,
    )


def _step(
    run_id: int,
    condition: str,
    step_t: int,
    *,
    oracle: bool = False,
    judge: str = "fail",
    prompt: str = "prompt",
    policy_decisions: list[str] | None = None,
    tool_trace: list[dict[str, object]] | None = None,
    retrieval_trace: dict[str, object] | None = None,
) -> StepEvent:
    return StepEvent(
        experiment_id=EXP,
        run_id=run_id,
        condition=condition,
        step_t=step_t,
        timestamp_ms=step_t * 10,
        prompt=prompt,
        response="response",
        tool_trace=tool_trace or [],
        policy_decisions=policy_decisions or [],
        oracle_verdict=oracle,
        judge_verdict=judge,
        retrieval_trace=retrieval_trace or {},
    )


def _write_completed_run(
    store: EventStore,
    run_id: int,
    condition: str,
    seed: int,
    *,
    breach: bool = False,
    q2b: int = 5,
    prompt: str = "prompt",
) -> None:
    store.write_run_event(_run_event(run_id, condition, seed, "started"))
    for step_t in range(1, q2b + 1):
        is_breach_step = breach and step_t == q2b
        store.write_step_event(
            _step(
                run_id,
                condition,
                step_t,
                oracle=is_breach_step,
                judge="success" if is_breach_step else "fail",
                prompt=prompt,
            )
        )
    stop = _stop(
        run_id,
        condition,
        seed,
        stop_reason="breach_stop" if breach else "budget_exhausted",
        q2b=q2b,
        censored=not breach,
    )
    store.write_stop_event(stop)
    store.write_run_event(
        _run_event(
            run_id,
            condition,
            seed,
            "completed",
            stop_reason=stop.stop_reason,
            last_step_t=q2b,
        )
    )


def test_holm_preserves_nan_and_adjusts_finite_values() -> None:
    adjusted = thesis.holm_adjust_p_values([0.01, float("nan"), 0.03, 0.20])

    assert adjusted[0] == pytest.approx(0.03)
    assert math.isnan(adjusted[1])
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[3] == pytest.approx(0.20)


def test_holm_can_use_planned_family_size_for_unavailable_tests() -> None:
    adjusted = thesis.holm_adjust_p_values(
        [0.01, float("nan"), 0.03, 0.20],
        planned_m=4,
    )

    assert adjusted[0] == pytest.approx(0.04)
    assert math.isnan(adjusted[1])
    assert adjusted[2] == pytest.approx(0.09)
    assert adjusted[3] == pytest.approx(0.40)


def test_fisher_odds_ratio_is_rag_relative_to_stm() -> None:
    odds_ratio, p_value = thesis.fisher_rag_vs_stm(
        rag_successes=4,
        rag_total=4,
        stm_successes=1,
        stm_total=4,
    )

    assert math.isinf(odds_ratio)
    assert 0.0 <= p_value <= 1.0


def test_pairwise_comparison_handles_all_censored_survival(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    store = EventStore(db_path)
    _manifest(store)
    _write_completed_run(store, 1, STM, seed=1)
    _write_completed_run(store, 2, RAG, seed=1)
    _write_completed_run(store, 3, RAG, seed=2)
    _write_completed_run(store, 4, STM, seed=2)

    source = _source(db_path)
    payload = thesis.load_block_payload(source)
    row = thesis.build_pairwise_comparison(source, payload)

    assert row["stm_breach"] == 0
    assert row["rag_breach"] == 0
    assert math.isnan(row["asr_odds_ratio_rag_vs_stm"])
    assert math.isnan(row["q2b_logrank_p"])
    assert math.isnan(row["t2b_logrank_p"])


def test_block_validation_detects_bad_stops_and_oracle_mismatch(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    store = EventStore(db_path)
    _manifest(store)
    _write_completed_run(store, 1, STM, seed=1, breach=True, q2b=2)
    _write_completed_run(store, 2, RAG, seed=1)
    store.write_stop_event(
        _stop(
            2,
            RAG,
            seed=1,
            stop_reason="budget_exhausted",
            q2b=4,
            censored=False,
        )
    )

    source = _source(db_path)
    payload = thesis.load_block_payload(source)
    codes = {issue.code for issue in thesis.validate_block(source, payload)}

    assert "duplicate_condition_seed" in codes
    assert "budget_not_censored" in codes
    assert "budget_q2b_mismatch" in codes
    assert "step_sequence_mismatch" in codes


def test_lifecycle_marks_recovered_and_unrecovered_aborts(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    store = EventStore(db_path)
    _manifest(store)
    _write_completed_run(store, 1, STM, seed=1)
    _write_completed_run(store, 2, RAG, seed=1)
    store.write_run_event(_run_event(99, RAG, seed=1, status="aborted", error="retry ok"))
    store.write_run_event(_run_event(100, RAG, seed=2, status="aborted", error="lost run"))

    source = _source(db_path)
    payload = thesis.load_block_payload(source)
    issues = thesis.validate_run_lifecycle(source, payload["stops"], payload["run_events"])
    by_code = {issue.code: issue for issue in issues}

    assert by_code["recovered_run_attempt"].level == "warning"
    assert by_code["unrecovered_run_attempt"].level == "error"


def test_ltm_audit_matches_reconstructed_completed_rag_episodes(tmp_path: Path) -> None:
    chromadb = pytest.importorskip("chromadb")
    db_path = tmp_path / "events.db"
    chroma_path = tmp_path / "chroma"
    store = EventStore(db_path)
    _manifest(store)
    store.write_run_event(_run_event(1, RAG, seed=1, status="started"))
    store.write_step_event(
        _step(
            1,
            RAG,
            1,
            oracle=False,
            judge="partial",
            prompt="[strategy=table_read] SELECT id, role FROM users;",
            tool_trace=[
                {
                    "tool": "sql",
                    "operation": "SELECT",
                    "target": "users",
                    "raw_input": "SELECT id, role FROM users;",
                    "output": "[]",
                    "error": None,
                }
            ],
            policy_decisions=["allow"],
        )
    )
    store.write_stop_event(_stop(1, RAG, seed=1, stop_reason="budget_exhausted", q2b=1, censored=True))
    store.write_run_event(_run_event(1, RAG, seed=1, status="completed", stop_reason="budget_exhausted", last_step_t=1))

    source = _source(db_path, expected_runs=1)
    payload = thesis.load_block_payload(source)
    expected_ids = thesis.expected_ltm_document_ids(source, payload)
    assert len(expected_ids) == 1

    collection_name = thesis.ltm_scope_name(EXP, RAG)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(collection_name)
    collection.add(
        ids=list(expected_ids),
        embeddings=[[0.0, 0.0]],
        documents=["episode"],
        metadatas=[{"run_id": 1, "step_t": 1, "memory_kind": "positive"}],
    )

    row, issues = thesis.audit_chroma_ltm(source, payload, chroma_path)

    assert issues == []
    assert row["status"] == "passed"
    assert row["expected_document_count"] == 1
    assert row["actual_document_count"] == 1

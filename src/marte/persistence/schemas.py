from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class StepEvent(BaseModel):
    """Single attack step written to the step_events table in EventStore.

    Fields mirror the SQLite schema defined in ARCHITECTURE.md §EventStore.
    ``event_id`` is None before insertion; SQLite assigns it on write.
    """

    event_id: Optional[int] = Field(default=None, description="Auto-assigned by SQLite on insert.")
    experiment_id: str = Field(description="Unique ID of the experiment run batch.")
    run_id: int = Field(description="Sequential run number within the experiment.")
    condition: str = Field(description="Condition string, e.g. 'sql_moderate_stm_only'.")
    step_t: int = Field(description="Step number within the run (1-indexed).")
    timestamp_ms: int = Field(description="Milliseconds since run start.")
    prompt: str = Field(description="Attacker prompt sent to the target.")
    response: str = Field(description="Target agent response text.")
    tool_trace: list[Any] = Field(default_factory=list, description="Tool call records.")
    policy_decisions: list[Any] = Field(default_factory=list, description="Guardrail decisions.")
    oracle_verdict: bool = Field(description="True if a breach was detected.")
    judge_verdict: str = Field(description="'success', 'partial', or 'fail'.")
    memory_context: dict[str, Any] = Field(default_factory=dict, description="STM/LTM context available to the planner at this step.")
    plan: dict[str, Any] = Field(default_factory=dict, description="Planner output used by the mutator.")
    pre_snapshot: dict[str, Any] = Field(default_factory=dict, description="Canonical sandbox snapshot before target execution.")
    post_snapshot: dict[str, Any] = Field(default_factory=dict, description="Canonical sandbox snapshot after target execution.")
    input_decision: str = Field(default="", description="Input guardrail decision recorded before target execution.")
    output_decision: str = Field(default="", description="Output guardrail decision recorded after target execution.")
    input_reason: str = Field(default="", description="Reason attached to the input guardrail decision.")
    output_reason: str = Field(default="", description="Reason attached to the output guardrail decision.")
    retrieval_trace: dict[str, Any] = Field(default_factory=dict, description="Auditable planner and mutator retrieval batches.")
    timings: dict[str, Any] = Field(default_factory=dict, description="Per-phase timing data for the step.")
    errors: list[Any] = Field(default_factory=list, description="Recoverable phase errors recorded during the step.")
    planner_debug: dict[str, Any] = Field(default_factory=dict, description="Planner raw output, fallback/error state, and diagnostics.")
    mutator_debug: dict[str, Any] = Field(default_factory=dict, description="Mutator raw output, fallback/error state, and diagnostics.")
    judge_debug: dict[str, Any] = Field(default_factory=dict, description="Judge raw output, Oracle-gating state, rationale, and diagnostics.")


class StopEvent(BaseModel):
    """Run-termination record written to the stop_events table in EventStore.

    All 8 fields mirror the SQLite schema defined in ARCHITECTURE.md §EventStore.
    ``censored_flag=True`` marks runs that exhausted the budget without breach —
    these are right-censored observations for Kaplan-Meier analysis.
    """

    experiment_id: str = Field(description="Unique ID of the experiment run batch.")
    run_id: int = Field(description="Sequential run number within the experiment.")
    condition: str = Field(description="Condition string, e.g. 'sql_moderate_rag_ltm'.")
    stop_reason: str = Field(description="'breach_stop' or 'budget_exhausted'.")
    q2b: int = Field(description="Queries to breach (or max_budget if censored).")
    censored_flag: bool = Field(description="True when run exhausted budget without breach.")
    timestamp_ms: int = Field(description="Run duration in milliseconds.")
    seed: int = Field(description="RNG seed used for this run (for reproducibility).")


class RunEvent(BaseModel):
    """Append-only lifecycle record for run audit and crash detection.

    ``run_events`` intentionally does not replace ``stop_events``. Completed
    runs still write exactly one StopEvent for metrics; lifecycle events add
    the audit trail needed to detect started-but-not-terminal runs after a
    process crash.
    """

    event_id: Optional[int] = Field(default=None, description="Auto-assigned by SQLite.")
    experiment_id: str = Field(description="Unique ID of the experiment run batch.")
    run_id: int = Field(description="Sequential run number within the experiment.")
    condition: str = Field(description="Condition string, e.g. 'sql_moderate_rag_ltm'.")
    seed: int = Field(description="RNG seed used for this run.")
    status: str = Field(description="'started', 'completed', 'aborted', or 'retry_aborted'.")
    timestamp_ms: int = Field(description="Wall-clock milliseconds since epoch.")
    last_step_t: int = Field(default=0, description="Last persisted step index for this run.")
    stop_reason: str = Field(default="", description="Final stop reason, if terminal.")
    error: str = Field(default="", description="Abort/error message, if any.")


class ExperimentManifest(BaseModel):
    """Immutable configuration recorded once for an experiment ID."""

    experiment_id: str = Field(description="Unique ID of the experiment batch.")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Canonical fixed configuration required for safe resume.",
    )
    created_at_ms: int = Field(description="Wall-clock creation time in milliseconds.")


class Episode(BaseModel):
    """Single episode stored in STM and (for rag_ltm) in VectorDB.

    Schema mirrors the VectorDB episode schema from ARCHITECTURE.md §Layer 3.
    The `embedding` field is optional: STM does not require it; RAGAdapter
    populates it before upsert into the vector store.
    """

    document_id: str = Field(description="Unique episode ID.")
    run_id: int = Field(description="Run this episode belongs to.")
    step_t: int = Field(description="Step number within the run.")
    scenario: str = Field(description="'sql' or 'file'.")
    guardrail_level: str = Field(description="'moderate' or 'enterprise'.")
    strategy: str = Field(default="", description="Planner strategy used for this episode.")
    mutation: str = Field(description="The attacker prompt sent to the target.")
    response_summary: str = Field(description="Short summary of the target's response.")
    tool_effect: str = Field(description="What the tool actually did (empty if no tool call).")
    success_label: str = Field(description="'success', 'partial', or 'fail'.")
    memory_kind: str = Field(default="positive", description="'positive' or 'blocked'.")
    policy_stage: str = Field(default="", description="'input', 'tool', or empty.")
    policy_reason: str = Field(default="", description="Guardrail reason for blocked episodes.")
    embedding: Optional[list[float]] = Field(default=None, description="Dense embedding vector; populated by RAGAdapter before VectorDB upsert.")

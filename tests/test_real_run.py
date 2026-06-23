from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from marte.attack.judge import Judge
from marte.attack.mutator import Mutator
from marte.attack.oracle import BreachOracle, SandboxPolicy
from marte.attack.orchestrator import RunOrchestrator
from marte.attack.planner import Planner
from marte.attack.target_executor import TargetExecutor
from marte.cli.env import _load_env_file_without_dotenv
from marte.memory.gateway import MemoryGateway
from marte.memory.rag_adapter import RAGAdapter
from marte.persistence.event_store import EventStore
from marte.persistence.schemas import StepEvent, StopEvent
from marte.target.agent import TargetAgent
from marte.target.guardrail import GuardrailLevel
from marte.target.model_adapter import OpenAIClientBackend
from marte.target.sandbox import SandboxBoundary

_ROOT = Path(__file__).resolve().parent.parent


_load_env_file_without_dotenv(_ROOT / ".env")


_RUN_REAL = os.environ.get("RUN_REAL_LLM_TESTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

pytestmark = pytest.mark.skipif(
    not _RUN_REAL,
    reason="Real LLM tests disabled. Set RUN_REAL_LLM_TESTS=1.",
)


@dataclass
class CountingBackend:
    backend: OpenAIClientBackend
    role: str
    calls: int = 0
    latency_ms_total: float = 0.0

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        t0 = time.monotonic()
        try:
            return self.backend.complete(messages, **kwargs)
        finally:
            self.calls += 1
            self.latency_ms_total += (time.monotonic() - t0) * 1000.0

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_total / self.calls if self.calls else 0.0


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _new_backend(role: str) -> OpenAIClientBackend:
    base_url = _env(f"{role}_LLM_BASE_URL", "LLM_BASE_URL")
    api_key = _env(
        f"{role}_LLM_API_KEY",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "MYLAB_API_KEY",
        default="ignored",
    )
    timeout_raw = _env(f"{role}_LLM_TIMEOUT_S", "LLM_TIMEOUT_S", default="60")
    timeout_s = float(timeout_raw)
    if not base_url:
        raise RuntimeError(f"Missing base URL for {role}. Set {role}_LLM_BASE_URL or LLM_BASE_URL.")
    return OpenAIClientBackend(base_url=base_url, api_key=api_key, timeout_s=timeout_s)


def test_real_single_run() -> None:
    runtime_dir = Path(tempfile.mkdtemp(prefix="marte_real_run_"))
    db_path = runtime_dir / "events.db"

    sandbox = SandboxBoundary(
        seed_sql_path=_ROOT / "sandbox_data" / "sql" / "seed.sql",
        seed_fs_path=_ROOT / "sandbox_data" / "fs" / "seed",
        runtime_dir=runtime_dir / "sandbox_runtime",
    )
    sandbox.reset_sandbox()

    attack_model = _env("ATTACK_LLM_MODEL", "LLM_MODEL", default="chat-medium")
    target_model = _env("TARGET_LLM_MODEL", "LLM_MODEL", default="chat-small")

    planner_backend = CountingBackend(_new_backend("ATTACK"), role="planner")
    mutator_backend = CountingBackend(_new_backend("ATTACK"), role="mutator")
    target_backend = CountingBackend(_new_backend("TARGET"), role="target")

    planner = Planner(
        model_backend=planner_backend,
        model_cfg={"model": attack_model, "temperature": 0.7, "seed": 42, "max_tokens": 512},
    )
    mutator = Mutator(
        model_backend=mutator_backend,
        model_cfg={"model": attack_model, "temperature": 0.7, "seed": 42, "max_tokens": 512},
    )

    target_agent = TargetAgent(sandbox=sandbox, model_backend=target_backend)
    target_agent.init_config(
        guardrail_level=GuardrailLevel.MODERATE,
        tool_allowlist=["sql"],
        model_cfg={"model": target_model, "temperature": 0.0, "seed": 42, "max_tokens": 512},
    )

    rag = RAGAdapter(
        collection_name=f"real_run_{int(time.time() * 1000)}",
        top_k=3,
        persist_path=runtime_dir / "chroma",
    )
    memory_gateway = MemoryGateway(strategy=rag, stm_capacity=5)

    executor = TargetExecutor(target_agent=target_agent)
    oracle = BreachOracle()
    # For this validation run we keep SQL protection relaxed to ensure full 3-step execution.
    sandbox_policy = SandboxPolicy(
        scenario="sql",
        protected_tables=(),
        allowed_operations={},
    )
    judge = Judge(memory_gateway=memory_gateway)

    orchestrator = RunOrchestrator(
        planner=planner,
        mutator=mutator,
        executor=executor,
        judge=judge,
        snapshot_provider=sandbox,
        oracle=oracle,
        memory_gateway=memory_gateway,
    )

    t0 = time.monotonic()
    result = orchestrator.run(
        max_budget=3,
        sandbox_policy=sandbox_policy,
        initial_run_state={
            "scenario": "sql",
            "guardrail_level": "moderate",
            "memory_mode": "rag_ltm",
            "goal": "Inspect non-sensitive users table records for validation.",
            "seed": 42,
            "run_id": 42,
        },
    )
    duration_ms = int((time.monotonic() - t0) * 1000)

    for ev in result.step_events:
        step = int(ev["step_index"])
        strategy = str(ev.get("planner_selected_strategy") or "")
        prompt = str(ev.get("prompt") or "")
        oracle_verdict = ev.get("oracle_verdict")
        judge_verdict = str(ev.get("judge_verdict") or "")
        print(
            f"Step {step}: Planner={strategy[:50]!r}, Prompt={prompt[:50]!r}, "
            f"Oracle={oracle_verdict}, Judge={judge_verdict}"
        )

    store = EventStore(db_path=db_path)
    experiment_id = f"real_run_{int(time.time() * 1000)}"
    condition = "sql_moderate_rag_ltm"

    for ev in result.step_events:
        store.write_step_event(
            StepEvent(
                experiment_id=experiment_id,
                run_id=42,
                condition=condition,
                step_t=int(ev["step_index"]),
                timestamp_ms=int(ev["timestamp_ms"]),
                prompt=str(ev.get("prompt", "")),
                response=str(ev.get("response_text", "")),
                tool_trace=list(ev.get("tool_trace", [])),
                policy_decisions=list(ev.get("policy_decisions", [])),
                oracle_verdict=bool(ev.get("oracle_verdict", False)),
                judge_verdict=str(ev.get("judge_verdict", "fail")),
            )
        )

    store.write_stop_event(
        StopEvent(
            experiment_id=experiment_id,
            run_id=42,
            condition=condition,
            stop_reason=result.stop_reason,
            q2b=result.q2b,
            censored_flag=result.censored_flag,
            timestamp_ms=int(result.stop_event["timestamp_ms"]),
            seed=42,
        )
    )

    stop_events = store.get_stop_events(experiment_id)
    step_events = store.get_step_events(experiment_id)

    # Requested checks
    assert planner_backend.calls > 0
    assert mutator_backend.calls > 0
    assert target_backend.calls > 0

    for ev in result.step_events:
        assert str(ev.get("planner_selected_strategy") or "").strip()
        assert str(ev.get("prompt") or "").strip()
        assert str(ev.get("response_text") or "").strip()
        assert isinstance(ev.get("tool_trace"), list)
        assert isinstance(ev.get("oracle_verdict"), bool)
        assert str(ev.get("judge_verdict") or "") in {"success", "partial", "fail"}

    assert rag._collection.count() > 0, "MemoryGateway.upsert_episode() did not persist episodes into ChromaDB."

    assert len(stop_events) == 1
    assert len(step_events) == 3

    for sev in step_events:
        assert sev.prompt
        assert sev.response
        assert sev.timestamp_ms > 0

    total_api_calls = planner_backend.calls + mutator_backend.calls + target_backend.calls

    print(
        f"Run beendet: stop_reason={result.stop_reason}, q2b={result.q2b}, censored={result.censored_flag}"
    )
    print(f"Gesamtdauer: {duration_ms}ms")
    print(
        "API-Calls: "
        f"{total_api_calls} "
        f"(planner={planner_backend.calls}, mutator={mutator_backend.calls}, target={target_backend.calls})"
    )
    print(
        "Durchschnittliche Latenz je Rolle (ms): "
        f"planner={planner_backend.avg_latency_ms:.1f}, "
        f"mutator={mutator_backend.avg_latency_ms:.1f}, "
        f"target={target_backend.avg_latency_ms:.1f}"
    )

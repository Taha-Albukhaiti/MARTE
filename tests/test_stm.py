from __future__ import annotations

import pytest

from marte.memory.stm import STMRingBuffer
from marte.persistence.schemas import Episode


def _ep(step_t: int, label: str = "fail") -> Episode:
    return Episode(
        document_id=f"ep-{step_t}",
        run_id=1,
        step_t=step_t,
        scenario="sql",
        guardrail_level="moderate",
        mutation=f"prompt-{step_t}",
        response_summary=f"response-{step_t}",
        tool_effect="",
        success_label=label,
    )


def test_append_under_capacity() -> None:
    buf = STMRingBuffer(capacity=5)
    for i in range(1, 4):
        buf.append(_ep(i))
    assert len(buf) == 3
    ctx = buf.get_context()
    assert len(ctx) == 3
    assert [ep.step_t for ep in ctx] == [1, 2, 3]


def test_append_overflow() -> None:
    buf = STMRingBuffer(capacity=3)
    for i in range(1, 6):
        buf.append(_ep(i))
    assert len(buf) == 3
    ctx = buf.get_context()
    # Oldest three that fit: steps 3, 4, 5
    assert [ep.step_t for ep in ctx] == [3, 4, 5]


def test_get_context_order() -> None:
    buf = STMRingBuffer(capacity=5)
    for i in [10, 20, 30]:
        buf.append(_ep(i))
    ctx = buf.get_context()
    assert [ep.step_t for ep in ctx] == [10, 20, 30]


def test_reset() -> None:
    buf = STMRingBuffer(capacity=5)
    for i in range(1, 4):
        buf.append(_ep(i))
    buf.reset()
    assert len(buf) == 0
    assert buf.get_context() == []


def test_len() -> None:
    buf = STMRingBuffer(capacity=4)
    assert len(buf) == 0
    buf.append(_ep(1))
    assert len(buf) == 1
    buf.append(_ep(2))
    assert len(buf) == 2
    # overflow: capacity=4, add 5 total → len stays at 4
    for i in range(3, 7):
        buf.append(_ep(i))
    assert len(buf) == 4


def test_capacity_one_keeps_only_last() -> None:
    buf = STMRingBuffer(capacity=1)
    buf.append(_ep(1))
    buf.append(_ep(2))
    assert len(buf) == 1
    assert buf.get_context()[0].step_t == 2


def test_invalid_capacity_raises() -> None:
    with pytest.raises(ValueError):
        STMRingBuffer(capacity=0)


def test_get_context_returns_copy() -> None:
    buf = STMRingBuffer(capacity=3)
    buf.append(_ep(1))
    ctx = buf.get_context()
    ctx.clear()           # mutating the returned list must not affect the buffer
    assert len(buf) == 1

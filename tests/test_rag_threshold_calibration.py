from __future__ import annotations

from pathlib import Path

import pytest

from marte.memory.calibration import (
    CALIBRATED_BLOCKED_MIN_SIMILARITY,
    CALIBRATED_POSITIVE_MIN_SIMILARITY,
)
from scripts.calibrate_rag_thresholds import (
    STRATA,
    _calibrate_channel,
    _load_records,
)


def _records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for scenario, guardrail in STRATA:
        records.extend(
            [
                {
                    "channel": "positive",
                    "scenario": scenario,
                    "guardrail_level": guardrail,
                    "similarity": 0.8,
                    "relevant": True,
                },
                {
                    "channel": "positive",
                    "scenario": scenario,
                    "guardrail_level": guardrail,
                    "similarity": 0.2,
                    "relevant": False,
                },
            ]
        )
    return records


def test_calibration_maximizes_macro_f1_and_uses_higher_tie_threshold() -> None:
    result = _calibrate_channel(
        _records(),
        "positive",
        allow_small_set=True,
    )

    assert result["threshold"] == pytest.approx(0.8)
    assert result["macro_f1"] == pytest.approx(1.0)


def test_calibration_rejects_too_small_relevance_set_by_default() -> None:
    with pytest.raises(ValueError, match="at least 20 relevant"):
        _calibrate_channel(
            _records(),
            "positive",
            allow_small_set=False,
        )


def test_frozen_thresholds_match_the_reviewed_calibration_dataset() -> None:
    root = Path(__file__).resolve().parents[1]
    records = _load_records(root / "retrieval_calibration" / "rag_pairs.json")

    positive = _calibrate_channel(records, "positive", allow_small_set=False)
    blocked = _calibrate_channel(records, "blocked", allow_small_set=False)

    assert positive["threshold"] == pytest.approx(
        CALIBRATED_POSITIVE_MIN_SIMILARITY
    )
    assert positive["macro_f1"] == pytest.approx(0.7435897435897436)
    assert blocked["threshold"] == pytest.approx(
        CALIBRATED_BLOCKED_MIN_SIMILARITY
    )
    assert blocked["macro_f1"] == pytest.approx(0.8331168831168831)

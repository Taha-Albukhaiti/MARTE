from __future__ import annotations

from collections import Counter

from scripts.build_rag_calibration_dataset import STRATA, _build_labeled_pairs


def test_calibration_dataset_has_balanced_required_coverage() -> None:
    records = _build_labeled_pairs()

    assert len(records) == 80
    counts = Counter(
        (
            record["channel"],
            record["scenario"],
            record["guardrail_level"],
            record["relevant"],
        )
        for record in records
    )
    for channel in ("positive", "blocked"):
        for scenario, guardrail in STRATA:
            assert counts[(channel, scenario, guardrail, True)] == 5
            assert counts[(channel, scenario, guardrail, False)] == 5


def test_calibration_dataset_uses_production_text_serialization() -> None:
    records = _build_labeled_pairs()

    assert all(record["query"] for record in records)
    assert all("strategy:" in record["episode_text"] for record in records)
    assert all(record["annotation_reason"] for record in records)
    assert all("similarity" not in record for record in records)

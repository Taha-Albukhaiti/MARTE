from __future__ import annotations

import pytest

from marte.cli.component_factory import _load_rag_settings
from marte.memory.calibration import (
    CALIBRATED_BLOCKED_MIN_SIMILARITY,
    CALIBRATED_POSITIVE_MIN_SIMILARITY,
)

_RAG_ENV_KEYS = (
    "MARTE_RAG_DIAGNOSTIC_MODE",
    "MARTE_RAG_POSITIVE_TOP_K",
    "MARTE_RAG_BLOCKED_TOP_K",
    "MARTE_RAG_POSITIVE_FETCH_K",
    "MARTE_RAG_BLOCKED_FETCH_K",
    "MARTE_RAG_POSITIVE_MIN_SIMILARITY",
    "MARTE_RAG_BLOCKED_MIN_SIMILARITY",
)


def _clear_rag_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _RAG_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_local_rag_config_uses_frozen_calibrated_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_rag_env(monkeypatch)

    settings = _load_rag_settings("local")

    assert settings.positive_top_k == 3
    assert settings.blocked_top_k == 1
    assert settings.positive_fetch_k == 12
    assert settings.blocked_fetch_k == 4
    assert settings.positive_min_similarity == pytest.approx(
        CALIBRATED_POSITIVE_MIN_SIMILARITY
    )
    assert settings.blocked_min_similarity == pytest.approx(
        CALIBRATED_BLOCKED_MIN_SIMILARITY
    )


def test_api_rag_config_uses_frozen_calibrated_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_rag_env(monkeypatch)

    settings = _load_rag_settings("api")

    assert settings.require_calibrated_thresholds is True
    assert settings.positive_min_similarity == pytest.approx(
        CALIBRATED_POSITIVE_MIN_SIMILARITY
    )
    assert settings.blocked_min_similarity == pytest.approx(
        CALIBRATED_BLOCKED_MIN_SIMILARITY
    )


def test_api_diagnostic_mode_allows_disabled_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_rag_env(monkeypatch)
    monkeypatch.setenv("MARTE_RAG_DIAGNOSTIC_MODE", "1")

    settings = _load_rag_settings("api")

    assert settings.positive_min_similarity is None
    assert settings.blocked_min_similarity is None
    assert settings.require_calibrated_thresholds is False


def test_rag_diagnostic_config_allows_explicit_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_rag_env(monkeypatch)
    monkeypatch.setenv("MARTE_RAG_DIAGNOSTIC_MODE", "1")
    monkeypatch.setenv("MARTE_RAG_POSITIVE_TOP_K", "3")
    monkeypatch.setenv("MARTE_RAG_BLOCKED_TOP_K", "1")
    monkeypatch.setenv("MARTE_RAG_POSITIVE_FETCH_K", "12")
    monkeypatch.setenv("MARTE_RAG_BLOCKED_FETCH_K", "4")
    monkeypatch.setenv("MARTE_RAG_POSITIVE_MIN_SIMILARITY", "0.61")
    monkeypatch.setenv("MARTE_RAG_BLOCKED_MIN_SIMILARITY", "0.72")

    settings = _load_rag_settings("api")

    assert settings.positive_min_similarity == pytest.approx(0.61)
    assert settings.blocked_min_similarity == pytest.approx(0.72)
    assert settings.require_calibrated_thresholds is False


def test_regular_rag_config_rejects_threshold_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_rag_env(monkeypatch)
    monkeypatch.setenv("MARTE_RAG_POSITIVE_MIN_SIMILARITY", "0.61")

    with pytest.raises(ValueError, match="is frozen"):
        _load_rag_settings("api")


@pytest.mark.parametrize("value", ["-0.1", "1.1", "not-a-number"])
def test_rag_config_rejects_invalid_similarity(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    _clear_rag_env(monkeypatch)
    monkeypatch.setenv("MARTE_RAG_POSITIVE_MIN_SIMILARITY", value)

    with pytest.raises(ValueError, match="MARTE_RAG_POSITIVE_MIN_SIMILARITY"):
        _load_rag_settings("local")


def test_rag_config_rejects_fetch_limit_below_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_rag_env(monkeypatch)
    monkeypatch.setenv("MARTE_RAG_POSITIVE_TOP_K", "4")
    monkeypatch.setenv("MARTE_RAG_POSITIVE_FETCH_K", "3")

    with pytest.raises(ValueError, match="POSITIVE_FETCH_K"):
        _load_rag_settings("local")

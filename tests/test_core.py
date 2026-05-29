"""Tests for dqkit.core: config, logging, results, validation, source."""

from __future__ import annotations

import logging

import pytest

from dqkit.core.config import Settings, get_settings
from dqkit.core.logging import get_logger
from dqkit.core.results import AnomalyReport, EvaluationMetrics, ResolutionReport
from dqkit.core.source import DataSource
from dqkit.core.validation import require_columns, require_numeric


def test_get_settings_defaults_and_cache():
    """Defaults are correct and get_settings returns a cached singleton."""
    settings = get_settings()
    assert settings.app_name == "dqkit"
    assert settings.zscore_threshold == pytest.approx(3.0)
    assert settings.iqr_multiplier == pytest.approx(1.5)
    assert get_settings() is settings


def test_settings_reads_environment(monkeypatch):
    """Settings pick up DQKIT_-prefixed environment variables."""
    monkeypatch.setenv("DQKIT_APP_NAME", "custom")
    monkeypatch.setenv("DQKIT_ZSCORE_THRESHOLD", "2.5")
    settings = Settings()
    assert settings.app_name == "custom"
    assert settings.zscore_threshold == pytest.approx(2.5)


def test_settings_rejects_non_positive_threshold():
    """A non-positive threshold violates the gt=0 constraint."""
    with pytest.raises(ValueError, match="zscore_threshold"):
        Settings(zscore_threshold=0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("dqkit", "dqkit"),
        ("dqkit.detect", "dqkit.detect"),
        ("plain", "dqkit.plain"),
    ],
)
def test_get_logger_namespacing(name, expected):
    """get_logger keeps everything under the dqkit namespace."""
    assert get_logger(name).name == expected


def test_get_logger_is_idempotent():
    """Repeated calls never stack duplicate handlers on the root logger."""
    get_logger("one")
    get_logger("two")
    assert len(logging.getLogger("dqkit").handlers) == 1


def test_evaluation_metrics_typical():
    """from_counts computes precision, recall, and F1 from confusion counts."""
    metrics = EvaluationMetrics.from_counts(tp=8, fp=2, fn=4)
    recall = 8 / 12
    assert metrics.precision == pytest.approx(0.8)
    assert metrics.recall == pytest.approx(recall)
    assert metrics.f1 == pytest.approx(2 * 0.8 * recall / (0.8 + recall))
    assert metrics.support == 12


def test_evaluation_metrics_perfect():
    """Perfect counts give precision = recall = f1 = 1."""
    metrics = EvaluationMetrics.from_counts(tp=5, fp=0, fn=0)
    assert (metrics.precision, metrics.recall, metrics.f1, metrics.support) == (
        1.0,
        1.0,
        1.0,
        5,
    )


def test_evaluation_metrics_all_zero():
    """Empty counts yield zeros rather than dividing by zero."""
    metrics = EvaluationMetrics.from_counts(tp=0, fp=0, fn=0)
    assert (metrics.precision, metrics.recall, metrics.f1, metrics.support) == (
        0.0,
        0.0,
        0.0,
        0,
    )


def test_evaluation_metrics_zero_recall_and_f1():
    """Predictions but no true positives: precision defined, recall/F1 zero."""
    metrics = EvaluationMetrics.from_counts(tp=0, fp=5, fn=0)
    assert (metrics.precision, metrics.recall, metrics.f1) == (0.0, 0.0, 0.0)


def test_result_dataclasses_hold_values():
    """Result dataclasses store their fields (no runtime DataFrame check)."""
    anomaly = AnomalyReport(
        flagged=object(), detector="z", column="c", n_flagged=1, threshold=3.0
    )
    resolution = ResolutionReport(clusters=object(), n_records=2, n_entities=1)
    assert anomaly.detector == "z"
    assert anomaly.n_flagged == 1
    assert resolution.n_records == 2
    assert resolution.n_entities == 1


def test_datasource_is_runtime_checkable():
    """A class exposing name + load satisfies the DataSource protocol."""

    class _Source:
        name = "x"

        def load(self, spark):
            return spark

    assert isinstance(_Source(), DataSource)
    assert not isinstance(object(), DataSource)


def test_require_columns(spark):
    """require_columns passes when present and raises when missing."""
    df = spark.createDataFrame([(1, 2)], ["a", "b"])
    require_columns(df, ["a", "b"])
    with pytest.raises(ValueError, match="missing required"):
        require_columns(df, ["a", "c"])


def test_require_numeric(spark):
    """require_numeric enforces presence and numeric type."""
    df = spark.createDataFrame([(1, "x")], ["num", "txt"])
    require_numeric(df, "num")
    with pytest.raises(ValueError, match="must be numeric"):
        require_numeric(df, "txt")
    with pytest.raises(ValueError, match="missing required"):
        require_numeric(df, "missing")

"""Tests for dqkit.detect detectors, registry, and pipeline."""

from __future__ import annotations

import pytest

from dqkit.core.results import AnomalyReport
from dqkit.detect import (
    DetectionPipeline,
    GaussianDetector,
    IQRDetector,
    ZScoreDetector,
    available_detectors,
    get_detector,
    register_detector,
)


@pytest.fixture(scope="module")
def amounts(spark):
    """Tight normal values with one clear global outlier (id 100)."""
    rows = [(i, float(10 + (i % 3))) for i in range(30)]
    rows.append((100, 1000.0))
    return spark.createDataFrame(rows, ["id", "value"])


@pytest.fixture(scope="module")
def constant_df(spark):
    """A zero-variance column."""
    return spark.createDataFrame([(i, 5.0) for i in range(10)], ["id", "value"])


@pytest.fixture(scope="module")
def grouped_df(spark):
    """Two tight groups (A, B), a within-A outlier, and a singleton group C."""
    rows = []
    for i in range(15):
        rows.append((i, "A", 10.0 + (i % 2)))
        rows.append((100 + i, "B", 100.0 + (i % 2)))
    rows.append((999, "A", 500.0))
    rows.append((998, "C", 7.0))
    return spark.createDataFrame(rows, ["id", "grp", "value"])


def test_registry_contains_default_detectors():
    """Importing the package registers the parameter-free detectors."""
    names = available_detectors()
    assert "zscore" in names
    assert "iqr" in names


def test_get_detector_returns_instance():
    """A registered detector is retrievable by name."""
    assert get_detector("zscore").name == "zscore"


def test_get_detector_missing_raises():
    """Unknown detector names raise KeyError."""
    with pytest.raises(KeyError, match="no detector"):
        get_detector("does-not-exist")


def test_register_detector_new_and_duplicate():
    """A new detector registers once; a duplicate name is rejected."""

    class _Fake:
        name = "_fake_detector"

        def detect(self, df, column):
            raise NotImplementedError

    register_detector(_Fake())
    assert "_fake_detector" in available_detectors()
    with pytest.raises(ValueError, match="already registered"):
        register_detector(_Fake())


def test_zscore_flags_global_outlier(amounts):
    """The global z-score detector flags the extreme value."""
    report = ZScoreDetector().detect(amounts, "value")
    assert isinstance(report, AnomalyReport)
    assert report.threshold == pytest.approx(3.0)
    assert 100 in {row["id"] for row in report.flagged.collect()}


def test_zscore_custom_threshold_constant_column(constant_df):
    """A custom threshold is honored; zero variance flags nothing."""
    report = ZScoreDetector(threshold=2.0).detect(constant_df, "value")
    assert report.threshold == pytest.approx(2.0)
    assert report.n_flagged == 0


def test_iqr_flags_outlier_and_handles_constant(amounts, constant_df):
    """IQR flags the outlier; a degenerate IQR flags nothing."""
    report = IQRDetector().detect(amounts, "value")
    assert 100 in {row["id"] for row in report.flagged.collect()}
    assert IQRDetector(multiplier=2.0).detect(constant_df, "value").n_flagged == 0


def test_gaussian_per_group(grouped_df):
    """The per-group detector flags within-group outliers, not singletons."""
    report = GaussianDetector(group_col="grp").detect(grouped_df, "value")
    flagged = {row["id"] for row in report.flagged.collect()}
    assert 999 in flagged
    assert 998 not in flagged


def test_gaussian_custom_threshold_and_missing_group(grouped_df):
    """Custom threshold is stored; a missing group column raises."""
    assert GaussianDetector("grp", threshold=2.5).threshold == pytest.approx(2.5)
    with pytest.raises(ValueError, match="missing required"):
        GaussianDetector(group_col="nope").detect(grouped_df, "value")


def test_detection_pipeline_runs_each_detector(amounts):
    """The pipeline returns one report per detector."""
    reports = DetectionPipeline((ZScoreDetector(), IQRDetector())).run(amounts, "value")
    assert set(reports) == {"zscore", "iqr"}
    assert all(isinstance(report, AnomalyReport) for report in reports.values())

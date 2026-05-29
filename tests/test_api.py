"""Tests for the high-level convenience API."""

from __future__ import annotations

import pytest

from dqkit import detect_anomalies, resolve_entities
from dqkit.core.results import AnomalyReport, ResolutionReport
from dqkit.sources import SyntheticCustomers


def test_detect_anomalies_zscore_and_iqr(spark):
    """The facade dispatches to global detectors and flags the outlier."""
    rows = [(i, float(10 + i % 3)) for i in range(30)] + [(100, 1000.0)]
    df = spark.createDataFrame(rows, ["id", "value"])

    report = detect_anomalies(df, "value", method="zscore")
    assert isinstance(report, AnomalyReport)
    assert report.detector == "zscore"
    assert 100 in {row["id"] for row in report.flagged.collect()}

    iqr_report = detect_anomalies(df, "value", method="iqr", multiplier=2.0)
    assert iqr_report.detector == "iqr"
    assert 100 in {row["id"] for row in iqr_report.flagged.collect()}


def test_detect_anomalies_gaussian_forwards_group(spark):
    """The facade forwards group_col to the per-group detector."""
    rows = [(i, "A", 10.0 + i % 2) for i in range(15)] + [
        (99, "A", 500.0),
        (98, "B", 7.0),
    ]
    df = spark.createDataFrame(rows, ["id", "grp", "value"])

    report = detect_anomalies(df, "value", method="gaussian", group_col="grp")
    assert report.detector == "gaussian"
    assert 99 in {row["id"] for row in report.flagged.collect()}


def test_detect_anomalies_unknown_method(spark):
    """An unknown method name raises a helpful error."""
    df = spark.createDataFrame([(1, 1.0)], ["id", "value"])
    with pytest.raises(ValueError, match="unknown method"):
        detect_anomalies(df, "value", method="nope")


def test_resolve_entities_default_and_threshold(spark):
    """The facade resolves entities, with and without a custom threshold."""
    customers = SyntheticCustomers(n_entities=30, dup_rate=0.5, seed=5).load(spark)

    report = resolve_entities(customers)
    assert isinstance(report, ResolutionReport)
    assert report.n_records > report.n_entities

    tuned = resolve_entities(customers, threshold=0.6)
    assert tuned.n_records == customers.count()

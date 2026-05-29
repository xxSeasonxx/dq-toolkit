"""Tests for dqkit.eval metrics."""

from __future__ import annotations

import pytest

from dqkit.eval import anomaly_metrics, pairwise_resolution_metrics


def test_anomaly_metrics(spark):
    """Flagged rows are scored against the binary ground-truth label."""
    truth = spark.createDataFrame(
        [(1, 1), (2, 1), (3, 0), (4, 0)], ["txn_id", "is_anomaly"]
    )
    flagged = spark.createDataFrame([(1,), (3,)], ["txn_id"])
    metrics = anomaly_metrics(truth, flagged)
    # tp=1 (id1), fp=1 (id3), fn=1 (id2)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.support == 2


def test_pairwise_resolution_metrics_with_duplicates(spark):
    """A correctly merged pair scores perfectly."""
    clusters = spark.createDataFrame(
        [(1, 1), (2, 1), (3, 3)], ["record_id", "entity_id"]
    )
    truth = spark.createDataFrame(
        [(1, 10), (2, 10), (3, 30)], ["record_id", "true_entity_id"]
    )
    metrics = pairwise_resolution_metrics(clusters, truth)
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)


def test_pairwise_resolution_metrics_all_singletons(spark):
    """With no predicted or true pairs, metrics are zero (falsy-sum branch)."""
    clusters = spark.createDataFrame([(1, 1), (2, 2)], ["record_id", "entity_id"])
    truth = spark.createDataFrame([(1, 10), (2, 20)], ["record_id", "true_entity_id"])
    metrics = pairwise_resolution_metrics(clusters, truth)
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.support == 0

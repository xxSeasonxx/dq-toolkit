"""Evaluation: score detector and resolver output against ground-truth labels.

Metrics build on :class:`dqkit.core.results.EvaluationMetrics`.
"""

from __future__ import annotations

from dqkit.eval.metrics import anomaly_metrics, pairwise_resolution_metrics

__all__ = ["anomaly_metrics", "pairwise_resolution_metrics"]

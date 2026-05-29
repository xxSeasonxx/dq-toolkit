"""Immutable result types returned by detectors, resolvers, and evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


@dataclass(frozen=True, kw_only=True)
class AnomalyReport:
    """Outcome of running a detector over a column.

    Attributes:
        flagged: Rows judged anomalous, carrying the original columns plus an
            ``anomaly_score`` column.
        detector: Name of the detector that produced the report.
        column: The column that was scored.
        n_flagged: Number of rows flagged.
        threshold: The score threshold applied.
    """

    flagged: DataFrame
    detector: str
    column: str
    n_flagged: int
    threshold: float


@dataclass(frozen=True, kw_only=True)
class ResolutionReport:
    """Outcome of an entity-resolution run.

    Attributes:
        clusters: Maps each input record id to a resolved ``entity_id``.
        n_records: Number of input records.
        n_entities: Number of distinct resolved entities.
    """

    clusters: DataFrame
    n_records: int
    n_entities: int


@dataclass(frozen=True, kw_only=True)
class EvaluationMetrics:
    """Precision / recall / F1 against ground-truth labels.

    Attributes:
        precision: ``TP / (TP + FP)``.
        recall: ``TP / (TP + FN)``.
        f1: Harmonic mean of precision and recall.
        support: Number of ground-truth positives (``TP + FN``).
    """

    precision: float
    recall: float
    f1: float
    support: int

    @classmethod
    def from_counts(cls, *, tp: int, fp: int, fn: int) -> EvaluationMetrics:
        """Build metrics from raw confusion counts.

        Args:
            tp: True positives.
            fp: False positives.
            fn: False negatives.

        Returns:
            Populated :class:`EvaluationMetrics`, with zeros where a ratio is
            undefined (no predicted or no actual positives).
        """
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom else 0.0
        return cls(precision=precision, recall=recall, f1=f1, support=tp + fn)

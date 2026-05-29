"""Evaluation metrics computed against ground-truth labels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from dqkit.core.logging import get_logger
from dqkit.core.results import EvaluationMetrics

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

logger = get_logger(__name__)


def anomaly_metrics(
    truth: DataFrame,
    flagged: DataFrame,
    *,
    id_col: str = "txn_id",
    label_col: str = "is_anomaly",
) -> EvaluationMetrics:
    """Score flagged rows against a binary ground-truth label.

    Args:
        truth: All rows, carrying the ground-truth ``label_col`` (1 = anomaly).
        flagged: The detector's flagged subset, identified by ``id_col``.
        id_col: Unique row identifier shared by both frames.
        label_col: Ground-truth label column in ``truth``.

    Returns:
        Precision / recall / F1 of the flagged set.
    """
    predicted = flagged.select(id_col).distinct().withColumn("_pred", F.lit(1))
    joined = truth.join(predicted, id_col, "left").withColumn(
        "_pred", F.coalesce(F.col("_pred"), F.lit(0))
    )
    counts = joined.select(
        F.sum(((F.col("_pred") == 1) & (F.col(label_col) == 1)).cast("long")).alias(
            "tp"
        ),
        F.sum(((F.col("_pred") == 1) & (F.col(label_col) == 0)).cast("long")).alias(
            "fp"
        ),
        F.sum(((F.col("_pred") == 0) & (F.col(label_col) == 1)).cast("long")).alias(
            "fn"
        ),
    ).collect()[0]
    return EvaluationMetrics.from_counts(
        tp=int(counts["tp"]), fp=int(counts["fp"]), fn=int(counts["fn"])
    )


def pairwise_resolution_metrics(
    clusters: DataFrame,
    truth: DataFrame,
    *,
    id_col: str = "record_id",
    truth_col: str = "true_entity_id",
) -> EvaluationMetrics:
    """Pairwise precision / recall / F1 for entity resolution.

    Two records form a *positive pair* if they share a predicted ``entity_id``,
    and a *true pair* if they share ``truth_col``. Rather than materialize pairs,
    counts use the closed form ``sum k*(k-1)/2`` over group sizes.

    Args:
        clusters: ``id_col`` -> predicted ``entity_id`` mapping.
        truth: ``id_col`` -> ``truth_col`` ground truth.
        id_col: Unique record identifier.
        truth_col: Ground-truth entity column.

    Returns:
        Pairwise precision / recall / F1 of the resolution.
    """
    labeled = clusters.join(truth.select(id_col, truth_col), id_col)

    def _pairs(*group_cols: str) -> int:
        sizes = labeled.groupBy(*group_cols).count()
        total = sizes.select(
            F.sum(F.col("count") * (F.col("count") - 1) / 2).alias("p")
        ).collect()[0]["p"]
        return int(total or 0)

    predicted = _pairs("entity_id")
    actual = _pairs(truth_col)
    tp = _pairs("entity_id", truth_col)
    return EvaluationMetrics.from_counts(tp=tp, fp=predicted - tp, fn=actual - tp)

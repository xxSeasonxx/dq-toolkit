"""Z-score anomaly detector: flag values far from the column mean."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from dqkit.core.config import get_settings
from dqkit.core.logging import get_logger
from dqkit.core.results import AnomalyReport
from dqkit.core.validation import require_numeric
from dqkit.detect.base import register_detector

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

logger = get_logger(__name__)


class ZScoreDetector:
    """Flag rows lying more than ``threshold`` standard deviations from the mean.

    Mean and standard deviation are computed once with native aggregate
    expressions, then attached to every row via a broadcast cross-join to a
    one-row stats frame. This is deliberately *not* a global
    ``Window`` (one with no ``partitionBy``): a global window funnels all rows
    into a single partition and defeats Spark on real data, whereas the
    broadcast cross-join stays shuffle-free and scales horizontally.

    Attributes:
        name: Registry key, ``"zscore"``.
        threshold: ``|z|`` above which a row is flagged.
    """

    name = "zscore"

    def __init__(self, threshold: float | None = None) -> None:
        """Initialize the detector.

        Args:
            threshold: Override for ``|z|`` cutoff. Falls back to
                :attr:`Settings.zscore_threshold` when ``None``.
        """
        self.threshold = (
            threshold if threshold is not None else get_settings().zscore_threshold
        )

    def detect(self, df: DataFrame, column: str) -> AnomalyReport:
        """Score ``column`` and return rows beyond the z-score threshold.

        Args:
            df: Input data.
            column: Numeric column to score.

        Returns:
            An :class:`AnomalyReport` whose ``flagged`` frame carries an
            ``anomaly_score`` column (the absolute z-score). When the column is
            constant (zero variance) nothing is flagged.
        """
        require_numeric(df, column)

        stats = df.select(
            F.mean(column).alias("_mu"),
            F.stddev_samp(column).alias("_sigma"),
        )
        # Broadcast a 1-row stats frame onto every record: shuffle-free, and it
        # avoids the single-partition trap of a global Window.
        scored = (
            df.crossJoin(F.broadcast(stats))
            .withColumn(
                "anomaly_score",
                F.when(
                    F.col("_sigma").isNull() | (F.col("_sigma") == 0),
                    F.lit(0.0),
                ).otherwise(F.abs((F.col(column) - F.col("_mu")) / F.col("_sigma"))),
            )
            .drop("_mu", "_sigma")
        )
        flagged = scored.filter(F.col("anomaly_score") > self.threshold)

        n_flagged = flagged.count()
        logger.info(
            "zscore: flagged %d row(s) in %r at threshold %.2f",
            n_flagged,
            column,
            self.threshold,
        )
        return AnomalyReport(
            flagged=flagged,
            detector=self.name,
            column=column,
            n_flagged=n_flagged,
            threshold=self.threshold,
        )


register_detector(ZScoreDetector())

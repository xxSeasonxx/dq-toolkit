"""IQR (Tukey fence) anomaly detector."""

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


class IQRDetector:
    """Flag values beyond Tukey fences ``[Q1 - k*IQR, Q3 + k*IQR]``.

    Quartiles are estimated with ``percentile_approx`` in a single aggregation
    and broadcast onto the rows (the same shuffle-free pattern as the z-score
    detector). Being quantile-based, it resists the masking effect that extreme
    outliers have on mean/standard-deviation methods.

    The ``anomaly_score`` is the number of IQRs a value lies beyond the nearer
    fence (``0`` inside the fences), so the unified decision rule
    ``anomaly_score > threshold`` holds with ``threshold = 0``.

    Attributes:
        name: Registry key, ``"iqr"``.
        multiplier: Fence multiplier ``k``.
    """

    name = "iqr"

    def __init__(self, multiplier: float | None = None) -> None:
        """Initialize the detector.

        Args:
            multiplier: Tukey fence multiplier ``k``. Falls back to
                :attr:`Settings.iqr_multiplier` when ``None``.
        """
        self.multiplier = (
            multiplier if multiplier is not None else get_settings().iqr_multiplier
        )

    def detect(self, df: DataFrame, column: str) -> AnomalyReport:
        """Score ``column`` against Tukey fences and return rows beyond them.

        Args:
            df: Input data.
            column: Numeric column to score.

        Returns:
            An :class:`AnomalyReport`; ``flagged`` carries ``anomaly_score``
            (IQRs beyond the nearer fence). A degenerate ``IQR <= 0`` flags
            nothing.
        """
        require_numeric(df, column)

        stats = df.select(
            F.percentile_approx(column, 0.25).alias("_q1"),
            F.percentile_approx(column, 0.75).alias("_q3"),
        )
        scored = (
            df.crossJoin(F.broadcast(stats))
            .withColumn("_iqr", F.col("_q3") - F.col("_q1"))
            .withColumn("_lower", F.col("_q1") - F.lit(self.multiplier) * F.col("_iqr"))
            .withColumn("_upper", F.col("_q3") + F.lit(self.multiplier) * F.col("_iqr"))
            .withColumn(
                "anomaly_score",
                F.when(F.col("_iqr") <= 0, F.lit(0.0)).otherwise(
                    F.greatest(
                        F.lit(0.0),
                        (F.col("_lower") - F.col(column)) / F.col("_iqr"),
                        (F.col(column) - F.col("_upper")) / F.col("_iqr"),
                    )
                ),
            )
            .drop("_q1", "_q3", "_iqr", "_lower", "_upper")
        )
        flagged = scored.filter(F.col("anomaly_score") > 0)

        n_flagged = flagged.count()
        logger.info(
            "iqr: flagged %d row(s) in %r at k=%.2f", n_flagged, column, self.multiplier
        )
        return AnomalyReport(
            flagged=flagged,
            detector=self.name,
            column=column,
            n_flagged=n_flagged,
            threshold=0.0,
        )


register_detector(IQRDetector())

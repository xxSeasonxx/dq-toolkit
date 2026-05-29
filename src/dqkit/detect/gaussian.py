"""Per-group Gaussian anomaly detector."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import Window
from pyspark.sql import functions as F

from dqkit.core.config import get_settings
from dqkit.core.logging import get_logger
from dqkit.core.results import AnomalyReport
from dqkit.core.validation import require_columns, require_numeric

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

logger = get_logger(__name__)


class GaussianDetector:
    """Flag values far from their *group's* mean — a per-group z-score.

    Global statistics blur heterogeneous populations: a $5k transaction is
    routine for one customer and a glaring outlier for another. This detector
    computes mean and standard deviation **within each group** using
    ``Window.partitionBy(group_col)`` — the case where a window *is* the right,
    scalable tool, in deliberate contrast to the global z-score detector, which
    avoids a partition-less window.

    Because it needs a grouping column it is constructed explicitly rather than
    auto-registered (there is no sensible global default group).

    Attributes:
        name: Detector name, ``"gaussian"``.
        group_col: Column whose values define the groups.
        threshold: Within-group ``|z|`` cutoff.
    """

    name = "gaussian"

    def __init__(self, group_col: str, threshold: float | None = None) -> None:
        """Initialize the detector.

        Args:
            group_col: Column defining the groups.
            threshold: Within-group ``|z|`` cutoff. Falls back to
                :attr:`Settings.zscore_threshold` when ``None``.
        """
        self.group_col = group_col
        self.threshold = (
            threshold if threshold is not None else get_settings().zscore_threshold
        )

    def detect(self, df: DataFrame, column: str) -> AnomalyReport:
        """Score ``column`` against per-group statistics.

        Args:
            df: Input data.
            column: Numeric column to score.

        Returns:
            An :class:`AnomalyReport`; ``flagged`` carries the within-group
            absolute z-score as ``anomaly_score``. Groups with one row or zero
            variance contribute no flags.
        """
        require_numeric(df, column)
        require_columns(df, [self.group_col])

        window = Window.partitionBy(self.group_col)
        mu = F.mean(column).over(window)
        sigma = F.stddev_samp(column).over(window)
        scored = df.withColumn(
            "anomaly_score",
            F.when(sigma.isNull() | (sigma == 0), F.lit(0.0)).otherwise(
                F.abs((F.col(column) - mu) / sigma)
            ),
        )
        flagged = scored.filter(F.col("anomaly_score") > self.threshold)

        n_flagged = flagged.count()
        logger.info(
            "gaussian: flagged %d row(s) in %r by %r at threshold %.2f",
            n_flagged,
            column,
            self.group_col,
            self.threshold,
        )
        return AnomalyReport(
            flagged=flagged,
            detector=self.name,
            column=column,
            n_flagged=n_flagged,
            threshold=self.threshold,
        )

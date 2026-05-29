"""Compose multiple detectors over a single column."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dqkit.core.logging import get_logger

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

    from dqkit.core.results import AnomalyReport
    from dqkit.detect.base import Detector

logger = get_logger(__name__)


@dataclass(frozen=True)
class DetectionPipeline:
    """Apply a sequence of detectors to one column.

    Each detector is independent, so the pipeline applies them in turn and
    returns one report per detector. This is composition, not inheritance: any
    object satisfying the :class:`~dqkit.detect.base.Detector` protocol drops in
    without the pipeline knowing its internals.

    Attributes:
        detectors: The detectors to apply, in order.
    """

    detectors: tuple[Detector, ...]

    def run(self, df: DataFrame, column: str) -> dict[str, AnomalyReport]:
        """Apply every detector to ``column``.

        Args:
            df: Input data.
            column: Numeric column to score.

        Returns:
            Mapping of detector name to its :class:`AnomalyReport`.
        """
        reports = {det.name: det.detect(df, column) for det in self.detectors}
        logger.info("pipeline: ran %d detector(s) on %r", len(self.detectors), column)
        return reports

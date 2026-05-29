"""High-level convenience API: the easy entry point over the modular core.

These two functions are the common-case front door: choose a method by name,
pass a DataFrame, get a typed result. They are a thin facade — they compose the
same protocols, detectors, scorers, and pipelines the package exposes directly,
so power users lose nothing by starting here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dqkit.detect import GaussianDetector, IQRDetector, ZScoreDetector
from dqkit.resolve import MatchScorer, ResolutionPipeline

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

    from dqkit.core.results import AnomalyReport, ResolutionReport

_DETECTORS = {
    "zscore": ZScoreDetector,
    "iqr": IQRDetector,
    "gaussian": GaussianDetector,
}


def detect_anomalies(
    df: DataFrame,
    column: str,
    *,
    method: str = "zscore",
    **detector_kwargs: object,
) -> AnomalyReport:
    """Detect anomalies in ``column`` with a named detector.

    Args:
        df: Input data.
        column: Numeric column to score.
        method: One of ``"zscore"``, ``"iqr"``, ``"gaussian"``.
        **detector_kwargs: Forwarded to the detector's constructor — e.g.
            ``threshold=`` (z-score / gaussian), ``multiplier=`` (IQR),
            ``group_col=`` (gaussian).

    Returns:
        An :class:`~dqkit.core.results.AnomalyReport`.

    Raises:
        ValueError: If ``method`` is not a known detector.
    """
    try:
        builder = _DETECTORS[method]
    except KeyError:
        raise ValueError(
            f"unknown method {method!r}; choose from {sorted(_DETECTORS)}"
        ) from None
    return builder(**detector_kwargs).detect(df, column)


def resolve_entities(
    customers: DataFrame,
    *,
    threshold: float | None = None,
    id_col: str = "record_id",
) -> ResolutionReport:
    """Resolve duplicate records into entities.

    Args:
        customers: Records to resolve.
        threshold: Optional match-score cutoff; defaults to the scorer's own
            default when ``None``.
        id_col: Unique record identifier.

    Returns:
        A :class:`~dqkit.core.results.ResolutionReport`.
    """
    scorer = MatchScorer() if threshold is None else MatchScorer(threshold=threshold)
    return ResolutionPipeline(scorer=scorer, id_col=id_col).resolve(customers)

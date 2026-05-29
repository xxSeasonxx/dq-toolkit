"""The detector contract and a small registry that keeps detection open/closed."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dqkit.core.logging import get_logger

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

    from dqkit.core.results import AnomalyReport

logger = get_logger(__name__)


@runtime_checkable
class Detector(Protocol):
    """Scores a numeric column and flags anomalous rows.

    A detector is any object exposing :meth:`detect`. Keeping the surface this
    narrow (interface segregation) means callers, the registry, and the
    evaluation harness never depend on a detector's internals.

    Attributes:
        name: Registry key for the detector.
    """

    name: str

    def detect(self, df: DataFrame, column: str) -> AnomalyReport:
        """Flag anomalous rows in ``column``.

        Args:
            df: Input data.
            column: Numeric column to score.

        Returns:
            An :class:`~dqkit.core.results.AnomalyReport`.
        """
        ...


_REGISTRY: dict[str, Detector] = {}


def register_detector(detector: Detector) -> Detector:
    """Register a detector instance under its ``name``.

    Args:
        detector: The detector to register.

    Returns:
        The same detector, so the call can wrap a construction expression.

    Raises:
        ValueError: If a detector is already registered under that name.
    """
    if detector.name in _REGISTRY:
        raise ValueError(f"detector {detector.name!r} already registered")
    _REGISTRY[detector.name] = detector
    logger.debug("registered detector %r", detector.name)
    return detector


def get_detector(name: str) -> Detector:
    """Look up a registered detector by name.

    Args:
        name: Registered detector name.

    Returns:
        The detector instance.

    Raises:
        KeyError: If no detector is registered under ``name``.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no detector {name!r}; registered: {available_detectors()}"
        ) from None


def available_detectors() -> list[str]:
    """Return the sorted names of all registered detectors.

    Returns:
        Sorted list of registered detector names.
    """
    return sorted(_REGISTRY)

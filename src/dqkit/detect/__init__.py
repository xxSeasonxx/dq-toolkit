"""Anomaly detection: pluggable detectors over a shared protocol.

Importing this package registers the parameter-free detectors (``zscore``,
``iqr``) in the registry. :class:`GaussianDetector` needs a grouping column, so
it is imported for direct construction but not auto-registered.
"""

from __future__ import annotations

from dqkit.detect.base import (
    Detector,
    available_detectors,
    get_detector,
    register_detector,
)
from dqkit.detect.gaussian import GaussianDetector
from dqkit.detect.iqr import IQRDetector
from dqkit.detect.pipeline import DetectionPipeline
from dqkit.detect.zscore import ZScoreDetector

__all__ = [
    "DetectionPipeline",
    "Detector",
    "GaussianDetector",
    "IQRDetector",
    "ZScoreDetector",
    "available_detectors",
    "get_detector",
    "register_detector",
]

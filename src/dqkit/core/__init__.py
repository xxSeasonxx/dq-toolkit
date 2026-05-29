"""Shared foundation: configuration, logging, validation, and result types.

Everything in :mod:`dqkit.core` is domain-agnostic. Detection and resolution
build on these contracts; nothing here depends on them (dependency inversion).
"""

from __future__ import annotations

from dqkit.core.config import Settings, get_settings
from dqkit.core.logging import get_logger
from dqkit.core.results import AnomalyReport, EvaluationMetrics, ResolutionReport
from dqkit.core.source import DataSource

__all__ = [
    "AnomalyReport",
    "DataSource",
    "EvaluationMetrics",
    "ResolutionReport",
    "Settings",
    "get_logger",
    "get_settings",
]

"""dqkit — modular, Spark-efficient data-quality toolkit.

Anomaly detection and entity resolution on a shared, typed ``core`` foundation.
The two functions re-exported here are the easy entry point; the README shows
the full, composable API underneath them.
"""

from __future__ import annotations

from dqkit.api import detect_anomalies, resolve_entities

__version__ = "0.1.0"

__all__ = ["__version__", "detect_anomalies", "resolve_entities"]

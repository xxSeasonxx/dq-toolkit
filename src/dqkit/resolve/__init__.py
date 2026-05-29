"""Entity resolution: blocking, comparison, scoring, and clustering.

Each stage is a focused, independently testable unit; :class:`ResolutionPipeline`
sequences them. Resolution depends on :mod:`dqkit.core`; nothing depends on its
internals.
"""

from __future__ import annotations

from dqkit.resolve.block import candidate_pairs
from dqkit.resolve.cluster import connected_components
from dqkit.resolve.compare import compare_pairs
from dqkit.resolve.pipeline import ResolutionPipeline
from dqkit.resolve.score import MatchScorer

__all__ = [
    "MatchScorer",
    "ResolutionPipeline",
    "candidate_pairs",
    "compare_pairs",
    "connected_components",
]

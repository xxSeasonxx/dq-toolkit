"""End-to-end entity-resolution pipeline: block -> compare -> score -> cluster."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dqkit.core.logging import get_logger
from dqkit.core.results import ResolutionReport
from dqkit.resolve.block import candidate_pairs
from dqkit.resolve.cluster import connected_components
from dqkit.resolve.compare import compare_pairs
from dqkit.resolve.score import MatchScorer

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

logger = get_logger(__name__)


@dataclass(frozen=True)
class ResolutionPipeline:
    """Wire the four resolution stages into one call.

    Each stage is a small, independently testable unit; the pipeline only
    sequences them. Swap the scorer or re-block by constructing with different
    components — no stage needs editing.

    Attributes:
        scorer: The match scorer applied to compared pairs.
        id_col: Unique record identifier.
    """

    scorer: MatchScorer = field(default_factory=MatchScorer)
    id_col: str = "record_id"

    def resolve(self, customers: DataFrame) -> ResolutionReport:
        """Resolve ``customers`` into entities.

        Args:
            customers: Records to resolve.

        Returns:
            A :class:`ResolutionReport` with the id->entity mapping and counts.
        """
        pairs = candidate_pairs(customers, id_col=self.id_col)
        compared = compare_pairs(pairs)
        matches = self.scorer.matches(compared)
        clusters = connected_components(customers, matches, id_col=self.id_col)

        n_records = clusters.count()
        n_entities = clusters.select("entity_id").distinct().count()
        logger.info("resolution: %d records -> %d entities", n_records, n_entities)
        return ResolutionReport(
            clusters=clusters, n_records=n_records, n_entities=n_entities
        )

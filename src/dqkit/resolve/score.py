"""Combine comparison features into a match score and decision."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from dqkit.core.logging import get_logger

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

logger = get_logger(__name__)

_DEFAULT_WEIGHTS = {
    "name_sim": 0.4,
    "email_exact": 0.3,
    "phone_exact": 0.2,
    "city_sim": 0.1,
}


@dataclass(frozen=True)
class MatchScorer:
    """Weighted-sum scorer over comparison features.

    The score is a weight-normalized linear blend of the feature columns, so it
    always lands in ``[0, 1]`` regardless of the weights supplied. A transparent
    linear rule is deliberately chosen over a black-box model: every match is
    explainable by its feature contributions.

    Attributes:
        weights: Feature-name to weight. Defaults favor name then email.
        threshold: Minimum score to declare a match. Agreement on the
            non-identifying fields alone (name + city) caps at 0.5 under the
            default weights, so any threshold above 0.5 keeps distinct entities
            apart; the default sits just above that to maximize recall.
    """

    weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    threshold: float = 0.6

    def score(self, compared: DataFrame) -> DataFrame:
        """Add ``match_score`` and boolean ``is_match`` columns.

        Args:
            compared: Pairs carrying every feature column in :attr:`weights`.

        Returns:
            The input with ``match_score`` and ``is_match`` added.
        """
        total = sum(self.weights.values())
        blended = F.lit(0.0)
        for feature, weight in self.weights.items():
            blended = blended + F.lit(weight) * F.col(feature)
        return compared.withColumn("match_score", blended / F.lit(total)).withColumn(
            "is_match", F.col("match_score") >= F.lit(self.threshold)
        )

    def matches(self, compared: DataFrame) -> DataFrame:
        """Return only the pairs scored at or above :attr:`threshold`.

        Args:
            compared: Pairs carrying every feature column in :attr:`weights`.

        Returns:
            The matched subset, with scoring columns attached.
        """
        return self.score(compared).filter(F.col("is_match"))

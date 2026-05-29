"""Blocking: generate candidate record pairs without an O(n^2) comparison.

Comparing every pair of N records is N*(N-1)/2 comparisons — infeasible at
scale. Blocking first partitions records into blocks that share a cheap key and
only compares within a block, cutting the space to the sum of per-block squares.
The key trades recall (missed matches across blocks) for speed; here it is the
leading name character plus a city prefix, which the synthetic perturbations
preserve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from dqkit.core.logging import get_logger

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame

logger = get_logger(__name__)

_COMPARE_FIELDS = ("name", "email", "phone", "city")


def _block_key() -> Column:
    """Build the blocking key column (leading name char + 3-char city prefix)."""
    return F.concat_ws(
        "|",
        F.substring(F.lower(F.col("name")), 1, 1),
        F.substring(F.lower(F.col("city")), 1, 3),
    )


def candidate_pairs(
    df: DataFrame,
    *,
    id_col: str = "record_id",
    fields: tuple[str, ...] = _COMPARE_FIELDS,
) -> DataFrame:
    """Generate within-block candidate pairs as a self-join.

    Args:
        df: Records to pair, including ``id_col`` and every name in ``fields``.
        id_col: Unique record identifier.
        fields: Columns carried onto each side of the pair (suffixed ``_a`` /
            ``_b``) for downstream comparison.

    Returns:
        One row per unordered candidate pair (``id_a < id_b``), with each field
        present as ``<field>_a`` and ``<field>_b``.
    """
    keyed = df.withColumn("_block_key", _block_key())
    left = keyed.alias("a")
    right = keyed.alias("b")

    selected: list[Column] = [
        F.col(f"a.{id_col}").alias(f"{id_col}_a"),
        F.col(f"b.{id_col}").alias(f"{id_col}_b"),
    ]
    for name in fields:
        selected.append(F.col(f"a.{name}").alias(f"{name}_a"))
        selected.append(F.col(f"b.{name}").alias(f"{name}_b"))

    pairs = left.join(
        right,
        (F.col("a._block_key") == F.col("b._block_key"))
        & (F.col(f"a.{id_col}") < F.col(f"b.{id_col}")),
    ).select(*selected)

    logger.info("blocking: generated candidate pairs within shared blocks")
    return pairs

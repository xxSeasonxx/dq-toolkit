"""Cluster matched pairs into entities via connected components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql.types import LongType, StructField, StructType

from dqkit.core.logging import get_logger

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

logger = get_logger(__name__)


class _UnionFind:
    """Disjoint-set with path halving; roots are the smallest id in a set."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        """Return the representative (minimum id) of ``x``'s set."""
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path halving
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        """Merge the sets of ``a`` and ``b``, keeping the smaller id as root."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)


def connected_components(
    records: DataFrame,
    matches: DataFrame,
    *,
    id_col: str = "record_id",
) -> DataFrame:
    """Assign every record an ``entity_id`` (the min id in its component).

    Matched pairs are sparse after blocking and scoring, so components are
    resolved with union-find on the driver and the small id->entity mapping is
    sent back as a DataFrame. For billion-edge graphs you would switch to
    GraphFrames or iterative label propagation; the contract here would not
    change.

    Args:
        records: All records (ensures singletons get their own entity).
        matches: Matched pairs with ``<id_col>_a`` / ``<id_col>_b``.
        id_col: Unique record identifier.

    Returns:
        DataFrame of ``id_col`` and ``entity_id``.
    """
    uf = _UnionFind()
    for row in matches.select(f"{id_col}_a", f"{id_col}_b").collect():
        uf.union(row[0], row[1])

    ids = [row[0] for row in records.select(id_col).collect()]
    mapping = [(record_id, uf.find(record_id)) for record_id in ids]

    schema = StructType(
        [
            StructField(id_col, LongType(), nullable=False),
            StructField("entity_id", LongType(), nullable=False),
        ]
    )
    clusters = records.sparkSession.createDataFrame(mapping, schema=schema)
    logger.info(
        "clustering: %d records -> %d entities",
        len(ids),
        len({entity for _, entity in mapping}),
    )
    return clusters

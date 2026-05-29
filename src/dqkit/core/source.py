"""The ingestion contract: anything that can yield a Spark DataFrame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


@runtime_checkable
class DataSource(Protocol):
    """A source of records for the toolkit.

    Implementations decouple *where data comes from* (a generator, Parquet, a
    warehouse table) from *what the toolkit does with it*. Callers depend only
    on this protocol, so swapping the source never ripples outward.

    Attributes:
        name: Stable identifier for the source.
    """

    name: str

    def load(self, spark: SparkSession) -> DataFrame:
        """Materialize the source as a Spark DataFrame.

        Args:
            spark: Active Spark session.

        Returns:
            The source data as a DataFrame.
        """
        ...

"""Small, composable guards for DataFrame preconditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql.types import NumericType

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def require_columns(df: DataFrame, columns: list[str]) -> None:
    """Ensure every named column is present.

    Args:
        df: DataFrame to inspect.
        columns: Columns that must exist.

    Raises:
        ValueError: If any column is missing.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing required column(s): {missing}; have {df.columns}")


def require_numeric(df: DataFrame, column: str) -> None:
    """Ensure a column exists and has a numeric type.

    Args:
        df: DataFrame to inspect.
        column: Column expected to be numeric.

    Raises:
        ValueError: If the column is missing or not a numeric type.
    """
    require_columns(df, [column])
    dtype = df.schema[column].dataType
    if not isinstance(dtype, NumericType):
        raise ValueError(f"column {column!r} must be numeric, got {dtype}")

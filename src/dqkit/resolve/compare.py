"""Pairwise field comparison features for candidate pairs.

Every comparator is a native Spark column expression — no Python UDFs — so the
work stays in the JVM and the Catalyst optimizer can see through it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as F

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame


def _norm(col: Column) -> Column:
    """Lower-case and trim a string column."""
    return F.lower(F.trim(col))


def _string_sim(a: Column, b: Column) -> Column:
    """Normalized Levenshtein similarity in ``[0, 1]`` (1.0 when both empty)."""
    na, nb = _norm(a), _norm(b)
    distance = F.levenshtein(na, nb)
    max_len = F.greatest(F.length(na), F.length(nb))
    return F.when(max_len == 0, F.lit(1.0)).otherwise(1 - distance / max_len)


def _digits(col: Column) -> Column:
    """Strip everything but digits from a string column."""
    return F.regexp_replace(col, r"\D", "")


def compare_pairs(pairs: DataFrame) -> DataFrame:
    """Add similarity features to candidate pairs.

    Args:
        pairs: Candidate pairs with ``name_a``/``name_b`` etc.

    Returns:
        The input with ``name_sim``, ``city_sim``, ``email_exact``, and
        ``phone_exact`` columns added. ``phone_exact`` is ``0.0`` when either
        phone is blank (absence is not evidence of a match).
    """
    phone_a, phone_b = _digits(F.col("phone_a")), _digits(F.col("phone_b"))
    return (
        pairs.withColumn("name_sim", _string_sim(F.col("name_a"), F.col("name_b")))
        .withColumn("city_sim", _string_sim(F.col("city_a"), F.col("city_b")))
        .withColumn(
            "email_exact",
            (_norm(F.col("email_a")) == _norm(F.col("email_b"))).cast("double"),
        )
        .withColumn(
            "phone_exact",
            F.when(
                (F.length(phone_a) == 0) | (F.length(phone_b) == 0), F.lit(0.0)
            ).otherwise((phone_a == phone_b).cast("double")),
        )
    )

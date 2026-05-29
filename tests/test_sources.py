"""Tests for dqkit.sources synthetic generators."""

from __future__ import annotations

import pytest

from dqkit.core.source import DataSource
from dqkit.sources import SyntheticCustomers, SyntheticTransactions


def test_transactions_schema_and_labels(spark):
    """Transactions have the expected schema, size, and planted anomalies."""
    source = SyntheticTransactions(
        n_customers=20, txns_per_customer=50, anomaly_rate=0.05, seed=1
    )
    df = source.load(spark)
    assert set(df.columns) == {"txn_id", "customer_id", "amount", "is_anomaly"}
    assert df.count() == 20 * 50
    assert df.filter("is_anomaly = 1").count() > 0
    assert isinstance(source, DataSource)


def test_transactions_are_deterministic(spark):
    """The same seed yields identical amounts."""
    first = SyntheticTransactions(n_customers=5, txns_per_customer=10, seed=42).load(
        spark
    )
    second = SyntheticTransactions(n_customers=5, txns_per_customer=10, seed=42).load(
        spark
    )
    assert [r.amount for r in first.orderBy("txn_id").collect()] == [
        r.amount for r in second.orderBy("txn_id").collect()
    ]


def test_customers_plant_duplicates(spark):
    """Customers carry the schema, the right entity count, and duplicates."""
    df = SyntheticCustomers(n_entities=50, dup_rate=0.6, seed=2).load(spark)
    assert set(df.columns) == {
        "record_id",
        "name",
        "email",
        "phone",
        "city",
        "true_entity_id",
    }
    assert df.select("true_entity_id").distinct().count() == 50
    assert df.count() > 50


@pytest.mark.parametrize(
    ("draw", "last", "contains"),
    [
        (0.1, "smith", " "),  # abbreviate first name to an initial
        (0.9, "smith", "james"),  # typo: drop an interior char (len > 2)
        (0.9, "li", "li"),  # typo branch but last name too short to edit
    ],
)
def test_perturb_name_branches(draw, last, contains):
    """_perturb_name covers abbreviation and typo paths."""

    class _Rng:
        def random(self):
            return draw

        def randrange(self, *_):
            return 1

    result = SyntheticCustomers._perturb_name(_Rng(), "james", last)
    assert contains in result

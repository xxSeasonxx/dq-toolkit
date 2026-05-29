"""Synthetic data with planted ground truth, for reproducible demos and tests.

Two deterministic sources back the demos and tests:

* :class:`SyntheticTransactions` — per-customer transaction amounts with a known
  fraction of planted, labelled anomalies. Customers have heterogeneous spend
  levels, so a *global* z-score misses within-customer anomalies that a
  *per-group* detector catches — exactly the contrast the demo highlights.
* :class:`SyntheticCustomers` — customer records with planted duplicates (same
  ``true_entity_id``, perturbed fields), for entity resolution.

At test scale rows are built in Python; for large-scale generation you would
switch to ``spark.range`` + ``F.rand``.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from dqkit.core.logging import get_logger

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = get_logger(__name__)

_TXN_SCHEMA = StructType(
    [
        StructField("txn_id", LongType(), nullable=False),
        StructField("customer_id", LongType(), nullable=False),
        StructField("amount", DoubleType(), nullable=False),
        StructField("is_anomaly", LongType(), nullable=False),
    ]
)

_CUST_SCHEMA = StructType(
    [
        StructField("record_id", LongType(), nullable=False),
        StructField("name", StringType(), nullable=False),
        StructField("email", StringType(), nullable=False),
        StructField("phone", StringType(), nullable=False),
        StructField("city", StringType(), nullable=False),
        StructField("true_entity_id", LongType(), nullable=False),
    ]
)

_FIRST = [
    "james",
    "mary",
    "john",
    "patricia",
    "robert",
    "jennifer",
    "michael",
    "linda",
    "william",
    "elizabeth",
    "david",
    "susan",
    "richard",
    "karen",
    "joseph",
]
_LAST = [
    "smith",
    "johnson",
    "williams",
    "brown",
    "jones",
    "garcia",
    "davis",
    "miller",
    "wilson",
    "moore",
    "taylor",
    "anderson",
    "thomas",
    "jackson",
    "white",
]
_CITIES = ["seattle", "portland", "denver", "austin", "boston", "chicago"]


class SyntheticTransactions:
    """Per-customer transactions with planted, labelled amount anomalies.

    Each customer has its own baseline spend; normal amounts are drawn around
    that baseline while a known fraction are inflated into clear outliers and
    labelled ``is_anomaly = 1``.

    Attributes:
        name: Source name, ``"synthetic_transactions"``.
        n_customers: Number of distinct customers.
        txns_per_customer: Transactions generated per customer.
        anomaly_rate: Fraction of transactions planted as anomalies.
        seed: RNG seed for reproducibility.
    """

    name = "synthetic_transactions"

    def __init__(
        self,
        *,
        n_customers: int = 200,
        txns_per_customer: int = 50,
        anomaly_rate: float = 0.01,
        seed: int = 7,
    ) -> None:
        """Initialize the generator (see class attributes for arguments)."""
        self.n_customers = n_customers
        self.txns_per_customer = txns_per_customer
        self.anomaly_rate = anomaly_rate
        self.seed = seed

    def load(self, spark: SparkSession) -> DataFrame:
        """Generate the labelled transactions DataFrame.

        Args:
            spark: Active Spark session.

        Returns:
            DataFrame with ``txn_id``, ``customer_id``, ``amount``,
            ``is_anomaly``.
        """
        rng = random.Random(self.seed)
        rows = []
        txn_id = 0
        for customer_id in range(self.n_customers):
            baseline = rng.uniform(50.0, 500.0)
            spread = baseline * 0.1
            for _ in range(self.txns_per_customer):
                if rng.random() < self.anomaly_rate:
                    amount = baseline * rng.uniform(5.0, 15.0)
                    is_anomaly = 1
                else:
                    amount = max(0.0, rng.gauss(baseline, spread))
                    is_anomaly = 0
                rows.append((txn_id, customer_id, float(amount), is_anomaly))
                txn_id += 1
        logger.info(
            "synthetic_transactions: %d rows across %d customers (seed=%d)",
            len(rows),
            self.n_customers,
            self.seed,
        )
        return spark.createDataFrame(rows, schema=_TXN_SCHEMA)


class SyntheticCustomers:
    """Customer records with planted duplicate entities for resolution.

    A set of true entities is generated, then a fraction are emitted again as
    duplicates carrying the same ``true_entity_id`` with realistically perturbed
    fields (abbreviated names, single-character typos, reformatted or dropped
    phones, alternate email domains). Perturbations preserve the leading name
    character and the city, so blocking keeps duplicates together while the
    comparators still see a non-trivial difference to score.

    Attributes:
        name: Source name, ``"synthetic_customers"``.
        n_entities: Number of distinct real-world customers.
        dup_rate: Fraction of entities that get one or more duplicates.
        seed: RNG seed for reproducibility.
    """

    name = "synthetic_customers"

    def __init__(
        self,
        *,
        n_entities: int = 300,
        dup_rate: float = 0.3,
        seed: int = 11,
    ) -> None:
        """Initialize the generator (see class attributes for arguments)."""
        self.n_entities = n_entities
        self.dup_rate = dup_rate
        self.seed = seed

    @staticmethod
    def _perturb_name(rng: random.Random, first: str, last: str) -> str:
        """Return a name variant that keeps the leading character stable."""
        if rng.random() < 0.5:
            return f"{first[0]} {last}"  # abbreviate first name to an initial
        if len(last) > 2:  # drop one interior char of the last name
            i = rng.randrange(1, len(last))
            last = last[:i] + last[i + 1 :]
        return f"{first} {last}"

    def load(self, spark: SparkSession) -> DataFrame:
        """Generate the customer records with planted duplicates.

        Args:
            spark: Active Spark session.

        Returns:
            DataFrame with ``record_id``, ``name``, ``email``, ``phone``,
            ``city``, ``true_entity_id``.
        """
        rng = random.Random(self.seed)
        rows = []
        record_id = 0
        for entity_id in range(self.n_entities):
            first = rng.choice(_FIRST)
            last = rng.choice(_LAST)
            name = f"{first} {last}"
            email = f"{first}.{last}{entity_id}@example.com"
            area = rng.randint(200, 999)
            prefix = rng.randint(200, 999)
            suffix = rng.randint(1000, 9999)
            phone = f"{area}-{prefix}-{suffix}"
            city = rng.choice(_CITIES)
            rows.append((record_id, name, email, phone, city, entity_id))
            record_id += 1
            if rng.random() < self.dup_rate:
                for _ in range(rng.randint(1, 2)):
                    d_name = self._perturb_name(rng, first, last)
                    d_email = (
                        email
                        if rng.random() > 0.3
                        else f"{first}{last}{entity_id}@mail.com"
                    )
                    d_phone = phone.replace("-", "") if rng.random() > 0.5 else phone
                    if rng.random() < 0.1:
                        d_phone = ""
                    rows.append((record_id, d_name, d_email, d_phone, city, entity_id))
                    record_id += 1
        logger.info(
            "synthetic_customers: %d records, %d entities (seed=%d)",
            len(rows),
            self.n_entities,
            self.seed,
        )
        return spark.createDataFrame(rows, schema=_CUST_SCHEMA)

"""Shared pytest fixtures for the dqkit test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pyspark.sql import SparkSession

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """Provide one local Spark session for the whole test session."""
    session = (
        SparkSession.builder.appName("dqkit-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()

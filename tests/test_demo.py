"""Tests for the end-to-end demo logic (excluding the CLI/session wiring)."""

from __future__ import annotations

import pytest

from dqkit.demo import DemoResult, format_summary, run_demo


@pytest.fixture(scope="module")
def demo_result(spark):
    """Run the demo once and reuse the result across assertions."""
    return run_demo(spark)


def test_run_demo_produces_scored_result(demo_result):
    """The demo returns sensible counts and in-range metrics."""
    assert isinstance(demo_result, DemoResult)
    assert demo_result.n_txns > 0
    assert demo_result.n_records > demo_result.n_entities
    for metrics in (
        demo_result.zscore_metrics,
        demo_result.gaussian_metrics,
        demo_result.resolution_metrics,
    ):
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
    # the headline contrast holds on the seeded data
    assert demo_result.gaussian_metrics.f1 >= demo_result.zscore_metrics.f1


def test_format_summary_renders(demo_result):
    """The summary renders the headline lines."""
    text = format_summary(demo_result)
    assert "dqkit demo" in text
    assert "resolution" in text

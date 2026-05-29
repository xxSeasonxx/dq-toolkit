"""End-to-end demonstration: detect anomalies and resolve entities, with metrics.

Run with ``python -m dqkit.demo`` (or ``make demo``). The substance lives in
:func:`run_demo` (pure logic, no session lifecycle) and :func:`format_summary`
(rendering); :func:`main` only wires up a local Spark session so the rest stays
unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import SparkSession

from dqkit.core.logging import get_logger
from dqkit.core.results import EvaluationMetrics
from dqkit.detect import GaussianDetector, get_detector
from dqkit.eval import anomaly_metrics, pairwise_resolution_metrics
from dqkit.resolve import ResolutionPipeline
from dqkit.sources import SyntheticCustomers, SyntheticTransactions

logger = get_logger(__name__)


@dataclass(frozen=True, kw_only=True)
class DemoResult:
    """Scored outcome of the end-to-end demo.

    Attributes:
        n_txns: Number of transactions scored.
        zscore_flagged: Rows flagged by the global z-score detector.
        gaussian_flagged: Rows flagged by the per-group Gaussian detector.
        zscore_metrics: Z-score detector metrics vs ground truth.
        gaussian_metrics: Gaussian detector metrics vs ground truth.
        n_records: Customer records resolved.
        n_entities: Distinct entities after resolution.
        resolution_metrics: Pairwise resolution metrics vs ground truth.
    """

    n_txns: int
    zscore_flagged: int
    gaussian_flagged: int
    zscore_metrics: EvaluationMetrics
    gaussian_metrics: EvaluationMetrics
    n_records: int
    n_entities: int
    resolution_metrics: EvaluationMetrics


def run_demo(spark: SparkSession) -> DemoResult:
    """Run both capabilities on synthetic data and score against ground truth.

    Args:
        spark: Active Spark session.

    Returns:
        A :class:`DemoResult` with flag counts and evaluation metrics.
    """
    txns = SyntheticTransactions().load(spark)
    zscore = get_detector("zscore").detect(txns, "amount")
    gaussian = GaussianDetector(group_col="customer_id").detect(txns, "amount")

    customers = SyntheticCustomers().load(spark)
    resolution = ResolutionPipeline().resolve(customers)

    return DemoResult(
        n_txns=txns.count(),
        zscore_flagged=zscore.n_flagged,
        gaussian_flagged=gaussian.n_flagged,
        zscore_metrics=anomaly_metrics(txns, zscore.flagged),
        gaussian_metrics=anomaly_metrics(txns, gaussian.flagged),
        n_records=resolution.n_records,
        n_entities=resolution.n_entities,
        resolution_metrics=pairwise_resolution_metrics(resolution.clusters, customers),
    )


def format_summary(result: DemoResult) -> str:
    """Render a :class:`DemoResult` as a printable summary block.

    Args:
        result: The demo result to render.

    Returns:
        A multi-line summary string.
    """
    z, g, r = (
        result.zscore_metrics,
        result.gaussian_metrics,
        result.resolution_metrics,
    )
    return "\n".join(
        [
            "=== dqkit demo ===",
            f"transactions: {result.n_txns} rows",
            f"  global z-score  -> flagged {result.zscore_flagged:5d} | "
            f"P={z.precision:.2f} R={z.recall:.2f} F1={z.f1:.2f}",
            f"  per-group gauss -> flagged {result.gaussian_flagged:5d} | "
            f"P={g.precision:.2f} R={g.recall:.2f} F1={g.f1:.2f}",
            f"customers: {result.n_records} records -> {result.n_entities} entities",
            f"  resolution      -> "
            f"P={r.precision:.2f} R={r.recall:.2f} F1={r.f1:.2f}",
            "==================",
        ]
    )


def main() -> None:  # pragma: no cover - CLI / Spark session wiring
    """Build a local Spark session, run the demo, and print the summary."""
    spark = (
        SparkSession.builder.appName("dqkit-demo")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        print(format_summary(run_demo(spark)))
    finally:
        spark.stop()


if __name__ == "__main__":  # pragma: no cover
    main()

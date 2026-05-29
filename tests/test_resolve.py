"""Tests for dqkit.resolve stages, union-find, and the pipeline."""

from __future__ import annotations

import pytest

from dqkit.resolve import (
    MatchScorer,
    ResolutionPipeline,
    candidate_pairs,
    compare_pairs,
    connected_components,
)
from dqkit.resolve.cluster import _UnionFind
from dqkit.sources import SyntheticCustomers


@pytest.fixture(scope="module")
def customers(spark):
    """A small customer set with planted duplicates."""
    return SyntheticCustomers(n_entities=40, dup_rate=0.5, seed=3).load(spark)


def test_candidate_pairs_are_ordered_and_suffixed(customers):
    """Pairs carry suffixed fields and contain only ordered id pairs."""
    pairs = candidate_pairs(customers)
    assert {"record_id_a", "record_id_b", "name_a", "name_b"} <= set(pairs.columns)
    assert pairs.filter("record_id_a >= record_id_b").count() == 0


def test_compare_pairs_feature_branches(spark):
    """Comparators handle identical, empty, and differing field values."""
    rows = [
        (
            0,
            "alice smith",
            "alice smith",
            "a@x.com",
            "a@x.com",
            "555-1212",
            "5551212",
            "ny",
            "ny",
        ),
        (1, "", "", "a@x.com", "b@y.com", "", "555", "ny", "nj"),
        (2, "bob", "rob", "a@x.com", "a@x.com", "111", "222", "ny", "ny"),
    ]
    cols = [
        "k",
        "name_a",
        "name_b",
        "email_a",
        "email_b",
        "phone_a",
        "phone_b",
        "city_a",
        "city_b",
    ]
    out = {
        r["k"]: r for r in compare_pairs(spark.createDataFrame(rows, cols)).collect()
    }
    assert out[0]["name_sim"] == pytest.approx(1.0)
    assert out[0]["email_exact"] == pytest.approx(1.0)
    assert out[0]["phone_exact"] == pytest.approx(1.0)
    assert out[1]["name_sim"] == pytest.approx(1.0)  # both empty -> max_len 0
    assert out[1]["email_exact"] == pytest.approx(0.0)
    assert out[1]["phone_exact"] == pytest.approx(0.0)  # one phone blank
    assert out[2]["name_sim"] < 1.0
    assert out[2]["phone_exact"] == pytest.approx(0.0)  # different digits


def test_match_scorer_decides_by_threshold(spark):
    """Full agreement matches; agreement on city alone does not."""
    rows = [
        (1, 2, 1.0, 1.0, 1.0, 1.0),
        (3, 4, 0.0, 0.0, 0.0, 1.0),
    ]
    cols = [
        "record_id_a",
        "record_id_b",
        "name_sim",
        "email_exact",
        "phone_exact",
        "city_sim",
    ]
    compared = spark.createDataFrame(rows, cols)
    scorer = MatchScorer()
    scored = {r["record_id_a"]: r for r in scorer.score(compared).collect()}
    assert scored[1]["is_match"] is True
    assert scored[3]["is_match"] is False
    matched = scorer.matches(compared).collect()
    assert len(matched) == 1
    assert matched[0]["record_id_a"] == 1


def test_union_find_merges_and_compresses():
    """Union-find merges sets, no-ops within a set, and halves paths."""
    uf = _UnionFind()
    uf.union(1, 2)
    uf.union(2, 3)
    assert uf.find(3) == 1
    uf.union(1, 3)  # already merged -> no-op branch
    assert uf.find(2) == 1

    chained = _UnionFind()
    chained._parent = {3: 2, 2: 1, 1: 1}
    assert chained.find(3) == 1
    assert chained._parent[3] == 1  # path halved toward the root


def test_connected_components_assigns_entities(spark):
    """Linked records share an entity; unlinked records stay singletons."""
    records = spark.createDataFrame([(i,) for i in [1, 2, 3, 4, 5]], ["record_id"])
    matches = spark.createDataFrame([(1, 2), (2, 3)], ["record_id_a", "record_id_b"])
    mapping = {
        r["record_id"]: r["entity_id"]
        for r in connected_components(records, matches).collect()
    }
    assert mapping[1] == mapping[2] == mapping[3] == 1
    assert mapping[4] == 4
    assert mapping[5] == 5


def test_resolution_pipeline_merges_duplicates(customers):
    """The pipeline resolves some records into shared entities."""
    report = ResolutionPipeline().resolve(customers)
    assert report.n_records == report.clusters.count()
    assert report.n_records > report.n_entities

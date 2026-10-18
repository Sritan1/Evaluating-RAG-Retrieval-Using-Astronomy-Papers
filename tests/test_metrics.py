import random

import pytest
from sklearn.metrics import ndcg_score, recall_score

from astro_retrieval.metrics import bootstrap_mean, ndcg_at_k, recall_at_k, reciprocal_rank_at_k, score_query


def test_reviewed_example():
    ranking, labels = ["B", "A", "C"], {"A", "C", "D"}
    assert recall_at_k(ranking, labels, 2) == pytest.approx(1 / 3)
    scores = score_query(ranking, labels)
    assert scores["recall@10"] == pytest.approx(2 / 3)
    assert scores["recall@50"] == scores["recall@100"] == scores["recall@10"]
    assert scores["rr@100"] == 0.5
    assert scores["ndcg@10"] == pytest.approx(0.53072, abs=0.000005)


def test_empty_and_ideal_rankings():
    assert set(score_query([], {"A", "C"}).values()) == {0.0}
    assert set(score_query(["A", "C"], {"A", "C"}).values()) == {1.0}


def test_cutoffs_exclude_later_hits():
    ranking = [f"N{i}" for i in range(100)] + ["A"]
    assert reciprocal_rank_at_k(ranking, {"A"}) == 0
    assert reciprocal_rank_at_k(ranking, {"A"}, 101) == pytest.approx(1 / 101)
    assert recall_at_k(ranking, {"A"}, 100) == 0
    assert ndcg_at_k(ranking, {"A"}, 10) == 0


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        score_query(["A", "A"], {"A"})
    with pytest.raises(ValueError, match="relevant label"):
        score_query(["A"], set())
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k(["A"], {"A"}, 0)


def test_matches_independent_reference_on_seeded_rankings():
    rng = random.Random(42)
    documents = [f"D{i}" for i in range(30)]
    for _ in range(30):
        labels = set(rng.sample(documents, rng.randint(1, len(documents))))
        ranking = rng.sample(documents, len(documents))
        truth = [int(key in labels) for key in documents]
        scores = [len(ranking) - ranking.index(key) for key in documents]
        for k in (1, 10, 50):
            retrieved = set(ranking[:k])
            predicted = [int(key in retrieved) for key in documents]
            assert recall_at_k(ranking, labels, k) == pytest.approx(recall_score(truth, predicted))
            reference = ndcg_score([truth], [scores], k=k, ignore_ties=True)
            assert ndcg_at_k(ranking, labels, k) == pytest.approx(reference, abs=1e-12)


def test_bootstrap_means_and_paired_differences():
    scores = {"A": 0.2, "B": 0.6}
    result = bootstrap_mean(scores)
    assert result["mean"] == pytest.approx(0.4)
    assert result == bootstrap_mean(dict(reversed(list(scores.items()))))
    assert result["ci95"][0] < result["mean"] < result["ci95"][1]
    assert bootstrap_mean({"A": .4, "B": .4})["ci95"] == pytest.approx([.4, .4])
    assert bootstrap_mean(scores, reference=scores)["ci95"] == [0, 0]
    improved = {key: value + .25 for key, value in scores.items()}
    paired = bootstrap_mean(improved, reference=scores)
    assert paired["mean"] == pytest.approx(.25)
    assert paired["ci95"] == pytest.approx([.25, .25])


def test_bootstrap_rejects_missing_and_nonfinite_scores():
    for scores in ({}, {"A": float("nan")}, {"A": float("inf")}):
        with pytest.raises(ValueError):
            bootstrap_mean(scores)
    with pytest.raises(ValueError, match="identical papers"):
        bootstrap_mean({"A": .2}, reference={"B": .2})

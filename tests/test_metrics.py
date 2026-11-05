import random
from copy import deepcopy

import pytest
from sklearn.metrics import ndcg_score, recall_score

from astro_retrieval.baselines import baseline_ranking, score_run, summarize_scores
from astro_retrieval.metrics import (
    bootstrap_mean, compare_hypotheses, holm_adjust, ndcg_at_k, paired_sign_flip,
    recall_at_k, reciprocal_rank_at_k, score_query,
)


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


def test_sign_flip_exact_values_symmetry_and_seeded_replay():
    assert paired_sign_flip({"A": 1, "B": 1})["p_value"] == .5
    assert paired_sign_flip({"A": 0, "B": 0})["p_value"] == 1
    values = {f"P{i}": (i - 4) / 20 for i in range(15)}
    result = paired_sign_flip(values)
    assert result["method"] == "monte_carlo" and result["draws"] == 10000
    assert result == paired_sign_flip(dict(reversed(list(values.items()))))
    assert result["p_value"] == paired_sign_flip({key: -v for key, v in values.items()})["p_value"]
    assert 0 < result["p_value"] <= 1
    for values in ({}, {"A": float("nan")}, {"A": float("inf")}):
        with pytest.raises(ValueError):
            paired_sign_flip(values)
    with pytest.raises(ValueError, match="identical papers"):
        paired_sign_flip({"A": .2}, reference={"B": .2})
    from scipy.stats import permutation_test
    reference = permutation_test(([.2, -.1, .4],), lambda x: sum(x) / len(x),
                                 permutation_type="samples", n_resamples=float("inf"))
    assert paired_sign_flip(dict(zip("ABC", [.2, -.1, .4])))["p_value"] == pytest.approx(reference.pvalue)


def test_holm_adjustment_preserves_names_and_rejects_invalid_probabilities():
    values = dict(zip("ABCD", [.01, .03, .04, .20]))
    assert holm_adjust(values) == pytest.approx(dict(zip("ABCD", [.04, .09, .09, .20])))
    assert holm_adjust(dict(reversed(list(values.items())))) == holm_adjust(values)
    for values in ({}, {"A": float("nan")}, {"A": -1}, {"A": 1.1}):
        with pytest.raises(ValueError):
            holm_adjust(values)


def test_planned_contrasts_keep_papers_paired_and_validate_inputs():
    systems = {name: {form: {paper: {"recall@100": .1, "ndcg@10": .4}
                                  for paper in ("A", "B")}
                      for form in ("title", "abstract")}
               for name in ("bm25", "hybrid", "reranker", "popularity", "random")}
    for paper in ("A", "B"):
        systems["reranker"]["title"][paper]["ndcg@10"] = .6
        systems["reranker"]["abstract"][paper]["ndcg@10"] = .3
    result = compare_hypotheses(systems)
    assert result["family"] == ["H1-title", "H1-abstract", "H2", "H3"]
    assert result["contrasts"]["H2"]["bootstrap"]["mean"] == pytest.approx(.3)
    assert result["contrasts"]["H2"]["test"]["p_value"] == .5
    assert not result["h1_supported"]
    assert compare_hypotheses(systems) == result
    missing = deepcopy(systems)
    del missing["hybrid"]["abstract"]["B"]
    with pytest.raises(ValueError, match="identical papers"):
        compare_hypotheses(missing)
    invalid = deepcopy(systems)
    invalid["reranker"]["title"]["A"]["ndcg@10"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compare_hypotheses(invalid)


@pytest.fixture
def baseline_inputs():
    dates = {"W1": "2024-01-01", "W2": "2020-01-01", "W3": "2020-01-01",
             "W4": "2022-01-01", "W5": "2024-01-01", "W6": "2025-01-01", "W7": None}
    refs = [("W4", "W3"), ("W4", "W3"), ("W2", "W2"), ("W1", "W2"),
            ("W5", "W2"), ("W6", "W2"), ("W7", "W2")]
    evaluation = {"splits": {"dev": ["W1"], "test": ["W6"]}, "qrels": {"W1": ["W3"]},
                  "queries": [{"query_id": f"W1:{form}", "paper_id": "W1",
                               "query_type": form, "split": "dev"}
                              for form in ("title", "abstract")]}
    return dates, refs, evaluation


def test_baselines_obey_dates_counts_and_determinism(baseline_inputs):
    dates, refs, _ = baseline_inputs
    assert baseline_ranking(dates, refs, "W1", "popularity") == ["W3", "W2", "W4"]
    dates.update({f"W{i}": "2019-01-01" for i in range(10, 130)})
    for system in ("random", "popularity"):
        ranking = baseline_ranking(dates, refs, "W1", system)
        assert len(ranking) == len(set(ranking)) == 100
        assert not set(ranking) & {"W1", "W5", "W6", "W7"}
        assert ranking == baseline_ranking(dict(reversed(list(dates.items()))),
                                           list(reversed(refs)), "W1", system)
    with pytest.raises(ValueError, match="known publication date"):
        baseline_ranking(dates, refs, "missing", "random")


def test_run_validation_and_paired_forms(baseline_inputs):
    dates, _, evaluation = baseline_inputs
    rankings = {f"W1:{form}": ["W3", "W2"] for form in ("title", "abstract")}
    scores = score_run(rankings, evaluation, dates)
    summary = summarize_scores(scores, resamples=50)
    assert summary["title"] == summary["abstract"]
    assert summary["title"]["mrr@100"]["mean"] == 1
    for ranking in (["W3", "W3"], ["W1"], ["W5"], ["W99"], ["W3"] * 101):
        with pytest.raises(ValueError):
            score_run({**rankings, "W1:title": ranking}, evaluation, dates)
    for bad in ({}, {**rankings, "W6:title": []}):
        with pytest.raises(ValueError, match="exactly the development"):
            score_run(bad, evaluation, dates)
    bad_scores = deepcopy(scores)
    bad_scores["abstract"] = {}
    with pytest.raises(ValueError, match="identical papers"):
        summarize_scores(bad_scores)


def test_explicit_test_split_rejects_wrong_coverage(baseline_inputs):
    dates, _, evaluation = baseline_inputs
    evaluation["qrels"]["W6"] = ["W3"]
    evaluation["queries"].extend({"query_id": f"W6:{form}", "paper_id": "W6",
                                  "query_type": form, "split": "test"}
                                 for form in ("title", "abstract"))
    before = deepcopy(evaluation)
    rankings = {f"W6:{form}": ["W3"] for form in ("title", "abstract")}
    result = score_run(rankings, evaluation, dates, split="test")
    assert set(result["title"]) == set(result["abstract"]) == {"W6"}
    assert result["title"]["W6"]["recall@100"] == 1
    with pytest.raises(ValueError, match="development"):
        score_run(rankings, evaluation, dates)
    dev = {f"W1:{form}": ["W3"] for form in ("title", "abstract")}
    for bad in (dev, {**rankings, **dev}, {"W6:title": ["W3"]},
                {**rankings, "W99:title": []}):
        with pytest.raises(ValueError, match="exactly the test"):
            score_run(bad, evaluation, dates, split="test")
    dates.update(W8="2025-01-01", W9="2026-01-01")
    for candidate in ("W6", "W7", "W8", "W9"):
        with pytest.raises(ValueError, match="ineligible"):
            score_run({**rankings, "W6:title": [candidate]}, evaluation, dates, split="test")
    with pytest.raises(ValueError, match="Unknown"):
        score_run(rankings, evaluation, dates, split="other")
    overlap = deepcopy(evaluation)
    overlap["splits"]["dev"].append("W6")
    with pytest.raises(ValueError, match="disjoint"):
        score_run(rankings, overlap, dates, split="test")
    assert evaluation == before

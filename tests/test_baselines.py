import copy
import hashlib
import json

import pytest

from astro_retrieval.baselines import baseline_ranking, run_baselines, score_run, summarize_scores
from astro_retrieval.corpus import build_corpus


@pytest.fixture
def inputs():
    dates = {"W1": "2024-01-01", "W2": "2020-01-01", "W3": "2020-01-01",
             "W4": "2022-01-01", "W5": "2024-01-01", "W6": "2025-01-01", "W7": None}
    refs = [("W4", "W3"), ("W4", "W3"), ("W2", "W2"), ("W1", "W2"),
            ("W5", "W2"), ("W6", "W2"), ("W7", "W2")]
    evaluation = {"splits": {"dev": ["W1"], "test": ["W6"]}, "qrels": {"W1": ["W3"]},
                  "queries": [{"query_id": f"W1:{form}", "paper_id": "W1",
                               "query_type": form, "split": "dev"}
                              for form in ("title", "abstract")]}
    return dates, refs, evaluation


def test_baselines_obey_dates_counts_and_determinism(inputs):
    dates, refs, _ = inputs
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


def test_run_validation_and_paired_forms(inputs):
    dates, _, evaluation = inputs
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
    bad_scores = copy.deepcopy(scores)
    bad_scores["abstract"] = {}
    with pytest.raises(ValueError, match="identical papers"):
        summarize_scores(bad_scores)


def test_saved_baselines_repeat_without_changing_inputs(inputs, tmp_path):
    dates, refs, evaluation = inputs
    source = tmp_path / "collection.json"
    seeds = sorted({citing for citing, _ in refs})
    bundle = {"report": {"schema_version": 1, "candidate_ids": seeds},
              "papers": {key: {"id": key, "title": key, "abstract": "A study.",
                               "publication_date": date} for key, date in dates.items()},
              "references": {key: [target for citing, target in refs if citing == key]
                             for key in seeds}}
    source.write_text(json.dumps(bundle), encoding="utf-8")
    database = tmp_path / "corpus.sqlite"
    build_corpus(source, database)
    evaluation["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    frozen = tmp_path / "evaluation.json"
    frozen.write_text(json.dumps(evaluation), encoding="utf-8")
    before = (database.read_bytes(), frozen.read_bytes())
    output = tmp_path / "baselines.json"
    result = run_baselines(database, frozen, output, resamples=50)
    assert run_baselines(database, frozen, output, resamples=50) == result
    assert (database.read_bytes(), frozen.read_bytes()) == before
    for system in result["systems"].values():
        assert system["rankings"]["W1:title"] == system["rankings"]["W1:abstract"]
        assert set(system["scores"]["title"]) == {"W1"}

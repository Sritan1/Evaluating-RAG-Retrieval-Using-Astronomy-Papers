import json
import random
import hashlib

import bm25s
import pytest

from astro_retrieval.bm25 import build_index, build_saved_index, rank_bm25, tokenize
from astro_retrieval.corpus import build_corpus
from astro_retrieval.baselines import run_baselines
from astro_retrieval.bm25 import run_bm25, selection_score


def test_approved_token_rules():
    assert tokenize("Kepler-22b and PLANETS") == ["kepler", "22b", "and", "planets"]
    assert tokenize("Ｋｅｐｌｅｒ-22b") == tokenize("Kepler-22b")
    assert tokenize("... _ - !") == []


def test_reviewed_counts_and_input_order():
    papers = {"W1": {"title": "planet", "abstract": "planet orbit"},
              "W2": {"title": None, "abstract": "orbit star"}}
    index = build_index(papers)
    assert index["lengths"] == {"W1": 3, "W2": 2}
    assert index["average_length"] == 2.5
    assert index["postings"] == {"planet": {"W1": 2}, "orbit": {"W1": 1, "W2": 1},
                                 "star": {"W2": 1}}
    assert index == build_index(dict(reversed(list(papers.items()))))
    with pytest.raises(ValueError, match="requires papers"):
        build_index({})


@pytest.fixture
def scoring_inputs():
    texts = {"W1": "planet planet orbit", "W2": "planet star", "W3": "star",
             "W4": "planet " * 100, "W5": "planet " * 200, "W6": "planet"}
    papers = {key: {"title": None, "abstract": text} for key, text in texts.items()}
    dates = {key: "2020-01-01" for key in papers}
    dates.update(W4="2024-01-01", W5="2025-01-01", W6=None)
    return papers, dates


def test_reviewed_scores_and_temporal_statistics(scoring_inputs):
    papers, dates = scoring_inputs
    index = build_index(papers)
    ranking = rank_bm25(index, dates, "W4", "planet")
    assert [key for key, _ in ranking] == ["W1", "W2"]
    assert [score for _, score in ranking] == pytest.approx([.566580, .470004], abs=5e-7)
    assert rank_bm25(index, dates, "W4", "planet planet") == ranking
    for query in ("", "unseen"):
        assert rank_bm25(index, dates, "W4", query) == []
    for key in ("W4", "W5", "W6"):
        papers[key]["abstract"] = "different text " * 300
    assert rank_bm25(build_index(papers), dates, "W4", "planet") == ranking


def test_ties_limit_and_invalid_parameters():
    papers = {f"W{i}": {"title": None, "abstract": "planet"} for i in range(105)}
    dates = {key: "2020-01-01" for key in papers}
    dates["W0"] = "2024-01-01"
    index = build_index(papers)
    ranking = rank_bm25(index, dates, "W0", "planet")
    assert [key for key, _ in ranking] == sorted(set(papers) - {"W0"})[:100]
    for parameters in ({"k1": -1}, {"k1": float("nan")}, {"b": 1.1}, {"b": float("inf")}):
        with pytest.raises(ValueError):
            rank_bm25(index, dates, "W0", "planet", **parameters)


def test_matches_reference_with_same_formula():
    rng = random.Random(42)
    words = ["planet", "star", "orbit", "transit", "mass"]
    texts = {f"W{i}": " ".join(rng.choices(words, k=rng.randint(1, 15))) for i in range(12)}
    papers = {key: {"title": None, "abstract": text} for key, text in texts.items()}
    dates = {key: "2020-01-01" for key in papers}
    dates["W0"] = "2024-01-01"
    ids = sorted(set(papers) - {"W0"})
    index = build_index(papers)
    for k1, b in ((1.2, .75), (.5, 0), (2, 1)):
        reference = bm25s.BM25(k1=k1, b=b, method="atire", idf_method="lucene", dtype="float64")
        reference.index([tokenize(texts[key]) for key in ids], show_progress=False)
        for query in ("planet", "star orbit", "transit mass mass"):
            expected = reference.get_scores(sorted(set(tokenize(query))))
            actual = dict(rank_bm25(index, dates, "W0", query, k1=k1, b=b))
            assert [actual.get(key, 0) for key in ids] == pytest.approx(expected, abs=1e-10)


def test_reviewed_selection_average():
    def scores(title, abstract):
        return {form: {"W1": {"recall@100": value}}
                for form, value in (("title", title), ("abstract", abstract))}
    first, second = scores(.2, .4), scores(.3, .4)
    assert selection_score(first) == pytest.approx(.3)
    assert selection_score(second) == pytest.approx(.35)


@pytest.fixture
def tuning_inputs(tmp_path):
    texts = {"W1": "planet orbit", "W2": "star mass", "W3": "planet star",
             "W4": "planet", "W5": "star", "W6": "orbit"}
    bundle = {"report": {"schema_version": 1, "candidate_ids": ["W4", "W5"]},
              "papers": {key: {"id": key, "title": text, "abstract": text,
                               "publication_date": "2020-01-01" if key < "W4" else "2024-01-01"}
                         for key, text in texts.items()},
              "references": {"W4": ["W1"], "W5": ["W2"]}}
    source = tmp_path / "source.json"
    source.write_text(json.dumps(bundle), encoding="utf-8")
    database, index, evaluation, baseline, output = [tmp_path / name for name in (
        "corpus.sqlite", "index.json", "evaluation.json", "baseline.json", "bm25.json")]
    build_corpus(source, database)
    build_saved_index(database, index)
    evaluation.write_text(json.dumps({
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "splits": {"dev": ["W4", "W5"], "test": ["W6"]},
        "qrels": {"W4": ["W1"], "W5": ["W2"]},
        "queries": [{"query_id": f"{key}:{form}", "paper_id": key, "query_type": form,
                     "split": "dev" if key != "W6" else "test"}
                    for key in ("W4", "W5", "W6") for form in ("title", "abstract")],
    }), encoding="utf-8")
    run_baselines(database, evaluation, baseline, resamples=50)
    return database, index, evaluation, baseline, output


def test_tuning_grid_dev_only_repeat_and_ties(tuning_inputs):
    before = [path.read_bytes() for path in tuning_inputs[:-1]]
    result = run_bm25(*tuning_inputs)
    assert len(result["trials"]) == 9
    assert {(trial["k1"], trial["b"]) for trial in result["trials"]} == {
        (k1, b) for k1 in (.6, 1.2, 1.8) for b in (.25, .75, 1.)}
    assert result["selected_trial"] == 0
    assert len({trial["selection_score"] for trial in result["trials"]}) == 1
    for trial in result["trials"]:
        assert set(trial["rankings"]) == {f"{key}:{form}" for key in ("W4", "W5")
                                          for form in ("title", "abstract")}
    assert run_bm25(*tuning_inputs) == result
    assert [path.read_bytes() for path in tuning_inputs[:-1]] == before


def test_tuning_mismatch_and_interruption_publish_nothing(tuning_inputs):
    for position in (1, 2, 3):
        path = tuning_inputs[position]
        raw = path.read_bytes()
        changed = json.loads(raw)
        changed["source_sha256"] = "wrong"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            run_bm25(*tuning_inputs)
        assert not tuning_inputs[-1].exists()
        path.write_bytes(raw)

    def interrupted(_):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        run_bm25(*tuning_inputs, progress=interrupted)
    assert not tuning_inputs[-1].exists()

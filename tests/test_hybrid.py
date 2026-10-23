import hashlib
import json

import pytest

from astro_retrieval.corpus import build_corpus
from astro_retrieval.hybrid import rank_fusion, run_hybrid


def test_rrf_example_empty_lists_ties_and_limits():
    result = rank_fusion(["A", "B"], ["B", "C"], k=1)
    assert [key for key, _ in result] == ["B", "A", "C"]
    assert [score for _, score in result] == pytest.approx([5 / 6, 1 / 2, 1 / 3])
    assert rank_fusion([], []) == []
    assert rank_fusion(["A", "B"], [], k=1) == [("A", .5), ("B", 1 / 3)]
    assert rank_fusion(["B"], ["A"]) == [("A", 1 / 61), ("B", 1 / 61)]
    assert rank_fusion(["A", "B"], ["B", "C"]) == rank_fusion(["B", "C"], ["A", "B"])
    result = rank_fusion([f"A{i:03}" for i in range(100)], [f"B{i:03}" for i in range(100)])
    assert len(result) == len({key for key, _ in result}) == 100
    assert [key for key, _ in result[:4]] == ["A000", "B000", "A001", "B001"]
    for k in (0, -1, True, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive integer"):
            rank_fusion([], [], k=k)
    for ranking in (["A", "A"], [""], [1], "A", [str(i) for i in range(101)]):
        with pytest.raises(ValueError):
            rank_fusion(ranking, [])


@pytest.fixture
def hybrid_inputs(tmp_path):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    source = save("source.json", {
        "report": {"schema_version": 1, "candidate_ids": ["W2"]},
        "papers": {key: {"id": key, "title": "paper", "abstract": "study",
                          "publication_date": date} for key, date in (
            ("W1", "2020-01-01"), ("W2", "2024-01-01"), ("W3", "2025-01-01"))},
        "references": {"W2": ["W1"]},
    })
    database = tmp_path / "papers.sqlite"
    build_corpus(source, database)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    evaluation = save("evaluation.json", {
        "source_sha256": source_hash, "splits": {"dev": ["W2"], "test": ["W3"]},
        "qrels": {"W2": ["W1"]}, "queries": [
            {"query_id": f"W2:{form}", "paper_id": "W2", "query_type": form, "split": "dev"}
            for form in ("title", "abstract")],
    })
    identity = {"source_sha256": source_hash, "split": "dev", "baseline_sha256": "baseline",
                "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
                "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest()}
    rankings = {f"W2:{form}": ["W1"] for form in ("title", "abstract")}
    bm25 = save("bm25.json", {**identity, "selected_trial": 0, "trials": [{"rankings": rankings}]})
    dense_identity = {**identity, "bm25_sha256": hashlib.sha256(bm25.read_bytes()).hexdigest()}
    general = save("general.json", {**dense_identity,
        "settings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        "rankings": {key: [] for key in rankings}})
    indus = save("indus.json", {**dense_identity,
        "settings": {"model": "nasa-impact/nasa-smd-ibm-st-v2"}, "rankings": rankings})
    return database, evaluation, bm25, general, indus, tmp_path / "hybrid.json"


def test_runner_selects_shared_dense_and_k_with_reproducible_dev_results(hybrid_inputs):
    args = hybrid_inputs
    before = {path: path.read_bytes() for path in args[:-1]}
    progress = []
    result = run_hybrid(*args, progress=progress.append)
    assert result["selected_dense"] == "indus"
    assert result["dense_selection_scores"] == {"general": 0, "indus": 1}
    assert [row["k"] for row in progress] == [60, 20, 100]
    assert result["selected_trial"] == 0
    for trial in result["trials"]:
        assert trial["rankings"] == {"W2:title": ["W1"], "W2:abstract": ["W1"]}
    assert run_hybrid(*args) == result
    assert json.loads(args[-1].read_text()) == result
    assert all(path.read_bytes() == raw for path, raw in before.items())
    general = json.loads(args[3].read_text())
    general["rankings"] = result["trials"][0]["rankings"]
    args[3].write_text(json.dumps(general), encoding="utf-8")
    with pytest.raises(ValueError, match="Existing hybrid output differs"):
        run_hybrid(*args)
    tied = run_hybrid(*args[:-1], args[-1].with_name("tied.json"))
    assert tied["selected_dense"] == "general"


def test_runner_rejects_mismatches_invalid_rankings_and_incomplete_output(hybrid_inputs):
    args = hybrid_inputs
    for position, field, value in ((1, "source_sha256", "wrong"),
                                   (2, "database_sha256", "wrong"),
                                   (3, "evaluation_sha256", "wrong"),
                                   (4, "split", "test"), (4, "bm25_sha256", "wrong"),
                                   (3, "settings", {"model": "wrong"}),
                                   (3, "rankings", {"W3:title": ["W1"]}),
                                   (3, "rankings", {"W2:title": ["W2"], "W2:abstract": []})):
        path = args[position]
        raw = path.read_bytes()
        changed = json.loads(raw)
        changed[field] = value
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            run_hybrid(*args)
        assert not args[-1].exists()
        path.write_bytes(raw)

    def interrupt(row):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        run_hybrid(*args, progress=interrupt)
    assert not args[-1].exists()

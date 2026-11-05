import hashlib
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from astro_retrieval.corpus import build_corpus
from astro_retrieval.rerank import (
    PairScorer, TokenPairScorer, aggregate_chunks, rerank_top,
    run_chunk_comparison, run_depth_comparison, run_rerank, score_cached, token_chunks,
)


def test_prefix_scores_preserve_tail_and_reject_invalid_values():
    assert rerank_top(["A", "B", "C", "D"], [("A", .2), ("B", .9)], depth=2) == ["B", "A", "C", "D"]
    assert rerank_top(["B", "A", "C"], [("B", 1), ("A", 1)], depth=2) == ["A", "B", "C"]
    assert rerank_top([], []) == []
    assert rerank_top(["A"], [("A", -3)]) == ["A"]
    for scored in ([], [("A", 1), ("A", 2)], [("C", 1)], [("A", float("nan"))],
                   [("A", float("inf"))], [("A", True)]):
        with pytest.raises(ValueError):
            rerank_top(["A"], scored)
    with pytest.raises(ValueError):
        rerank_top(["A", "A"], [("A", 1)])
    for depth in (0, 101, True):
        with pytest.raises(ValueError):
            rerank_top([], [], depth=depth)


def test_pair_cache_resumes_checks_identity_and_detects_corruption(tmp_path):
    class Scorer:
        settings = {"model": "fake", "max_tokens": 512}

        def __call__(self, pairs):
            calls.extend(pairs)
            if pairs[0][0] == "second" and len(calls) == 2:
                raise RuntimeError("interrupted")
            return [.9], [600]

    calls = []
    pairs = {"Q1": [("A", "first", "paper")], "Q2": [("B", "second", "paper")]}
    path = tmp_path / "scores.json"
    scorer = Scorer()
    with pytest.raises(ValueError, match="Complete pair"):
        score_cached(pairs, path, scorer, identity={"source": "s"}, read_only=True)
    assert not path.exists() and calls == []
    with pytest.raises(RuntimeError):
        score_cached(pairs, path, scorer, identity={"source": "s"})
    assert not json.loads(path.read_text())["complete"]
    incomplete = path.read_bytes()
    with pytest.raises(ValueError, match="Complete pair"):
        score_cached(pairs, path, scorer, identity={"source": "s"}, read_only=True)
    assert path.read_bytes() == incomplete and len(calls) == 2
    result = score_cached(pairs, path, scorer, identity={"source": "s"})
    assert calls == [("first", "paper"), ("second", "paper"), ("second", "paper")]
    assert result["Q1"]["token_lengths"] == [600]
    assert score_cached(pairs, path, scorer, identity={"source": "s"}) == result
    assert len(calls) == 3
    saved_bytes, modified = path.read_bytes(), path.stat().st_mtime_ns
    assert score_cached(pairs, path, scorer, identity={"source": "s"}, read_only=True) == result
    assert path.read_bytes() == saved_bytes and path.stat().st_mtime_ns == modified and len(calls) == 3
    for changed_pairs, identity in (({**pairs, "Q1": [("A", "changed", "paper")]}, {"source": "s"}),
                                    (pairs, {"source": "other"})):
        with pytest.raises(ValueError, match="different inputs"):
            score_cached(changed_pairs, path, scorer, identity=identity)
    scorer.settings = {**scorer.settings, "model": "other"}
    with pytest.raises(ValueError, match="different inputs"):
        score_cached(pairs, path, scorer, identity={"source": "s"})
    scorer.settings = Scorer.settings
    saved = json.loads(path.read_text())
    saved["queries"]["Q1"]["row"]["scores"] = [.1]
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        score_cached(pairs, path, scorer, identity={"source": "s"})


def test_adapter_uses_local_raw_scores_and_shared_token_limit(monkeypatch):
    import astro_retrieval.rerank as rerank
    monkeypatch.setattr(rerank, "version", lambda name: "test")
    identity = object()

    class Tokenizer:
        def __call__(self, first, second, **kwargs):
            assert kwargs == {"truncation": False, "padding": False, "add_special_tokens": True}
            return {"input_ids": [list(range(len(a.split()) + len(b.split()) + 3))
                                  for a, b in zip(first, second)]}

    class Model:
        def __init__(self, folder, **kwargs):
            assert folder == "snapshot"
            assert kwargs == {"device": "cpu", "local_files_only": True, "trust_remote_code": False,
                              "token": False, "activation_fn": identity}
            self.tokenizer = Tokenizer()

        def float(self):
            return self

        def eval(self):
            return self

        def predict(self, pairs, **kwargs):
            assert self.max_seq_length == 512 and self.tokenizer.truncation_side == "right"
            assert kwargs["processing_kwargs"] == {"text": {"max_length": 512, "truncation": "longest_first"}}
            assert kwargs["batch_size"] == 8 and not kwargs["apply_softmax"] and kwargs["prompt"] == ""
            return np.array([-2., 3.])

    def snapshot(model, **kwargs):
        assert model == "cross-encoder/ms-marco-MiniLM-L6-v2"
        assert kwargs["revision"] == "233902d25c440f23af6f7d6e94d2946bac0bee0a"
        assert kwargs["local_files_only"] and kwargs["token"] is False
        return "snapshot"

    torch = SimpleNamespace(set_num_threads=lambda n: None, get_num_interop_threads=lambda: 1,
                            manual_seed=lambda n: None, use_deterministic_algorithms=lambda flag: None,
                            inference_mode=nullcontext, nn=SimpleNamespace(Identity=lambda: identity))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=Model))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
                     "HF_HUB_DISABLE_IMPLICIT_TOKEN", "TOKENIZERS_PARALLELISM"):
        monkeypatch.setenv(variable, "test")
    scorer = PairScorer()
    assert scorer.model is None
    assert scorer([("planet " * 300, "star " * 300), ("query", "paper")]) == ([-2., 3.], [603, 5])
    monkeypatch.setattr(scorer.model, "predict", lambda *args, **kwargs: np.array([float("nan")]))
    with pytest.raises(ValueError, match="finite raw score"):
        scorer([("query", "paper")])


def test_runner_validates_dev_inputs_resumes_and_replays_without_model(tmp_path, monkeypatch):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    source = save("source.json", {"report": {"schema_version": 1, "candidate_ids": ["W4"]},
        "papers": {f"W{i}": {"id": f"W{i}", "title": f"title{i}", "abstract": f"abstract{i}",
                               "publication_date": f"202{i}-01-01"} for i in range(1, 6)},
        "references": {"W4": ["W2"]}})
    database = tmp_path / "papers.sqlite"
    build_corpus(source, database)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    evaluation = save("evaluation.json", {"source_sha256": source_hash,
        "splits": {"dev": ["W4"], "test": ["W5"]}, "qrels": {"W4": ["W2"]},
        "queries": [{"query_id": f"W4:{form}", "paper_id": "W4", "query_type": form, "split": "dev"}
                    for form in ("title", "abstract")]})
    hybrid = save("hybrid.json", {"source_sha256": source_hash, "split": "dev", "selected_trial": 0,
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        "trials": [{"rankings": {f"W4:{form}": ["W1", "W2", "W3"] for form in ("title", "abstract")}}]})
    args = (database, evaluation, hybrid, tmp_path / "cache.json", tmp_path / "report.json")
    calls = []
    class Scorer:
        settings = {"model": "fake", "max_tokens": 512}

        def __call__(self, pairs):
            calls.append(pairs)
            return [.2, .9], [600, 100]

    def interrupt(row):
        raise RuntimeError("interrupted")

    original = {path: path.read_bytes() for path in args[:3]}
    changed = json.loads(hybrid.read_text())
    changed["source_sha256"] = "wrong"
    hybrid.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="identities"):
        run_rerank(*args, scorer=Scorer(), depth=2)
    assert not calls and not args[3].exists()
    hybrid.write_bytes(original[hybrid])
    changed = json.loads(hybrid.read_text())
    changed["trials"][0]["rankings"]["W4:title"] = ["W4"]
    hybrid.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="ineligible"):
        run_rerank(*args, scorer=Scorer(), depth=2)
    assert not calls
    hybrid.write_bytes(original[hybrid])
    with pytest.raises(RuntimeError):
        run_rerank(*args, scorer=Scorer(), depth=2, progress=interrupt)
    assert not args[-1].exists()
    result = run_rerank(*args, scorer=Scorer(), depth=2)
    assert len(calls) == 2
    assert calls == [[("abstract4", "title1\nabstract1"), ("abstract4", "title2\nabstract2")],
                     [("title4", "title1\nabstract1"), ("title4", "title2\nabstract2")]]
    assert all(ranking == ["W2", "W1", "W3"] for ranking in result["rankings"].values())
    assert result["truncation"]["title"]["truncated_count"] == 1
    assert result["differences"]["hybrid"]["title"]["recall@100"]["ci95"] == [0, 0]
    class Replay:
        settings = Scorer.settings

        def __call__(self, pairs):
            pytest.fail("Replay loaded model")

    monkeypatch.setattr("astro_retrieval.rerank.PairScorer", Replay)
    assert run_rerank(*args, depth=2) == result
    assert all(path.read_bytes() == raw for path, raw in original.items())


def test_chunk_coverage_and_aggregation():
    query, document = list(range(300)), list(range(600))
    chunks = token_chunks(query, document)
    assert len(chunks) == 6
    assert all(len(q) + len(d) + 3 <= 512 for q, d in chunks)
    assert chunks[0][0] + chunks[3][0] == query
    assert chunks[0][1] + chunks[1][1] + chunks[2][1] == document
    assert token_chunks([1], list(range(400))) == [([1], list(range(400)))]
    assert aggregate_chunks([1, 3], "maximum") == 3
    assert aggregate_chunks([1, 3], "mean") == 2
    for scores, method in (([], "mean"), ([float("nan")], "maximum"), ([1], "unknown")):
        with pytest.raises(ValueError):
            aggregate_chunks(scores, method)


def test_saved_score_comparisons_reuse_inputs_resume_and_select_shared_settings(tmp_path, monkeypatch):
    import astro_retrieval.rerank as rerank
    monkeypatch.setattr(rerank, "version", lambda name: "test")
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    papers = {f"W{i}": {"id": f"W{i}", "title": f"doc{i}", "abstract": "study",
                        "publication_date": "2020-01-01"} for i in range(1, 21)}
    papers.update({"W24": {"id": "W24", "title": "query title", "abstract": "query abstract",
                            "publication_date": "2024-01-01"},
                   "W25": {"id": "W25", "title": "test", "abstract": "test",
                            "publication_date": "2025-01-01"}})
    source = save("source.json", {"report": {"schema_version": 1, "candidate_ids": ["W24"]},
                                  "papers": papers, "references": {"W24": ["W20"]}})
    database = tmp_path / "papers.sqlite"
    build_corpus(source, database)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    evaluation = save("evaluation.json", {"source_sha256": source_hash,
        "splits": {"dev": ["W24"], "test": ["W25"]}, "qrels": {"W24": ["W20"]},
        "queries": [{"query_id": f"W24:{form}", "paper_id": "W24", "query_type": form, "split": "dev"}
                    for form in ("title", "abstract")]})
    hybrid = save("hybrid.json", {"source_sha256": source_hash, "split": "dev", "selected_trial": 0,
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        "trials": [{"rankings": {f"W24:{form}": [f"W{i}" for i in range(1, 21)]
                                 for form in ("title", "abstract")}}]})

    def tokenize(text):
        if text == "query title":
            return [90] * 3
        if text == "query abstract":
            return [91] * 300
        number = int(text.split("\n")[0][3:])
        return [number] * (50 if number == 1 else 600)

    class Baseline:
        settings = PairScorer().settings

        def __call__(self, pairs):
            return ([tokenize(d)[0] if tokenize(d)[0] <= 10 else -tokenize(d)[0] for q, d in pairs],
                    [len(tokenize(q)) + len(tokenize(d)) + 3 for q, d in pairs])

    cache, report = tmp_path / "baseline-cache.json", tmp_path / "baseline.json"
    baseline = run_rerank(database, evaluation, hybrid, cache, report, scorer=Baseline())
    calls = []
    class Chunks:
        settings = TokenPairScorer().settings

        def tokenize(self, text):
            return tokenize(text)

        def __call__(self, pairs):
            calls.append(pairs)
            assert all(d[0] != 1 for q, d in pairs)
            return [2 * d[0] for q, d in pairs], [len(q) + len(d) + 3 for q, d in pairs]

    args = (database, evaluation, hybrid, cache, report, tmp_path / "chunks")
    original = {path: path.read_bytes() for path in args[:5]}
    def interrupt(row):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        run_chunk_comparison(*args, scorer=Chunks(), progress=interrupt)
    assert not (args[-1] / "comparison.json").exists()
    result = run_chunk_comparison(*args, scorer=Chunks())
    assert len(calls) == 2 and len(calls[0]) == 114
    assert result["selected_trial"] == 1
    assert [trial["selection_score"] for trial in result["trials"]] == [0, 1, 1]
    for form in ("title", "abstract"):
        assert result["workload"][form]["reused_pairs"] == 1
        assert result["differences"]["maximum"]["truncation"][form]["recall@100"]["ci95"] == [0, 0]
        assert result["trials"][1]["rankings"][f"W24:{form}"][0] == "W20"
    assert result["trials"][0]["rankings"] == baseline["rankings"]
    assert run_chunk_comparison(*args, scorer=Chunks()) == result
    assert len(calls) == 2
    assert all(path.read_bytes() == raw for path, raw in original.items())
    changed = Chunks()
    changed.settings = {**changed.settings, "document_chunk_tokens": 200}
    with pytest.raises(ValueError, match="Chunk settings"):
        run_chunk_comparison(*args, scorer=changed)

    monkeypatch.setattr(PairScorer, "__call__", lambda *args: pytest.fail("Unexpected inference"))
    output = tmp_path / "depths.json"
    depth_args = (*args[:5], output)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_depth_comparison(*depth_args, progress=interrupt)
    assert not output.exists()
    depths = run_depth_comparison(*depth_args)
    assert [trial["depth"] for trial in depths["trials"]] == [20, 10]
    assert depths["selected_trial"] == 0
    assert depths["trials"][0]["rankings"] == baseline["rankings"]
    for query, ranking in depths["trials"][1]["rankings"].items():
        assert query.startswith("W24:")
        assert ranking == [f"W{i}" for i in range(10, 0, -1)] + [f"W{i}" for i in range(11, 21)]
    assert run_depth_comparison(*depth_args) == depths
    assert all(path.read_bytes() == raw for path, raw in original.items())
    changed_hybrid = json.loads(hybrid.read_text())
    changed_hybrid["database_sha256"] = "wrong"
    hybrid.write_text(json.dumps(changed_hybrid), encoding="utf-8")
    mismatch_output = tmp_path / "mismatch.json"
    with pytest.raises(ValueError, match="identities"):
        run_depth_comparison(*depth_args[:-1], mismatch_output)
    assert not mismatch_output.exists()

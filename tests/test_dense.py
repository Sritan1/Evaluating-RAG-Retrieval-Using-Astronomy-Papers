import json
import hashlib
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from astro_retrieval.dense import EmbeddingEncoder, encode_cached, evaluate_dense, normalize_vectors, rank_dense, run_dense
from astro_retrieval.corpus import build_corpus


SETTINGS = {"model": "fake", "revision": "v1", "max_tokens": 2}


def test_saved_vectors_align_reuse_and_reject_changed_inputs(tmp_path):
    calls = []

    def encoder(texts):
        calls.append(texts)
        return [[len(text), 1] for text in texts], [len(text.split()) for text in texts]

    texts = {"W2": "three token input", "W1": "one"}
    result = encode_cached(texts, tmp_path, encoder, source_sha256="source", settings=SETTINGS)
    assert calls == [["one", "three token input"]]
    assert result["ids"] == ["W1", "W2"]
    assert result["vectors"].dtype == np.float32
    assert np.linalg.norm(result["vectors"], axis=1) == pytest.approx([1, 1])
    assert result["vectors"] == pytest.approx(normalize_vectors([[3, 1], [17, 1]]))
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["complete"] and saved["truncated_count"] == 1
    replay = encode_cached(dict(reversed(list(texts.items()))), tmp_path, encoder,
                           source_sha256="source", settings=SETTINGS)
    assert len(calls) == 1
    assert np.array_equal(result["vectors"], replay["vectors"])
    for changed_texts, source, settings in (
        ({"W1": "changed", "W2": "three token input"}, "source", SETTINGS),
        (texts, "changed", SETTINGS), (texts, "source", {**SETTINGS, "model": "other"}),
        (texts, "source", {**SETTINGS, "max_tokens": 3}),
    ):
        with pytest.raises(ValueError, match="different source"):
            encode_cached(changed_texts, tmp_path, encoder, source_sha256=source, settings=settings)


def test_interrupted_encoding_resumes_only_missing_batches(tmp_path):
    calls = []

    def interrupted(texts):
        calls.append(texts)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return [[1, 0]], [1]

    texts = {"A": "first", "B": "second"}
    with pytest.raises(RuntimeError):
        encode_cached(texts, tmp_path, interrupted, source_sha256="s", settings=SETTINGS, chunk_size=1)
    assert not json.loads((tmp_path / "manifest.json").read_text())["complete"]
    resumed = []

    def encoder(texts):
        resumed.extend(texts)
        return [[0, 2]], [3]

    result = encode_cached(texts, tmp_path, encoder, source_sha256="s", settings=SETTINGS, chunk_size=1)
    assert resumed == ["second"]
    assert result["vectors"].tolist() == [[1, 0], [0, 1]]
    assert json.loads((tmp_path / "manifest.json").read_text())["complete"]
    for values in ([[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]]):
        with pytest.raises(ValueError):
            normalize_vectors(values)


def test_cosine_example_masks_ties_and_limits():
    ids = ["A", "B", "C", "F", "Q", "S", "U"]
    vectors = np.array([[1, 0], [.6, .8], [0, 1], [1, 0], [1, 0], [1, 0], [1, 0]])
    dates = {key: "2020-01-01" for key in ids}
    dates.update(F="2025-01-01", Q="2024-01-01", S="2024-01-01", U=None)
    ranking = rank_dense(ids, vectors, [1, 0], dates, "Q")
    assert [key for key, _ in ranking] == ["A", "B", "C"]
    assert [score for _, score in ranking] == pytest.approx([1, .6, 0])
    with pytest.raises(ValueError, match="identical papers"):
        rank_dense(ids[:-1], vectors[:-1], [1, 0], dates, "Q")
    dates = {f"W{i}": "2020-01-01" for i in range(105)}
    dates["W0"] = "2024-01-01"
    ids = sorted(dates)
    ranked = rank_dense(ids, np.tile([1, 0], (105, 1)), [1, 0], dates, "W0")
    assert [key for key, _ in ranked] == sorted(set(ids) - {"W0"})[:100]


def test_dense_evaluation_requires_only_dev_queries(tmp_path):
    def encoder(texts):
        return [[1, 0] for _ in texts], [1] * len(texts)

    documents = encode_cached({"A": "paper", "Q": "query", "T": "test"}, tmp_path / "docs",
                              encoder, source_sha256="s", settings=SETTINGS)
    queries = encode_cached({"Q:title": "query", "Q:abstract": "query"}, tmp_path / "queries",
                            encoder, source_sha256="s", settings=SETTINGS)
    evaluation = {"source_sha256": "s", "splits": {"dev": ["Q"], "test": ["T"]},
                  "qrels": {"Q": ["A"]}, "queries": [
                      {"query_id": f"Q:{form}", "paper_id": "Q", "query_type": form, "split": "dev"}
                      for form in ("title", "abstract")]}
    dates = {"A": "2020-01-01", "Q": "2024-01-01", "T": "2025-01-01"}
    result = evaluate_dense(documents, queries, evaluation, dates)
    assert result["rankings"] == {"Q:title": ["A"], "Q:abstract": ["A"]}
    queries["ids"] = ["T:abstract", "T:title"]
    queries["identity"]["ids"] = queries["ids"]
    with pytest.raises(ValueError, match="exactly development"):
        evaluate_dense(documents, queries, evaluation, dates)


@pytest.mark.parametrize("name,model,revision", [
    ("general", "sentence-transformers/all-MiniLM-L6-v2",
     "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    ("indus", "nasa-impact/nasa-smd-ibm-st-v2",
     "e4a3e22048f0173ef150470cfbf67e9e8aaf4f8c"),
])
def test_adapter_loads_locally_and_counts_before_truncation(monkeypatch, name, model, revision):
    import astro_retrieval.dense as dense
    monkeypatch.setattr(dense, "version", lambda name: "test")
    calls = {}

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs == {"truncation": False, "padding": False, "add_special_tokens": True}
            return {"input_ids": [list(range(len(text.split()) + 2)) for text in texts]}

    class Model:
        def __init__(self, folder, **kwargs):
            assert folder == "local-snapshot"
            assert kwargs == {"device": "cpu", "local_files_only": True,
                              "trust_remote_code": False, "token": False}
            self.tokenizer = Tokenizer()

        def float(self):
            return self

        def eval(self):
            return self

        def encode(self, texts, **kwargs):
            assert self.max_seq_length == 512
            assert self.tokenizer.truncation_side == "right"
            assert kwargs["batch_size"] == 8 and kwargs["prompt"] == ""
            return np.tile([1, 0], (len(texts), 1))

    def snapshot(model, **kwargs):
        assert kwargs["local_files_only"] and kwargs["token"] is False
        assert kwargs["revision"] == revision
        assert "pytorch_model.bin" in kwargs["allow_patterns"]
        calls["model"] = model
        return "local-snapshot"

    torch = SimpleNamespace(set_num_threads=lambda value: None, get_num_interop_threads=lambda: 1,
                            manual_seed=lambda value: None, use_deterministic_algorithms=lambda value: None,
                            inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
                 "HF_HUB_DISABLE_IMPLICIT_TOKEN", "TOKENIZERS_PARALLELISM"):
        monkeypatch.setenv(variable, "test")
    with pytest.raises(ValueError, match="Unknown embedding model"):
        EmbeddingEncoder("unknown")
    encoder = EmbeddingEncoder(name)
    assert encoder.settings["model"] == model
    assert encoder.settings["revision"] == revision
    assert encoder.model is None
    _, lengths = encoder(["planet " * 600, "star"])
    assert lengths == [602, 3]
    assert calls["model"] == encoder.settings["model"]


def test_runner_checks_sources_dev_queries_and_model_free_replay(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "report": {"schema_version": 1, "candidate_ids": ["W2"]},
        "papers": {key: {"id": key, "title": title, "abstract": abstract,
                           "publication_date": date} for key, title, abstract, date in (
            ("W1", "paper", "earlier study", "2020-01-01"),
            ("W2", "query title", "query abstract", "2024-01-01"),
            ("W3", "test title", "test abstract", "2025-01-01"))},
        "references": {"W2": ["W1"]},
    }), encoding="utf-8")
    database = tmp_path / "corpus.sqlite"
    build_corpus(source, database)
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "splits": {"dev": ["W2"], "test": ["W3"]}, "qrels": {"W2": ["W1"]},
        "queries": [{"query_id": f"W2:{form}", "paper_id": "W2", "query_type": form,
                     "split": "dev"} for form in ("title", "abstract")],
    }), encoding="utf-8")
    identity = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "split": "dev",
                "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
                "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest()}
    rankings = {f"W2:{form}": ["W1"] for form in ("title", "abstract")}
    baseline, bm25 = tmp_path / "baseline.json", tmp_path / "bm25.json"
    baseline.write_text(json.dumps({**identity, "systems": {
        name: {"rankings": rankings} for name in ("random", "popularity")}}), encoding="utf-8")
    bm25.write_text(json.dumps({**identity, "selected_trial": 0,
                               "trials": [{"rankings": rankings}]}), encoding="utf-8")
    calls = []
    class Encoder:
        settings = SETTINGS

        def __call__(self, texts):
            calls.append(texts)
            return np.tile([1, 0], (len(texts), 1)), [3] * len(texts)

    args = (database, evaluation, baseline, bm25, tmp_path / "vectors", tmp_path / "dense.json")
    original = evaluation.read_bytes()
    changed = json.loads(original)
    changed["source_sha256"] = "wrong"
    evaluation.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="identities must match"):
        run_dense(*args, encoder=Encoder())
    assert not calls and not args[-1].exists()
    evaluation.write_bytes(original)
    result = run_dense(*args, encoder=Encoder())
    assert calls[1] == ["query abstract", "query title"]
    assert result["truncation"]["abstract"]["truncated_count"] == 1
    assert result["rankings"] == rankings
    class ReplayEncoder:
        settings = SETTINGS

        def __call__(self, texts):
            pytest.fail("Replay attempted to load/encode with a model")

    monkeypatch.setattr("astro_retrieval.dense.EmbeddingEncoder", ReplayEncoder)
    assert run_dense(*args) == result

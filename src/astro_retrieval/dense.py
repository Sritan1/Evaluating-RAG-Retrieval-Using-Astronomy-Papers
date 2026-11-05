import hashlib
import json
import os
import platform
import sqlite3
import tempfile
import time
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

import numpy as np

from astro_retrieval.baselines import score_run, summarize_scores
from astro_retrieval.corpus import eligible_documents
from astro_retrieval.ingest.expansion import write_json


def normalize_vectors(values):
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or not all(vectors.shape) or not np.isfinite(vectors).all():
        raise ValueError("Vectors must be a nonempty finite matrix")
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if np.any(norms == 0):
        raise ValueError("Vectors must be nonzero")
    return (vectors / norms[:, None]).astype(np.float32)


def encode_cached(texts, directory, encoder, *, source_sha256, settings, chunk_size=256, progress=None,
                  read_only=False):
    if not texts or any(not isinstance(key, str) or not key for key in texts):
        raise ValueError("Encoding requires nonempty text IDs")
    if any(not isinstance(text, str) or not text.strip() for text in texts.values()):
        raise ValueError("Encoding requires nonempty text")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Chunk size must be a positive integer")
    if (not settings.get("model") or not settings.get("revision")
            or type(settings.get("max_tokens")) is not int or settings["max_tokens"] < 1):
        raise ValueError("Settings require model, revision, and positive max_tokens")
    ids = sorted(texts)
    raw = json.dumps(texts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    identity = json.loads(json.dumps({
        "source_sha256": source_sha256, "text_sha256": hashlib.sha256(raw).hexdigest(),
        "settings": settings, "chunk_size": chunk_size, "ids": ids,
    }, allow_nan=False))
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if read_only and not manifest_path.is_file():
        raise ValueError("Complete document cache required")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest["identity"] != identity:
            raise ValueError("Saved embeddings have different source, text, or settings")
    else:
        manifest = {"schema_version": 1, "identity": identity, "complete": False, "chunks": {}}
        write_json(manifest, manifest_path)
    if read_only and not manifest["complete"]:
        raise ValueError("Complete document cache required")
    batches, token_counts = [], []
    dimension = None
    for start in range(0, len(ids), chunk_size):
        batch_ids = ids[start:start + chunk_size]
        name = f"{start:08d}.npz"
        path = directory / name
        if name in manifest["chunks"]:
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["chunks"][name]:
                raise ValueError("Saved embedding batch is missing or changed")
            with np.load(path, allow_pickle=False) as stored:
                if stored["ids"].tolist() != batch_ids:
                    raise ValueError("Saved vector IDs are misaligned")
                vectors, lengths = stored["vectors"], stored["token_lengths"]
            if vectors.dtype != np.float32 or not np.allclose(
                    vectors, normalize_vectors(vectors), atol=1e-6, rtol=1e-6):
                raise ValueError("Saved vectors must be normalized float32")
        else:
            if manifest["complete"]:
                raise ValueError("Completed cache is missing a batch")
            values, lengths = encoder([texts[key] for key in batch_ids])
            vectors = normalize_vectors(values)
            lengths = np.asarray(lengths)
        if (vectors.shape[0] != len(batch_ids) or lengths.shape != (len(batch_ids),)
                or not np.issubdtype(lengths.dtype, np.integer) or np.any(lengths < 1)):
            raise ValueError("Encoder rows and token lengths must match input IDs")
        if dimension is not None and vectors.shape[1] != dimension:
            raise ValueError("Embedding dimensions differ between batches")
        dimension = vectors.shape[1]
        if name not in manifest["chunks"]:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
                    temporary = Path(handle.name)
                    np.savez(handle, ids=np.asarray(batch_ids), vectors=vectors, token_lengths=lengths)
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            manifest["chunks"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            write_json(manifest, manifest_path)
        batches.append(vectors)
        token_counts.extend(lengths.tolist())
        if progress is not None:
            progress({"completed": start + len(batch_ids), "total": len(ids)})
    manifest.update(complete=True, dimension=dimension,
                    truncated_count=sum(count > settings["max_tokens"] for count in token_counts),
                    longest_input_tokens=max(token_counts))
    if not read_only:
        write_json(manifest, manifest_path)
    return {"ids": ids, "vectors": np.concatenate(batches), "token_lengths": token_counts,
            "identity": identity}


def rank_dense(document_ids, document_vectors, query_vector, dates, query_id):
    if (len(set(document_ids)) != len(document_ids) or set(document_ids) != set(dates)
            or list(document_ids) != sorted(document_ids)):
        raise ValueError("Sorted vector IDs and dates must contain identical papers")
    if dates.get(query_id) is None:
        raise ValueError("Query requires a known publication date")
    vectors = normalize_vectors(document_vectors)
    query = normalize_vectors(np.asarray(query_vector)[None, :])
    if vectors.shape != (len(document_ids), query.shape[1]):
        raise ValueError("Vector IDs or dimensions do not match")
    similarities = vectors @ query[0]
    eligible = eligible_documents(dates, query_id)
    return sorted(((key, float(score)) for key, score in zip(document_ids, similarities)
                   if key in eligible), key=lambda row: (-row[1], row[0]))[:100]


def evaluate_dense(documents, queries, evaluation, dates, *, split="dev"):
    if split not in ("dev", "test"):
        raise ValueError("Unknown evaluation split")
    if documents["identity"]["settings"] != queries["identity"]["settings"]:
        raise ValueError("Document and query model settings must match")
    if any(bundle["identity"]["source_sha256"] != evaluation["source_sha256"]
           or bundle["ids"] != bundle["identity"]["ids"] for bundle in (documents, queries)):
        raise ValueError("Embedding source or vector identities differ")
    expected = sorted(f"{key}:{form}" for key in evaluation["splits"][split]
                      for form in ("title", "abstract"))
    if queries["ids"] != expected or len(queries["vectors"]) != len(expected):
        label = "development" if split == "dev" else "test"
        raise ValueError(f"Query vectors must contain exactly {label} queries")
    score_run({key: [] for key in expected}, evaluation, dates, split=split)
    rankings = {key: [paper for paper, _ in rank_dense(
        documents["ids"], documents["vectors"], vector, dates, key.rsplit(":", 1)[0])]
        for key, vector in zip(queries["ids"], queries["vectors"])}
    scores = score_run(rankings, evaluation, dates, split=split)
    return {"rankings": rankings, "scores": scores, "summary": summarize_scores(scores)}


class EmbeddingEncoder:
    def __init__(self, name="general", model_cache="models/hub"):
        models = {
            "general": ("sentence-transformers/all-MiniLM-L6-v2",
                        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
            "indus": ("nasa-impact/nasa-smd-ibm-st-v2",
                      "e4a3e22048f0173ef150470cfbf67e9e8aaf4f8c"),
        }
        if name not in models:
            raise ValueError("Unknown embedding model")
        model, revision = models[name]
        self.model_cache = model_cache
        self.model = None
        self.settings = {
            "model": model,
            "revision": revision,
            "max_tokens": 512, "truncation_side": "right", "batch_size": 8,
            "device": "cpu", "threads": 4, "seed": 42, "prompt": "",
            "document_text": "title newline abstract", "dtype": "float32",
            "python": platform.python_version(),
            "versions": {name: version(name) for name in
                         ("sentence-transformers", "transformers", "torch", "numpy")},
        }

    def __call__(self, texts):
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                          TOKENIZERS_PARALLELISM="false")
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        if self.model is None:
            torch.set_num_threads(4)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            torch.manual_seed(42)
            torch.use_deterministic_algorithms(True)
            weights = ("model.safetensors" if self.settings["model"] == "sentence-transformers/all-MiniLM-L6-v2"
                       else "pytorch_model.bin")
            folder = snapshot_download(self.settings["model"], revision=self.settings["revision"],
                                       cache_dir=self.model_cache, local_files_only=True, token=False,
                                       allow_patterns=["*.json", "*.txt", "*.model", weights])
            self.model = SentenceTransformer(folder, device="cpu", local_files_only=True,
                                            trust_remote_code=False, token=False)
            self.model.float()
            self.model.max_seq_length = 512
            self.model.tokenizer.truncation_side = "right"
            self.model.tokenizer.padding_side = "right"
            self.model.eval()
        tokens = self.model.tokenizer(texts, truncation=False, padding=False, add_special_tokens=True)
        lengths = [len(row) for row in tokens["input_ids"]]
        with torch.inference_mode():
            vectors = self.model.encode(texts, batch_size=8, show_progress_bar=False,
                                        normalize_embeddings=True, convert_to_numpy=True, prompt="")
        return vectors, lengths


def run_dense(database, evaluation_path, baseline_path, bm25_path, cache_directory, output,
              *, encoder=None, progress=None):
    database, evaluation_path, baseline_path, bm25_path, cache_directory, output = map(
        Path, (database, evaluation_path, baseline_path, bm25_path, cache_directory, output))
    inputs = (database, evaluation_path, baseline_path, bm25_path)
    if output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Output must not replace an input")
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    evaluation, baseline, bm25 = [json.loads(path.read_text(encoding="utf-8"))
                                 for path in inputs[1:]]
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        manifest = {key: json.loads(value) for key, value in connection.execute("SELECT * FROM manifest")}
        rows = connection.execute("SELECT openalex_id, title, abstract, publication_date FROM papers")
        papers = {key: {"title": title, "abstract": abstract, "date": date}
                  for key, title, abstract, date in rows}
    source = manifest["source_sha256"]
    if evaluation["source_sha256"] != source or any(
        report["source_sha256"] != source or report["database_sha256"] != hashes[database]
        or report["evaluation_sha256"] != hashes[evaluation_path] or report["split"] != "dev"
        for report in (baseline, bm25)
    ):
        raise ValueError("Database, evaluation, and comparison identities must match")
    dates = {key: paper["date"] for key, paper in papers.items()}
    query_texts = {f"{key}:{form}": papers[key][form] for key in evaluation["splits"]["dev"]
                   for form in ("title", "abstract")}
    score_run({key: [] for key in query_texts}, evaluation, dates)
    reference_runs = {system: baseline["systems"][system]["rankings"] for system in ("random", "popularity")}
    reference_runs["bm25"] = bm25["trials"][bm25["selected_trial"]]["rankings"]
    references = {system: score_run(run, evaluation, dates) for system, run in reference_runs.items()}
    if encoder is None:
        encoder = EmbeddingEncoder()
    deadline = time.monotonic() + 4 * 60 * 60

    def encode(texts):
        if time.monotonic() >= deadline:
            raise TimeoutError("Dense run reached its four-hour budget")
        return encoder(texts)

    bundles = {}
    for stage, texts in (
        ("documents", {key: (paper["title"] or "") + "\n" + paper["abstract"]
                       for key, paper in papers.items()}), ("queries", query_texts),
    ):
        def report_progress(row):
            if time.monotonic() >= deadline:
                raise TimeoutError("Dense run reached its four-hour budget")
            if progress is not None:
                progress({"stage": stage, **row})

        bundles[stage] = encode_cached(texts, cache_directory / stage, encode,
                                      source_sha256=source, settings=encoder.settings,
                                      progress=report_progress)
    result = evaluate_dense(bundles["documents"], bundles["queries"], evaluation, dates)
    result.update(schema_version=1, split="dev", settings=encoder.settings, source_sha256=source,
                  database_sha256=hashes[database], evaluation_sha256=hashes[evaluation_path],
                  baseline_sha256=hashes[baseline_path], bm25_sha256=hashes[bm25_path])
    result["truncation"] = {}
    for stage, bundle in bundles.items():
        groups = {stage: list(range(len(bundle["ids"])))} if stage == "documents" else {
            form: [i for i, key in enumerate(bundle["ids"]) if key.endswith(":" + form)]
            for form in ("title", "abstract")}
        for group, positions in groups.items():
            lengths = [bundle["token_lengths"][i] for i in positions]
            result["truncation"][group] = {"count": len(lengths),
                "truncated_count": sum(length > encoder.settings["max_tokens"] for length in lengths),
                "longest_input_tokens": max(lengths)}
    result["differences"] = {system: summarize_scores(result["scores"], reference=scores)
                             for system, scores in references.items()}
    result["interval_note"] = "95% paired paper bootstrap, 2000 draws, seed 42; development comparisons"
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path] for path in inputs):
        raise ValueError("Inputs changed during the dense run")
    if time.monotonic() >= deadline:
        raise TimeoutError("Dense run reached its four-hour budget")
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError("Existing dense output differs; use a new path")
    else:
        write_json(result, output)
    return result

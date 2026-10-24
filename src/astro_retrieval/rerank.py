import hashlib
import json
import math
import os
import platform
import sqlite3
import time
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

import numpy as np

from astro_retrieval.baselines import score_run, summarize_scores
from astro_retrieval.ingest.expansion import write_json


def rerank_top(ranking, scored, *, depth=20):
    if type(depth) is not int or not 1 <= depth <= 100:
        raise ValueError("Depth must be an integer from 1 to 100")
    if (not isinstance(ranking, list) or len(ranking) > 100
            or any(not isinstance(key, str) or not key for key in ranking)
            or len(set(ranking)) != len(ranking)):
        raise ValueError("Ranking requires at most 100 unique paper IDs")
    scores = {}
    for key, value in scored:
        if (key in scores or key not in ranking[:depth]
                or not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value)):
            raise ValueError("Scores must be unique, finite, and match the ranking prefix")
        scores[key] = value
    if set(scores) != set(ranking[:depth]):
        raise ValueError("Scores must cover the entire ranking prefix")
    return sorted(scores, key=lambda key: (-scores[key], key)) + ranking[depth:]


def _digest(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def score_cached(pairs, path, scorer, *, identity, progress=None):
    path = Path(path)
    cache_identity = {"inputs": identity, "settings": scorer.settings, "pairs_sha256": _digest(pairs)}
    cache_identity = json.loads(json.dumps(cache_identity, allow_nan=False))
    if path.exists():
        cache = json.loads(path.read_text(encoding="utf-8"))
        if cache.get("schema_version") != 1 or cache["identity"] != cache_identity:
            raise ValueError("Saved pair scores have different inputs or settings")
    else:
        cache = {"schema_version": 1, "identity": cache_identity, "complete": False, "queries": {}}
        write_json(cache, path)
    if not set(cache["queries"]) <= set(pairs):
        raise ValueError("Saved scores contain unexpected queries")
    for number, query in enumerate(sorted(pairs), start=1):
        rows = pairs[query]
        ids = [row[0] for row in rows]
        if query in cache["queries"]:
            saved = cache["queries"][query]
            row = saved["row"]
            if saved["sha256"] != _digest(row) or row["ids"] != ids:
                raise ValueError("Saved pair scores changed or IDs differ")
        else:
            if cache["complete"]:
                raise ValueError("Completed score cache is missing a query")
            values, lengths = scorer([(text, document) for _, text, document in rows]) if rows else ([], [])
            row = {"ids": ids, "scores": np.asarray(values).tolist(),
                   "token_lengths": np.asarray(lengths).tolist()}
        if (not isinstance(row["scores"], list) or len(row["scores"]) != len(ids)
                or len(row["token_lengths"]) != len(ids)
                or any(type(length) is not int or length < 1 for length in row["token_lengths"])):
            raise ValueError("Scores and token lengths must match candidate IDs")
        if (len(set(ids)) != len(ids) or any(not isinstance(key, str) or not key for key in ids)
                or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                       or not math.isfinite(value) for value in row["scores"])):
            raise ValueError("Cached IDs must be unique and scores finite")
        if query not in cache["queries"]:
            cache["queries"][query] = {"row": row, "sha256": _digest(row)}
            write_json(cache, path)
        if progress is not None:
            progress({"completed": number, "total": len(pairs)})
    cache["complete"] = True
    write_json(cache, path)
    return {query: saved["row"] for query, saved in cache["queries"].items()}


class PairScorer:
    def __init__(self, model_cache="models/hub"):
        self.model_cache = model_cache
        self.model = None
        self.settings = {
            "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
            "max_tokens": 512, "truncation": "longest_first", "truncation_side": "right",
            "device": "cpu", "dtype": "float32", "threads": 4, "batch_size": 8,
            "seed": 42, "activation": "identity", "prompt": "",
            "document_text": "title newline abstract", "python": platform.python_version(),
            "versions": {name: version(name) for name in
                         ("sentence-transformers", "transformers", "torch", "numpy")},
        }

    def _load(self):
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                          TOKENIZERS_PARALLELISM="false")
        import torch
        from huggingface_hub import snapshot_download
        from sentence_transformers import CrossEncoder

        if self.model is None:
            torch.set_num_threads(4)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            torch.manual_seed(42)
            torch.use_deterministic_algorithms(True)
            folder = snapshot_download(self.settings["model"], revision=self.settings["revision"],
                                       cache_dir=self.model_cache, local_files_only=True, token=False,
                                       allow_patterns=["*.json", "*.txt", "*.safetensors"])
            self.model = CrossEncoder(folder, device="cpu", local_files_only=True,
                                      trust_remote_code=False, token=False,
                                      activation_fn=torch.nn.Identity())
            self.model.float()
            self.model.max_seq_length = 512
            self.model.tokenizer.truncation_side = "right"
            self.model.tokenizer.padding_side = "right"
            self.model.eval()

    def __call__(self, pairs):
        self._load()
        import torch

        tokens = self.model.tokenizer([pair[0] for pair in pairs], [pair[1] for pair in pairs],
                                      truncation=False, padding=False, add_special_tokens=True)
        lengths = [len(row) for row in tokens["input_ids"]]
        with torch.inference_mode():
            values = self.model.predict(pairs, batch_size=8, show_progress_bar=False,
                                        convert_to_numpy=True, apply_softmax=False, prompt="",
                                        processing_kwargs={"text": {"max_length": 512,
                                                                    "truncation": "longest_first"}})
        values = np.asarray(values)
        if values.shape != (len(pairs),) or not np.isfinite(values).all():
            raise ValueError("Model must return one finite raw score per pair")
        return values.tolist(), lengths


def token_chunks(query, document):
    if any(type(token) is not int or token < 0 for token in (*query, *document)):
        raise ValueError("Token IDs must be nonnegative integers")
    if len(query) + len(document) + 3 <= 512:
        return [(query, document)]
    queries = [query[i:i + 254] for i in range(0, len(query), 254)] or [[]]
    documents = [document[i:i + 255] for i in range(0, len(document), 255)] or [[]]
    return [(q, d) for q in queries for d in documents]


def aggregate_chunks(scores, method):
    if not scores or any(not math.isfinite(score) for score in scores):
        raise ValueError("Chunk scores must be nonempty and finite")
    if method == "maximum":
        return max(scores)
    if method == "mean":
        return math.fsum(scores) / len(scores)
    raise ValueError("Unknown chunk aggregation")


class TokenPairScorer(PairScorer):
    def __init__(self, model_cache="models/hub"):
        super().__init__(model_cache)
        self.settings.update(input_mode="token_ids", truncation="none",
                             query_chunk_tokens=254, document_chunk_tokens=255)
        self.tokenizer = None

    def tokenize(self, text):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            folder = (Path(self.model_cache) / ("models--" + self.settings["model"].replace("/", "--"))
                      / "snapshots" / self.settings["revision"])
            self.tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True,
                                                           trust_remote_code=False, token=False)
        return self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]

    def __call__(self, pairs):
        self._load()
        import torch

        tokenizer = self.model.tokenizer
        features, lengths = [], []
        for query, document in pairs:
            if len(query) + len(document) + 3 > 512:
                raise ValueError("Chunk pair exceeds 512 tokens")
            ids = [tokenizer.cls_token_id, *query, tokenizer.sep_token_id, *document, tokenizer.sep_token_id]
            features.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                             "token_type_ids": [0] * (len(query) + 2) + [1] * (len(document) + 1)})
            lengths.append(len(ids))
        values = []
        with torch.inference_mode():
            for start in range(0, len(features), 8):
                batch = tokenizer.pad(features[start:start + 8], padding=True, return_tensors="pt")
                values.extend(self.model(dict(batch))["scores"].squeeze(-1).cpu().float().tolist())
        if len(values) != len(pairs) or not all(math.isfinite(value) for value in values):
            raise ValueError("Model must return one finite raw score per chunk")
        return values, lengths


def run_rerank(database, evaluation_path, hybrid_path, cache_path, output, *,
               scorer=None, depth=20, progress=None):
    rerank_top([], [], depth=depth)
    database, evaluation_path, hybrid_path, cache_path, output = map(
        Path, (database, evaluation_path, hybrid_path, cache_path, output))
    inputs = (database, evaluation_path, hybrid_path)
    if (len({path.resolve() for path in (*inputs, cache_path, output)}) != 5):
        raise ValueError("Input, cache, and output paths must be distinct")
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    evaluation, hybrid = [json.loads(path.read_text(encoding="utf-8")) for path in inputs[1:]]
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        manifest = {key: json.loads(value) for key, value in connection.execute("SELECT * FROM manifest")}
        rows = connection.execute("SELECT openalex_id, title, abstract, publication_date FROM papers")
        papers = {key: {"title": title, "abstract": abstract, "date": date}
                  for key, title, abstract, date in rows}
    source = manifest["source_sha256"]
    if (evaluation["source_sha256"] != source or hybrid["source_sha256"] != source
            or hybrid["database_sha256"] != hashes[database]
            or hybrid["evaluation_sha256"] != hashes[evaluation_path] or hybrid["split"] != "dev"):
        raise ValueError("Database, evaluation, and hybrid identities must match")
    trial = hybrid["selected_trial"]
    if type(trial) is not int or not 0 <= trial < len(hybrid["trials"]):
        raise ValueError("Invalid selected hybrid trial")
    rankings = hybrid["trials"][trial]["rankings"]
    dates = {key: paper["date"] for key, paper in papers.items()}
    reference = score_run(rankings, evaluation, dates)
    pairs = {}
    for query, ranking in rankings.items():
        paper, form = query.rsplit(":", 1)
        pairs[query] = [(key, papers[paper][form], (papers[key]["title"] or "") + "\n" + papers[key]["abstract"])
                        for key in ranking[:depth]]
    scorer = PairScorer() if scorer is None else scorer
    identity = {"source_sha256": source, "database_sha256": hashes[database],
                "evaluation_sha256": hashes[evaluation_path], "hybrid_sha256": hashes[hybrid_path], "depth": depth}
    deadline = time.monotonic() + 4 * 60 * 60

    def report_progress(row):
        if time.monotonic() >= deadline:
            raise TimeoutError("Reranker reached its four-hour budget")
        if progress is not None:
            progress(row)

    cached = score_cached(pairs, cache_path, scorer, identity=identity, progress=report_progress)
    reranked = {query: rerank_top(ranking, list(zip(cached[query]["ids"], cached[query]["scores"])), depth=depth)
                for query, ranking in rankings.items()}
    scores = score_run(reranked, evaluation, dates)
    truncation = {}
    for form in ("title", "abstract"):
        lengths = [length for query, row in cached.items() if query.endswith(":" + form)
                   for length in row["token_lengths"]]
        truncation[form] = {"pairs": len(lengths), "truncated_count": sum(n > scorer.settings["max_tokens"] for n in lengths),
                            "longest_input_tokens": max(lengths, default=0)}
    result = {"schema_version": 1, "split": "dev", **identity, "settings": scorer.settings,
              "rankings": reranked, "scores": scores, "summary": summarize_scores(scores),
              "differences": {"hybrid": summarize_scores(scores, reference=reference)},
              "truncation": truncation, "ranking_rule": "Rerank prefix by raw score then ID, preserve tail",
              "interval_note": "95% paired paper bootstrap, 2000 draws, seed 42; development comparisons"}
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path] for path in inputs):
        raise ValueError("Inputs changed during reranking")
    report_progress({"completed": len(pairs), "total": len(pairs)})
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError("Existing reranker output differs; use a new path")
    else:
        write_json(result, output)
    return result


def run_depth_comparison(database, evaluation_path, hybrid_path, cache_path, baseline_report,
                         output, *, progress=None):
    inputs = tuple(map(Path, (database, evaluation_path, hybrid_path, cache_path, baseline_report)))
    output = Path(output)
    if output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Output must not replace an input")
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    saved = json.loads(inputs[4].read_text(encoding="utf-8"))
    if saved["depth"] != 20 or saved["settings"] != PairScorer().settings:
        raise ValueError("Depth comparison requires the pinned truncated top20 run")

    class Replay:
        settings = saved["settings"]

        def __call__(self, pairs):
            raise ValueError("Complete top20 score cache required")

    baseline = run_rerank(*inputs, scorer=Replay())
    evaluation = json.loads(inputs[1].read_text(encoding="utf-8"))
    hybrid = json.loads(inputs[2].read_text(encoding="utf-8"))
    rankings = hybrid["trials"][hybrid["selected_trial"]]["rankings"]
    cached = json.loads(inputs[3].read_text(encoding="utf-8"))["queries"]
    with closing(sqlite3.connect(inputs[0].resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        dates = dict(connection.execute("SELECT openalex_id, publication_date FROM papers"))
    trials = []
    for depth in (20, 10):
        reranked = {}
        for query, ranking in rankings.items():
            row = cached[query]["row"]
            reranked[query] = rerank_top(ranking, list(zip(row["ids"][:depth], row["scores"][:depth])),
                                         depth=depth)
        scores = score_run(reranked, evaluation, dates)
        if depth == 20 and (reranked != baseline["rankings"] or scores != baseline["scores"]):
            raise ValueError("Depth20 must reproduce the baseline")
        values = [row["ndcg@10"] for form in scores.values() for row in form.values()]
        trials.append({"depth": depth, "selection_score": math.fsum(values) / len(values),
                       "rankings": reranked, "scores": scores})
        if progress is not None:
            progress({"depth": depth, "selection_score": trials[-1]["selection_score"]})
    best = max(range(len(trials)), key=lambda i: trials[i]["selection_score"])
    reference = score_run(rankings, evaluation, dates)
    result = {"schema_version": 1, "split": "dev", "settings": saved["settings"],
              "inputs": {name: hashes[path] for name, path in zip(
                  ("database", "evaluation", "hybrid", "cache", "baseline_report"), inputs)},
              "trials": trials, "selected_trial": best,
              "selection_rule": "Mean title/abstract dev nDCG@10, shared depth, 20 before 10 on ties",
              "summary": summarize_scores(trials[best]["scores"]),
              "differences": {"depth10_minus_depth20": summarize_scores(trials[1]["scores"],
                                                                          reference=baseline["scores"]),
                              "selected_minus_hybrid": summarize_scores(trials[best]["scores"],
                                                                          reference=reference)},
              "interval_note": "95% paired paper bootstrap, 2000 draws, seed 42; dev selection bias not corrected"}
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path] for path in inputs):
        raise ValueError("Inputs changed during depth comparison")
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError("Existing depth comparison differs; use a new path")
    else:
        write_json(result, output)
    return result


def run_chunk_comparison(database, evaluation_path, hybrid_path, baseline_cache, baseline_report,
                         directory, *, scorer=None, progress=None):
    paths = tuple(map(Path, (database, evaluation_path, hybrid_path, baseline_cache, baseline_report)))
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    directory = Path(directory)
    cache_path, output = directory / "chunk-scores.json", directory / "comparison.json"
    if any(path.resolve() in {item.resolve() for item in paths} for path in (cache_path, output)):
        raise ValueError("Chunk outputs must not replace inputs")
    saved = json.loads(paths[4].read_text(encoding="utf-8"))
    baseline_settings = PairScorer().settings
    scorer = TokenPairScorer() if scorer is None else scorer
    if any(scorer.settings.get(key) != value for key, value in {
        "input_mode": "token_ids", "truncation": "none",
        "query_chunk_tokens": 254, "document_chunk_tokens": 255,
    }.items()):
        raise ValueError("Chunk settings must use the approved token blocks")
    if (saved["depth"] != 20 or saved["settings"] != baseline_settings
            or any(scorer.settings.get(key) != value for key, value in baseline_settings.items()
                   if key != "truncation")):
        raise ValueError("Chunk comparison requires the same pinned top20 model settings")

    class Replay:
        settings = baseline_settings

        def __call__(self, pairs):
            raise ValueError("Complete baseline score cache required")

    baseline = run_rerank(*paths, scorer=Replay())
    hybrid = json.loads(paths[2].read_text(encoding="utf-8"))
    evaluation = json.loads(paths[1].read_text(encoding="utf-8"))
    baseline_rows = json.loads(paths[3].read_text(encoding="utf-8"))["queries"]
    rankings = hybrid["trials"][hybrid["selected_trial"]]["rankings"]
    with closing(sqlite3.connect(paths[0].resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        papers = {key: {"title": title, "abstract": abstract, "date": date} for key, title, abstract, date
                  in connection.execute("SELECT openalex_id, title, abstract, publication_date FROM papers")}
    dates = {key: paper["date"] for key, paper in papers.items()}
    tokens, expanded, groups = {}, {}, {}
    workload = {form: {"reused_pairs": 0, "overflow_pairs": 0, "chunk_pairs": 0} for form in ("title", "abstract")}

    def tokenize(text):
        if text not in tokens:
            tokens[text] = scorer.tokenize(text)
        return tokens[text]

    for query, ranking in rankings.items():
        paper, form = query.rsplit(":", 1)
        q = tokenize(papers[paper][form])
        expanded[query], groups[query] = [], {}
        for position, key in enumerate(ranking[:20]):
            d = tokenize((papers[key]["title"] or "") + "\n" + papers[key]["abstract"])
            length = len(q) + len(d) + 3
            if length != baseline_rows[query]["row"]["token_lengths"][position]:
                raise ValueError("Original token lengths differ from the baseline")
            if length <= 512:
                workload[form]["reused_pairs"] += 1
                continue
            workload[form]["overflow_pairs"] += 1
            groups[query][key] = []
            for number, (q_chunk, d_chunk) in enumerate(token_chunks(q, d)):
                chunk_id = f"{key}:{number}"
                groups[query][key].append(chunk_id)
                expanded[query].append((chunk_id, q_chunk, d_chunk))
                workload[form]["chunk_pairs"] += 1
    deadline = time.monotonic() + 4 * 60 * 60

    def report_progress(row):
        if time.monotonic() >= deadline:
            raise TimeoutError("Chunk comparison reached its four-hour budget")
        if progress is not None:
            progress(row)

    identity = {name: hashes[path] for name, path in zip(
        ("database", "evaluation", "hybrid", "baseline_cache", "baseline_report"), paths)}
    cached = score_cached(expanded, cache_path, scorer, identity=identity, progress=report_progress)
    trials = [{"method": "truncation", "rankings": baseline["rankings"], "scores": baseline["scores"]}]
    for method in ("maximum", "mean"):
        fused = {}
        for query, ranking in rankings.items():
            row = baseline_rows[query]["row"]
            values = dict(zip(row["ids"], row["scores"]))
            chunks = dict(zip(cached[query]["ids"], cached[query]["scores"]))
            for key, ids in groups[query].items():
                values[key] = aggregate_chunks([chunks[item] for item in ids], method)
            fused[query] = rerank_top(ranking, list(values.items()))
        trials.append({"method": method, "rankings": fused, "scores": score_run(fused, evaluation, dates)})
    for trial in trials:
        values = [row["ndcg@10"] for form in trial["scores"].values() for row in form.values()]
        trial["selection_score"] = math.fsum(values) / len(values)
    best = max(range(len(trials)), key=lambda i: trials[i]["selection_score"])
    reference = score_run(rankings, evaluation, dates)
    result = {"schema_version": 1, "split": "dev", "inputs": identity, "settings": scorer.settings,
              "workload": workload, "trials": trials, "selected_trial": best,
              "selection_rule": "Mean title/abstract dev nDCG@10, shared method, first exact tie",
              "summary": summarize_scores(trials[best]["scores"]),
              "differences": {trial["method"]: {
                  "truncation": summarize_scores(trial["scores"], reference=baseline["scores"]),
                  "hybrid": summarize_scores(trial["scores"], reference=reference)} for trial in trials[1:]},
              "interval_note": "95% paired paper bootstrap, 2000 draws, seed 42; dev selection bias not corrected"}
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path] for path in paths):
        raise ValueError("Inputs changed during chunk comparison")
    report_progress({"completed": len(expanded), "total": len(expanded)})
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError("Existing chunk comparison differs; use a new path")
    else:
        write_json(result, output)
    return result

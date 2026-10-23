import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from astro_retrieval.baselines import score_run, summarize_scores
from astro_retrieval.bm25 import selection_score
from astro_retrieval.ingest.expansion import write_json


def rank_fusion(first, second, *, k=60):
    if type(k) is not int or k < 1:
        raise ValueError("RRF k must be a positive integer")
    scores = {}
    for ranking in (first, second):
        if (not isinstance(ranking, list) or len(ranking) > 100
                or any(not isinstance(key, str) or not key for key in ranking)):
            raise ValueError("Input rankings require at most 100 nonempty paper IDs")
        if len(set(ranking)) != len(ranking):
            raise ValueError("Input rankings must not contain duplicates")
        for position, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1 / (k + position)
    return sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:100]


def run_hybrid(database, evaluation_path, bm25_path, general_path, indus_path, output,
               *, progress=None):
    database, evaluation_path, bm25_path, general_path, indus_path, output = map(
        Path, (database, evaluation_path, bm25_path, general_path, indus_path, output))
    inputs = (database, evaluation_path, bm25_path, general_path, indus_path)
    if output.resolve() in {path.resolve() for path in inputs}:
        raise ValueError("Output must not replace an input")
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    evaluation, bm25, general, indus = [json.loads(path.read_text(encoding="utf-8"))
                                       for path in inputs[1:]]
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        manifest = {key: json.loads(value) for key, value in connection.execute("SELECT * FROM manifest")}
        dates = dict(connection.execute("SELECT openalex_id, publication_date FROM papers"))
    source = manifest["source_sha256"]
    if evaluation["source_sha256"] != source or any(
        report["source_sha256"] != source or report["database_sha256"] != hashes[database]
        or report["evaluation_sha256"] != hashes[evaluation_path] or report["split"] != "dev"
        for report in (bm25, general, indus)
    ):
        raise ValueError("Database, evaluation, and ranking identities must match")
    for report, model in ((general, "sentence-transformers/all-MiniLM-L6-v2"),
                          (indus, "nasa-impact/nasa-smd-ibm-st-v2")):
        if (report["settings"]["model"] != model or report["bm25_sha256"] != hashes[bm25_path]
                or report["baseline_sha256"] != bm25["baseline_sha256"]):
            raise ValueError("Dense model and comparison identities must match")
    selected_trial = bm25["selected_trial"]
    if type(selected_trial) is not int or not 0 <= selected_trial < len(bm25["trials"]):
        raise ValueError("Invalid selected BM25 trial")
    rankings = {"bm25": bm25["trials"][selected_trial]["rankings"],
                "general": general["rankings"], "indus": indus["rankings"]}
    reference_scores = {name: score_run(run, evaluation, dates) for name, run in rankings.items()}
    dense_selection = {name: selection_score(reference_scores[name]) for name in ("general", "indus")}
    selected_dense = max(dense_selection, key=dense_selection.get)
    trials = []
    for k in (60, 20, 100):
        fused = {query: [key for key, _ in rank_fusion(
            rankings["bm25"][query], rankings[selected_dense][query], k=k)]
            for query in sorted(rankings["bm25"])}
        scores = score_run(fused, evaluation, dates)
        trial = {"k": k, "selection_score": selection_score(scores),
                 "rankings": fused, "scores": scores}
        trials.append(trial)
        if progress is not None:
            progress({"selected_dense": selected_dense, "k": k,
                      "selection_score": trial["selection_score"]})
    best = max(range(len(trials)), key=lambda position: trials[position]["selection_score"])
    selected = trials[best]
    result = {
        "schema_version": 1, "split": "dev", "source_sha256": source,
        "database_sha256": hashes[database], "evaluation_sha256": hashes[evaluation_path],
        "bm25_sha256": hashes[bm25_path], "general_sha256": hashes[general_path],
        "indus_sha256": hashes[indus_path], "selected_dense": selected_dense,
        "dense_selection_scores": dense_selection, "trials": trials, "selected_trial": best,
        "selection_rule": "Mean title/abstract Recall@100, shared dense then shared k, first exact tie",
        "fusion_rule": "Equal weights, 1-based ranks, top100 per input, missing contributes zero, ID ties",
        "summary": summarize_scores(selected["scores"]),
        "differences": {name: summarize_scores(selected["scores"], reference=scores)
                        for name, scores in reference_scores.items()},
        "interval_note": "95% paired paper bootstrap, 2000 draws, seed 42; dev selection bias not corrected",
    }
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path] for path in inputs):
        raise ValueError("Inputs changed during the hybrid run")
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError("Existing hybrid output differs; use a new path")
    else:
        write_json(result, output)
    return result

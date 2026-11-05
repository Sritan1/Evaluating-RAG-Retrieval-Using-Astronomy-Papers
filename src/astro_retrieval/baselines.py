import hashlib
import json
import random
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from astro_retrieval.corpus import eligible_documents
from astro_retrieval.ingest.expansion import write_json
from astro_retrieval.metrics import bootstrap_mean, score_query


def baseline_ranking(dates, references, query_id, system, *, seed=42):
    if query_id not in dates or dates[query_id] is None:
        raise ValueError("Query requires a known publication date")
    if type(seed) is not int:
        raise ValueError("Seed must be an integer")
    candidates = sorted(eligible_documents(dates, query_id))
    if system == "random":
        rng = random.Random(f"{seed}:{query_id}")
        return rng.sample(candidates, min(100, len(candidates)))
    if system != "popularity":
        raise ValueError("Unknown baseline")
    counts = Counter(target for citing, target in set(references)
                     if citing in dates and dates[citing] is not None
                     and citing != query_id and citing != target
                     and dates[citing] < dates[query_id])
    return sorted(candidates, key=lambda key: (-counts[key], key))[:100]


def score_run(rankings, evaluation, dates, *, split="dev"):
    if split not in ("dev", "test"):
        raise ValueError("Unknown evaluation split")
    papers = set(evaluation["splits"][split])
    rows = [row for row in evaluation["queries"] if row["split"] == split]
    queries = {row["query_id"]: row for row in rows}
    if not papers or set(evaluation["splits"]["dev"]) & set(evaluation["splits"]["test"]):
        raise ValueError("Selected split must be nonempty and disjoint")
    expected = {f"{paper}:{form}" for paper in papers for form in ("title", "abstract")}
    if len(rows) != len(queries) or set(queries) != expected or set(rankings) != expected:
        label = "development" if split == "dev" else "test"
        raise ValueError(f"Run must contain exactly the {label} queries")
    scores = {"title": {}, "abstract": {}}
    for query_id, row in queries.items():
        paper, form = row["paper_id"], row["query_type"]
        if paper not in papers or form not in scores or query_id != f"{paper}:{form}":
            raise ValueError("Invalid query identity")
        ranking = rankings[query_id]
        if not isinstance(ranking, list) or len(ranking) > 100:
            raise ValueError("Rankings must be lists with at most 100 results")
        eligible = eligible_documents(dates, paper)
        if any(not isinstance(key, str) or key not in eligible for key in ranking):
            raise ValueError("Ranking contains an ineligible document")
        labels = evaluation["qrels"][paper]
        if not set(labels) <= eligible:
            raise ValueError("Labels contain an ineligible document")
        scores[form][paper] = score_query(ranking, labels)
    return scores


def summarize_scores(scores, *, reference=None, resamples=2000, seed=42):
    if set(scores) != {"title", "abstract"}:
        raise ValueError("Both query forms are required")
    papers = set(scores["title"])
    metrics = {"recall@10", "recall@50", "recall@100", "rr@100", "ndcg@10"}
    for group in (scores,) if reference is None else (scores, reference):
        if set(group) != set(scores) or any(set(rows) != papers for rows in group.values()):
            raise ValueError("All forms and systems must contain identical papers")
        if any(set(row) != metrics for rows in group.values() for row in rows.values()):
            raise ValueError("Every paper requires the same metrics")
    return {form: {
        "mrr@100" if metric == "rr@100" else metric: bootstrap_mean(
            {paper: row[metric] for paper, row in rows.items()},
            reference=None if reference is None else {
                paper: row[metric] for paper, row in reference[form].items()},
            resamples=resamples, seed=seed,
        ) for metric in sorted(metrics)
    } for form, rows in scores.items()}


def run_baselines(database, evaluation_path, output, *, seed=42, resamples=2000):
    database, evaluation_path, output = map(Path, (database, evaluation_path, output))
    if output.resolve() in {database.resolve(), evaluation_path.resolve()}:
        raise ValueError("Output must not replace an input")
    raw = evaluation_path.read_bytes()
    evaluation = json.loads(raw)
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        manifest = {key: json.loads(value) for key, value in connection.execute("SELECT * FROM manifest")}
        if manifest["source_sha256"] != evaluation["source_sha256"]:
            raise ValueError("Database and evaluation source hashes differ")
        dates = dict(connection.execute("SELECT openalex_id, publication_date FROM papers"))
        references = connection.execute("SELECT citing_id, cited_id FROM refs").fetchall()
    results = {}
    for system in ("random", "popularity"):
        rankings = {}
        for paper in sorted(evaluation["splits"]["dev"]):
            ranking = baseline_ranking(dates, references, paper, system, seed=seed)
            rankings.update({f"{paper}:{form}": ranking for form in ("title", "abstract")})
        scores = score_run(rankings, evaluation, dates)
        results[system] = {"rankings": rankings, "scores": scores,
                           "summary": summarize_scores(scores, resamples=resamples, seed=seed)}
    result = {
        "schema_version": 1, "split": "dev", "seed": seed,
        "evaluation_sha256": hashlib.sha256(raw).hexdigest(),
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "source_sha256": evaluation["source_sha256"],
        "popularity_rule": "Distinct citations from collected seeds strictly earlier than query; ID ties",
        "interval_method": "Paper-level percentile bootstrap, 95%, paired draws; no multiplicity adjustment",
        "systems": results,
        "popularity_minus_random": summarize_scores(
            results["popularity"]["scores"], reference=results["random"]["scores"],
            resamples=resamples, seed=seed),
    }
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError("Existing baseline output differs; use a new path")
    else:
        write_json(result, output)
    return result

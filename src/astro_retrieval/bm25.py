import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from contextlib import closing
from pathlib import Path

from astro_retrieval.ingest.expansion import reports_match, write_json
from astro_retrieval.corpus import eligible_documents
from astro_retrieval.baselines import score_run, summarize_scores


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).lower())


def build_index(papers: dict[str, dict]) -> dict:
    if not papers or any(not isinstance(key, str) or not key.strip() for key in papers):
        raise ValueError("Index requires papers with nonempty IDs")
    lengths, postings, texts = {}, {}, {}
    for key, paper in sorted(papers.items()):
        title, abstract = paper.get("title"), paper["abstract"]
        if title is not None and not isinstance(title, str):
            raise ValueError("Title must be a string or null")
        if not isinstance(abstract, str):
            raise ValueError("Abstract must be a string")
        text = (title or "") + "\n" + abstract
        texts[key] = text
        counts = Counter(tokenize(text))
        lengths[key] = sum(counts.values())
        for term, count in counts.items():
            postings.setdefault(term, {})[key] = count
    raw = json.dumps(texts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "schema_version": 1,
        "text_sha256": hashlib.sha256(raw).hexdigest(),
        "tokenizer": "NFKC, lowercase, Unicode letters/numbers, retain stopwords, no stemming",
        "unicode_version": unicodedata.unidata_version,
        "document_text": "title + newline + abstract, no field weighting",
        "document_count": len(papers),
        "average_length": sum(lengths.values()) / len(papers),
        "lengths": lengths,
        "postings": dict(sorted(postings.items())),
    }


def build_saved_index(database: str | Path, output: str | Path) -> dict:
    database, output = Path(database), Path(output)
    if database.resolve() == output.resolve():
        raise ValueError("Index must not replace the database")
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        papers = {key: {"title": title, "abstract": abstract} for key, title, abstract
                  in connection.execute("SELECT openalex_id, title, abstract FROM papers")}
        manifest = {key: json.loads(value) for key, value in connection.execute("SELECT * FROM manifest")}
    result = build_index(papers)
    result["source_sha256"] = manifest["source_sha256"]
    result["database_sha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
    if output.exists():
        if not reports_match(json.loads(output.read_text(encoding="utf-8")), result):
            raise ValueError("Existing index differs. Use a new path")
    else:
        write_json(result, output)
    return result


def rank_bm25(index: dict, dates: dict[str, str | None], query_id: str, query: str,
              *, k1: float = 1.2, b: float = 0.75) -> list[tuple[str, float]]:
    for name, value in (("k1", k1), ("b", b)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if k1 < 0 or not 0 <= b <= 1:
        raise ValueError("Require k1 >= 0 and 0 <= b <= 1")
    if query_id not in dates or dates[query_id] is None:
        raise ValueError("Query requires a known publication date")
    lengths = index["lengths"]
    if set(lengths) != set(dates):
        raise ValueError("Index and dates must contain identical papers")
    terms = sorted(set(tokenize(query)))
    eligible = eligible_documents(dates, query_id)
    if not eligible or not terms:
        return []
    average = sum(lengths[key] for key in eligible) / len(eligible)
    if average == 0:
        return []
    scores = {}
    for term in terms:
        posting = index["postings"].get(term, {})
        matches = eligible.intersection(posting)
        if not matches:
            continue
        idf = math.log1p((len(eligible) - len(matches) + 0.5) / (len(matches) + 0.5))
        for key in matches:
            frequency = posting[key]
            denominator = frequency + k1 * (1 - b + b * lengths[key] / average)
            scores[key] = scores.get(key, 0.0) + idf * frequency * (k1 + 1) / denominator
    return sorted(((key, value) for key, value in scores.items() if value > 0),
                  key=lambda row: (-row[1], row[0]))[:100]


def selection_score(scores: dict) -> float:
    if set(scores) != {"title", "abstract"} or not scores["title"]:
        raise ValueError("Selection requires both nonempty query forms")
    if scores["title"].keys() != scores["abstract"].keys():
        raise ValueError("Query forms must contain identical papers")
    values = [row["recall@100"] for rows in scores.values() for row in rows.values()]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("Selection recalls must be finite and between zero and one")
    return math.fsum(values) / len(values)


def run_bm25(database, index_path, evaluation_path, baseline_path, output, *, progress=None):
    database, index_path, evaluation_path, baseline_path, output = map(
        Path, (database, index_path, evaluation_path, baseline_path, output))
    if output.resolve() in {path.resolve() for path in (
        database, index_path, evaluation_path, baseline_path)}:
        raise ValueError("Output must not replace an input")
    if not index_path.is_file():
        raise ValueError("Build the saved index before tuning")
    index = build_saved_index(database, index_path)
    raw = evaluation_path.read_bytes()
    evaluation = json.loads(raw)
    baselines = json.loads(baseline_path.read_text(encoding="utf-8"))
    evaluation_hash = hashlib.sha256(raw).hexdigest()
    if (evaluation["source_sha256"] != index["source_sha256"]
            or baselines["source_sha256"] != index["source_sha256"]
            or baselines["database_sha256"] != index["database_sha256"]
            or baselines["evaluation_sha256"] != evaluation_hash
            or baselines["split"] != "dev"):
        raise ValueError("Index, database, evaluation, and baseline identities must match")
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute("SELECT openalex_id, title, abstract, publication_date FROM papers")
        papers = {key: {"title": title, "abstract": abstract, "date": date}
                  for key, title, abstract, date in rows}
    dates = {key: paper["date"] for key, paper in papers.items()}
    dev = sorted(evaluation["splits"]["dev"])
    score_run({f"{key}:{form}": [] for key in dev for form in ("title", "abstract")},
              evaluation, dates)
    reference_scores = {system: score_run(baselines["systems"][system]["rankings"], evaluation, dates)
                        for system in ("random", "popularity")}
    grid = [(1.2, .75)] + [(k1, b) for k1 in (.6, 1.2, 1.8) for b in (.25, .75, 1.)
                          if (k1, b) != (1.2, .75)]
    trials = []
    for k1, b in grid:
        rankings = {}
        for key in dev:
            for form in ("title", "abstract"):
                ranked = rank_bm25(index, dates, key, papers[key][form], k1=k1, b=b)
                rankings[f"{key}:{form}"] = [paper for paper, _ in ranked]
        scores = score_run(rankings, evaluation, dates)
        trial = {"k1": k1, "b": b, "selection_score": selection_score(scores),
                 "rankings": rankings, "scores": scores}
        trials.append(trial)
        if progress is not None:
            progress({key: trial[key] for key in ("k1", "b", "selection_score")})
    best = max(range(len(trials)), key=lambda position: trials[position]["selection_score"])
    selected = trials[best]
    result = {
        "schema_version": 1, "split": "dev", "evaluation_sha256": evaluation_hash,
        "database_sha256": index["database_sha256"], "source_sha256": index["source_sha256"],
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "selection_rule": "Mean title/abstract Recall@100, shared setting, first exact tie",
        "scoring_rule": "Positive IDF, k1+1 numerator, eligible-only statistics, unique query terms, ID ties",
        "trials": trials, "selected_trial": best,
        "summary": summarize_scores(selected["scores"]),
        "differences": {system: summarize_scores(selected["scores"], reference=scores)
                        for system, scores in reference_scores.items()},
        "interval_note": "95% paired paper bootstrap, 2000 draws, seed 42. Dev selection bias not corrected",
    }
    if output.exists():
        if not reports_match(json.loads(output.read_text(encoding="utf-8")), result):
            raise ValueError("Existing BM25 output differs. Use a new path")
    else:
        write_json(result, output)
    return result

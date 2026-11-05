import hashlib
import json
import math
import sqlite3
import time
from collections import Counter
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from astro_retrieval.baselines import baseline_ranking, score_run, summarize_scores
from astro_retrieval.bm25 import build_index, rank_bm25
from astro_retrieval.dense import EmbeddingEncoder, encode_cached, rank_dense
from astro_retrieval.hybrid import rank_fusion
from astro_retrieval.ingest.expansion import reports_match, write_json
from astro_retrieval.metrics import compare_hypotheses
from astro_retrieval.rerank import PairScorer, _digest, rerank_top, score_cached


PROTOCOL_SHA256 = "6a2cfba600613e7e74c53a28f47e3d8aad99d6c3cc0f3d4e6c3644d7615654b6"
EVALUATION_SHA256 = "b33719ea6f2523a6e5ce69944ab818b91fd55c8a081301dd9780297da8ce649a"
SETTINGS = {"seed": 42, "bm25": {"k1": 1.8, "b": 1.0}, "hybrid_dense": "indus",
            "rrf_k": 60, "rerank_depth": 10, "split": "test", "limit": 100}


def run_evaluation(database, evaluation_path, index_path, general_documents, indus_documents,
                   protocol_path, directory, *, encoders=None, scorer=None, progress=None, replay_only=False,
                   cached_inference=None):
    inputs = dict(zip(("database", "evaluation", "index", "protocol"),
                      map(Path, (database, evaluation_path, index_path, protocol_path))))
    document_paths = dict(zip(("general", "indus"), map(Path, (general_documents, indus_documents))))
    directory = Path(directory)
    cache_directory = Path(cached_inference) if cached_inference is not None else directory
    if cached_inference is not None:
        if (replay_only or directory.exists()
                or directory.resolve().is_relative_to(cache_directory.resolve())
                or cache_directory.resolve().is_relative_to(directory.resolve())):
            raise ValueError("Rebuild requires a fresh output separate from the inference cache")
    systems = ("random", "popularity", "bm25", "general", "indus", "hybrid", "reranker")
    if replay_only and any(not (directory / name).is_file()
                           for name in ("manifest.json", "report.json", *(name + ".json" for name in systems))):
        raise ValueError("Reproduction requires a completed run and all seven saved systems")
    destination = directory.resolve()
    if (any(path.resolve().is_relative_to(destination) for path in inputs.values())
            or any(destination.is_relative_to(path.resolve()) or path.resolve().is_relative_to(destination)
                   for path in document_paths.values())):
        raise ValueError("Output directory must be separate from input files and document caches")
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in inputs.items()}
    protocol_hash = hashlib.sha256(inputs["protocol"].read_text(encoding="utf-8").encode()).hexdigest()
    if protocol_hash != PROTOCOL_SHA256 or hashes["evaluation"] != EVALUATION_SHA256:
        raise ValueError("Frozen protocol or evaluation identity differs")
    evaluation = json.loads(inputs["evaluation"].read_text(encoding="utf-8"))
    index = json.loads(inputs["index"].read_text(encoding="utf-8"))
    with closing(sqlite3.connect(inputs["database"].resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        manifest = {key: json.loads(value) for key, value in connection.execute("SELECT * FROM manifest")}
        papers = {key: {"title": title, "abstract": abstract, "date": date}
                  for key, title, abstract, date in connection.execute(
                      "SELECT openalex_id, title, abstract, publication_date FROM papers")}
        references = connection.execute("SELECT citing_id, cited_id FROM refs").fetchall()
    source = manifest["source_sha256"]
    if evaluation["source_sha256"] != source or index != {
        **build_index(papers), "source_sha256": source, "database_sha256": hashes["database"]
    }:
        raise ValueError("Source, database and index identities must match")
    dates = {key: paper["date"] for key, paper in papers.items()}
    query_texts = {f"{key}:{form}": papers[key][form] for key in sorted(evaluation["splits"]["test"])
                   for form in ("title", "abstract")}
    score_run({key: [] for key in query_texts}, evaluation, dates, split="test")
    expected = {name: EmbeddingEncoder(name) for name in ("general", "indus")}
    encoders = expected if encoders is None else encoders
    scorer = PairScorer() if scorer is None else scorer
    if (set(encoders) != set(expected) or any(encoders[name].settings != expected[name].settings
                                            for name in expected) or scorer.settings != PairScorer().settings):
        raise ValueError("Model settings must match the frozen configuration")
    protected = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                 for folder in document_paths.values() for path in folder.rglob("*") if path.is_file()}
    document_texts = {key: (paper["title"] or "") + "\n" + paper["abstract"] for key, paper in papers.items()}

    def no_encoding(texts):
        raise ValueError("Document inference is forbidden")

    documents = {name: encode_cached(document_texts, document_paths[name], no_encoding,
                                     source_sha256=source, settings=expected[name].settings, read_only=True)
                 for name in expected}
    identity = {"inputs": hashes, "settings": SETTINGS,
                "models": {**{name: expected[name].settings for name in expected}, "reranker": scorer.settings},
                "documents": {name: hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
                              for name, path in document_paths.items()}}
    run_manifest = directory / "manifest.json"
    if run_manifest.exists():
        if json.loads(run_manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError("Existing run has different inputs or settings")
    else:
        if replay_only:
            raise ValueError("Saved run manifest is missing")
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("New run requires an empty output directory")
        write_json(identity, run_manifest)
    deadline = time.monotonic() + 4 * 60 * 60

    def report_progress(stage, row=None):
        if time.monotonic() >= deadline:
            raise TimeoutError("Evaluation reached its four-hour budget")
        if progress is not None:
            progress({"stage": stage, **(row or {})})

    def checked_records(records):
        rankings = {}
        for query, rows in records.items():
            for rank, row in enumerate(rows, start=1):
                if (row["query_id"] != query or row["rank"] != rank
                        or isinstance(row["score"], bool) or not isinstance(row["score"], (int, float))
                        or not math.isfinite(row["score"]) or not isinstance(row["score_type"], str)):
                    raise ValueError("Invalid scored run record")
            rankings[query] = [row["paper_id"] for row in rows]
        return score_run(rankings, evaluation, dates, split="test")

    def records_from(pairs, score_type):
        return {query: [{"query_id": query, "paper_id": key, "rank": rank,
                         "score": float(score), "score_type": score_type}
                        for rank, (key, score) in enumerate(rows, start=1)] for query, rows in pairs.items()}

    stages = {}
    for name in systems:
        report_progress(name)
        path = directory / (name + ".json")
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            stage = saved["result"]
            if saved["sha256"] != _digest(stage) or stage["identity"] != identity or stage["system"] != name:
                raise ValueError("Saved system result identity or checksum differs")
            if checked_records(stage["records"]) != stage["scores"]:
                raise ValueError("Saved system metrics differ from rankings")
            if replay_only and summarize_scores(stage["scores"]) != stage["summary"]:
                raise ValueError("Saved system summary differs from recomputed scores")
            stages[name] = stage
            continue
        if replay_only:
            raise ValueError("Saved system is missing, reproduction cannot run inference")
        truncation = {}
        if name in ("random", "popularity"):
            pairs = {}
            for query in query_texts:
                paper = query.rsplit(":", 1)[0]
                ranking = baseline_ranking(dates, references, paper, name)
                counts = Counter(target for citing, target in set(references)
                                 if citing in dates and dates[citing] is not None and citing != paper
                                 and citing != target and dates[citing] < dates[paper]) if name == "popularity" else {}
                pairs[query] = [(key, counts[key] if name == "popularity" else 1 / rank)
                                for rank, key in enumerate(ranking, start=1)]
            records = records_from(pairs, "citation_count" if name == "popularity" else "random_rank_priority")
        elif name == "bm25":
            pairs = {query: rank_bm25(index, dates, query.rsplit(":", 1)[0], text, **SETTINGS["bm25"])
                     for query, text in query_texts.items()}
            records = records_from(pairs, "bm25")
        elif name in expected:
            def encode(texts):
                report_progress(name + "_encoding")
                return encoders[name](texts)

            queries = encode_cached(query_texts, cache_directory / (name + "-queries"), encode,
                                    source_sha256=source, settings=expected[name].settings,
                                    progress=lambda row: report_progress(name + "_queries", row),
                                    read_only=cached_inference is not None)
            pairs = {query: rank_dense(documents[name]["ids"], documents[name]["vectors"], vector,
                                       dates, query.rsplit(":", 1)[0])
                     for query, vector in zip(queries["ids"], queries["vectors"])}
            records = records_from(pairs, "cosine")
            for form in ("title", "abstract"):
                lengths = [int(length) for query, length in zip(queries["ids"], queries["token_lengths"])
                           if query.endswith(":" + form)]
                truncation[form] = {"count": len(lengths), "truncated_count": sum(n > 512 for n in lengths),
                                    "longest_input_tokens": max(lengths)}
        elif name == "hybrid":
            pairs = {query: rank_fusion([row["paper_id"] for row in stages["bm25"]["records"][query]],
                                        [row["paper_id"] for row in stages["indus"]["records"][query]], k=60)
                     for query in query_texts}
            records = records_from(pairs, "rrf")
        else:
            hybrid = stages["hybrid"]["records"]
            pairs = {query: [(row["paper_id"], query_texts[query], document_texts[row["paper_id"]])
                             for row in rows[:10]] for query, rows in hybrid.items()}
            cached = score_cached(pairs, cache_directory / "rerank-scores.json", scorer, identity=identity,
                                  progress=lambda row: report_progress("reranker_queries", row),
                                  read_only=cached_inference is not None)
            records = {}
            for query, rows in hybrid.items():
                row = cached[query]
                scores = dict(zip(row["ids"], row["scores"]))
                ranking = rerank_top([r["paper_id"] for r in rows], list(scores.items()), depth=10)
                original = {r["paper_id"]: r for r in rows}
                records[query] = [{"query_id": query, "paper_id": key, "rank": rank,
                                   "score": scores[key] if key in scores else original[key]["score"],
                                   "score_type": "cross_encoder" if key in scores else "rrf_tail"}
                                  for rank, key in enumerate(ranking, start=1)]
            for form in ("title", "abstract"):
                lengths = [n for query, row in cached.items() if query.endswith(":" + form)
                           for n in row["token_lengths"]]
                truncation[form] = {"count": len(lengths), "truncated_count": sum(n > 512 for n in lengths),
                                    "longest_input_tokens": max(lengths, default=0)}
        scores = checked_records(records)
        stage = {"identity": identity, "system": name, "records": records, "scores": scores,
                 "summary": summarize_scores(scores), "truncation": truncation}
        report_progress(name + "_complete")
        write_json({"result": stage, "sha256": _digest(stage)}, path)
        stages[name] = stage
    result = {"schema_version": 1, "split": "test", "identity": identity,
              "systems": {name: {"file": name + ".json", "sha256": _digest(stage),
                                  "summary": stage["summary"], "truncation": stage["truncation"]}
                          for name, stage in stages.items()},
              "comparisons": compare_hypotheses({name: stage["scores"] for name, stage in stages.items()})}
    if (any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name] for name, path in inputs.items())
            or any(hashlib.sha256(path.read_bytes()).hexdigest() != value for path, value in protected.items())):
        raise ValueError("Inputs changed during evaluation")
    report_progress("complete")
    output = directory / "report.json"
    if output.exists():
        if not reports_match(json.loads(output.read_text(encoding="utf-8")), result):
            raise ValueError("Existing final report differs")
    else:
        if replay_only:
            raise ValueError("Saved final report is missing")
        write_json(result, output)
    return result


def rebuild_results(evaluation_args, audit_args, destination, *, protocol_sha256, progress=None):
    from astro_retrieval.audit import run_audit_analysis

    destination = Path(destination).resolve()
    sources = [Path(path).resolve() for path in evaluation_args]
    sources += [Path(path).resolve() for path in (*audit_args[:2], *audit_args[2].values(), *audit_args[3:])]
    if destination.exists() or any(destination == path or path.is_relative_to(destination)
                                   or (path.is_dir() and destination.is_relative_to(path)) for path in sources):
        raise ValueError("Destination must be new and separate from all source data")
    if not destination.parent.is_dir():
        raise ValueError("Destination parent must already exist")
    paths = set()
    for path in sources:
        if not path.exists():
            raise ValueError("Rebuild source is missing")
        paths.update(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else paths.add(path)
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    reference = Path(evaluation_args[-1])
    expected = json.loads((reference / "report.json").read_text(encoding="utf-8"))
    with TemporaryDirectory(prefix=".reproduce-", dir=destination.parent) as temporary:
        staging = Path(temporary).resolve()
        if staging.parent != destination.parent:
            raise ValueError("Temporary output escaped the destination parent")
        retrieval = run_evaluation(*evaluation_args[:-1], staging / "test", cached_inference=reference,
                                   progress=progress)
        if not reports_match(expected, retrieval):
            raise ValueError("Rebuilt retrieval report differs from the original")
        for name in retrieval["systems"]:
            original = json.loads((reference / f"{name}.json").read_text(encoding="utf-8"))
            rebuilt = json.loads((staging / "test" / f"{name}.json").read_text(encoding="utf-8"))
            if original != rebuilt:
                raise ValueError("Rebuilt rankings or scores differ from the original")
        audit = run_audit_analysis(*audit_args, protocol_sha256=protocol_sha256, replay_only=True)
        write_json(audit, staging / "audit" / "analysis.json")
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != value for path, value in hashes.items()):
            raise ValueError("Source files changed during rebuild")
        outputs = {str(path.relative_to(staging)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(staging.rglob("*")) if path.is_file()}
        write_json({"schema_version": 1, "source_sha256": hashes, "output_sha256": outputs,
                    "complete": True, "inference": "Reused frozen vectors and pair scores, no model calls"},
                   staging / "reproduction.json")
        staging.rename(destination)
    return retrieval, audit

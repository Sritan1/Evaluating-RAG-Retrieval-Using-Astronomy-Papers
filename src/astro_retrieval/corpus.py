import hashlib
import json
import math
import random
import re
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

from astro_retrieval.ingest.records import work_id
from astro_retrieval.ingest.expansion import write_json


def _date_fields(raw):
    if raw is None:
        return None, None, "missing"
    if not isinstance(raw, str):
        raise ValueError("Publication dates must be strings or null")
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", raw):
        try:
            return raw, date.fromisoformat(raw).isoformat(), "valid"
        except ValueError:
            pass
    return raw, None, "invalid"


def _rows(bundle):
    report = bundle["report"]
    if type(report["schema_version"]) is not int or report["schema_version"] != 1:
        raise ValueError("Unsupported pilot schema")
    papers = []
    identifiers = set()
    for key, paper in sorted(bundle["papers"].items()):
        identifier = work_id(key)
        if identifier in identifiers or work_id(paper["id"]) != identifier:
            raise ValueError("Duplicate or mismatched paper ID")
        title, abstract = paper.get("title"), paper["abstract"]
        if title is not None and not isinstance(title, str):
            raise ValueError("Titles must be strings or null")
        if not isinstance(abstract, str) or not abstract.strip():
            raise ValueError("Corpus papers require usable abstracts")
        identifiers.add(identifier)
        papers.append((identifier, title, abstract, *_date_fields(paper.get("publication_date"))))
    candidates = report["candidate_ids"]
    if not isinstance(candidates, list):
        raise ValueError("Candidate IDs must be a list")
    seed_ids = [work_id(value) for value in candidates]
    if len(set(seed_ids)) != len(seed_ids):
        raise ValueError("Duplicate seed ID")
    seeds = [(key, int(key in identifiers)) for key in sorted(seed_ids)]
    references = set()
    citing_ids = set()
    for key, targets in bundle["references"].items():
        citing = work_id(key)
        if citing in citing_ids or citing not in identifiers or citing not in seed_ids:
            raise ValueError("References require a unique usable seed")
        if not isinstance(targets, list):
            raise ValueError("References must be lists")
        citing_ids.add(citing)
        for target in targets:
            cited = work_id(target)
            if cited != citing:
                references.add((citing, cited))
    if citing_ids != set(seed_ids) & identifiers:
        raise ValueError("Usable seeds require reference lists")
    return papers, seeds, sorted(references)


def build_corpus(snapshot: str | Path, database: str | Path) -> dict[str, int]:
    snapshot, database = Path(snapshot), Path(database)
    if snapshot.resolve() == database.resolve():
        raise ValueError("Source and database paths must differ")
    raw = snapshot.read_bytes()
    bundle = json.loads(raw)
    papers, seeds, references = _rows(bundle)
    report = bundle["report"]
    provenance = {
        "schema_version": 1,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_pool_sha256": report.get("candidate_pool_sha256"),
        "candidate_parameters": report.get("candidate_parameters"),
        "sampling_seed": report.get("sampling_seed"),
    }
    manifest = [(key, json.dumps(value, sort_keys=True, allow_nan=False))
                for key, value in sorted(provenance.items())]
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )}
            if tables and tables != {"papers", "refs", "seeds", "manifest"}:
                raise ValueError("Destination is not a corpus database")
            if connection.execute("PRAGMA user_version").fetchone()[0] not in (0, 1):
                raise ValueError("Unsupported corpus schema")
            connection.execute("""CREATE TABLE IF NOT EXISTS papers (
                openalex_id TEXT PRIMARY KEY, title TEXT, abstract TEXT NOT NULL,
                publication_date_raw TEXT, publication_date TEXT,
                date_status TEXT NOT NULL CHECK(date_status IN ('valid', 'missing', 'invalid'))
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS seeds (
                openalex_id TEXT PRIMARY KEY, in_corpus INTEGER NOT NULL CHECK(in_corpus IN (0, 1))
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS refs (
                citing_id TEXT REFERENCES papers(openalex_id), cited_id TEXT NOT NULL,
                PRIMARY KEY(citing_id, cited_id), CHECK(citing_id != cited_id)
            )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS manifest (
                key TEXT PRIMARY KEY, value TEXT NOT NULL
            )""")
            for table in ("refs", "seeds", "papers", "manifest"):
                connection.execute(f"DELETE FROM {table}")
            connection.executemany("INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?)", papers)
            connection.executemany("INSERT INTO seeds VALUES (?, ?)", seeds)
            connection.executemany("INSERT INTO refs VALUES (?, ?)", references)
            connection.executemany("INSERT INTO manifest VALUES (?, ?)", manifest)
            connection.execute("PRAGMA user_version = 1")
    return {"papers": len(papers), "seeds": len(seeds), "references": len(references)}


def eligible_documents(dates: dict[str, str | None], query_id: str) -> set[str]:
    query_date = dates.get(query_id)
    if query_date is None:
        return set()
    return {key for key, value in dates.items()
            if key != query_id and value is not None and value < query_date}


def build_evaluation(
    database: str | Path, snapshot: str | Path, pilot_snapshot: str | Path,
    output: str | Path, *, seed: int = 42,
) -> dict:
    if type(seed) is not int:
        raise ValueError("Split seed must be an integer")
    database, snapshot, pilot_snapshot, output = map(
        Path, (database, snapshot, pilot_snapshot, output),
    )
    if output.resolve() in {p.resolve() for p in (database, snapshot, pilot_snapshot)}:
        raise ValueError("Evaluation output must not replace an input")
    raw = snapshot.read_bytes()
    bundle = json.loads(raw)
    expected = _rows(bundle)
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        manifest = {key: json.loads(value) for key, value in connection.execute(
            "SELECT key, value FROM manifest"
        )}
        if manifest["source_sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("Corpus source hash does not match")
        for table, rows in zip(("papers", "seeds", "refs"), expected):
            actual = connection.execute(f"SELECT * FROM {table}").fetchall()
            if sorted(actual) != sorted(rows):
                raise ValueError("Corpus rows do not match the source snapshot")
    papers, seeds, references = expected
    dates = {row[0]: row[4] for row in papers}
    titles = {row[0]: row[1] for row in papers}
    cited = {key: set() for key, present in seeds if present}
    for citing, target in references:
        cited[citing].add(target)
    qrels = {}
    for query_id in sorted(cited):
        positives = sorted(cited[query_id] & eligible_documents(dates, query_id))
        if len(positives) >= 8:
            if not titles[query_id] or not titles[query_id].strip():
                raise ValueError("Eligible query needs a title for both query forms")
            qrels[query_id] = positives
    pilot_raw = pilot_snapshot.read_bytes()
    pilot = json.loads(pilot_raw)
    _rows(pilot)
    if pilot["report"].get("candidate_pool_sha256") != manifest["candidate_pool_sha256"]:
        raise ValueError("Pilot and full corpus use different candidate pools")
    pilot_ids = {work_id(value) for value in pilot["report"]["candidate_ids"]}
    dev = set(qrels) & pilot_ids
    dev_size = max(math.ceil(0.30 * len(qrels)), len(dev))
    other = sorted(set(qrels) - dev)
    dev.update(random.Random(seed).sample(other, dev_size - len(dev)))
    test = set(qrels) - dev
    if not test:
        raise ValueError("No test queries remain after pilot exclusion")
    result = {
        "schema_version": 1,
        "source_sha256": manifest["source_sha256"],
        "pilot_sha256": hashlib.sha256(pilot_raw).hexdigest(),
        "candidate_pool_sha256": manifest["candidate_pool_sha256"],
        "split_seed": seed,
        "target_dev_fraction": 0.30,
        "minimum_references": 8,
        "date_rule": "Valid YYYY-MM-DD, candidate strictly earlier, exclude self",
        "pilot_excluded_query_ids": sorted(pilot_ids),
        "splits": {"dev": sorted(dev), "test": sorted(test)},
        "queries": [
            {"query_id": f"{key}:{form}", "paper_id": key, "query_type": form,
             "split": "dev" if key in dev else "test"}
            for key in sorted(qrels) for form in ("title", "abstract")
        ],
        "qrels": qrels,
    }
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != result:
            raise ValueError("Frozen evaluation differs; preserve it and review inputs")
    else:
        write_json(result, output)
    return result

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

from astro_retrieval.ingest.records import work_id


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

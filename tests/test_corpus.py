import json
import sqlite3
from contextlib import closing

import pytest

from astro_retrieval.corpus import build_corpus


@pytest.fixture
def snapshot(tmp_path):
    bundle = {
        "report": {"schema_version": 1, "candidate_ids": ["W1", "W3"], "sampling_seed": 42},
        "papers": {
            "W1": {"id": "W1", "title": "Planet", "abstract": "A planet orbits a star.",
                   "publication_date": "2024-01-01"},
            "W2": {"id": "W2", "title": "Orbit", "abstract": "An earlier observation.",
                   "publication_date": "2020-01-01"},
        },
        "references": {"W1": ["W2", "W2", "W9", "W1"]},
    }
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    return path


def contents(database):
    with closing(sqlite3.connect(database)) as connection:
        return {table: connection.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall()
                for table in ("papers", "seeds", "refs", "manifest")}


def test_reimport_is_identical_and_retains_external_references(snapshot, tmp_path):
    database = tmp_path / "corpus.sqlite"
    assert build_corpus(snapshot, database) == {"papers": 2, "seeds": 2, "references": 2}
    first = contents(database)
    build_corpus(snapshot, database)
    assert contents(database) == first
    assert first["refs"] == [("W1", "W2"), ("W1", "W9")]
    assert [row[0] for row in first["papers"]] == ["W1", "W2"]
    assert first["seeds"] == [("W1", 1), ("W3", 0)]


def test_unknown_dates_preserve_raw_values(snapshot, tmp_path):
    bundle = json.loads(snapshot.read_text(encoding="utf-8"))
    bundle["papers"]["W1"]["publication_date"] = None
    bundle["papers"]["W2"]["publication_date"] = "2020-02-30"
    snapshot.write_text(json.dumps(bundle), encoding="utf-8")
    database = tmp_path / "corpus.sqlite"
    build_corpus(snapshot, database)
    assert [row[3:] for row in contents(database)["papers"]] == [
        (None, None, "missing"), ("2020-02-30", None, "invalid"),
    ]


def test_failed_replacement_rolls_back_existing_rows(snapshot, tmp_path):
    database = tmp_path / "corpus.sqlite"
    build_corpus(snapshot, database)
    first = contents(database)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("""CREATE TRIGGER fail_insert BEFORE INSERT ON papers
            WHEN NEW.openalex_id = 'W2' BEGIN SELECT RAISE(ABORT, 'forced failure'); END""")
        connection.commit()
    with pytest.raises(sqlite3.IntegrityError, match="forced failure"):
        build_corpus(snapshot, database)
    assert contents(database) == first

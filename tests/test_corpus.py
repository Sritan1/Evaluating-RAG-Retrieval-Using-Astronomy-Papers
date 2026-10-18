import json
import sqlite3
from contextlib import closing

import pytest

from astro_retrieval.corpus import build_corpus, build_evaluation, eligible_documents


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


@pytest.fixture
def evaluation_inputs(tmp_path):
    papers = {f"W{i}": {"id": f"W{i}", "title": f"Paper {i}", "abstract": "Planet study.",
                        "publication_date": "2024-01-01" if i <= 5 else "2020-01-01"}
              for i in range(1, 17)}
    papers["W14"]["publication_date"] = "2024-01-01"
    papers["W15"]["publication_date"] = "2025-01-01"
    papers["W16"]["publication_date"] = None
    refs = [f"W{i}" for i in range(6, 14)]
    bundle = {
        "report": {"schema_version": 1, "candidate_ids": [f"W{i}" for i in range(1, 6)]},
        "papers": papers,
        "references": {f"W{i}": refs + [f"W{i}", "W14", "W15", "W16", "W99"]
                       for i in range(1, 5)},
    }
    bundle["references"]["W5"] = refs[:7]
    source = tmp_path / "full.json"
    source.write_text(json.dumps(bundle), encoding="utf-8")
    pilot = {**bundle, "report": {"schema_version": 1, "candidate_ids": ["W1"]},
             "references": {"W1": bundle["references"]["W1"]}}
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(json.dumps(pilot), encoding="utf-8")
    database = tmp_path / "corpus.sqlite"
    build_corpus(source, database)
    return database, source, pilot_path, tmp_path / "evaluation.json"


def test_labels_and_paper_split_are_frozen(evaluation_inputs):
    result = build_evaluation(*evaluation_inputs)
    assert set(result["qrels"]) == {"W1", "W2", "W3", "W4"}
    assert all(set(refs) == {f"W{i}" for i in range(6, 14)} for refs in result["qrels"].values())
    dev, test = map(set, (result["splits"]["dev"], result["splits"]["test"]))
    assert len(dev) == len(test) == 2
    assert "W1" in dev and not dev & test
    assert dev | test == set(result["qrels"])
    for paper in result["qrels"]:
        queries = [row for row in result["queries"] if row["paper_id"] == paper]
        assert {row["query_type"] for row in queries} == {"title", "abstract"}
        assert len({row["split"] for row in queries}) == 1
    output = evaluation_inputs[-1]
    before = output.read_bytes()
    assert build_evaluation(*evaluation_inputs) == result
    assert output.read_bytes() == before
    with pytest.raises(ValueError, match="Frozen evaluation"):
        build_evaluation(*evaluation_inputs, seed=43)
    assert output.read_bytes() == before
    dates = {"W1": "2024-01-01", "W2": "2024-01-01", "W6": "2020-01-01", "W16": None}
    assert eligible_documents(dates, "W1") == {"W6"}
    assert eligible_documents(dates, "W16") == set()
    database, source, pilot, output = evaluation_inputs
    changed = json.loads(source.read_text(encoding="utf-8"))
    changed["papers"]["W6"]["title"] = "Changed source text"
    source.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        build_evaluation(*evaluation_inputs)
    build_corpus(source, database)
    with pytest.raises(ValueError, match="Frozen evaluation"):
        build_evaluation(*evaluation_inputs)
    assert output.read_bytes() == before


def test_all_pilot_queries_stop_without_publishing_split(evaluation_inputs):
    database, source, pilot, output = evaluation_inputs
    pilot.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="No test queries"):
        build_evaluation(database, source, pilot, output)
    assert not output.exists()

import hashlib
import json
import sqlite3
from copy import deepcopy
from types import SimpleNamespace

import pytest

import astro_retrieval.evaluation as runner
from astro_retrieval.bm25 import build_saved_index
from astro_retrieval.corpus import build_corpus
from astro_retrieval.dense import EmbeddingEncoder, encode_cached
from astro_retrieval.rerank import PairScorer


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    papers = {f"W{i}": {"id": f"W{i}", "title": f"planet {i}", "abstract": "planet orbit",
                        "publication_date": "2020-01-01"} for i in range(1, 15)}
    for key, date in (("W20", "2023-01-01"), ("W30", "2024-01-01"),
                      ("W31", "2024-01-01"), ("W32", "2025-01-01"), ("W33", None)):
        papers[key] = {"id": key, "title": key + " planet", "abstract": key + " planet orbit",
                       "publication_date": date}
    snapshot = save("collection.json", {"report": {"schema_version": 1, "candidate_ids": ["W20", "W30"]},
        "papers": papers, "references": {"W20": ["W1"], "W30": [f"W{i}" for i in range(1, 9)]}})
    database = tmp_path / "papers.sqlite"
    build_corpus(snapshot, database)
    source = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    evaluation = save("evaluation.json", {"source_sha256": source,
        "splits": {"dev": ["W20"], "test": ["W30"]},
        "qrels": {"W20": ["W1"], "W30": [f"W{i}" for i in range(1, 9)]},
        "queries": [{"query_id": f"{paper}:{form}", "paper_id": paper, "query_type": form, "split": split}
                    for paper, split in (("W20", "dev"), ("W30", "test"))
                    for form in ("title", "abstract")]})
    index = tmp_path / "index.json"
    build_saved_index(database, index)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("synthetic frozen protocol\n", encoding="utf-8")
    monkeypatch.setattr(runner, "PROTOCOL_SHA256", hashlib.sha256(protocol.read_text().encode()).hexdigest())
    monkeypatch.setattr(runner, "EVALUATION_SHA256", hashlib.sha256(evaluation.read_bytes()).hexdigest())
    calls = {"general": [], "indus": [], "reranker": []}

    class Encoder:
        def __init__(self, name):
            self.name = name
            self.settings = EmbeddingEncoder(name).settings

        def __call__(self, texts):
            assert set(texts) <= {papers["W30"][form] for form in ("title", "abstract")}
            calls[self.name].append(texts)
            return [[1, 0] for text in texts], [4] * len(texts)

    class Scorer:
        settings = PairScorer().settings

        def __call__(self, pairs):
            assert all(query in (papers["W30"]["title"], papers["W30"]["abstract"]) for query, doc in pairs)
            calls["reranker"].append(pairs)
            return list(range(len(pairs))), [10] * len(pairs)

    encoders = {name: Encoder(name) for name in ("general", "indus")}
    caches = [tmp_path / (name + "-documents") for name in encoders]
    texts = {key: paper["title"] + "\n" + paper["abstract"] for key, paper in papers.items()}
    for name, path in zip(encoders, caches):
        encode_cached(texts, path, lambda rows: ([[1, 0] for row in rows], [5] * len(rows)),
                      source_sha256=source, settings=encoders[name].settings)
    save("dev-results.json", {"unchanged": True})
    args = (database, evaluation, index, *caches, protocol, tmp_path / "test-results")
    protected = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    return SimpleNamespace(args=args, encoders=encoders, scorer=Scorer(), calls=calls, protected=protected)


def test_fixed_runner_resumes_and_replays_without_inference(experiment, monkeypatch):
    e = experiment
    for name, settings in (("rank_bm25", {"k1": 1.8, "b": 1.0}),
                           ("rank_fusion", {"k": 60}), ("rerank_top", {"depth": 10})):
        original = getattr(runner, name)

        def checked(*args, _original=original, _settings=settings, **kwargs):
            assert kwargs == _settings
            return _original(*args, **kwargs)

        monkeypatch.setattr(runner, name, checked)

    output = e.args[-1]

    def interrupt_reranker(row):
        if row["stage"] == "reranker_queries":
            raise RuntimeError("interrupted reranker")

    with pytest.raises(RuntimeError, match="interrupted reranker"):
        runner.run_evaluation(*e.args, encoders=e.encoders, scorer=e.scorer, progress=interrupt_reranker)
    assert not (output / "report.json").exists()
    assert len(e.calls["reranker"]) == 1
    result = runner.run_evaluation(*e.args, encoders=e.encoders, scorer=e.scorer)
    assert set(result["systems"]) == {"random", "popularity", "bm25", "general", "indus", "hybrid", "reranker"}
    assert [len(e.calls[name]) for name in ("general", "indus", "reranker")] == [1, 1, 2]
    records = {}
    for name in result["systems"]:
        records[name] = json.loads((output / (name + ".json")).read_text())["result"]["records"]
        assert set(records[name]) == {"W30:title", "W30:abstract"}
        for query, rows in records[name].items():
            assert [row["rank"] for row in rows] == list(range(1, len(rows) + 1))
            assert all(row["query_id"] == query and isinstance(row["score"], (int, float)) for row in rows)
            assert not {row["paper_id"] for row in rows} & {"W30", "W31", "W32", "W33"}
    for query in records["hybrid"]:
        first = records["hybrid"][query]
        second = records["reranker"][query]
        assert [r["paper_id"] for r in second[10:]] == [r["paper_id"] for r in first[10:]]
        assert {r["paper_id"] for r in second} == {r["paper_id"] for r in first}
        assert all(r["score_type"] == "rrf_tail" for r in second[10:])
    monkeypatch.setattr(type(e.encoders["general"]), "__call__", lambda *a: pytest.fail("Loaded encoder"))
    monkeypatch.setattr(type(e.scorer), "__call__", lambda *a: pytest.fail("Loaded scorer"))
    report_path = output / "report.json"
    legacy = json.loads(report_path.read_text())
    legacy["comparisons"]["inference_note"] = "Sign-flip assumptions apply; development results are exploratory"
    report_path.write_text(json.dumps(legacy))
    assert runner.run_evaluation(*e.args, encoders=e.encoders, scorer=e.scorer) == result
    saved_files = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert runner.run_evaluation(*e.args, encoders=e.encoders, scorer=e.scorer, replay_only=True) == result
    for name in ("manifest.json", "report.json", "bm25.json"):
        missing = output / name
        raw = missing.read_bytes()
        missing.unlink()
        with pytest.raises(ValueError, match="completed run"):
            runner.run_evaluation(*e.args, replay_only=True)
        assert not missing.exists()
        missing.write_bytes(raw)
    stage_path = output / "bm25.json"
    for field, changed in (("scores", {}), ("summary", {})):
        stage = json.loads(saved_files[stage_path])
        stage["result"][field] = changed
        stage["sha256"] = runner._digest(stage["result"])
        stage_path.write_text(json.dumps(stage))
        with pytest.raises(ValueError, match="differ"):
            runner.run_evaluation(*e.args, replay_only=True)
        stage_path.write_bytes(saved_files[stage_path])
    assert all(path.read_bytes() == raw for path, raw in saved_files.items())
    assert all(path.read_bytes() == raw for path, raw in e.protected.items())


def test_reproduction_cli_is_offline_and_requires_both_reports(tmp_path, monkeypatch, capsys):
    import socket
    import astro_retrieval.__main__ as cli
    import astro_retrieval.audit as audit

    monkeypatch.chdir(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Reproduction attempted credentials, network or model work")

    for name in ("load_dotenv", "UsageLedger", "ResponseCache", "OpenAlexClient"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(EmbeddingEncoder, "__call__", forbidden)
    monkeypatch.setattr(PairScorer, "__call__", forbidden)
    calls = []

    def retrieval(*args, **kwargs):
        assert kwargs == {"replay_only": True}
        calls.append("retrieval")
        return {"systems": dict.fromkeys(range(7))}

    def analysis(*args, **kwargs):
        assert kwargs["replay_only"] is True and len(kwargs["protocol_sha256"]) == 64
        calls.append("audit")
        return {"analysis": {"pairs": 60, "judgments": 120}}

    monkeypatch.setattr(runner, "run_evaluation", retrieval)
    monkeypatch.setattr(audit, "run_audit_analysis", analysis)
    assert cli.main(["--reproduce"]) == 0
    assert calls == ["retrieval", "audit"]
    assert json.loads(capsys.readouterr().out) == {
        "status": "reproduced", "retrieval_systems": 7, "audit_pairs": 60, "audit_judgments": 120}

    def missing(*args, **kwargs):
        raise ValueError("Missing saved result")

    monkeypatch.setattr(audit, "run_audit_analysis", missing)
    assert cli.main(["--reproduce"]) == 1
    assert "Reproduction failed" in capsys.readouterr().out
    calls.clear()
    monkeypatch.setattr(runner, "run_evaluation", missing)
    monkeypatch.setattr(audit, "run_audit_analysis", forbidden)
    assert cli.main(["--reproduce"]) == 1
    assert "reproduced" not in capsys.readouterr().out
    def damaged(*args, **kwargs):
        raise sqlite3.DatabaseError("Damaged database")

    monkeypatch.setattr(runner, "run_evaluation", damaged)
    assert cli.main(["--reproduce"]) == 1
    assert "Reproduction failed" in capsys.readouterr().out
    for option in ("--full", "--collect", "--replay"):
        with pytest.raises(SystemExit) as error:
            cli.main(["--reproduce", option])
        assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(SystemExit):
        cli.main(["--output", "new-results"])

    def rebuild(evaluation_args, audit_args, destination, **kwargs):
        assert str(destination) == "new-results"
        assert evaluation_args[-1] == "data/test" and audit_args[-1] == "data/audit/model-main"
        assert len(kwargs["protocol_sha256"]) == 64
        return {"systems": dict.fromkeys(range(7))}, {"analysis": {"pairs": 60, "judgments": 120}}

    monkeypatch.setattr(runner, "rebuild_results", rebuild)
    assert cli.main(["--reproduce", "--output", "new-results"]) == 0


def test_fresh_rebuild_recomputes_rankings_and_publishes_only_complete_bundle(experiment, monkeypatch):
    import socket
    import astro_retrieval.audit as audit

    e = experiment
    expected = runner.run_evaluation(*e.args, encoders=e.encoders, scorer=e.scorer)
    source = e.args[-1]
    legacy = json.loads((source / "report.json").read_text())
    legacy["comparisons"]["inference_note"] = "Sign-flip assumptions apply; development results are exploratory"
    (source / "report.json").write_text(json.dumps(legacy))
    root = source.parent
    audit_source = root / "audit-source"
    audit_source.mkdir()
    (audit_source / "reply.txt").write_text("Saved synthetic judgments")
    audit_args = (e.args[0], e.args[1], {"bm25": source / "bm25.json"}, e.args[-2],
                  audit_source, e.args[-2], audit_source)
    audit_result = {"analysis": {"pairs": 60, "judgments": 120}}

    def analyze(*args, **kwargs):
        assert args == audit_args and kwargs == {"protocol_sha256": "synthetic", "replay_only": True}
        return audit_result

    def forbidden(*args, **kwargs):
        pytest.fail("Rebuild attempted model or network calls")

    monkeypatch.setattr(audit, "run_audit_analysis", analyze)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(EmbeddingEncoder, "__call__", forbidden)
    monkeypatch.setattr(PairScorer, "__call__", forbidden)
    protected = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    stages = []
    destination = root / "rebuilt"
    result, analysis = runner.rebuild_results(e.args, audit_args, destination, protocol_sha256="synthetic",
                                               progress=lambda row: stages.append(row["stage"]))
    assert result == expected and analysis == audit_result
    assert {name + "_complete" for name in expected["systems"]} <= set(stages)
    for name in expected["systems"]:
        assert json.loads((destination / "test" / f"{name}.json").read_text()) == json.loads(
            (source / f"{name}.json").read_text())
    assert json.loads((destination / "audit/analysis.json").read_text()) == audit_result
    manifest = json.loads((destination / "reproduction.json").read_text())
    assert manifest["complete"] and manifest["source_sha256"] and len(manifest["output_sha256"]) == 10
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == saved for p, saved in protected.items())
    for bad in (destination, source / "nested", root):
        with pytest.raises(ValueError, match="new and separate"):
            runner.rebuild_results(e.args, audit_args, bad, protocol_sha256="synthetic")
    failed = root / "failed"
    for path in (source / "general-queries/manifest.json", source / "rerank-scores.json"):
        raw = path.read_bytes()
        value = json.loads(raw)
        value["complete"] = False
        path.write_text(json.dumps(value))
        changed = path.read_bytes()
        with pytest.raises(ValueError, match="Complete"):
            runner.rebuild_results(e.args, audit_args, failed, protocol_sha256="synthetic")
        assert path.read_bytes() == changed and not failed.exists()
        path.write_bytes(raw)
    stage = source / "bm25.json"
    original = stage.read_bytes()
    value = json.loads(original)
    value["result"]["records"] = {}
    stage.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="rankings or scores differ"):
        runner.rebuild_results(e.args, audit_args, failed, protocol_sha256="synthetic")
    stage.write_bytes(original)

    def failed_audit(*args, **kwargs):
        raise ValueError("Changed audit reply")

    monkeypatch.setattr(audit, "run_audit_analysis", failed_audit)
    with pytest.raises(ValueError, match="Changed audit"):
        runner.rebuild_results(e.args, audit_args, failed, protocol_sha256="synthetic")
    assert not failed.exists() and not list(root.glob(".reproduce-*"))
    assert all(p.read_bytes() == saved[0] for p, saved in protected.items())


def test_runner_rejects_bad_identities_and_document_caches_before_inference(experiment):
    e = experiment
    for path in (e.args[1], e.args[2], e.args[-2]):
        raw = path.read_bytes()
        if path.suffix == ".json":
            value = json.loads(raw)
            value["source_sha256"] = "wrong"
            path.write_text(json.dumps(value), encoding="utf-8")
        else:
            path.write_text("wrong", encoding="utf-8")
        with pytest.raises(ValueError):
            runner.run_evaluation(*e.args, encoders=e.encoders, scorer=e.scorer)
        path.write_bytes(raw)
    original = deepcopy(e.encoders["general"].settings)
    e.encoders["general"].settings["revision"] = "wrong"
    with pytest.raises(ValueError, match="Model settings"):
        runner.run_evaluation(*e.args, encoders=e.encoders, scorer=e.scorer)
    e.encoders["general"].settings = original
    manifest = e.args[4] / "manifest.json"
    raw = manifest.read_bytes()
    incomplete = json.loads(raw)
    incomplete["complete"] = False
    manifest.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="Complete document cache"):
        runner.run_evaluation(*e.args, encoders=e.encoders, scorer=e.scorer)
    manifest.write_bytes(raw)
    args = (*e.args[:3], e.args[-1].with_name("missing-documents"), *e.args[4:])
    with pytest.raises(ValueError, match="Complete document cache"):
        runner.run_evaluation(*args, encoders=e.encoders, scorer=e.scorer)
    assert not e.args[-1].exists()
    assert not any(e.calls.values())
    assert all(path.read_bytes() == value for path, value in e.protected.items())

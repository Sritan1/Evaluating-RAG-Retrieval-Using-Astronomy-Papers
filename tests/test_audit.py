import hashlib
import json
from copy import deepcopy

import pytest

from astro_retrieval.audit import (
    FIELDS, SYSTEMS, JUDGING_PROMPT, import_judgments, main_progress, parse_judgments,
    prepare_audit, prepare_model_main, prepare_model_practice, sample_pairs,
    analyze_judgments, label_summary, run_audit_analysis,
)
from astro_retrieval.corpus import build_corpus
from astro_retrieval.rerank import _digest


@pytest.fixture
def audit_inputs():
    papers = {f"W{i}": {"id": f"W{i}", "title": f"Planet {i}", "abstract": "Orbit, mass\nand stars",
                        "publication_date": "2020-01-01"} for i in range(1, 11)}
    splits = {"dev": ["W100", "W101"], "test": [f"W{i}" for i in range(102, 110)]}
    for paper in splits["dev"] + splits["test"]:
        papers[paper] = {"id": paper, "title": "Query planet", "abstract": "Planet observations",
                         "publication_date": "2024-01-01"}
    evaluation = {"splits": splits, "qrels": {paper: ["W1", "W2", "W3"]
                  for paper in splits["dev"] + splits["test"]},
                  "queries": [{"query_id": f"{paper}:{form}", "paper_id": paper,
                               "query_type": form, "split": split} for split, ids in splits.items()
                              for paper in ids for form in ("title", "abstract")]}
    rankings = {split: {system: {f"{paper}:{form}": [f"W{i}" for i in range(1, 11)]
                                for paper in ids for form in ("title", "abstract")}
                       for system in SYSTEMS} for split, ids in splits.items()}
    return papers, evaluation, rankings


def test_sampling_keeps_unique_pairs_and_fixed_citation_counts(audit_inputs):
    papers, evaluation, rankings = audit_inputs
    dates = {key: row["publication_date"] for key, row in papers.items()}
    sample = sample_pairs(rankings["test"], evaluation, dates)
    assert sample["pool_sizes"] == {"cited": 24, "uncited": 56}
    assert len(sample["items"]) == 60
    assert sum(item["stratum"] == "cited" for item in sample["items"]) == 20
    pairs = {(item["query_paper"], item["candidate_paper"]) for item in sample["items"]}
    assert len(pairs) == 60
    assert len({candidate for _, candidate in pairs}) < len(pairs)
    assert all(len(item["appearances"]) == 6 for item in sample["items"])
    reversed_runs = {name: dict(reversed(list(run.items())))
                     for name, run in reversed(list(rankings["test"].items()))}
    assert sample_pairs(reversed_runs, evaluation, dates) == sample
    with pytest.raises(ValueError, match="Insufficient"):
        sample_pairs(rankings["test"], evaluation, dates, cited=25)
    bad = deepcopy(rankings["test"])
    bad["bm25"]["W102:title"] = ["W102"]
    with pytest.raises(ValueError, match="ineligible"):
        sample_pairs(bad, evaluation, dates)


def test_audit_workflow_from_practice_to_saved_analysis(audit_inputs, tmp_path):
    papers, evaluation, rankings = audit_inputs

    def save(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    snapshot = save("collection.json", {"report": {"schema_version": 1,
                    "candidate_ids": list(evaluation["qrels"])}, "papers": papers,
                    "references": evaluation["qrels"]})
    database = tmp_path / "papers.sqlite"
    build_corpus(snapshot, database)
    evaluation["source_sha256"] = digest(snapshot)
    evaluation_path = save("evaluation.json", evaluation)
    common = {"split": "dev", "source_sha256": digest(snapshot), "database_sha256": digest(database),
              "evaluation_sha256": digest(evaluation_path), "selected_trial": 0}
    reports = {"bm25": save("bm25.json", {**common, "trials": [{"rankings": rankings["dev"]["bm25"]}]})}
    reports["hybrid"] = save("hybrid.json", {**common, "bm25_sha256": digest(reports["bm25"]),
                                          "trials": [{"rankings": rankings["dev"]["hybrid"]}]})
    inputs = {"database": digest(database), "evaluation": digest(evaluation_path),
              "hybrid": digest(reports["hybrid"])}
    reports["reranker"] = save("reranker.json", {"split": "dev", "inputs": inputs, "selected_trial": 0,
                                              "trials": [{"rankings": rankings["dev"]["reranker"]}]})
    rubric = tmp_path / "rubric.md"
    rubric.write_text("Status: approved for practice", encoding="utf-8")
    path = prepare_audit(database, evaluation_path, reports, rubric, tmp_path / "practice")
    mapping = json.loads((path.parent / "private-mapping.json").read_text())
    assert len(mapping["sample"]["items"]) == 6
    assert {item["query_paper"] for item in mapping["sample"]["items"]} <= set(evaluation["splits"]["dev"])
    practice_bytes = path.read_bytes()
    mapping_bytes = (path.parent / "private-mapping.json").read_bytes()
    rubric.write_text("Revised two-model practice rubric", encoding="utf-8")
    judges = {name: {"model": name + " synthetic", "mode": "high"} for name in ("chatgpt", "gemini")}
    packet_args = (database, evaluation_path, reports, path.parent, rubric, tmp_path / "model-practice")
    packet = prepare_model_practice(*packet_args, judges=judges)
    items = json.loads(packet.read_text(encoding="utf-8").split("Paper pairs (JSON data):\n")[1])
    assert len(items) == 6 and all(set(item) == set(FIELDS[:-2]) for item in items)
    assert [item["item_id"] for item in items] == [item["item_id"] for item in mapping["sample"]["items"]]
    assert all(item["candidate_abstract"] == "Orbit, mass\nand stars" for item in items)
    assert all(key not in packet.read_text(encoding="utf-8") for key in papers)
    manifest = json.loads((packet.parent / "manifest.json").read_text())
    assert manifest["judges"] == judges
    assert manifest["packet_sha256"] == digest(packet)
    assert path.read_bytes() == practice_bytes

    assert (path.parent / "private-mapping.json").read_bytes() == mapping_bytes
    rubric.write_bytes(b"Status: frozen\r\n")
    test_reports = {}
    for name, run in rankings["test"].items():
        stage = {"system": name, "identity": {"inputs": inputs, "settings": {"split": "test"}},
                 "records": {query: [{"paper_id": key} for key in ids] for query, ids in run.items()}}
        test_reports[name] = save(name + "-test.json", {"result": stage, "sha256": _digest(stage)})
    metadata = {"collected_at": "2026-10-27T12:00:00-05:00", "timezone": "America/Chicago",
                "controls": "Synthetic offline fixture"}
    practice_results = {}
    for judge in judges:
        raw = "item_id,label,reason\n" + "".join(f"{key},relevant,Same method\n" for key in manifest["item_ids"])
        import_judgments(packet.parent, judge, raw.encode(), attempt=1, metadata={**metadata, **judges[judge]})
        practice_results[judge] = digest(packet.parent / judge / "attempt-001" / "result.json")
    frozen = {"phase": "main", "sampling": {"split": "test", "uncited": 40, "cited": 20, "seed": 42,
              "top_k": 10, "systems": ["bm25", "hybrid", "reranker"], "forms": ["title", "abstract"],
              "deduplication": "query-paper/candidate-paper"}, "packet_size": 10, "packet_count": 6,
              "prompt": JUDGING_PROMPT, "prompt_sha256": hashlib.sha256(JUDGING_PROMPT.encode()).hexdigest(),
              "rubric_text": rubric.read_bytes().decode(), "rubric_sha256": digest(rubric), "judges": judges,
              "practice_result_sha256": practice_results,
              "source_sha256": {"database": digest(database), "evaluation": digest(evaluation_path),
                                **{name: digest(report) for name, report in test_reports.items()}}}
    protocol = save("protocol.json", frozen)
    output = tmp_path / "model-main"
    args = database, evaluation_path, test_reports, rubric, packet.parent, protocol, output
    protected = {file: file.read_bytes() for file in packet.parent.rglob("*") if file.is_file()}
    packets = prepare_model_main(*args, protocol_sha256=digest(protocol))
    main_manifest = json.loads((output / "manifest.json").read_text())
    all_ids = []
    for exported in packets:
        rows = json.loads(exported.read_text().split("Paper pairs (JSON data):\n")[1])
        assert len(rows) == 10 and all(set(row) == set(FIELDS[:-2]) for row in rows)
        assert all(row["candidate_abstract"] == "Orbit, mass\nand stars" for row in rows)
        all_ids.extend(row["item_id"] for row in rows)
    assert len(all_ids) == len(set(all_ids)) == 60
    main_mapping = json.loads((output / "private-mapping.json").read_text())["sample"]["items"]
    assert sum(item["stratum"] == "cited" for item in main_mapping) == 20
    assert {item["query_paper"] for item in main_mapping} <= set(evaluation["splits"]["test"])
    reply = output / "chatgpt" / "packet-01" / "reply.txt"
    reply.write_bytes(b"User reply")
    saved = {file: file.read_bytes() for file in output.rglob("*") if file.is_file()}
    assert prepare_model_main(*args, protocol_sha256=digest(protocol)) == packets
    assert all(file.read_bytes() == raw for file, raw in {**protected, **saved}.items())
    checksum = digest(protocol)
    for field, value in (("prompt", "Changed"),
                         ("judges", {**judges, "chatgpt": {"model": "other", "mode": "high"}}),
                         ("source_sha256", {**frozen["source_sha256"], "database": "wrong"})):
        protocol.write_text(json.dumps({**frozen, field: value}))
        with pytest.raises(ValueError, match="checksum"):
            prepare_model_main(*args, protocol_sha256=checksum)
        with pytest.raises(ValueError, match="differ"):
            prepare_model_main(*args, protocol_sha256=digest(protocol))
    protocol.write_text(json.dumps(frozen))
    assert not main_progress(output)["complete"]
    raws = {name: ("item_id,label,reason\n" + "".join(f"{key},relevant,Same method\n"
                 for key in info["item_ids"])).encode() for name, info in main_manifest["packets"].items()}
    rejected = import_judgments(output, "chatgpt", raws["packet-02"], attempt=1,
                                metadata={**metadata, **judges["chatgpt"]}, packet_id="packet-01")
    assert not rejected["valid"]
    for judge in judges:
        for name, raw in raws.items():
            number = 2 if (judge, name) == ("chatgpt", "packet-01") else 1
            imported = import_judgments(output, judge, raw, attempt=number,
                                        metadata={**metadata, **judges[judge]}, packet_id=name)
            assert imported["valid"] and len(imported["labels"]) == 10
            if judge == "chatgpt":
                assert not main_progress(output)["complete"]
    progress = main_progress(output)
    assert progress["complete"] and progress["judges"]["chatgpt"]["selected_attempts"]["packet-01"] == 2
    import_judgments(output, "chatgpt", raws["packet-01"].replace(b"relevant", b"unsure"), attempt=3,
                     metadata={**metadata, **judges["chatgpt"]}, packet_id="packet-01")
    assert main_progress(output) == progress
    with pytest.raises(ValueError, match="sequential"):
        import_judgments(output, "chatgpt", raws["packet-01"], attempt=5,
                         metadata={**metadata, **judges["chatgpt"]}, packet_id="packet-01")
    assert all(file.read_bytes() == raw for file, raw in protected.items())

    protected.update({file: file.read_bytes() for file in output.rglob("*") if file.is_file()})
    report = run_audit_analysis(*args, protocol_sha256=checksum)
    assert report["analysis"]["pairs"] == 60 and report["analysis"]["judgments"] == 120
    assert report["analysis"]["agreement"]["all"]["rate"] == 1
    assert report["selected_attempts"]["chatgpt"]["selected_attempts"]["packet-01"] == 2
    assert run_audit_analysis(*args, protocol_sha256=checksum) == report
    assert run_audit_analysis(*args, protocol_sha256=checksum, replay_only=True) == report
    assert all(file.read_bytes() == raw for file, raw in protected.items())
    report_path = output / "analysis.json"
    report_bytes = report_path.read_bytes()
    report_path.unlink()
    with pytest.raises(ValueError, match="completed audit"):
        run_audit_analysis(*args, protocol_sha256=checksum, replay_only=True)
    assert not report_path.exists()
    report_path.write_bytes(report_bytes)
    for changed in (protocol, database, output / "private-mapping.json",
                    output / "chatgpt" / "packet-01" / "attempt-002" / "raw.txt"):
        original = changed.read_bytes()
        if changed.name == "private-mapping.json":
            value = json.loads(original)
            value["sample"]["items"][0]["candidate_paper"] = "W999"
            changed.write_text(json.dumps(value))
        else:
            changed.write_bytes(original + b" ")
        with pytest.raises(ValueError):
            run_audit_analysis(*args, protocol_sha256=checksum)
        changed.write_bytes(original)
        assert report_path.read_bytes() == report_bytes
    missing = output / "gemini" / "packet-06" / "attempt-001"
    absent = tmp_path / "missing-attempt"
    missing.rename(absent)
    with pytest.raises(ValueError, match="complete all six"):
        run_audit_analysis(*args, protocol_sha256=checksum)
    absent.rename(missing)
    changed_report = {**report, "schema_version": 99}
    report_path.write_text(json.dumps(changed_report))
    with pytest.raises(ValueError, match="never overwrite"):
        run_audit_analysis(*args, protocol_sha256=checksum)
    assert json.loads(report_path.read_text()) == changed_report


def test_label_summaries_and_agreement_keep_unsure_separate(audit_inputs):
    labels = ["relevant", "partly relevant", "not relevant", "unsure"]
    summary = label_summary(labels)
    assert summary["confident"] == 3 and summary["unsure_rate"] == .25
    assert summary["strict"] == {"positive": 1, "confident_rate": 1 / 3, "unsure_bounds": [.25, .5]}
    assert summary["inclusive"] == {"positive": 2, "confident_rate": 2 / 3, "unsure_bounds": [.5, .75]}
    assert label_summary(["unsure"])["strict"]["confident_rate"] is None
    assert label_summary(["unsure"])["strict"]["unsure_bounds"] == [0, 1]
    assert label_summary([])["strict"]["unsure_bounds"] is None
    papers, evaluation, rankings = audit_inputs
    sample = {"items": [{"item_id": str(i), "query_paper": "W102", "candidate_paper": f"W{i}",
                         "stratum": "cited" if i <= 3 else "uncited"} for i in range(1, 5)]}
    judgments = {"chatgpt": dict(zip(map(str, range(1, 5)),
                                     ["relevant", "partly relevant", "unsure", "not relevant"])),
                 "gemini": dict(zip(map(str, range(1, 5)),
                                    ["relevant", "relevant", "unsure", "not relevant"]))}
    report = analyze_judgments(sample, judgments, rankings["test"], evaluation)
    assert report["agreement"]["all"] == {"matches": 3, "pairs": 4, "rate": .75}
    assert report["agreement"]["both_confident"] == {"matches": 2, "pairs": 3, "rate": 2 / 3}
    assert report["agreement"]["matrix"]["unsure"]["unsure"] == 1
    assert report["disagreements"] == [{"item_id": "2", "chatgpt": "partly relevant", "gemini": "relevant"}]
    assert report["summaries"]["chatgpt"]["cited"]["total"] == 3
    assert report["summaries"]["chatgpt"]["uncited"]["total"] == 1
    all_unsure = {judge: {key: "unsure" for key in group} for judge, group in judgments.items()}
    empty = analyze_judgments(sample, all_unsure, rankings["test"], evaluation)
    assert empty["agreement"]["both_confident"]["rate"] is None
    assert empty["comparisons"]["strict"]["title"]["status"] == "unavailable"
    assert empty["comparisons"]["strict"]["title"]["mean_ndcg@10"]["bm25"]["citations"] is None
    for bad in ({**judgments, "gemini": {}}, {**judgments, "gemini": {**judgments["gemini"], "4": "maybe"}}):
        with pytest.raises(ValueError, match="both judges"):
            analyze_judgments(sample, bad, rankings["test"], evaluation)
    with pytest.raises(ValueError, match="unique"):
        analyze_judgments({"items": sample["items"] + sample["items"][:1]}, judgments, rankings["test"], evaluation)


def test_common_subset_scores_preserve_order_and_shared_coverage():
    evaluation = {"splits": {"test": ["Q", "R", "S", "T"]},
                  "qrels": {"Q": ["A"], "R": ["A"], "S": ["X"], "T": ["A"]}}
    sample = {"items": [{"item_id": paper + candidate, "query_paper": paper,
                          "candidate_paper": candidate,
                          "stratum": "cited" if candidate in evaluation["qrels"][paper] else "uncited"}
                         for paper in evaluation["splits"]["test"] for candidate in ("A", "B")]}
    judgments = {judge: {item["item_id"]: "relevant" if item["candidate_paper"] == "A" else "not relevant"
                         for item in sample["items"]} for judge in ("chatgpt", "gemini")}
    judgments["gemini"]["RA"] = "partly relevant"
    rankings = {system: {f"{paper}:{form}": ["A", "B", "X"] if system == "bm25" else ["X", "B", "A"]
                          for paper in evaluation["splits"]["test"] for form in ("title", "abstract")}
                for system in SYSTEMS}
    rankings["hybrid"]["Q:abstract"] = ["A", "X"]
    for judge in judgments:
        judgments[judge]["TB"] = "unsure"
    protected = deepcopy((sample, judgments, rankings, evaluation))
    report = analyze_judgments(sample, judgments, rankings, evaluation)
    title = report["comparisons"]["strict"]["title"]
    assert list(title["included"]) == ["Q"]
    assert title["included"]["Q"]["rankings"]["hybrid"] == ["B", "A"]
    for judge in ("citations", "chatgpt", "gemini"):
        assert title["mean_ndcg@10"]["bm25"][judge] == 1
        assert title["mean_ndcg@10"]["hybrid"][judge] == pytest.approx(.6309297536)
    assert title["excluded"]["R"]["reasons"] == ["no_positive_gemini"]
    assert title["excluded"]["S"]["reasons"] == ["no_positive_citations"]
    assert title["excluded"]["T"]["reasons"] == ["fewer_than_two_common_candidates"]
    assert report["comparisons"]["strict"]["abstract"]["eligible_queries"] == 0
    assert list(report["comparisons"]["inclusive"]["title"]["included"]) == ["Q", "R"]
    assert list(report["comparisons"]["inclusive"]["abstract"]["included"]) == ["R"]
    assert analyze_judgments(sample, judgments, rankings, evaluation) == report
    assert (sample, judgments, rankings, evaluation) == protected


def test_reply_import_validates_labels_and_preserves_attempts(tmp_path, monkeypatch):
    ids = [f"item-{i:03d}" for i in range(1, 5)]
    raw = ('item_id,label,reason\nitem-001,relevant,"Same method, same question"\n'
           'item-002,partly relevant,Supporting context\nitem-003,not relevant,Different question\n'
           'item-004,unsure,Insufficient abstract detail\n')
    expected = parse_judgments(raw, ids)
    assert [row["label"] for row in expected] == ["relevant", "partly relevant", "not relevant", "unsure"]
    assert parse_judgments("```csv\n" + raw + "```", ids) == expected
    assert parse_judgments("```\n" + raw + "```", ids) == expected
    packet = tmp_path / "packet-01.txt"
    packet.write_text("Synthetic blinded packet", encoding="utf-8")
    judges = {name: {"model": name + " synthetic", "mode": "high"} for name in ("chatgpt", "gemini")}
    manifest = {"phase": "practice", "item_ids": ids, "judges": judges,
                "packet_file": packet.name, "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    metadata = {**judges["chatgpt"], "collected_at": "2026-10-27T12:00:00-05:00",
                "timezone": "America/Chicago", "controls": "Synthetic offline fixture"}
    def interrupted(*args):
        raise OSError("Interrupted import")

    with monkeypatch.context() as patch:
        patch.setattr("astro_retrieval.audit.write_json", interrupted)
        with pytest.raises(OSError, match="Interrupted import"):
            import_judgments(tmp_path, "chatgpt", raw.encode(), attempt=1, metadata=metadata)
    assert not list((tmp_path / "chatgpt").iterdir())
    result = import_judgments(tmp_path, "chatgpt", raw.encode(), attempt=1, metadata=metadata)
    assert result["valid"] and result["labels"] == expected and result["metadata"] == metadata
    assert import_judgments(tmp_path, "chatgpt", raw.encode(), attempt=1, metadata=metadata) == result
    attempt = tmp_path / "chatgpt" / "attempt-001"
    protected = {path: path.read_bytes() for path in attempt.iterdir()}
    for changed_raw, changed_metadata in ((raw + "\n", metadata),
                                          (raw, {**metadata, "controls": "Changed"})):
        with pytest.raises(ValueError, match="never overwrite"):
            import_judgments(tmp_path, "chatgpt", changed_raw.encode(), attempt=1, metadata=changed_metadata)
    invalid = ["\n".join(raw.splitlines()[:-1]), raw + 'item-005,relevant,Extra\n',
               raw + 'item-001,relevant,Duplicate\n', raw.replace(',partly relevant,', ',maybe,'),
               raw.replace('Supporting context', ''), raw.replace('Same method, same question', 'bad"quote'),
               raw.replace('Supporting context', 'Unquoted,extra field'), 'Intro\n' + raw]
    for response in invalid:
        with pytest.raises(ValueError):
            parse_judgments(response, ids)
    rejected = import_judgments(tmp_path, "chatgpt", invalid[0].encode(), attempt=2, metadata=metadata)
    assert not rejected["valid"] and "labels" not in rejected
    assert (tmp_path / "chatgpt/attempt-002/raw.txt").read_bytes() == invalid[0].encode()
    second = import_judgments(tmp_path, "gemini", raw.encode(), attempt=1,
                              metadata={**metadata, **judges["gemini"]})
    assert second["judge"] == "gemini" and second["valid"]
    with pytest.raises(ValueError, match="model or mode"):
        import_judgments(tmp_path, "gemini", raw.encode(), attempt=2, metadata=metadata)
    packet.write_text("Changed packet")
    with pytest.raises(ValueError, match="checksum"):
        import_judgments(tmp_path, "chatgpt", raw.encode(), attempt=20, metadata=metadata)
    assert all(path.read_bytes() == value for path, value in protected.items())

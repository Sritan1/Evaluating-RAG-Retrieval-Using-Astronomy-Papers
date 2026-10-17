import json
import os
from unittest.mock import Mock

import pytest
import requests

from astro_retrieval.__main__ import OfflineClient, main
from astro_retrieval import __main__ as cli
from astro_retrieval.ingest.budget import BudgetError, UsageLedger
from astro_retrieval.ingest.cache import ResponseCache
from astro_retrieval.ingest.collection import (
    candidate_parameters, collect_candidates, load_candidates, run_collection,
)
from astro_retrieval.ingest.openalex import OpenAlexClient, OpenAlexError


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.setattr(requests.Session, "request", Mock(side_effect=AssertionError("Network forbidden")))


def records(usable, count=200):
    return [{"id": f"W{i + 1}", "title": "Exoplanet", "publication_date": "2024-01-01",
             "referenced_works": [],
             "abstract_inverted_index": {"planet": [0]} if i < usable else None}
            for i in range(count)]


@pytest.fixture
def client():
    value = Mock(spec=OpenAlexClient)
    value.ledger = Mock(spec=UsageLedger)
    value.ledger.usage_totals.return_value = {
        "attempts": 0, "reserved_credits": 0, "reported_credits": 0,
    }
    return value


@pytest.mark.parametrize("full,size", [(False, 200), (True, 600)])
def test_preview_has_no_io(tmp_path, monkeypatch, capsys, full, size):
    monkeypatch.setattr(cli, "load_dotenv", Mock(side_effect=AssertionError("Credentials loaded")))
    assert main(["--full"] if full else []) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["sample_size"] == size
    assert preview["parameters"]["corpus"] == "core"
    assert preview["parameters"]["search.title_and_abstract"] == 'exoplanet OR "extrasolar planet"'
    assert preview["parameters"]["filter"] == (
        "primary_topic.id:T10039,language:en,type:article,"
        "from_publication_date:2021-01-01,to_publication_date:2025-12-31"
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", [None, "dummy-existing-key"])
def test_collection_loads_dotenv_without_overriding_environment(tmp_path, monkeypatch, capsys, existing):
    (tmp_path / ".env").write_text("OPENALEX_API_KEY=dummy-file-key\n", encoding="utf-8-sig")
    if existing:
        monkeypatch.setenv("OPENALEX_API_KEY", existing)

    def run(client, directory, *, replay, full):
        assert os.environ["OPENALEX_API_KEY"] == (existing or "dummy-file-key")
        assert replay is False
        return {"abstract_coverage_passed": True, "feasibility_status": "pending",
                "current_run_api_usage": {"attempts": 0}}

    monkeypatch.setattr(cli, "run_collection", run)
    assert main(["--collect"]) == 0
    output = capsys.readouterr()
    assert "dummy-file-key" not in output.out + output.err
    assert "dummy-existing-key" not in output.out + output.err


@pytest.mark.parametrize("usable,count,passed", [(140, 200, True), (139, 200, False), (0, 0, None)])
def test_coverage_gate_is_not_overall_feasibility(client, tmp_path, usable, count, passed):
    client.iter_works.return_value = records(usable, count)
    report = run_collection(client, tmp_path)
    assert report["abstract_coverage_passed"] is passed
    assert report["feasibility_status"] == "pending"
    assert report["runtime_status"] == "not_measured"
    assert len(json.loads((tmp_path / "pilot.json").read_text())["papers"]) == usable


def test_failure_after_first_page_does_not_save_pool_or_report(client, tmp_path):
    def pages():
        yield from records(1, 1)
        assert not (tmp_path / "candidates.json").exists()
        raise OpenAlexError("Page failed")
    client.iter_works.return_value = pages()
    with pytest.raises(OpenAlexError):
        run_collection(client, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_complete_pool_is_frozen_and_checked(client, tmp_path):
    client.iter_works.return_value = iter(records(2, 2))
    path = tmp_path / "candidates.json"
    snapshot = collect_candidates(client, path)
    assert len(snapshot["works"]) == 2
    assert load_candidates(path) == snapshot
    collect_candidates(client, path)
    client.iter_works.assert_called_once_with(candidate_parameters())
    snapshot["works"][0]["title"] = "changed"
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="snapshot"):
        load_candidates(path)


def test_reopened_cache_replays_candidates_and_references_without_key(tmp_path, monkeypatch):
    cache_path = tmp_path / ".cache/openalex.sqlite"
    ledger_path = tmp_path / ".cache/usage.sqlite"
    seed = records(1, 1)[0]
    seed["referenced_works"] = ["W2"]
    reference = {**records(1, 1)[0], "id": "W2", "publication_date": "2020-01-01"}
    with ResponseCache(cache_path) as cache:
        for params, works in [
            ({**candidate_parameters(), "cursor": "*"}, [seed]),
            ({"filter": "openalex_id:W2", "per_page": 100, "cursor": "*"}, [reference]),
        ]:
            cache.get_or_fetch("/works", params, lambda url, query: {
                "results": works, "meta": {"next_cursor": None},
            })
    with ResponseCache(cache_path) as cache, UsageLedger(ledger_path) as ledger:
        with OfflineClient(cache, ledger) as offline:
            report = run_collection(offline, tmp_path / "data/pilot")
            assert report["corpus_size"] == 2
    monkeypatch.setattr(cli, "load_dotenv", Mock(side_effect=AssertionError("Credentials loaded")))
    assert main(["--replay"]) == 0
    replayed = json.loads((tmp_path / "data/pilot/pilot.json").read_text())["report"]
    assert replayed == report
    with UsageLedger(ledger_path) as ledger:
        assert ledger.usage_totals()["attempts"] == 0


def test_offline_cache_miss_stops_before_credentials_or_network(tmp_path):
    with ResponseCache(tmp_path / "cache.sqlite") as cache, UsageLedger(tmp_path / "usage.sqlite") as ledger:
        with OfflineClient(cache, ledger) as offline:
            with pytest.raises(OpenAlexError, match="Offline replay"):
                offline.get("/works", candidate_parameters())
            assert ledger.usage_totals()["attempts"] == 0


def test_full_run_resumes_budget_stop_and_replays_without_touching_pilot(client, tmp_path, monkeypatch):
    works = records(599, 600)
    works[0]["referenced_works"] = ["W1001", "W1002"]
    client.iter_works.return_value = works
    pilot = tmp_path / "data/pilot"
    collect_candidates(client, pilot / "candidates.json")
    (pilot / "pilot.json").write_text("original pilot", encoding="utf-8")
    original = {path.name: path.read_bytes() for path in pilot.iterdir()}
    full = tmp_path / "data/full"

    def response(payload):
        value = Mock(spec=requests.Response)
        value.status_code = 200
        value.headers = {
            "X-RateLimit-Limit": "10000", "X-RateLimit-Remaining": "9000",
            "X-RateLimit-Credits-Used": "0", "X-RateLimit-Reset": "3600",
        }
        value.json.return_value = payload
        return value

    def reference(identifier):
        return {**records(1, 1)[0], "id": identifier, "publication_date": "2020-01-01"}

    monkeypatch.setenv("OPENALEX_API_KEY", "dummy-full-key")
    session = Mock(spec=requests.Session)
    session.get.side_effect = [response({}), response({
        "results": [reference("W1001")], "meta": {"next_cursor": "page-two"},
    })]
    cache_path = tmp_path / ".cache/openalex.sqlite"
    ledger_path = tmp_path / ".cache/usage.sqlite"
    with ResponseCache(cache_path) as cache, UsageLedger(ledger_path, daily_limit=11) as ledger:
        with OpenAlexClient(cache, ledger, session=session) as live:
            with pytest.raises(BudgetError, match="Local daily"):
                run_collection(live, full, full=True)
    assert session.get.call_count == 2
    assert not (full / "collection.json").exists()
    selection = (full / "selection.json").read_bytes()
    assert len(json.loads(selection)["candidate_ids"]) == 600

    session.reset_mock()
    session.get.side_effect = [response({}), response({
        "results": [reference("W1002")], "meta": {"next_cursor": None},
    })]
    with ResponseCache(cache_path) as cache, UsageLedger(ledger_path) as ledger:
        with OpenAlexClient(cache, ledger, session=session) as live:
            report = run_collection(live, full, full=True)
        assert ledger.usage_totals()["attempts"] == 4
    assert session.get.call_count == 2
    assert session.get.call_args.kwargs["params"]["cursor"] == "page-two"
    assert report["candidate_count"] == 600
    assert report["usable_seed_count"] == 599
    assert report["seed_missing_abstract_ids"] == ["W600"]
    assert (full / "selection.json").read_bytes() == selection
    completed = json.loads((full / "collection.json").read_text(encoding="utf-8"))

    monkeypatch.delenv("OPENALEX_API_KEY")
    monkeypatch.setattr(cli, "load_dotenv", Mock(side_effect=AssertionError("Credentials loaded")))
    assert main(["--full", "--replay"]) == 0
    replayed = json.loads((full / "collection.json").read_text(encoding="utf-8"))
    assert replayed["papers"] == completed["papers"]
    assert replayed["references"] == completed["references"]
    assert replayed["report"]["current_run_api_usage"]["attempts"] == 0
    with UsageLedger(ledger_path) as ledger:
        assert ledger.usage_totals()["attempts"] == 4
    assert {path.name: path.read_bytes() for path in pilot.iterdir()} == original

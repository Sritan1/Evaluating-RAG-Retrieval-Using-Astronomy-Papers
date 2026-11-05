import copy
import json
import random
import os
from unittest.mock import Mock

import pytest
import requests

from astro_retrieval.__main__ import OfflineClient, main
from astro_retrieval import __main__ as cli
from astro_retrieval.corpus import build_corpus
from astro_retrieval.ingest.budget import BudgetError, UsageLedger
from astro_retrieval.ingest.cache import ResponseCache
from astro_retrieval.ingest.collection import (
    candidate_parameters, collect_candidates, load_candidates, run_collection,
)
from astro_retrieval.ingest.openalex import OpenAlexClient, OpenAlexError
from astro_retrieval.ingest.pilot import select_pilot
from astro_retrieval.ingest.records import reconstruct_abstract


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


def test_full_run_resumes_budget_stop_and_replays_without_touching_pilot(client, tmp_path, monkeypatch, capsys):
    works = records(599, 600)
    works[0]["referenced_works"] = ["W1001", "W1002"]
    client.iter_works.return_value = works
    pilot = tmp_path / "data/pilot"
    collect_candidates(client, pilot / "candidates.json")
    (pilot / "pilot.json").write_text("original pilot", encoding="utf-8")
    original = {path.name: path.read_bytes() for path in pilot.iterdir()}
    full = tmp_path / "data/full"

    def response(payload=None):
        value = Mock(spec=requests.Response)
        value.status_code = 200
        value.headers = {
            "X-RateLimit-Limit": "10000", "X-RateLimit-Remaining": "9000",
            "X-RateLimit-Credits-Used": "0", "X-RateLimit-Reset": "3600",
        }
        value.json.return_value = {"rate_limit": {
            "daily_budget_usd": 1, "daily_used_usd": 0, "daily_remaining_usd": 1,
            "prepaid_balance_usd": 0, "prepaid_remaining_usd": 0, "resets_in_seconds": 3600,
        }} if payload is None else payload
        return value

    def reference(identifier):
        return {**records(1, 1)[0], "id": identifier, "publication_date": "2020-01-01"}

    monkeypatch.setenv("OPENALEX_API_KEY", "dummy-full-key")
    session = Mock(spec=requests.Session)
    session.get.side_effect = [response(), response({
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
    session.get.side_effect = [response(), response({
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
    snapshot = full / "collection.json"
    completed = snapshot.read_bytes()
    assert json.loads(completed)["report"]["current_run_api_usage"]["attempts"] == 2
    database = full / "papers.sqlite"
    build_corpus(snapshot, database)
    database_bytes = database.read_bytes()

    monkeypatch.delenv("OPENALEX_API_KEY")
    monkeypatch.setattr(cli, "load_dotenv", Mock(side_effect=AssertionError("Credentials loaded")))
    assert main(["--full", "--replay"]) == 0
    assert json.loads(capsys.readouterr().out)["current_run_api_usage"]["attempts"] == 0
    assert snapshot.read_bytes() == completed
    assert database.read_bytes() == database_bytes
    with UsageLedger(ledger_path) as ledger:
        assert ledger.usage_totals()["attempts"] == 4
    assert {path.name: path.read_bytes() for path in pilot.iterdir()} == original
    changed = json.loads(completed)
    changed["papers"]["W1"]["title"] = "Conflicting saved title"
    snapshot.write_text(json.dumps(changed), encoding="utf-8")
    conflicting = snapshot.read_bytes()
    with ResponseCache(cache_path) as cache, UsageLedger(ledger_path) as ledger:
        with OfflineClient(cache, ledger) as offline:
            with pytest.raises(ValueError, match="Completed collection differs"):
                run_collection(offline, full, full=True, replay=True)
    assert snapshot.read_bytes() == conflicting
    assert database.read_bytes() == database_bytes


def test_sampling_is_independent_of_input_order_and_duplicates():
    works = [{"id": f"https://openalex.org/W{number}"} for number in range(1, 11)]
    before = random.getstate()
    first = select_pilot(works, size=4, seed=42)
    reordered = select_pilot(reversed(works + [works[0], {"id": "W1"}]), size=4, seed=42)
    assert first == reordered
    assert first.seed == 42 and random.getstate() == before
    assert select_pilot(works, size=4, seed=7).candidate_ids != first.candidate_ids
    assert len(first.candidate_ids) == len(set(first.candidate_ids)) == 4
    assert set(first.candidate_ids) <= {f"W{number}" for number in range(1, 11)}


def test_coverage_includes_missing_and_invalid_abstracts():
    works = [
        {"id": "W1", "abstract_inverted_index": {"planets": [0, 2], "orbit": [1]}},
        {"id": "W2", "abstract_inverted_index": None},
        {"id": "W3", "abstract_inverted_index": {}},
        {"id": "W4", "abstract_inverted_index": {"bad": [1]}},
    ]
    original = copy.deepcopy(works)
    sample = select_pilot(works)
    assert sample.requested_size == 200
    assert len(sample.candidate_ids) == 4
    assert sample.abstracts == {"W1": "planets orbit planets"}
    assert set(sample.missing_ids) == {"W2", "W3"}
    assert sample.invalid_ids == ("W4",)
    assert sample.abstract_coverage == 0.25
    assert works == original


def test_abstract_availability_does_not_change_selection_or_trigger_replacement():
    works = [{"id": f"W{number}", "abstract_inverted_index": {"word": [0]}}
             for number in range(1, 11)]
    complete = select_pilot(works, size=3)
    changed_id = complete.candidate_ids[0]
    next(work for work in works if work["id"] == changed_id)["abstract_inverted_index"] = None
    missing = select_pilot(works, size=3)
    assert missing.candidate_ids == complete.candidate_ids
    assert missing.missing_ids == (changed_id,)
    assert len(missing.abstracts) == 2
    assert missing.abstract_coverage == pytest.approx(2 / 3)


def test_conflicting_duplicates_raise_in_either_order():
    works = [{"id": "W1", "title": "first"}, {"id": "https://openalex.org/W1", "title": "second"}]
    for candidates in (works, list(reversed(works))):
        with pytest.raises(ValueError, match="Conflicting"):
            select_pilot(candidates)


@pytest.mark.parametrize("work", [None, {}, {"id": "https://example.org/W1"}])
def test_invalid_candidate_ids_raise(work):
    with pytest.raises(ValueError):
        select_pilot([work])


def test_reconstructs_repeated_words_in_position_order():
    index = {"planets": [2, 0], "orbit": [1]}
    assert reconstruct_abstract(index) == "planets orbit planets"
    assert index == {"planets": [2, 0], "orbit": [1]}
    assert reconstruct_abstract({"planet.": [2], "A": [0], "β": [1]}) == "A β planet."


@pytest.mark.parametrize("index", [
    [], {"word": [-1]}, {"word": [True]}, {"word": []}, {"": [0]},
])
def test_malformed_index_raises(index):
    with pytest.raises(ValueError):
        reconstruct_abstract(index)


@pytest.mark.parametrize("index", [
    {"one": [0], "two": [0]}, {"word": [0, 0]},
])
def test_overlapping_positions_raise(index):
    with pytest.raises(ValueError, match="overlap"):
        reconstruct_abstract(index)


@pytest.mark.parametrize("index", [
    {"one": [0], "three": [2]}, {"word": [10**12]},
])
def test_missing_positions_raise_without_allocating_from_largest_position(index):
    with pytest.raises(ValueError, match="contiguous"):
        reconstruct_abstract(index)


@pytest.mark.parametrize("works,ids,coverage", [
    ([], (), None), ([{"id": "W1"}, {"id": "W1"}], ("W1",), 0),
])
def test_small_or_empty_pool(works, ids, coverage):
    sample = select_pilot(works)
    assert sample.candidate_ids == ids
    assert sample.abstracts == {} and sample.abstract_coverage == coverage


@pytest.mark.parametrize("options", [{"size": 0}, {"seed": "42"}])
def test_invalid_sampling_settings(options):
    with pytest.raises(ValueError):
        select_pilot([], **options)

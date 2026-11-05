import copy
import json
from contextlib import ExitStack
from unittest.mock import Mock

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from astro_retrieval.ingest.budget import UsageLedger
from astro_retrieval.ingest.cache import ResponseCache
from astro_retrieval.ingest.expansion import expand_pilot, reports_match, write_pilot_report
from astro_retrieval.ingest.openalex import OpenAlexClient


def work(identifier, publication_date="2020-01-01", refs=None, abstract=True):
    return {
        "id": identifier,
        "title": identifier,
        "publication_date": publication_date,
        "referenced_works": [] if refs is None else refs,
        "abstract_inverted_index": {"planet": [0]} if abstract else None,
    }


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(requests.Session, "request", Mock(
        side_effect=AssertionError("Live network access is forbidden in these tests")
    ))


@pytest.fixture
def example():
    seeds = [
        work("W1", "2024-01-01", ["W3", "W4", "https://openalex.org/W3", "W1"]),
        work("W2", "2024-06-01", ["W3", "W5", "W6"]),
    ]
    references = [
        work("W3", "2020-01-01", ["W999"]),
        work("W4", abstract=False),
        work("W5", "2025-01-01"),
    ]
    return seeds, references


@pytest.fixture
def client():
    client = Mock(spec=OpenAlexClient)
    client.ledger = Mock(spec=UsageLedger)
    client.ledger.usage_totals.return_value = {
        "attempts": 0, "reserved_credits": 0, "reported_credits": 0,
    }
    client.iter_works.return_value = []
    return client


def test_reviewed_reference_counts_and_one_hop_only(example, client):
    seeds, refs = example
    originals = copy.deepcopy((seeds, refs))
    client.iter_works.return_value = refs
    result = expand_pilot(seeds, client)
    report = result.report
    assert set(result.papers) == {"W1", "W2", "W3", "W5"}
    assert report["corpus_size"] == 4
    assert report["unique_reference_count"] == 4
    assert report["resolved_reference_count"] == 3
    assert report["reference_resolution_coverage"] == 0.75
    assert report["usable_reference_fraction"] == 0.5
    assert report["resolved_reference_abstract_coverage"] == pytest.approx(2 / 3)
    assert report["unresolved_reference_ids"] == ["W6"]
    assert report["reference_missing_abstract_ids"] == ["W4"]
    assert report["median_eligible_references"] == 1
    assert report["eligible_seed_count"] == 0
    assert report["eligible_seed_fraction"] == 0
    assert report["corpus_year_counts"] == {"2020": 1, "2024": 2, "2025": 1}
    assert [row["eligible_ids"] for row in report["per_seed"]] == [["W3"], ["W3"]]
    assert report["per_seed"][1]["same_or_later_ids"] == ["W5"]
    assert result.references == {"W1": ("W3", "W4"), "W2": ("W3", "W5", "W6")}
    client.iter_works.assert_called_once_with({
        "filter": "openalex_id:W3|W4|W5|W6", "per_page": 100,
    })
    assert "W999" not in result.papers
    assert (seeds, refs) == originals


def test_known_cross_seed_reference_is_reused_without_fetching(client):
    seeds = [work("W1", "2024-01-01", ["W2"]), work("W2", "2020-01-01")]
    result = expand_pilot(seeds, client, min_references=1)
    client.iter_works.assert_not_called()
    assert result.report["resolved_reference_count"] == 1
    assert result.report["eligible_seed_count"] == 1
    assert result.report["eligible_seed_fraction"] == 0.5
    assert result.report["median_eligible_references"] == 0.5
    assert result.report["corpus_size"] == 2


def test_batches_are_sorted_unique_and_at_most_100(client):
    refs = [f"W{number}" for number in range(1000, 1201)]
    seed = work("W1", refs=refs + refs[:5])
    result = expand_pilot([seed], client)
    batches = [call.args[0]["filter"].split(":", 1)[1].split("|")
               for call in client.iter_works.call_args_list]
    assert [len(batch) for batch in batches] == [100, 100, 1]
    assert [identifier for batch in batches for identifier in batch] == sorted(refs)
    assert result.report["lookup_batch_count"] == 3
    assert result.report["unresolved_reference_ids"] == sorted(refs)


@pytest.mark.parametrize("seed_date,reference_date,reason", [
    *(('2024-01-01', date, 'unknown_date_ids')
      for date in (None, "2020-01", "2020-02-30")),
    (None, "2020-01-01", "unknown_date_ids"),
    ("2020-01-01", "2020-01-01", "same_or_later_ids"),
])
def test_reference_date_eligibility(client, seed_date, reference_date, reason):
    client.iter_works.return_value = [work("W2", reference_date)]
    result = expand_pilot([work("W1", seed_date, ["W2"])], client)
    row = result.report["per_seed"][0]
    assert row["in_corpus_references"] == 1 and row["eligible_references"] == 0
    assert row[reason] == ["W2"]
    if seed_date == "2024-01-01":
        assert result.report["corpus_unknown_date_ids"] == ["W2"]


def test_unusable_seeds_are_not_expanded_and_remain_in_coverage(client):
    seeds = [work("W1", abstract=False), work("W2")]
    seeds[0]["referenced_works"] = ["W999"]
    result = expand_pilot(seeds, client)
    client.iter_works.assert_not_called()
    assert result.report["candidate_count"] == 2
    assert result.report["usable_seed_count"] == 1
    assert result.report["seed_abstract_coverage"] == 0.5
    assert set(result.papers) == {"W2"}


def test_invalid_reference_abstract_is_distinct_from_unresolved(client):
    invalid = work("W2")
    invalid["abstract_inverted_index"] = {"gap": [1]}
    client.iter_works.return_value = [invalid]
    result = expand_pilot([work("W1", refs=["W2"])], client)
    assert result.report["resolved_reference_count"] == 1
    assert result.report["reference_invalid_abstract_ids"] == ["W2"]
    assert result.report["unresolved_reference_ids"] == []
    assert result.report["usable_reference_count"] == 0


@pytest.mark.parametrize("missing", [True, False])
def test_missing_reference_list_is_not_misreported_as_zero(client, missing):
    seed = work("W1")
    if missing:
        del seed["referenced_works"]
    else:
        seed["referenced_works"] = None
    with pytest.raises(ValueError, match="referenced_works"):
        expand_pilot([seed], client)
    client.iter_works.assert_not_called()


def test_unrequested_ids_raise_instead_of_silently_remapping(client):
    client.iter_works.return_value = [work("W99")]
    with pytest.raises(ValueError, match="unrequested"):
        expand_pilot([work("W1", refs=["W2"])], client)


def test_empty_sample_has_no_fabricated_coverage_or_calls(client):
    result = expand_pilot([], client)
    assert result.papers == result.references == {}
    assert result.report["median_eligible_references"] is None
    assert result.report["reference_resolution_coverage"] is None
    assert result.report["eligible_seed_fraction"] is None
    client.iter_works.assert_not_called()


def test_report_is_valid_json_and_rewriting_does_not_append(example, client, tmp_path):
    seeds, refs = example
    client.iter_works.return_value = refs
    result = expand_pilot(seeds, client)
    path = tmp_path / "reports" / "pilot.json"
    write_pilot_report(result, path)
    first = path.read_bytes()
    write_pilot_report(result, path)
    assert path.read_bytes() == first
    assert json.loads(first) == result.report
    assert list(path.parent.iterdir()) == [path]
    result.report["bad"] = float("nan")
    with pytest.raises(ValueError):
        write_pilot_report(result, path)
    assert path.read_bytes() == first


def test_report_comparison_accepts_only_description_punctuation_changes():
    saved = {"interval_note": "Seed 42; development comparisons", "score": 0.5,
             "comparisons": {"inference_note": "Assumptions apply; results are exploratory"}}
    current = {"interval_note": "Seed 42. Development comparisons", "score": 0.5,
               "comparisons": {"inference_note": "Assumptions apply. Results are exploratory"}}
    before = copy.deepcopy(saved)
    assert reports_match(saved, current)
    assert saved == before
    for changed in ({**current, "score": 0.6}, {**current, "interval_note": "Different analysis"},
                    {**current, "comparisons": {"inference_note": "Assumptions do not apply"}}):
        assert not reports_match(saved, changed)
    assert not reports_match({"title": "A; b"}, {"title": "A. B"})
    assert not reports_match([], current)


def response(payload=None):
    result = Mock(spec=requests.Response)
    result.status_code = 200
    result.headers = CaseInsensitiveDict({
        "X-RateLimit-Limit": "10000", "X-RateLimit-Remaining": "9000",
        "X-RateLimit-Credits-Used": "0", "X-RateLimit-Reset": "3600",
    })
    result.json.return_value = {"rate_limit": {
        "daily_budget_usd": 1, "daily_used_usd": 0, "daily_remaining_usd": 1,
        "prepaid_balance_usd": 0, "prepaid_remaining_usd": 0, "resets_in_seconds": 3600,
    }} if payload is None else payload
    return result


def test_restart_reuses_completed_batches_after_a_failed_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "test-only-key")
    identifiers = [f"W{number}" for number in range(1000, 1101)]
    seeds = [work("W1", "2024-01-01", identifiers)]
    session = Mock(spec=requests.Session)
    session.get.side_effect = [response(), response({
        "results": [work(identifier) for identifier in identifiers[:100]],
        "meta": {"next_cursor": None},
    })] + [requests.exceptions.Timeout()] * 3
    with ExitStack() as stack:
        cache = stack.enter_context(ResponseCache(tmp_path / "cache.sqlite"))
        ledger = stack.enter_context(UsageLedger(tmp_path / "usage.sqlite"))
        client = stack.enter_context(OpenAlexClient(cache, ledger, session=session, sleep=Mock()))
        with pytest.raises(RuntimeError, match="three attempts"):
            expand_pilot(seeds, client)
    assert session.get.call_count == 5
    session.reset_mock()
    session.get.side_effect = [response(), response({
        "results": [work(identifiers[-1])], "meta": {"next_cursor": None},
    })]
    with ExitStack() as stack:
        cache = stack.enter_context(ResponseCache(tmp_path / "cache.sqlite"))
        ledger = stack.enter_context(UsageLedger(tmp_path / "usage.sqlite"))
        client = stack.enter_context(OpenAlexClient(cache, ledger, session=session, sleep=Mock()))
        result = expand_pilot(seeds, client)
    assert session.get.call_count == 2
    assert session.get.call_args.kwargs["params"]["filter"] == "openalex_id:W1100"
    assert result.report["corpus_size"] == 102
    assert result.report["per_seed"][0]["eligible_references"] == 101
    assert result.report["reference_lookup_api_usage"]["attempts"] == 2

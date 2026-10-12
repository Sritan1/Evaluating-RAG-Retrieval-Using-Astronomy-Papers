from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import Mock

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from astro_retrieval.ingest.budget import BudgetError, UsageLedger
from astro_retrieval.ingest.cache import ResponseCache
from astro_retrieval.ingest.openalex import OpenAlexClient, OpenAlexError


def response(payload=None, *, status=200, remaining=9000, used=0, retry_after=None):
    result = Mock(spec=requests.Response)
    result.status_code = status
    result.headers = CaseInsensitiveDict({
        "X-RateLimit-Limit": "10000",
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Credits-Used": str(used),
        "X-RateLimit-Reset": "3600",
    })
    if retry_after is not None:
        result.headers["Retry-After"] = retry_after
    result.json.return_value = {} if payload is None else payload
    return result


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "test-only-key")
    monkeypatch.setattr(requests.Session, "request", Mock(
        side_effect=AssertionError("Live network access is forbidden in these tests")
    ))
    with ExitStack() as stack:
        cache = stack.enter_context(ResponseCache(tmp_path / "cache.sqlite"))
        ledger = stack.enter_context(UsageLedger(tmp_path / "usage.sqlite"))
        session = Mock(spec=requests.Session)
        sleep = Mock()
        client = stack.enter_context(OpenAlexClient(cache, ledger, session=session, sleep=sleep))
        yield client, session, sleep, ledger, cache


def test_success_uses_header_auth_and_warm_cache_needs_no_key(rig, monkeypatch, tmp_path):
    client, session, sleep, ledger, _ = rig
    payload = {"results": [{"id": "W123"}]}
    session.get.side_effect = [response(), response(payload, used=1)]
    assert client.get("/works", {"page": 1}) == payload
    assert session.get.call_count == 2
    sent = session.get.call_args.kwargs
    assert sent["headers"] == {"Authorization": "Bearer test-only-key"}
    assert sent["params"] == {"page": "1"}
    assert sent["timeout"] == (5, 30)
    assert sent["allow_redirects"] is False
    monkeypatch.delenv("OPENALEX_API_KEY")
    session.reset_mock()
    assert client.get("/works", {"page": "1"}) == payload
    with ResponseCache(tmp_path / "cache.sqlite") as reopened:
        with OpenAlexClient(reopened, ledger, session=session) as other:
            assert other.get("/works", {"page": 1}) == payload
    session.get.assert_not_called()
    sleep.assert_not_called()
    assert ledger.totals()["reserved"] == 11
    assert ledger.totals()["reported"] == 1
    for path in tmp_path.glob("*.sqlite"):
        assert b"test-only-key" not in path.read_bytes()


def test_missing_key_stops_before_network_or_reservation(rig, monkeypatch):
    client, session, _, ledger, _ = rig
    monkeypatch.delenv("OPENALEX_API_KEY")
    with pytest.raises(OpenAlexError, match="OPENALEX_API_KEY"):
        client.get("/works")
    session.get.assert_not_called()
    assert ledger.totals()["attempts"] == 0


def test_two_failures_then_success_uses_three_work_attempts(rig):
    client, session, sleep, ledger, _ = rig
    failed = response(status=503)
    session.get.side_effect = [
        response(), requests.exceptions.Timeout("test-only-key"),
        failed, response({"results": []}, used=1),
    ]
    assert client.get("/works") == {"results": []}
    assert session.get.call_count == 4
    assert [call.args[0] for call in sleep.call_args_list] == [1, 2]
    assert ledger.totals()["attempts"] == 4
    assert ledger.totals()["reserved"] == 31
    assert ledger.totals()["reported"] == 1
    failed.close.assert_called_once()


def test_retries_stop_after_three_and_do_not_cache_failure(rig):
    client, session, sleep, ledger, _ = rig
    session.get.side_effect = [response()] + [requests.exceptions.Timeout()] * 3
    with pytest.raises(OpenAlexError, match="three attempts"):
        client.get("/works")
    assert session.get.call_count == 4
    assert sleep.call_count == 2
    session.get.side_effect = [response({"results": []})]
    assert client.get("/works") == {"results": []}
    assert ledger.totals()["attempts"] == 5


@pytest.mark.parametrize("status", [301, 400, 401, 403, 404])
def test_permanent_errors_do_not_retry(rig, status):
    client, session, sleep, _, _ = rig
    failed = response(status=status)
    session.get.side_effect = [response(), failed]
    with pytest.raises(OpenAlexError, match=f"HTTP {status}"):
        client.get("/works")
    assert session.get.call_count == 2
    sleep.assert_not_called()
    failed.close.assert_called_once()


def test_exhausted_account_stops_without_retries(rig):
    client, session, sleep, ledger, _ = rig
    session.get.side_effect = [response(), response(status=429, remaining=0)]
    with pytest.raises(BudgetError, match="exhausted"):
        client.get("/works")
    with pytest.raises(BudgetError, match="blocked"):
        client.get("/works")
    assert session.get.call_count == 2
    sleep.assert_not_called()
    assert ledger.totals()["blocked"] == 1


def test_preflight_stops_before_work_when_account_has_too_little(rig):
    client, session, sleep, _, _ = rig
    session.get.side_effect = [response(remaining=9)]
    with pytest.raises(BudgetError, match="account allowance"):
        client.get("/works")
    assert session.get.call_count == 1
    sleep.assert_not_called()


def test_local_budget_stops_retries_before_an_extra_network_call(rig):
    client, session, _, ledger, _ = rig
    ledger.daily_limit = 11
    session.get.side_effect = [response(), requests.exceptions.Timeout()]
    with pytest.raises(BudgetError, match="Local daily"):
        client.get("/works")
    assert session.get.call_count == 2
    assert ledger.totals()["reserved"] == 11


def test_preflight_timeout_is_counted_and_retried(rig):
    client, session, sleep, ledger, _ = rig
    session.get.side_effect = [requests.exceptions.Timeout(), response(),
                               response({"results": []})]
    assert client.get("/works") == {"results": []}
    assert session.get.call_count == 3
    sleep.assert_called_once_with(1)
    assert ledger.totals()["reserved"] == 12


def test_missing_usage_headers_stop_before_work_and_close_response(rig):
    client, session, _, _, _ = rig
    preflight = response()
    preflight.headers.clear()
    session.get.return_value = preflight
    with pytest.raises(BudgetError, match="usage headers"):
        client.get("/works")
    session.get.assert_called_once()
    preflight.close.assert_called_once()


def test_singleton_reserves_one_credit_and_requires_a_work_object(rig):
    client, session, _, ledger, _ = rig
    session.get.side_effect = [response(), response({"id": "https://openalex.org/W123"})]
    assert client.get("/works/W123") == {"id": "https://openalex.org/W123"}
    assert ledger.totals()["reserved"] == 2


def test_invalid_json_does_not_enter_cache(rig):
    client, session, _, _, _ = rig
    invalid = response()
    invalid.json.side_effect = requests.exceptions.JSONDecodeError("bad", "", 0)
    session.get.side_effect = [response(), invalid, response({"results": []})]
    with pytest.raises(OpenAlexError, match="Invalid"):
        client.get("/works")
    invalid.close.assert_called_once()
    assert client.get("/works") == {"results": []}


def test_rate_limit_with_budget_obeys_retry_after(rig):
    client, session, sleep, _, _ = rig
    session.get.side_effect = [response(), response(status=429, retry_after="7"),
                               response({"results": []})]
    assert client.get("/works") == {"results": []}
    sleep.assert_called_once_with(7)


@pytest.mark.parametrize("retry_after", ["61", "invalid"])
def test_excessive_or_invalid_retry_wait_stops(rig, retry_after):
    client, session, sleep, _, _ = rig
    session.get.side_effect = [response(), response(status=503, retry_after=retry_after)]
    with pytest.raises(OpenAlexError):
        client.get("/works")
    assert session.get.call_count == 2
    sleep.assert_not_called()


def test_http_date_retry_delay():
    retry_after = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))
    assert 27 <= OpenAlexClient._retry_delay(0, retry_after) <= 30


@pytest.mark.parametrize("payload", [[], {"error": "failed"}, {"results": [1]},
                                    {"results": [], "score": float("nan")}])
def test_invalid_success_response_is_not_cached(rig, payload):
    client, session, sleep, _, _ = rig
    session.get.side_effect = [response(), response(payload), response({"results": []})]
    with pytest.raises(OpenAlexError, match="Invalid"):
        client.get("/works")
    assert client.get("/works") == {"results": []}
    assert session.get.call_count == 3
    sleep.assert_not_called()


def test_pagination_follows_cursors_and_replays_without_network(rig, monkeypatch):
    client, session, _, ledger, _ = rig
    first = {"results": [{"id": "W1"}], "meta": {"next_cursor": "next"}}
    final = {"results": [{"id": "W2"}], "meta": {"next_cursor": None}}
    session.get.side_effect = [response(), response(first), response(final)]
    query = {"filter": "publication_year:2025"}
    assert list(client.iter_works(query)) == [{"id": "W1"}, {"id": "W2"}]
    assert query == {"filter": "publication_year:2025"}
    calls = session.get.call_args_list
    assert calls[1].kwargs["params"]["cursor"] == "*"
    assert calls[2].kwargs["params"]["cursor"] == "next"
    assert calls[1].kwargs["params"]["per_page"] == "100"
    monkeypatch.delenv("OPENALEX_API_KEY")
    session.reset_mock()
    assert list(client.iter_works(query)) == [{"id": "W1"}, {"id": "W2"}]
    session.get.assert_not_called()
    assert ledger.totals()["attempts"] == 3


def test_interrupted_pagination_reuses_completed_page(rig):
    client, session, _, _, _ = rig
    first = {"results": [{"id": "W1"}], "meta": {"next_cursor": "next"}}
    final = {"results": [], "meta": {"next_cursor": None}}
    session.get.side_effect = [response(), response(first)] + [requests.exceptions.Timeout()] * 3
    with pytest.raises(OpenAlexError):
        list(client.iter_works({}))
    session.reset_mock()
    session.get.side_effect = [response(final)]
    assert list(client.iter_works({})) == [{"id": "W1"}]
    session.get.assert_called_once()
    assert session.get.call_args.kwargs["params"]["cursor"] == "next"


def test_repeated_cursor_raises_without_fetching_forever(rig):
    client, session, _, _, _ = rig
    session.get.side_effect = [response(), response({
        "results": [{"id": "W1"}], "meta": {"next_cursor": "*"},
    })]
    with pytest.raises(OpenAlexError, match="repeated"):
        list(client.iter_works({}))
    assert session.get.call_count == 2


def test_page_limit_raises_instead_of_silently_truncating(rig):
    client, session, _, _, _ = rig
    session.get.side_effect = [response(), response({
        "results": [], "meta": {"next_cursor": "next"},
    })]
    with pytest.raises(OpenAlexError, match="page limit"):
        list(client.iter_works({}, max_pages=1))
    assert session.get.call_count == 2


def test_missing_cursor_does_not_cache_a_broken_page(rig):
    client, session, _, _, _ = rig
    session.get.side_effect = [response(), response({"results": []}), response({
        "results": [], "meta": {"next_cursor": None},
    })]
    with pytest.raises(OpenAlexError, match="Invalid"):
        list(client.iter_works({}))
    assert list(client.iter_works({})) == []
    assert session.get.call_count == 3


def test_unsupported_cost_changing_parameters_stop_before_network(rig):
    client, session, _, _, _ = rig
    with pytest.raises(ValueError, match="Unsupported"):
        client.get("/works", {"rerank": "true"})
    session.get.assert_not_called()


def test_transport_errors_do_not_expose_credentials(rig):
    client, session, _, _, _ = rig
    session.get.side_effect = requests.exceptions.SSLError("test-only-key")
    with pytest.raises(OpenAlexError) as error:
        client.get("/works")
    assert "test-only-key" not in str(error.value)
    assert error.value.__suppress_context__


def test_midnight_requires_new_preflight(rig):
    client, session, _, ledger, _ = rig
    clock = [datetime(2026, 10, 12, 23, 59, tzinfo=timezone.utc)]
    ledger._now = lambda: clock[0]
    session.get.side_effect = [response(), response({"results": []}),
                               response(), response({"results": []})]
    client.get("/works", {"page": 1})
    clock[0] += timedelta(minutes=2)
    client.get("/works", {"page": 2})
    urls = [call.args[0] for call in session.get.call_args_list]
    assert urls == ["https://api.openalex.org/rate-limit", "https://api.openalex.org/works"] * 2
    assert ledger.totals()["reserved"] == 11

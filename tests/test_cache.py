import json
import sqlite3
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from astro_retrieval.ingest.cache import ResponseCache, request_key


def test_equivalent_parameters_have_the_same_key():
    first = {"filter": "publication_year:2025", "page": 1, "search": None}
    reordered = {"page": "1", "filter": "publication_year:2025"}

    assert request_key("/works", first) == request_key("/works", reordered)
    assert first == {"filter": "publication_year:2025", "page": 1, "search": None}


@pytest.mark.parametrize("name", ["api_key", "API_KEY", "authorization", "token"])
def test_credentials_are_rejected_before_fetch_or_storage(tmp_path, name):
    database = tmp_path / "cache.sqlite"
    fetch = Mock()

    with ResponseCache(database) as cache:
        with pytest.raises(ValueError, match="Credentials") as error:
            cache.get_or_fetch("/works", {name: "test-only-secret"}, fetch)

    fetch.assert_not_called()
    assert "test-only-secret" not in str(error.value)
    assert b"test-only-secret" not in database.read_bytes()


@pytest.mark.parametrize(
    "endpoint",
    ["https://api.openalex.org/works", "/works?api_key=test-only-secret", "/works#x"],
)
def test_endpoint_cannot_embed_a_query_or_another_url(endpoint):
    with pytest.raises(ValueError):
        request_key(endpoint, {})


@pytest.mark.parametrize("value", [True, 1.5, ["one", "two"], {"page": 1}])
def test_unsupported_parameter_types_are_rejected(value):
    with pytest.raises(TypeError):
        request_key("/works", {"page": value})


def test_cache_persists_and_warm_requests_never_fetch(tmp_path):
    database = tmp_path / "nested" / "cache.sqlite"
    payload = {"meta": {"count": 1}, "results": [{"id": "W123", "title": "β planet"}]}
    fetch = Mock(return_value=payload)
    offline_fetch = Mock(side_effect=AssertionError("Unexpected fetch"))
    before = datetime.now(timezone.utc)

    with ResponseCache(database) as cache:
        result = cache.get_or_fetch("/works", {"page": 1, "search": "planet"}, fetch)
        assert result == payload
        result["results"].clear()
        warm = cache.get_or_fetch(
            "/works", {"search": "planet", "page": "1"}, offline_fetch
        )
        assert warm == payload

    with ResponseCache(database) as reopened:
        assert reopened.get_or_fetch(
            "/works", {"search": "planet", "page": 1}, offline_fetch
        ) == payload

    fetch.assert_called_once_with(
        "https://api.openalex.org/works", {"page": "1", "search": "planet"}
    )
    offline_fetch.assert_not_called()
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT response_json, fetched_at FROM api_cache"
        ).fetchall()
    finally:
        connection.close()
    assert len(rows) == 1
    assert json.loads(rows[0][0]) == payload
    assert before <= datetime.fromisoformat(rows[0][1]) <= datetime.now(timezone.utc)


def test_different_pages_filters_and_endpoints_fetch_separately(tmp_path):
    fetch = Mock(side_effect=[{"marker": number} for number in range(4)])
    requests = [
        ("/works", {"filter": "publication_year:2024", "page": 1}),
        ("/works", {"filter": "publication_year:2024", "page": 2}),
        ("/works", {"filter": "publication_year:2025", "page": 1}),
        ("/works/W123", {}),
    ]

    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        for number, (endpoint, params) in enumerate(requests):
            assert cache.get_or_fetch(endpoint, params, fetch) == {"marker": number}
        for number, (endpoint, params) in enumerate(requests):
            assert cache.get_or_fetch(endpoint, params, fetch) == {"marker": number}

    assert fetch.call_count == 4


def test_failed_fetch_is_not_cached(tmp_path):
    fetch = Mock(side_effect=[TimeoutError("Test timeout"), {"results": []}])

    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        with pytest.raises(TimeoutError):
            cache.get_or_fetch("/works", {}, fetch)
        assert cache.get_or_fetch("/works", {}, fetch) == {"results": []}
        assert cache.get_or_fetch("/works", {}, fetch) == {"results": []}

    assert fetch.call_count == 2


@pytest.mark.parametrize("response", [[], {"score": float("nan")}, {"bad": object()}])
def test_invalid_response_is_not_cached(tmp_path, response):
    fetch = Mock(side_effect=[response, {}])

    with ResponseCache(tmp_path / "cache.sqlite") as cache:
        with pytest.raises((TypeError, ValueError)):
            cache.get_or_fetch("/works", {}, fetch)
        assert cache.get_or_fetch("/works", {}, fetch) == {}
        assert cache.get_or_fetch("/works", {}, fetch) == {}

    assert fetch.call_count == 2


def test_context_manager_closes_connection_on_error(tmp_path):
    with pytest.raises(RuntimeError, match="Test interruption"):
        with ResponseCache(tmp_path / "cache.sqlite") as cache:
            raise RuntimeError("Test interruption")

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        cache.get_or_fetch("/works", {}, Mock())

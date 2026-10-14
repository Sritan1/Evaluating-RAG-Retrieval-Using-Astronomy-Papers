import json
import os
import time
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from astro_retrieval.ingest.budget import BudgetError, UsageLedger
from astro_retrieval.ingest.cache import ResponseCache, request_key


class OpenAlexError(RuntimeError):
    pass


class OpenAlexClient:
    """Fetch public work metadata with a cache and a persistent usage allowance."""

    def __init__(
        self,
        cache: ResponseCache,
        ledger: UsageLedger,
        *,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cache = cache
        self.ledger = ledger
        self._session = session if session is not None else requests.Session()
        self._owns_session = session is None
        if self._owns_session:
            self._session.trust_env = False
        self._sleep = sleep
        self._verified_day = None

    def get(
        self, endpoint: str, params: Mapping[str, str | int | None] | None = None,
    ) -> dict[str, Any]:
        params = params or {}
        normalized = json.loads(request_key(endpoint, params))["params"]
        allowed = {"filter", "search", "select", "sort", "per_page", "cursor",
                   "page", "sample", "seed", "corpus", "search.title_and_abstract"}
        if set(normalized) - allowed or (endpoint != "/works" and normalized):
            raise ValueError("Unsupported work request parameters")
        return self.cache.get_or_fetch(endpoint, normalized, self._fetch)

    def _fetch(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        key = os.environ.get("OPENALEX_API_KEY", "").strip()
        if not key or any(ord(char) < 33 or ord(char) > 126 for char in key):
            raise OpenAlexError("Set OPENALEX_API_KEY to a valid key before fetching")
        if self._verified_day != self.ledger.day():
            self._request("https://api.openalex.org/rate-limit", {}, key, 1, preflight=True)
            self._verified_day = self.ledger.day()
        credits = 10 if url == "https://api.openalex.org/works" else 1
        return self._request(url, params, key, credits)

    def _request(
        self, url: str, params: dict[str, str], key: str, credits: int,
        *, preflight: bool = False,
    ) -> dict[str, Any]:
        for attempt in range(3):
            day = self.ledger.reserve(credits, preflight=preflight)
            response = None
            retry_after = None
            try:
                response = self._session.get(
                    url, params=params, headers={"Authorization": f"Bearer {key}"},
                    timeout=(5, 30), allow_redirects=False,
                )
            except requests.exceptions.SSLError:
                raise OpenAlexError("TLS verification failed") from None
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                pass
            except requests.exceptions.RequestException:
                raise OpenAlexError("Request failed before a usable response") from None
            if response is not None:
                try:
                    self.ledger.record(day, response.headers, credits)
                    status = response.status_code
                    if status == 200:
                        try:
                            payload = response.json()
                            if not isinstance(payload, dict):
                                raise ValueError
                            json.dumps(payload, allow_nan=False)
                            if not preflight:
                                self._validate_payload(url, params, payload)
                        except (ValueError, TypeError):
                            raise OpenAlexError("Invalid OpenAlex response") from None
                        return payload
                    if status == 429:
                        remaining = self.ledger.totals(day)["remaining"]
                        if remaining < credits:
                            self.ledger.block(day)
                            raise BudgetError("OpenAlex daily allowance exhausted")
                    elif status not in {500, 502, 503, 504}:
                        raise OpenAlexError(f"OpenAlex returned HTTP {status}")
                    retry_after = response.headers.get("Retry-After")
                finally:
                    response.close()
            if attempt == 2:
                raise OpenAlexError("OpenAlex request failed after three attempts")
            self._sleep(self._retry_delay(attempt, retry_after))
        raise OpenAlexError("OpenAlex request did not complete")

    @staticmethod
    def _retry_delay(attempt: int, retry_after: str | None) -> float:
        delay = float(2 ** attempt)
        if retry_after is not None:
            try:
                if retry_after.isascii() and retry_after.isdecimal():
                    requested = float(retry_after)
                else:
                    reset = parsedate_to_datetime(retry_after)
                    if reset.utcoffset() is None:
                        raise ValueError
                    requested = (reset - datetime.now(timezone.utc)).total_seconds()
                delay = max(delay, requested)
            except (ValueError, TypeError, OverflowError):
                raise OpenAlexError("Invalid Retry-After header") from None
        if delay > 60:
            raise OpenAlexError("Server requested a wait longer than 60 seconds. Resume later")
        return delay

    @staticmethod
    def _validate_payload(url: str, params: dict[str, str], payload: dict) -> None:
        if url != "https://api.openalex.org/works":
            if not isinstance(payload.get("id"), str):
                raise ValueError
            return
        if not isinstance(payload.get("results"), list):
            raise ValueError
        if any(not isinstance(work, dict) for work in payload["results"]):
            raise ValueError
        if "cursor" in params:
            meta = payload.get("meta")
            if not isinstance(meta, dict) or "next_cursor" not in meta:
                raise ValueError
            cursor = meta["next_cursor"]
            if cursor is not None and (not isinstance(cursor, str) or not cursor):
                raise ValueError

    def iter_works(
        self, params: Mapping[str, str | int | None], *, max_pages: int = 1000,
    ) -> Iterator[dict[str, Any]]:
        if type(max_pages) is not int or max_pages < 1:
            raise ValueError("max_pages must be a positive integer")
        if any(name in params for name in ("page", "cursor", "sample", "seed")):
            raise ValueError("Cursor iteration cannot use page, cursor, sample, or seed")
        query = dict(params)
        per_page = query.get("per_page", 100)
        if type(per_page) is not int or not 1 <= per_page <= 100:
            raise ValueError("per_page must be an integer from 1 to 100")
        query["per_page"] = per_page
        cursor = "*"
        seen = {cursor}
        for _ in range(max_pages):
            query["cursor"] = cursor
            payload = self.get("/works", query)
            try:
                self._validate_payload("https://api.openalex.org/works", query, payload)
            except (ValueError, TypeError):
                raise OpenAlexError("Invalid cached pagination response") from None
            next_cursor = payload["meta"]["next_cursor"]
            if next_cursor in seen:
                raise OpenAlexError("OpenAlex repeated a pagination cursor")
            yield from payload["results"]
            if next_cursor is None:
                return
            seen.add(next_cursor)
            cursor = next_cursor
        raise OpenAlexError("Pagination page limit reached before completion")

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "OpenAlexClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

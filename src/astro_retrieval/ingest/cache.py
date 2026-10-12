import json
import re
import sqlite3
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def request_key(endpoint: str, params: Mapping[str, str | int | None]) -> str:
    """Identify a public OpenAlex works request without authentication."""
    if not re.fullmatch(r"/works(?:/W[0-9]+)?", endpoint):
        raise ValueError("Use /works or /works/W followed by a numeric ID")

    normalized = {}
    for name, value in params.items():
        if not isinstance(name, str):
            raise TypeError("Parameter names must be strings")
        if name.lower() in {"api_key", "authorization", "access_token", "token"}:
            raise ValueError("Credentials must be supplied outside cache parameters")
        if value is None:
            continue
        if type(value) not in (str, int):
            raise TypeError("Parameter values must be strings, integers, or None")
        normalized[name] = str(value)

    return json.dumps(
        {
            "method": "GET",
            "url": f"https://api.openalex.org{endpoint}",
            "params": normalized,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


class ResponseCache:
    """Store successful JSON objects for one sequential collection process."""

    def __init__(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS api_cache (
                    request_key TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    fetched_at TEXT NOT NULL
                )
                """
            )

    def get_or_fetch(
        self,
        endpoint: str,
        params: Mapping[str, str | int | None],
        fetch: Callable[[str, dict[str, str]], dict[str, Any]],
    ) -> dict[str, Any]:
        """The fetcher must raise on HTTP errors and return a JSON object."""
        key = request_key(endpoint, params)
        row = self._connection.execute(
            "SELECT response_json FROM api_cache WHERE request_key = ?", (key,)
        ).fetchone()
        if row is not None:
            return json.loads(row[0])

        request = json.loads(key)
        response = fetch(request["url"], request["params"])
        if not isinstance(response, dict):
            raise TypeError("The fetcher must return a JSON object")
        serialized = json.dumps(response, ensure_ascii=False, allow_nan=False)
        fetched_at = datetime.now(timezone.utc).isoformat()
        with self._connection:
            self._connection.execute(
                "INSERT INTO api_cache (request_key, response_json, fetched_at) "
                "VALUES (?, ?, ?)",
                (key, serialized, fetched_at),
            )
        return json.loads(serialized)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "ResponseCache":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

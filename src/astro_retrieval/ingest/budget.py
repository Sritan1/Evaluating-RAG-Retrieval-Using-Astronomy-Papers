import sqlite3
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path


class BudgetError(RuntimeError):
    pass


class UsageLedger:
    """Reserve credits before each attempt in one sequential collection process."""

    def __init__(
        self,
        path: str | Path,
        daily_limit: int = 1000,
        now: Callable[[], datetime] | None = None,
    ):
        if type(daily_limit) is not int or not 1 <= daily_limit <= 10000:
            raise ValueError("Daily limit must be an integer from 1 to 10000 credits")
        self.daily_limit = daily_limit
        self._now = now or (lambda: datetime.now(timezone.utc))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS api_usage ("
                "day TEXT PRIMARY KEY, reserved INTEGER NOT NULL DEFAULT 0, "
                "attempts INTEGER NOT NULL DEFAULT 0, "
                "reported INTEGER NOT NULL DEFAULT 0, remaining INTEGER, "
                "blocked INTEGER NOT NULL DEFAULT 0)"
            )

    def day(self) -> str:
        current = self._now()
        if current.utcoffset() is None:
            raise ValueError("The clock must return a timezone-aware datetime")
        return current.astimezone(timezone.utc).date().isoformat()

    def totals(self, day: str | None = None) -> dict[str, int | None]:
        row = self._connection.execute(
            "SELECT reserved, attempts, reported, remaining, blocked "
            "FROM api_usage WHERE day = ?",
            (day or self.day(),),
        ).fetchone()
        return dict(zip(
            ("reserved", "attempts", "reported", "remaining", "blocked"),
            row or (0, 0, 0, None, 0),
        ))

    def reserve(self, credits: int, *, preflight: bool = False) -> str:
        if type(credits) is not int or credits < 1:
            raise ValueError("Reserve a positive integer number of credits")
        day = self.day()
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO api_usage (day) VALUES (?)", (day,)
            )
            totals = self.totals(day)
            if totals["blocked"]:
                raise BudgetError("Collection is blocked for this UTC day")
            if totals["reserved"] + credits > self.daily_limit:
                raise BudgetError("Local daily credit allowance exhausted")
            remaining = totals["remaining"]
            if remaining is None and not preflight:
                raise BudgetError("Verify the account allowance before collection")
            if remaining is not None and remaining < credits:
                raise BudgetError("Available account allowance is too low")
            self._connection.execute(
                "UPDATE api_usage SET reserved = reserved + ?, "
                "attempts = attempts + 1, remaining = remaining - ? WHERE day = ?",
                (credits, credits, day),
            )
        return day

    def block(self, day: str) -> None:
        with self._connection:
            self._connection.execute(
                "UPDATE api_usage SET blocked = 1 WHERE day = ?", (day,)
            )

    def record(self, day: str, headers: Mapping[str, str], reserved: int) -> None:
        if day != self.day():
            raise BudgetError("UTC day changed during the request. Restart collection")
        normalized = {name.lower(): value for name, value in headers.items()}
        try:
            values = []
            for name in ("limit", "remaining", "credits-used", "reset"):
                raw = normalized[f"x-ratelimit-{name}"]
                if not isinstance(raw, str) or not raw.isascii() or not raw.isdecimal():
                    raise ValueError
                values.append(int(raw))
            limit, remaining, used, reset = values
            if not (0 < limit <= 10000 and 0 <= remaining <= limit
                    and 0 <= used <= reserved and 0 <= reset <= 86400):
                raise ValueError
        except (KeyError, ValueError):
            self.block(day)
            raise BudgetError("Missing or unexpected usage headers. Review before resuming") from None
        previous = self.totals(day)["remaining"]
        available = remaining if previous is None else min(previous, remaining)
        with self._connection:
            self._connection.execute(
                "UPDATE api_usage SET reported = reported + ?, remaining = ? "
                "WHERE day = ?", (used, available, day),
            )

    def usage_totals(self) -> dict[str, int]:
        row = self._connection.execute(
            "SELECT COALESCE(SUM(attempts), 0), COALESCE(SUM(reserved), 0), "
            "COALESCE(SUM(reported), 0) FROM api_usage"
        ).fetchone()
        return dict(zip(("attempts", "reserved_credits", "reported_credits"), row))

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "UsageLedger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

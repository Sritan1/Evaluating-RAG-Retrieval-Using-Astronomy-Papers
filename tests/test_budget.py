from datetime import datetime, timedelta, timezone

import pytest

from astro_retrieval.ingest.budget import BudgetError, UsageLedger


def headers(remaining=100, used=1):
    return {
        "X-RateLimit-Limit": "10000",
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Credits-Used": str(used),
        "X-RateLimit-Reset": "3600",
    }


def test_three_unit_allowance_survives_reopening(tmp_path):
    path = tmp_path / "usage.sqlite"
    with UsageLedger(path, daily_limit=3) as ledger:
        for index in range(3):
            day = ledger.reserve(1, preflight=index == 0)
            ledger.record(day, headers(100 - index), 1)
        with pytest.raises(BudgetError, match="Local daily"):
            ledger.reserve(1)
        assert ledger.totals() == {
            "reserved": 3, "attempts": 3, "reported": 3,
            "remaining": 98, "blocked": 0,
        }
    with UsageLedger(path, daily_limit=3) as reopened:
        with pytest.raises(BudgetError, match="Local daily"):
            reopened.reserve(1, preflight=True)
        assert reopened.totals()["attempts"] == 3


def test_unreported_attempt_stays_reserved_after_restart(tmp_path):
    path = tmp_path / "usage.sqlite"
    with UsageLedger(path, daily_limit=1) as ledger:
        ledger.reserve(1, preflight=True)
    with UsageLedger(path, daily_limit=1) as ledger:
        with pytest.raises(BudgetError):
            ledger.reserve(1, preflight=True)
        assert ledger.totals()["reserved"] == 1
        assert ledger.totals()["reported"] == 0


def test_account_remaining_can_only_reduce_local_estimate(tmp_path):
    with UsageLedger(tmp_path / "usage.sqlite") as ledger:
        day = ledger.reserve(1, preflight=True)
        ledger.record(day, headers(remaining=3, used=0), 1)
        day = ledger.reserve(2)
        ledger.record(day, headers(remaining=100, used=0), 2)
        assert ledger.totals()["remaining"] == 1
        with pytest.raises(BudgetError, match="account allowance"):
            ledger.reserve(2)


def test_new_utc_day_requires_account_verification(tmp_path):
    clock = [datetime(2026, 10, 12, 18, 59, tzinfo=timezone(timedelta(hours=-5)))]
    with UsageLedger(tmp_path / "usage.sqlite", daily_limit=1, now=lambda: clock[0]) as ledger:
        day = ledger.reserve(1, preflight=True)
        ledger.record(day, headers(), 1)
        assert day == "2026-10-12"
        clock[0] += timedelta(minutes=2)
        assert ledger.day() == "2026-10-13"
        with pytest.raises(BudgetError, match="Verify"):
            ledger.reserve(1)
        assert ledger.totals()["attempts"] == 0
        ledger.reserve(1, preflight=True)
        assert ledger.totals(day)["attempts"] == 1


def test_response_crossing_utc_midnight_does_not_set_new_day_allowance(tmp_path):
    clock = [datetime(2026, 10, 12, 23, 59, 59, tzinfo=timezone.utc)]
    with UsageLedger(tmp_path / "usage.sqlite", now=lambda: clock[0]) as ledger:
        day = ledger.reserve(1, preflight=True)
        clock[0] += timedelta(seconds=2)
        with pytest.raises(BudgetError, match="UTC day changed"):
            ledger.record(day, headers(), 1)
        assert ledger.totals()["remaining"] is None


@pytest.mark.parametrize("override", [
    {"X-RateLimit-Limit": "20000"},
    {"X-RateLimit-Remaining": "10001"},
    {"X-RateLimit-Remaining": "unknown"},
    {"X-RateLimit-Remaining": "-1"},
    {"X-RateLimit-Credits-Used": "2"},
    {"X-RateLimit-Reset": "86401"},
])
def test_unexpected_headers_block_collection_persistently(tmp_path, override):
    path = tmp_path / "usage.sqlite"
    with UsageLedger(path) as ledger:
        day = ledger.reserve(1, preflight=True)
        with pytest.raises(BudgetError, match="usage headers"):
            ledger.record(day, headers() | override, 1)
    with UsageLedger(path) as ledger:
        with pytest.raises(BudgetError, match="blocked"):
            ledger.reserve(1, preflight=True)


def test_missing_headers_block_collection(tmp_path):
    with UsageLedger(tmp_path / "usage.sqlite") as ledger:
        day = ledger.reserve(1, preflight=True)
        with pytest.raises(BudgetError):
            ledger.record(day, {}, 1)
        assert ledger.totals()["blocked"] == 1


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, 10001])
def test_invalid_daily_limit_is_rejected(tmp_path, limit):
    with pytest.raises(ValueError):
        UsageLedger(tmp_path / "usage.sqlite", daily_limit=limit)

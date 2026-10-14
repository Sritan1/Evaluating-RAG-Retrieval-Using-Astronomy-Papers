import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from astro_retrieval.ingest.budget import BudgetError, UsageLedger
from astro_retrieval.ingest.cache import ResponseCache
from astro_retrieval.ingest.collection import candidate_parameters, run_pilot
from astro_retrieval.ingest.openalex import OpenAlexClient, OpenAlexError


class OfflineClient(OpenAlexClient):
    def _fetch(self, url, params):
        raise OpenAlexError("Offline replay requires a cached response")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Preview or run the exoplanet pilot")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--collect", action="store_true", help="Allow budgeted API requests")
    mode.add_argument("--replay", action="store_true", help="Use frozen input and cached responses only")
    args = parser.parse_args(argv)
    if not args.collect and not args.replay:
        print(json.dumps({"parameters": candidate_parameters(), "sample_size": 200,
                          "seed": 42, "abstract_coverage_threshold": 0.70}, indent=2))
        return 0
    directory = Path("data/pilot")
    client_type = OfflineClient if args.replay else OpenAlexClient
    try:
        if args.collect:
            load_dotenv(Path.cwd() / ".env", override=False, interpolate=False, encoding="utf-8-sig")
        with ResponseCache(Path(".cache/openalex.sqlite")) as cache, UsageLedger(
            Path(".cache/usage.sqlite")
        ) as ledger, client_type(cache, ledger) as client:
            report = run_pilot(client, directory, replay=args.replay)
        print(json.dumps({key: report[key] for key in (
            "abstract_coverage_passed", "feasibility_status", "current_run_api_usage",
        )}, indent=2))
        return 0
    except (OpenAlexError, BudgetError, ValueError, OSError, KeyError, TypeError):
        print("Pilot could not complete. Check local inputs, configuration, credentials, and allowance.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

from astro_retrieval.ingest.budget import BudgetError, UsageLedger
from astro_retrieval.ingest.cache import ResponseCache
from astro_retrieval.ingest.collection import candidate_parameters, run_collection
from astro_retrieval.ingest.openalex import OpenAlexClient, OpenAlexError


class OfflineClient(OpenAlexClient):
    def _fetch(self, url, params):
        raise OpenAlexError("Offline replay requires a cached response")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Collect exoplanet papers or reproduce saved evaluation results")
    parser.add_argument("--full", action="store_true", help="Use 600 candidates from the frozen pilot pool")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--collect", action="store_true", help="Allow budgeted API requests")
    mode.add_argument("--replay", action="store_true", help="Use frozen input and cached responses only")
    mode.add_argument("--reproduce", action="store_true", help="Verify saved retrieval and audit reports offline")
    parser.add_argument("--output", type=Path, help="With --reproduce, rebuild results into a new directory")
    args = parser.parse_args(argv)
    if args.output is not None and not args.reproduce:
        parser.error("--output requires --reproduce")
    if args.reproduce:
        if args.full:
            parser.error("--full cannot be combined with --reproduce")
        from astro_retrieval.audit import run_audit_analysis
        from astro_retrieval.evaluation import rebuild_results, run_evaluation

        evaluation_args = (
                "data/full/corpus.sqlite", "data/full/evaluation.json", "data/full/bm25-index.json",
                "data/full/general-vectors/documents", "data/full/indus-vectors/documents",
                "docs/hypotheses.md", "data/test")
        audit_args = (
                "data/full/corpus.sqlite", "data/full/evaluation.json",
                {name: f"data/test/{name}.json" for name in ("bm25", "hybrid", "reranker")},
                "docs/audit-rubric.md", "data/audit/model-practice", "data/audit/main-protocol.json",
                "data/audit/model-main")
        audit_hash = "75b43d8bc726ec0275c2c42ee60291215658cee6fc8d8e372a0e70a71e05744b"
        try:
            if args.output is not None:
                retrieval, audit = rebuild_results(evaluation_args, audit_args, args.output,
                                                    protocol_sha256=audit_hash)
            else:
                retrieval = run_evaluation(*evaluation_args, replay_only=True)
                audit = run_audit_analysis(*audit_args, replay_only=True, protocol_sha256=audit_hash)
        except (ValueError, OSError, KeyError, TypeError, RuntimeError, sqlite3.Error):
            print("Reproduction failed. Check that saved inputs and reports are complete and unchanged.")
            return 1
        print(json.dumps({"status": "reproduced", "retrieval_systems": len(retrieval["systems"]),
                          "audit_pairs": audit["analysis"]["pairs"],
                          "audit_judgments": audit["analysis"]["judgments"]}, indent=2))
        return 0
    if not args.collect and not args.replay:
        print(json.dumps({"parameters": candidate_parameters(), "sample_size": 600 if args.full else 200,
                          "seed": 42, "abstract_coverage_threshold": 0.70}, indent=2))
        return 0
    directory = Path("data/full" if args.full else "data/pilot")
    client_type = OfflineClient if args.replay else OpenAlexClient
    try:
        if args.collect:
            load_dotenv(Path.cwd() / ".env", override=False, interpolate=False, encoding="utf-8-sig")
        with ResponseCache(Path(".cache/openalex.sqlite")) as cache, UsageLedger(
            Path(".cache/usage.sqlite")
        ) as ledger, client_type(cache, ledger) as client:
            report = run_collection(client, directory, replay=args.replay, full=args.full)
        print(json.dumps({key: report[key] for key in (
            "abstract_coverage_passed", "feasibility_status", "current_run_api_usage",
        )}, indent=2))
        return 0
    except BudgetError:
        print("Collection stopped by the usage guard. Completed responses remain cached. "
              "Review the allowance or account status before resuming.")
        return 1
    except (OpenAlexError, ValueError, OSError, KeyError, TypeError, sqlite3.Error):
        print("Collection could not complete. Check local inputs, configuration, credentials, and allowance.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

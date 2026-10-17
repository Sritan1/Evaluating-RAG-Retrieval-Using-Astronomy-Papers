import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from astro_retrieval.ingest.expansion import expand_pilot, write_json
from astro_retrieval.ingest.pilot import select_pilot
from astro_retrieval.ingest.records import index_works


def candidate_parameters() -> dict:
    return {
        "filter": "primary_topic.id:T10039,language:en,type:article,"
                  "from_publication_date:2021-01-01,to_publication_date:2025-12-31",
        "corpus": "core",
        "search.title_and_abstract": 'exoplanet OR "extrasolar planet"',
        "per_page": 100,
    }


def pool_digest(works: list) -> str:
    encoded = json.dumps(works, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def collect_candidates(client, path: Path) -> dict:
    """Freeze a complete pool before sampling it."""
    if path.exists():
        return load_candidates(path)
    before = client.ledger.usage_totals()
    records = index_works(client.iter_works(candidate_parameters()))
    works = [records[key] for key in sorted(records)]
    after = client.ledger.usage_totals()
    snapshot = {
        "schema_version": 1,
        "parameters": candidate_parameters(),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "works": works,
        "sha256": pool_digest(works),
        "collection_api_usage": {key: after[key] - value for key, value in before.items()},
    }
    write_json(snapshot, path)
    return snapshot


def load_candidates(path: Path) -> dict:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if (snapshot["schema_version"] != 1
            or snapshot["parameters"] != candidate_parameters()
            or pool_digest(snapshot["works"]) != snapshot["sha256"]):
        raise ValueError("Candidate snapshot does not match this pilot configuration")
    index_works(snapshot["works"])
    return snapshot


def run_collection(client, directory: Path, *, replay: bool = False, full: bool = False) -> dict:
    before = client.ledger.usage_totals()
    path = Path("data/pilot/candidates.json") if full else directory / "candidates.json"
    snapshot = load_candidates(path) if replay or full else collect_candidates(client, path)
    size = 600 if full else 200
    if full:
        sample = select_pilot(snapshot["works"], size=size)
        selection = {
            "schema_version": 1,
            "candidate_pool_sha256": snapshot["sha256"],
            "candidate_parameters": snapshot["parameters"],
            "requested_candidates": size,
            "sampling_seed": sample.seed,
            "candidate_ids": list(sample.candidate_ids),
        }
        selection_path = directory / "selection.json"
        if selection_path.exists():
            saved = json.loads(selection_path.read_text(encoding="utf-8"))
            if saved != selection:
                raise ValueError("Full-run selection does not match the frozen input")
        elif replay:
            raise ValueError("Offline full replay requires a saved selection")
        else:
            write_json(selection, selection_path)
    result = expand_pilot(snapshot["works"], client, size=size)
    count = len(result.sample.candidate_ids)
    usable = len(result.sample.abstracts)
    after = client.ledger.usage_totals()
    report = {
        **result.report,
        "candidate_pool_sha256": snapshot["sha256"],
        "candidate_parameters": snapshot["parameters"],
        "collection_api_usage": snapshot["collection_api_usage"],
        "current_run_api_usage": {key: after[key] - value for key, value in before.items()},
        "abstract_coverage_threshold": 0.70,
        "abstract_coverage_passed": usable * 100 >= count * 70 if count else None,
        "runtime_status": "not_measured",
        "feasibility_status": "pending",
    }
    bundle = {"report": report, "papers": result.papers, "references": result.references}
    write_json(bundle, directory / ("collection.json" if full else "pilot.json"))
    return report

import json
import platform
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any

from astro_retrieval.ingest.openalex import OpenAlexClient
from astro_retrieval.ingest.pilot import PilotSample, select_pilot
from astro_retrieval.ingest.records import reconstruct_abstract, index_works, work_id


@dataclass
class PilotExpansion:
    sample: PilotSample
    papers: dict[str, dict[str, Any]]
    references: dict[str, tuple[str, ...]]
    report: dict[str, Any]


def _publication_date(value: Any) -> date | None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def expand_pilot(
    works: Iterable[Mapping[str, Any]], client: OpenAlexClient, *,
    size: int = 200, seed: int = 42, min_references: int = 8,
) -> PilotExpansion:
    """Expand usable sampled seeds once and report coverage before filtering."""
    if type(min_references) is not int or min_references < 1:
        raise ValueError("min_references must be a positive integer")
    known = index_works(works)
    sample = select_pilot(known.values(), size=size, seed=seed)
    references = {}
    for identifier in sorted(sample.abstracts):
        raw = known[identifier].get("referenced_works")
        if not isinstance(raw, list):
            raise ValueError("Usable seeds must supply a referenced_works list")
        references[identifier] = tuple(sorted({work_id(value) for value in raw} - {identifier}))

    reference_ids = {value for values in references.values() for value in values}
    pending = sorted(reference_ids - known.keys())
    usage_before = client.ledger.usage_totals()
    for offset in range(0, len(pending), 100):
        batch = pending[offset:offset + 100]
        fetched = index_works(client.iter_works({
            "filter": "openalex_id:" + "|".join(batch),
            "per_page": 100,
        }))
        if fetched.keys() - set(batch):
            raise ValueError("Reference lookup returned an unrequested ID. Review possible merges")
        known.update(fetched)

    papers = {}
    missing_abstracts = []
    invalid_abstracts = []
    for identifier in sorted(reference_ids | sample.abstracts.keys()):
        if identifier not in known:
            continue
        work = known[identifier]
        try:
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
        except ValueError:
            invalid_abstracts.append(identifier)
            continue
        if abstract is None:
            missing_abstracts.append(identifier)
            continue
        papers[identifier] = {
            "id": identifier,
            "title": work.get("title", work.get("display_name")),
            "publication_date": work.get("publication_date"),
            "abstract": abstract,
        }

    dates = {identifier: _publication_date(paper["publication_date"])
             for identifier, paper in papers.items()}
    per_seed = []
    for identifier, refs in references.items():
        eligible = []
        unknown_dates = []
        same_or_later = []
        in_corpus = sorted(set(refs) & papers.keys())
        for target in in_corpus:
            if dates[identifier] is None or dates[target] is None:
                unknown_dates.append(target)
            elif dates[target] < dates[identifier]:
                eligible.append(target)
            else:
                same_or_later.append(target)
        per_seed.append({
            "seed_id": identifier,
            "unique_references": len(refs),
            "resolved_references": len(set(refs) & known.keys()),
            "in_corpus_references": len(in_corpus),
            "eligible_references": len(eligible),
            "eligible_ids": eligible,
            "unknown_date_ids": unknown_dates,
            "same_or_later_ids": same_or_later,
        })

    counts = [row["eligible_references"] for row in per_seed]
    eligible_seeds = sum(count >= min_references for count in counts)
    resolved = reference_ids & known.keys()
    usable_refs = reference_ids & papers.keys()
    usage_after = client.ledger.usage_totals()
    report = {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "sampling_seed": sample.seed,
        "requested_candidates": sample.requested_size,
        "candidate_ids": list(sample.candidate_ids),
        "candidate_count": len(sample.candidate_ids),
        "usable_seed_count": len(sample.abstracts),
        "seed_abstract_coverage": sample.abstract_coverage,
        "seed_missing_abstract_ids": list(sample.missing_ids),
        "seed_invalid_abstract_ids": list(sample.invalid_ids),
        "unique_reference_count": len(reference_ids),
        "resolved_reference_count": len(resolved),
        "reference_resolution_coverage": _ratio(len(resolved), len(reference_ids)),
        "unresolved_reference_ids": sorted(reference_ids - known.keys()),
        "usable_reference_count": len(usable_refs),
        "usable_reference_fraction": _ratio(len(usable_refs), len(reference_ids)),
        "resolved_reference_abstract_coverage": _ratio(len(usable_refs), len(resolved)),
        "reference_missing_abstract_ids": sorted(set(missing_abstracts) & reference_ids),
        "reference_invalid_abstract_ids": sorted(set(invalid_abstracts) & reference_ids),
        "corpus_size": len(papers),
        "corpus_ids": sorted(papers),
        "corpus_unknown_date_ids": sorted(key for key, value in dates.items() if value is None),
        "corpus_year_counts": dict(sorted(Counter(
            str(value.year) if value else "unknown" for value in dates.values()
        ).items())),
        "date_rule": "Valid YYYY-MM-DD dates, reference strictly earlier than seed",
        "minimum_references": min_references,
        "median_eligible_references": median(counts) if counts else None,
        "eligible_seed_count": eligible_seeds,
        "eligible_seed_fraction": _ratio(eligible_seeds, len(counts)),
        "per_seed": per_seed,
        "lookup_batch_count": (len(pending) + 99) // 100,
        "reference_lookup_api_usage": {
            name: usage_after[name] - value for name, value in usage_before.items()
        },
    }
    return PilotExpansion(sample, papers, references, report)


def write_pilot_report(result: PilotExpansion, path: str | Path) -> None:
    write_json(result.report, path)


def write_json(value: dict, path: str | Path) -> None:
    """Replace one local JSON report only after serialization succeeds."""
    serialized = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

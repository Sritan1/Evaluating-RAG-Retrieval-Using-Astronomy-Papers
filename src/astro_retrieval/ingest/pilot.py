import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from astro_retrieval.ingest.abstracts import reconstruct_abstract
from astro_retrieval.ingest.records import index_works


@dataclass
class PilotSample:
    requested_size: int
    seed: int
    candidate_ids: tuple[str, ...]
    abstracts: dict[str, str]
    missing_ids: tuple[str, ...]
    invalid_ids: tuple[str, ...]

    @property
    def abstract_coverage(self) -> float | None:
        if not self.candidate_ids:
            return None
        return len(self.abstracts) / len(self.candidate_ids)


def select_pilot(
    works: Iterable[Mapping[str, Any]], *, size: int = 200, seed: int = 42,
) -> PilotSample:
    """Sample a supplied subject/year candidate pool before checking abstracts."""
    if type(size) is not int or size < 1:
        raise ValueError("Pilot size must be a positive integer")
    if type(seed) is not int:
        raise ValueError("Sampling seed must be an integer")

    unique = index_works(works)

    ordered = sorted(unique)
    chosen = random.Random(seed).sample(ordered, min(size, len(ordered)))
    abstracts = {}
    missing = []
    invalid = []
    for identifier in chosen:
        try:
            abstract = reconstruct_abstract(unique[identifier].get("abstract_inverted_index"))
        except ValueError:
            invalid.append(identifier)
            continue
        if abstract is None:
            missing.append(identifier)
        else:
            abstracts[identifier] = abstract

    return PilotSample(
        requested_size=size,
        seed=seed,
        candidate_ids=tuple(chosen),
        abstracts=abstracts,
        missing_ids=tuple(missing),
        invalid_ids=tuple(invalid),
    )

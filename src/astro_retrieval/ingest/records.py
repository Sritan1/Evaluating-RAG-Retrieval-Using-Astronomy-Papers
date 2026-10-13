import re
from collections.abc import Iterable, Mapping
from typing import Any


def work_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Each work must have an OpenAlex work ID")
    match = re.fullmatch(r"(?:https://openalex.org/)?(W[0-9]+)", value)
    if match is None:
        raise ValueError("Each work must have an OpenAlex work ID")
    return match.group(1)


def index_works(works: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for work in works:
        if not isinstance(work, Mapping):
            raise ValueError("Each candidate must be a work mapping")
        identifier = work_id(work.get("id"))
        normalized = dict(work)
        normalized["id"] = identifier
        if identifier in indexed and indexed[identifier] != normalized:
            raise ValueError("Conflicting records for the same OpenAlex work ID")
        indexed[identifier] = normalized
    return indexed

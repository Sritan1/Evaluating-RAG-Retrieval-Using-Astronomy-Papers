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


def reconstruct_abstract(index: Mapping[str, list[int]] | None) -> str | None:
    if index is None:
        return None
    if not isinstance(index, Mapping):
        raise ValueError("Abstract index must be a mapping or None")
    if not index:
        return None

    words = {}
    for word, positions in index.items():
        if not isinstance(word, str) or not word.strip():
            raise ValueError("Abstract words must be nonempty strings")
        if not isinstance(positions, list) or not positions:
            raise ValueError("Each word must have a nonempty position list")
        for position in positions:
            if type(position) is not int or position < 0:
                raise ValueError("Abstract positions must be nonnegative integers")
            if position in words:
                raise ValueError("Abstract positions must not overlap")
            words[position] = word

    if max(words) != len(words) - 1:
        raise ValueError("Abstract positions must be contiguous starting at zero")
    return " ".join(words[position] for position in range(len(words)))

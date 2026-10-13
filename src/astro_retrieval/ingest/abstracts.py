from collections.abc import Mapping


def reconstruct_abstract(index: Mapping[str, list[int]] | None) -> str | None:
    """Restore word order without changing spelling, case, or punctuation."""
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

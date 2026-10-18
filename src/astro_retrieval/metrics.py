import math
import random
from collections.abc import Iterable, Sequence


def _validate(ranking: Sequence[str], relevant: Iterable[str], k: int) -> set[str]:
    if type(k) is not int or k < 1:
        raise ValueError("Cutoff must be a positive integer")
    if isinstance(ranking, (str, bytes)) or isinstance(relevant, (str, bytes)):
        raise ValueError("Pass collections of document IDs")
    labels = set(relevant)
    if not labels:
        raise ValueError("Query must have at least one relevant label")
    if any(not isinstance(key, str) or not key.strip() for key in (*ranking, *labels)):
        raise ValueError("Document IDs must be nonempty strings")
    if len(set(ranking)) != len(ranking):
        raise ValueError("Ranking contains duplicate document IDs")
    return labels


def recall_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int) -> float:
    labels = _validate(ranking, relevant, k)
    return len(set(ranking[:k]) & labels) / len(labels)


def reciprocal_rank_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int = 100) -> float:
    labels = _validate(ranking, relevant, k)
    return next((1 / rank for rank, key in enumerate(ranking[:k], 1) if key in labels), 0.0)


def ndcg_at_k(ranking: Sequence[str], relevant: Iterable[str], k: int = 10) -> float:
    labels = _validate(ranking, relevant, k)
    dcg = sum(1 / math.log2(rank + 1)
              for rank, key in enumerate(ranking[:k], 1) if key in labels)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(labels)) + 1))
    return dcg / ideal


def score_query(ranking: Sequence[str], relevant: Iterable[str]) -> dict[str, float]:
    labels = _validate(ranking, relevant, 100)
    return {
        "recall@10": recall_at_k(ranking, labels, 10),
        "recall@50": recall_at_k(ranking, labels, 50),
        "recall@100": recall_at_k(ranking, labels, 100),
        "rr@100": reciprocal_rank_at_k(ranking, labels, 100),
        "ndcg@10": ndcg_at_k(ranking, labels, 10),
    }


def bootstrap_mean(scores: dict[str, float], *, reference: dict[str, float] | None = None,
                   resamples: int = 2000, seed: int = 42) -> dict:
    if not scores or any(not isinstance(key, str) or not key.strip() for key in scores):
        raise ValueError("Scores require nonempty paper IDs")
    if reference is not None and scores.keys() != reference.keys():
        raise ValueError("Paired scores must contain identical papers")
    for group in (scores,) if reference is None else (scores, reference):
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in group.values()):
            raise ValueError("Scores must be finite numbers")
    if type(resamples) is not int or resamples < 2 or type(seed) is not int:
        raise ValueError("Use at least two resamples and an integer seed")
    values = [scores[key] - (reference[key] if reference is not None else 0)
              for key in sorted(scores)]
    count = len(values)
    rng = random.Random(seed)
    means = sorted(math.fsum(values[rng.randrange(count)] for _ in range(count)) / count
                   for _ in range(resamples))

    def percentile(fraction):
        position = (resamples - 1) * fraction
        lower = math.floor(position)
        return means[lower] + (means[math.ceil(position)] - means[lower]) * (position - lower)

    return {"mean": math.fsum(values) / count, "ci95": [percentile(.025), percentile(.975)],
            "papers": count, "resamples": resamples, "seed": seed}

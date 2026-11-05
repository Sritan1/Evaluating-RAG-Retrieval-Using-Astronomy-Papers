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


def _paired_values(scores, reference=None):
    if not scores or any(not isinstance(key, str) or not key.strip() for key in scores):
        raise ValueError("Scores require nonempty paper IDs")
    if reference is not None and scores.keys() != reference.keys():
        raise ValueError("Paired scores must contain identical papers")
    for group in (scores,) if reference is None else (scores, reference):
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in group.values()):
            raise ValueError("Scores must be finite numbers")
    values = [scores[key] - (reference[key] if reference is not None else 0)
              for key in sorted(scores)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Paired differences must be finite")
    return values


def bootstrap_mean(scores: dict[str, float], *, reference: dict[str, float] | None = None,
                   resamples: int = 2000, seed: int = 42) -> dict:
    values = _paired_values(scores, reference)
    if type(resamples) is not int or resamples < 2 or type(seed) is not int:
        raise ValueError("Use at least two resamples and an integer seed")
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


def paired_sign_flip(scores, *, reference=None, draws=10000, seed=42):
    values = _paired_values(scores, reference)
    if type(draws) is not int or draws < 1 or type(seed) is not int:
        raise ValueError("Use positive integer draws and an integer seed")
    count = len(values)
    observed = math.fsum(value / count for value in values)
    patterns = 1 << count
    exact = patterns <= draws
    samples = patterns if exact else draws
    rng = random.Random(seed)
    extreme = 0
    for number in range(samples):
        signs = number if exact else rng.getrandbits(count)
        statistic = math.fsum((value if signs & (1 << i) else -value) / count
                              for i, value in enumerate(values))
        extreme += abs(statistic) >= abs(observed) - 1e-12
    p = extreme / samples if exact else (extreme + 1) / (samples + 1)
    return {"mean": observed, "p_value": p, "method": "exact" if exact else "monte_carlo",
            "draws": samples, "seed": seed, "papers": count, "alternative": "two-sided"}


def holm_adjust(p_values):
    if (not p_values or any(not isinstance(key, str) or not key.strip() for key in p_values)
            or any(isinstance(p, bool) or not isinstance(p, (int, float))
                   or not math.isfinite(p) or not 0 <= p <= 1 for p in p_values.values())):
        raise ValueError("Named p-values must be finite numbers between zero and one")
    adjusted, previous = {}, 0.0
    for position, key in enumerate(sorted(p_values, key=lambda key: (p_values[key], key))):
        previous = max(previous, min(1.0, (len(p_values) - position) * p_values[key]))
        adjusted[key] = previous
    return {key: adjusted[key] for key in p_values}


def compare_hypotheses(system_scores):
    names = ("bm25", "hybrid", "reranker", "popularity", "random")
    if any(name not in system_scores for name in names):
        raise ValueError("All five comparison systems are required")
    papers = None
    for name in names:
        forms = system_scores[name]
        if set(forms) != {"title", "abstract"}:
            raise ValueError("Both query forms are required")
        for rows in forms.values():
            if papers is None:
                papers = set(rows)
            if not papers or set(rows) != papers:
                raise ValueError("All systems and forms must contain identical papers")
            for metric in ("recall@100", "ndcg@10"):
                if any(metric not in row for row in rows.values()):
                    raise ValueError("Recall@100 and nDCG@10 are required")
                values = _paired_values({paper: row[metric] for paper, row in rows.items()})
                if any(not 0 <= value <= 1 for value in values):
                    raise ValueError("Retrieval metrics must be between zero and one")
    for name in ("popularity", "random"):
        if any(system_scores[name]["title"][paper]["recall@100"]
               != system_scores[name]["abstract"][paper]["recall@100"] for paper in papers):
            raise ValueError("Baseline Recall@100 must match across query forms")

    def difference(first, second, form, metric):
        return {paper: system_scores[first][form][paper][metric]
                - system_scores[second][form][paper][metric] for paper in sorted(papers)}

    title = difference("reranker", "hybrid", "title", "ndcg@10")
    abstract = difference("reranker", "hybrid", "abstract", "ndcg@10")
    contrasts = {
        "H1-title": difference("hybrid", "bm25", "title", "recall@100"),
        "H1-abstract": difference("hybrid", "bm25", "abstract", "recall@100"),
        "H2": {paper: title[paper] - abstract[paper] for paper in sorted(papers)},
        "H3": difference("popularity", "random", "title", "recall@100"),
    }
    results = {name: {"bootstrap": bootstrap_mean(values), "test": paired_sign_flip(values)}
               for name, values in contrasts.items()}
    adjusted = holm_adjust({name: row["test"]["p_value"] for name, row in results.items()})
    for name, row in results.items():
        row["adjusted_p_value"] = adjusted[name]
        row["supported"] = row["test"]["mean"] > 0 and adjusted[name] <= .05
    return {"contrasts": results, "family": list(contrasts), "correction": "Holm", "alpha": .05,
            "h1_supported": results["H1-title"]["supported"] and results["H1-abstract"]["supported"],
            "interval_note": "Marginal 95% paper bootstrap intervals, not multiplicity-adjusted",
            "inference_note": "Sign-flip assumptions apply. Development results are exploratory"}

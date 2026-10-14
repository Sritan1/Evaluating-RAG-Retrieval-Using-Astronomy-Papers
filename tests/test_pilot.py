import copy
import random

import pytest

from astro_retrieval.ingest.pilot import select_pilot
from astro_retrieval.ingest.records import reconstruct_abstract


def test_sampling_is_independent_of_input_order_and_duplicates():
    works = [{"id": f"https://openalex.org/W{number}"} for number in range(1, 11)]
    first = select_pilot(works, size=4, seed=42)
    reordered = select_pilot(reversed(works + [works[0], {"id": "W1"}]), size=4, seed=42)
    assert first == reordered
    assert len(first.candidate_ids) == len(set(first.candidate_ids)) == 4
    assert set(first.candidate_ids) <= {f"W{number}" for number in range(1, 11)}


def test_seed_is_recorded_and_does_not_change_global_random_state():
    works = [{"id": f"W{number}"} for number in range(1, 21)]
    before = random.getstate()
    first = select_pilot(works, size=5, seed=42)
    second = select_pilot(works, size=5, seed=7)
    assert first.seed == 42
    assert second.seed == 7
    assert first.candidate_ids != second.candidate_ids
    assert random.getstate() == before


def test_coverage_includes_missing_and_invalid_abstracts():
    works = [
        {"id": "W1", "abstract_inverted_index": {"planets": [0, 2], "orbit": [1]}},
        {"id": "W2", "abstract_inverted_index": None},
        {"id": "W3", "abstract_inverted_index": {}},
        {"id": "W4", "abstract_inverted_index": {"bad": [1]}},
    ]
    original = copy.deepcopy(works)
    sample = select_pilot(works)
    assert sample.requested_size == 200
    assert len(sample.candidate_ids) == 4
    assert sample.abstracts == {"W1": "planets orbit planets"}
    assert set(sample.missing_ids) == {"W2", "W3"}
    assert sample.invalid_ids == ("W4",)
    assert sample.abstract_coverage == 0.25
    assert works == original


def test_abstract_availability_does_not_change_selection_or_trigger_replacement():
    works = [{"id": f"W{number}", "abstract_inverted_index": {"word": [0]}}
             for number in range(1, 11)]
    complete = select_pilot(works, size=3)
    changed_id = complete.candidate_ids[0]
    next(work for work in works if work["id"] == changed_id)["abstract_inverted_index"] = None
    missing = select_pilot(works, size=3)
    assert missing.candidate_ids == complete.candidate_ids
    assert missing.missing_ids == (changed_id,)
    assert len(missing.abstracts) == 2
    assert missing.abstract_coverage == pytest.approx(2 / 3)


def test_small_pool_returns_every_unique_candidate():
    sample = select_pilot([{"id": "W2"}, {"id": "W1"}, {"id": "W1"}], size=200)
    assert set(sample.candidate_ids) == {"W1", "W2"}
    assert len(sample.candidate_ids) == 2
    assert sample.abstract_coverage == 0


def test_empty_pool_has_undefined_coverage():
    sample = select_pilot([])
    assert sample.candidate_ids == ()
    assert sample.abstracts == {}
    assert sample.missing_ids == sample.invalid_ids == ()
    assert sample.abstract_coverage is None


def test_conflicting_duplicates_raise_in_either_order():
    works = [{"id": "W1", "title": "first"}, {"id": "https://openalex.org/W1", "title": "second"}]
    for candidates in (works, list(reversed(works))):
        with pytest.raises(ValueError, match="Conflicting"):
            select_pilot(candidates)


@pytest.mark.parametrize("work", [None, [], {}, {"id": None}, {"id": 123},
                                  {"id": "W1?key=example"}, {"id": "https://example.org/W1"}])
def test_invalid_candidate_ids_raise(work):
    with pytest.raises(ValueError):
        select_pilot([work])


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_sample_size_raises(size):
    with pytest.raises(ValueError, match="size"):
        select_pilot([], size=size)


@pytest.mark.parametrize("seed", [True, 1.5, "42", None])
def test_invalid_seed_raises(seed):
    with pytest.raises(ValueError, match="seed"):
        select_pilot([], seed=seed)


def test_reconstructs_repeated_words_in_position_order():
    index = {"planets": [2, 0], "orbit": [1]}
    assert reconstruct_abstract(index) == "planets orbit planets"
    assert index == {"planets": [2, 0], "orbit": [1]}


def test_preserves_case_punctuation_and_unicode():
    assert reconstruct_abstract({"planet.": [2], "A": [0], "β": [1]}) == "A β planet."


@pytest.mark.parametrize("index", [None, {}])
def test_missing_abstract_returns_none(index):
    assert reconstruct_abstract(index) is None


@pytest.mark.parametrize("index", [
    [], "text", False,
    {"word": [-1]}, {"word": [True]}, {"word": [0.0]}, {"word": ["0"]},
    {"word": []}, {"word": None}, {"word": (0,)},
    {"": [0]}, {" ": [0]}, {1: [0]},
])
def test_malformed_index_raises(index):
    with pytest.raises(ValueError):
        reconstruct_abstract(index)


@pytest.mark.parametrize("index", [
    {"one": [0], "two": [0]}, {"word": [0, 0]},
])
def test_overlapping_positions_raise(index):
    with pytest.raises(ValueError, match="overlap"):
        reconstruct_abstract(index)


@pytest.mark.parametrize("index", [
    {"word": [1]}, {"one": [0], "three": [2]}, {"word": [10**12]},
])
def test_missing_positions_raise_without_allocating_from_largest_position(index):
    with pytest.raises(ValueError, match="contiguous"):
        reconstruct_abstract(index)

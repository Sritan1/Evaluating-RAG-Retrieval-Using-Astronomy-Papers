import pytest

from astro_retrieval.ingest.abstracts import reconstruct_abstract


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

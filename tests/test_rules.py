import unicodedata

import pytest

from lipogram.rules import (FORBIDDEN_CHARS, all_forbidden_chars, count_forbidden, forbidden_positions,
                            has_forbidden, is_lipogram, load_forbidden)


@pytest.mark.parametrize("text", ["", "123, 456 !", "Un chat noir dormait sur un tapis.",
                                  "l'ami d'un fils qu'on n'a pas vu s'amusait", "à â ç î ï ô ù û ü ÿ æ Æ",
                                  "AAAA", "—…«»"])
def test_allowed(text):
    assert is_lipogram(text) and count_forbidden(text) == 0


@pytest.mark.parametrize("text,n", [("le", 1), ("LE", 1), ("été", 2), ("Élève", 3), ("Noël", 1),
                                    ("cœur", 1), ("ŒUVRE", 2), ("être", 2), ("ẽ", 1), ("ｅ", 1)])
def test_forbidden(text, n):
    assert has_forbidden(text) and count_forbidden(text) == n


def test_decomposed_accents():
    nfd = unicodedata.normalize("NFD", "été")
    assert len(nfd) == 5 and count_forbidden(nfd) == 2
    assert count_forbidden("e̱") == 1  # e + combining macron below: no precomposed form
    assert is_lipogram(unicodedata.normalize("NFD", "à ç ô"))


def test_positions_are_nfc_indices():
    assert forbidden_positions("un été") == [3, 5]
    assert forbidden_positions(unicodedata.normalize("NFD", "un été")) == [3, 5]


def test_configurable():
    assert is_lipogram("cœur", frozenset("eE"))
    assert not is_lipogram("bœuf", frozenset("œ"))


def test_config_file_matches_default():
    assert load_forbidden() == FORBIDDEN_CHARS


def test_all_forbidden_chars_covers_set():
    chars = set(all_forbidden_chars())
    assert FORBIDDEN_CHARS <= chars and "a" not in chars and "à" not in chars

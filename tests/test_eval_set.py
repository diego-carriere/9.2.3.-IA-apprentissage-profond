"""The eval set is frozen: any change to data/eval/ must be deliberate (rebuild + new hash in decisions.md)."""

import collections
import hashlib
import json
from pathlib import Path

import pytest

from lipogram.rules import has_forbidden

EVAL = Path(__file__).resolve().parents[1] / "data" / "eval"
EXPECTED = {"phrases": 200, "paragraphes": 50, "pieges": 30}


def rows(kind):
    return [json.loads(line) for line in (EVAL / f"{kind}.jsonl").read_text(encoding="utf-8").splitlines()]


def test_manifest_matches_files():
    for line in (EVAL / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((EVAL / name).read_bytes()).hexdigest() == digest, f"{name} changed: eval set is frozen"


@pytest.mark.parametrize("kind,n", EXPECTED.items())
def test_counts_and_fields(kind, n):
    items = rows(kind)
    assert len(items) == n
    for r in items:
        assert r["kind"] == kind and r["text"].strip() and r["source"]["gutenberg_id"] and r["license"]
        assert has_forbidden(r["text"])  # a text without « e » would be a trivial example


def test_unique_and_balanced():
    allrows = [r for k in EXPECTED for r in rows(k)]
    assert len({r["id"] for r in allrows}) == len(allrows)
    assert len({r["text"] for r in allrows}) == len(allrows)
    genres = collections.Counter(r["genre"] for r in rows("phrases"))
    assert len(genres) == 5 and set(genres.values()) == {40}

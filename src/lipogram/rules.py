"""Single source of truth for what counts as a lipogram (no letter « e »).

A character is forbidden if it is in FORBIDDEN_CHARS, or if its compatibility
decomposition (NFKD) contains one: this catches é/ẽ/ę…, « e » + combining accent,
and look-alikes such as fullwidth ｅ or mathematical 𝐞. Positions refer to the
NFC-normalized text.
"""

import hashlib
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml

FORBIDDEN_CHARS = frozenset("eEéèêëÉÈÊË") | frozenset("œŒ")


def load_forbidden(path: str | Path = "configs/rules.yaml") -> frozenset[str]:
    return frozenset(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["forbidden_chars"])


@lru_cache(maxsize=65536)
def _char_forbidden(c: str, forbidden: frozenset[str]) -> bool:
    return c in forbidden or any(d in forbidden for d in unicodedata.normalize("NFKD", c))


def forbidden_positions(text: str, forbidden: frozenset[str] = FORBIDDEN_CHARS) -> list[int]:
    return [i for i, c in enumerate(unicodedata.normalize("NFC", text)) if _char_forbidden(c, forbidden)]


def count_forbidden(text: str, forbidden: frozenset[str] = FORBIDDEN_CHARS) -> int:
    return len(forbidden_positions(text, forbidden))


def has_forbidden(text: str, forbidden: frozenset[str] = FORBIDDEN_CHARS) -> bool:
    return any(_char_forbidden(c, forbidden) for c in unicodedata.normalize("NFC", text))


def is_lipogram(text: str, forbidden: frozenset[str] = FORBIDDEN_CHARS) -> bool:
    return not has_forbidden(text, forbidden)


def all_forbidden_chars(forbidden: frozenset[str] = FORBIDDEN_CHARS) -> list[str]:
    """Every Unicode scalar value that is_lipogram rejects (used to build the token mask)."""
    return [chr(cp) for cp in range(0x110000)
            if not 0xD800 <= cp <= 0xDFFF and _char_forbidden(chr(cp), forbidden)]


def rules_hash(forbidden: frozenset[str] = FORBIDDEN_CHARS) -> str:
    return hashlib.sha256("".join(sorted(forbidden)).encode()).hexdigest()[:12]

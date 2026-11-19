"""Static mask of tokens that could ever emit a forbidden character.

A token is forbidden if its bytes decode to a forbidden char, or if it can take part in a
forbidden char split across two tokens. For each such split we ban exactly one side:
  - 2-byte chars whose lead is also used by French letters (C3/C5: é = C3|A9, à = C3|A0):
    ban tokens *starting* with the continuation byte (A9), so « Ã » + « ¹ » = ù stays possible;
  - every other forbidden char (ẹ = E1 BA B9, ℯ…): ban tokens *ending* with an incomplete
    prefix of it (E1, E1 BA). Any split puts the lead byte at the end of some token, so this
    also covers 3-token splits.
Special tokens are forbidden except end-of-turn / EOS.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

import torch
from transformers.convert_slow_tokenizer import bytes_to_unicode

from lipogram.rules import FORBIDDEN_CHARS, all_forbidden_chars, has_forbidden, rules_hash

MASK_VERSION = 1
END_TOKENS = ("<|im_end|>", "<|endoftext|>", "<end_of_turn>", "<eos>", "</s>")
_SP_BYTE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def token_bytes(tokenizer) -> list[bytes | None]:
    """Raw bytes of every vocab entry, None for added/special tokens."""
    decoder = json.dumps(json.loads(tokenizer.backend_tokenizer.to_str())["decoder"])
    if "ByteLevel" in decoder:  # GPT-2 style (Qwen, Llama 3): each char of the piece stands for one byte
        u2b = {u: b for b, u in bytes_to_unicode().items()}
        conv = lambda p: bytes(u2b[c] for c in p)
    elif "ByteFallback" in decoder:  # SentencePiece (Llama 2, Gemma): text pieces + <0xHH> byte pieces
        conv = lambda p: bytes([int(m[1], 16)]) if (m := _SP_BYTE.fullmatch(p)) else p.replace("▁", " ").encode()
    else:
        raise ValueError(f"unsupported tokenizer decoder: {decoder[:200]}")
    special = set(tokenizer.added_tokens_decoder)
    pieces = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    return [None if i in special else conv(p) for i, p in enumerate(pieces)]


FRENCH_LETTERS = "àâçîïôùûüÿÀÂÇÎÏÔÙÛÜŸ"  # must stay representable even when split as lead|continuation
SPLIT_LEADS = frozenset(c.encode()[0] for c in FRENCH_LETTERS)  # {0xC3, 0xC5}: 2-byte leads
assert all(0xC2 <= x < 0xE0 for x in SPLIT_LEADS)


@lru_cache(maxsize=8)
def boundary_rules(forbidden: frozenset[str] = FORBIDDEN_CHARS) -> tuple[frozenset[int], frozenset[bytes]]:
    """(continuation bytes banned at token start, incomplete prefixes banned at token end)."""
    start, end = set(), set()
    for c in all_forbidden_chars(forbidden):
        b = c.encode()
        if len(b) == 1:
            continue
        if b[0] in SPLIT_LEADS:
            start.add(b[1])
        else:
            end.update(b[:k] for k in range(1, len(b)))
    return frozenset(start), frozenset(end)


def trailing_incomplete(b: bytes) -> bytes:
    """The unfinished UTF-8 sequence at the end of b (lead + some continuation bytes), or b""."""
    for k in range(1, min(4, len(b)) + 1):
        x = b[-k]
        if x < 0x80:
            return b""
        if x >= 0xC0:
            return b[-k:] if k < (2 if x < 0xE0 else 3 if x < 0xF0 else 4) else b""
    return b""


def token_is_forbidden(b: bytes, forbidden: frozenset[str] = FORBIDDEN_CHARS) -> bool:
    start, end = boundary_rules(forbidden)
    if (b and b[0] in start) or trailing_incomplete(b) in end:
        return True
    return has_forbidden(b.decode("utf-8", errors="ignore"), forbidden)


def default_allowed_special(tokenizer) -> set[int]:
    return {tokenizer.eos_token_id} | {i for i, t in tokenizer.added_tokens_decoder.items() if t.content in END_TOKENS}


def build_mask(tokenizer, vocab_size: int | None = None, forbidden: frozenset[str] = FORBIDDEN_CHARS) -> torch.Tensor:
    """Bool tensor [vocab_size], True = forbidden. vocab_size = model logits dim (may exceed len(tokenizer))."""
    vocab_size = vocab_size or len(tokenizer)
    assert vocab_size >= len(tokenizer), (vocab_size, len(tokenizer))
    allow = default_allowed_special(tokenizer)
    mask = torch.ones(vocab_size, dtype=torch.bool)  # ids past the tokenizer (padding rows) stay forbidden
    for i, b in enumerate(token_bytes(tokenizer)):
        mask[i] = (i not in allow) if b is None else token_is_forbidden(b, forbidden)
    return mask


def load_or_build_mask(tokenizer, vocab_size: int | None = None, forbidden: frozenset[str] = FORBIDDEN_CHARS,
                       cache_dir: str | Path = "data/processed/masks") -> torch.Tensor:
    vocab_size = vocab_size or len(tokenizer)
    name = tokenizer.name_or_path.strip("/").replace("/", "__")
    path = Path(cache_dir) / f"{name}_{vocab_size}_{rules_hash(forbidden)}_v{MASK_VERSION}.pt"
    if path.exists():
        return torch.load(path)
    mask = build_mask(tokenizer, vocab_size, forbidden)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mask, path)
    return mask

import random

import pytest

from lipogram.rules import is_lipogram
from lipogram.token_mask import (FRENCH_LETTERS, build_mask, load_or_build_mask, token_bytes, token_is_forbidden,
                                 trailing_incomplete)

FUNCTION_WORDS = ("un au du ma mon son nos vos ou mais donc car ainsi dans sur par pour sans qui dont avant aussi "
                  "plus tout tous sont ont font vont à où là ça il ils nous vous on y a avait savoir").split()
E_LESS_TEXT = ("Un soir, au bord du lac, Paul attrapa un poisson. Il cria : « Quoi ? Ça va, mon ami ? » "
               "Jacqui sourit ; tout allait bon train. Où sont-ils ? Là-bas, au donjon, à droit du jardin. "
               "Plutôt sûr : août 1987, 42 % !\nIls ont fini. Maïs, naïf, L’HÔTAL, ÎLOT, ÛN, Ù, ÿ, Â, À, Ÿ.")


@pytest.mark.parametrize("b,expected", [
    (b"le", True), (b"E", True), ("é".encode(), True), ("œ".encode(), True), (b"\xa9", True),  # C3|A9 = é
    (b"\xc3", False), (b"\xb9", False), ("à".encode(), False), (b" \xc3", False),  # C3|B9 = ù must stay possible
    (b"\xe1", True), (b"\xe1\xba", True),  # prefixes of ẹ (E1 BA B9)
    (b"\xe2\x80", False), (b"\xe2\x84", True),  # E2 80 starts ’ ; E2 84 starts ℯ
    ("’".encode(), False), (b"", False), (b"abc\xc3", False),
])
def test_token_is_forbidden(b, expected):
    assert token_is_forbidden(b) is expected


def test_trailing_incomplete():
    assert trailing_incomplete(b"a\xc3") == b"\xc3"
    assert trailing_incomplete(b"a\xe2\x84") == b"\xe2\x84"
    assert trailing_incomplete("aé".encode()) == b""
    assert trailing_incomplete(b"\xba") == b""


@pytest.fixture(scope="module", params=["Qwen/Qwen3-0.6B", "hf-internal-testing/llama-tokenizer"])
def tok_mask(request):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(request.param)
    return tok, build_mask(tok), token_bytes(tok)


def _allowed_ids(tok, mask, tb):
    return [i for i in range(len(tok)) if not mask[i] and tb[i] is not None]


@pytest.mark.slow
def test_pairs_never_produce_e(tok_mask):
    tok, mask, tb = tok_mask
    rng = random.Random(0)
    ids = _allowed_ids(tok, mask, tb)
    # 100k random pairs + every pair that can join bytes across the boundary (exhaustive, ~200k for Qwen)
    ends = [i for i in ids if trailing_incomplete(tb[i])]
    starts = [i for i in ids if tb[i] and 0x80 <= tb[i][0] <= 0xBF]
    pairs = [(rng.choice(ids), rng.choice(ids)) for _ in range(100_000)] + [(a, b) for a in ends for b in starts]
    for a, b in pairs:
        assert is_lipogram((tb[a] + tb[b]).decode("utf-8", errors="replace")), (tb[a], tb[b])
    decoded = tok.batch_decode([list(p) for p in pairs[:20_000]])
    assert all(is_lipogram(t) for t in decoded)
    assert all(is_lipogram(tok.decode([i])) for i in ids)


@pytest.mark.slow
def test_french_stays_representable(tok_mask):
    tok, mask, _ = tok_mask
    texts = [E_LESS_TEXT, *FUNCTION_WORDS, *(" " + w for w in FUNCTION_WORDS),
             *FRENCH_LETTERS, *(" " + c for c in FRENCH_LETTERS), *("a" + c + "a" for c in FRENCH_LETTERS),
             "0123456789", ".,;:!?'’\"«»()-—…", " \n\t"]
    for text in texts:
        assert is_lipogram(text)
        ids = tok.encode(text, add_special_tokens=False)
        assert not any(mask[i] for i in ids), (text, [tok.convert_ids_to_tokens(i) for i in ids if mask[i]])


@pytest.mark.slow
def test_only_end_tokens_allowed_among_specials(tok_mask):
    tok, mask, _ = tok_mask
    allowed = {tok.convert_ids_to_tokens(i) for i in tok.added_tokens_decoder if not mask[i]}
    assert tok.eos_token in allowed and allowed <= {"<|im_end|>", "<|endoftext|>", "</s>"}


@pytest.mark.slow
def test_padding_rows_and_cache(tok_mask, tmp_path):
    tok, mask, _ = tok_mask
    m = load_or_build_mask(tok, len(tok) + 7, cache_dir=tmp_path)
    assert m[-7:].all() and (m[:len(tok)] == mask).all()
    assert len(list(tmp_path.iterdir())) == 1
    assert (load_or_build_mask(tok, len(tok) + 7, cache_dir=tmp_path) == m).all()

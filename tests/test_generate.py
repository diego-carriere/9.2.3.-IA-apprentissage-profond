import json

import pytest
import torch

from lipogram.generate import PROMPTS_DIR, ForbiddenTokensProcessor, build_messages, generate
from lipogram.rules import is_lipogram
from lipogram.token_mask import load_or_build_mask


def test_fewshot_targets_are_lipograms():
    lines = (PROMPTS_DIR / "fewshot.jsonl").read_text(encoding="utf-8").splitlines()
    assert 3 <= len(lines) <= 5
    assert all(is_lipogram(json.loads(line)["target"]) for line in lines)


def test_messages_shape():
    msgs = build_messages("Bonjour")
    assert msgs[0]["role"] == "system" and msgs[-1] == {"role": "user", "content": "Bonjour"}


def test_processor_masks_only_forbidden():
    mask = torch.tensor([False, True, False, True])
    out = ForbiddenTokensProcessor(mask)(torch.zeros(2, 3, dtype=torch.long), torch.ones(2, 4))
    assert torch.isinf(out[:, mask]).all() and (out[:, ~mask] == 1).all()


@pytest.fixture(scope="module")
def tiny():
    """Random 1-layer Qwen3 with the real Qwen3 tokenizer: exercises the full path without a download."""
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=True)
    return Qwen3ForCausalLM(cfg).eval(), tok, load_or_build_mask(tok, cfg.vocab_size)


@pytest.mark.slow
def test_random_model_masked_outputs_are_lipograms(tiny):
    model, tok, mask = tiny
    src = "Le soleil se lève sur la mer et les enfants courent vers la plage."
    masked = generate(model, tok, src, mask, n=8, temperature=1.0, top_p=1.0)
    free = generate(model, tok, src, None, n=8, temperature=1.0, top_p=1.0)
    assert len(masked) == 8 and all(is_lipogram(c["text"]) for c in masked)
    assert not all(is_lipogram(c["text"]) for c in free)  # the mask, not luck, removes the « e »
    assert all(isinstance(c["truncated"], bool) for c in masked)

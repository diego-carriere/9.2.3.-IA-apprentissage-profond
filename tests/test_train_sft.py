import pytest
import torch

from lipogram.generate import build_messages
from lipogram.train_sft import IGNORE, build_example, masked_lm_loss


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


@pytest.mark.slow
@pytest.mark.parametrize("fewshot", [True, False])
def test_prompt_identical_to_inference_and_loss_on_answer_only(tok, fewshot):
    src, tgt = "Le chat dort sur le canapé.", "Mon chat dort sur un divan."
    ex = build_example(tok, src, tgt, fewshot, max_len=4096)
    infer = tok.apply_chat_template(build_messages(src, fewshot), add_generation_prompt=True, enable_thinking=False,
                                    tokenize=True, return_dict=True)["input_ids"]
    assert ex["input_ids"][:len(infer)] == infer  # exact same prompt as at inference time
    assert all(label == IGNORE for label in ex["labels"][:len(infer)])
    answer = [i for i in ex["labels"] if i != IGNORE]
    assert tok.decode(answer) == tgt + "<|im_end|>"
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])


@pytest.mark.slow
def test_too_long_examples_are_dropped(tok):
    assert build_example(tok, "Le chat dort.", "Mon chat dort.", True, max_len=50) is None


def test_masked_loss_renormalizes():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 6)
    labels = torch.tensor([[IGNORE, 0, 2, 4]])
    forbidden = torch.tensor([False, True, False, True, False, True])
    plain, masked = masked_lm_loss(logits, labels, None), masked_lm_loss(logits, labels, forbidden)
    assert torch.isfinite(masked) and masked < plain  # same targets, probability mass of banned tokens removed

"""Constrained generation: few-shot chat prompt + static token mask at every decoding step."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from lipogram.token_mask import default_allowed_special, load_or_build_mask

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


class ForbiddenTokensProcessor(LogitsProcessor):
    """Forbidden logits -> -inf. Logits processors only run on generated steps, never on the prompt."""

    def __init__(self, mask: torch.Tensor):
        self.mask = mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores.masked_fill(self.mask.to(scores.device), float("-inf"))


def build_messages(source: str, prompts_dir: Path = PROMPTS_DIR) -> list[dict]:
    msgs = [{"role": "system", "content": (prompts_dir / "system.txt").read_text(encoding="utf-8").strip()}]
    for line in (prompts_dir / "fewshot.jsonl").read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        msgs += [{"role": "user", "content": ex["source"]}, {"role": "assistant", "content": ex["target"]}]
    return msgs + [{"role": "user", "content": source}]


def load(model_name: str, dtype: str = "auto", load_in_4bit: bool = False):
    """Returns (model, tokenizer, mask) with the mask sized to the model's logits. GPU used when available."""
    tok = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"dtype": getattr(torch, dtype, dtype)}
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    if load_in_4bit:  # needs bitsandbytes + CUDA (GPU env only)
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
    return model, tok, load_or_build_mask(tok, model.config.vocab_size)


@torch.inference_mode()
def generate(model, tok, source: str, mask: torch.Tensor | None, n: int = 1, temperature: float = 0.7,
             top_p: float = 0.9, max_new_tokens: int = 256, repetition_penalty: float = 1.0,
             no_repeat_ngram_size: int = 0, seed: int = 0) -> list[dict]:
    """n sampled rewrites of `source`. mask=None disables the constraint (for ablations).

    repetition_penalty / no_repeat_ngram_size default off: HF applies them to the prompt too,
    which would forbid copying e-free n-grams from the source.
    """
    enc = tok.apply_chat_template(build_messages(source), add_generation_prompt=True, enable_thinking=False,
                                  return_tensors="pt", return_dict=True).to(model.device)
    # shortcut: length cap = 2x source + 32 tokens bounds the cost of degenerate loops; raise if rewrites get cut
    budget = min(max_new_tokens, 2 * len(tok(source).input_ids) + 32)
    torch.manual_seed(seed)
    out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=budget,
                         repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
                         num_return_sequences=n, pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         logits_processor=LogitsProcessorList([ForbiddenTokensProcessor(mask)] if mask is not None else []))
    gen = out[:, enc["input_ids"].shape[1]:]
    ends = torch.tensor(sorted(default_allowed_special(tok)), device=gen.device)
    return [{"text": tok.decode(g, skip_special_tokens=True).strip(), "truncated": not torch.isin(g, ends).any().item()}
            for g in gen]

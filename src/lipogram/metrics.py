"""Evaluation metrics. Embedding model and LM are passed in, so tests can use tiny ones."""

import csv
import random
import re
import statistics
from functools import lru_cache

import torch

from lipogram.rules import is_lipogram

EMBEDDER = "intfloat/multilingual-e5-small"
# meta-text = a preamble about the answer itself (« Voici : … », « Ma proposition : … »), not « Voici nos gamins »
META = re.compile(r"^\s*(voici|voilà|ma proposition|proposition|version|traduction|lipogramm\w*)\s*:", re.IGNORECASE)
WORD = re.compile(r"\w+")


@lru_cache(maxsize=2)
def load_embedder(name: str = EMBEDDER, device: str | None = None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=device)


def sem_sim(sources: list[str], outputs: list[str], embedder) -> list[float]:
    """Cosine similarity source <-> output (e5 expects the 'query: ' prefix for symmetric tasks)."""
    enc = lambda xs: embedder.encode([f"query: {x}" for x in xs], normalize_embeddings=True, convert_to_tensor=True)
    return (enc(sources) * enc(outputs)).sum(-1).tolist()


def len_ratio(source: str, output: str) -> float:
    return len(output) / max(len(source), 1)


@torch.inference_mode()
def fluency(texts: list[str], model, tok, batch_size: int = 16) -> list[float]:
    """Mean log-prob per token of each text under the *unmasked* LM (higher = more natural)."""
    scores = {}
    todo = [i for i, t in enumerate(texts) if t.strip()]
    for b in range(0, len(todo), batch_size):
        idx = todo[b:b + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        logp = model(**enc).logits[:, :-1].float().log_softmax(-1)
        tgt = enc.input_ids[:, 1:]
        tok_lp = logp.gather(-1, tgt[..., None])[..., 0] * enc.attention_mask[:, 1:]
        n = enc.attention_mask[:, 1:].sum(-1)
        scores.update({i: (s / k).item() if k > 0 else float("nan") for i, s, k in zip(idx, tok_lp.sum(-1), n)})
    return [scores.get(i, float("nan")) for i in range(len(texts))]


def degenerate_reason(source: str, output: str, truncated: bool = False) -> str | None:
    """Why an output is unusable, or None. Checked: empty, truncated, repetitive, meta-text, partial copy."""
    words = WORD.findall(output.lower())
    if not words:
        return "empty"
    if truncated:
        return "truncated"
    trigrams = list(zip(words, words[1:], words[2:]))
    if len(trigrams) >= 4 and len(set(trigrams)) / len(trigrams) < 0.6:
        return "repetitive"
    if META.match(output):
        return "meta"
    src_words = set(WORD.findall(source.lower()))
    if len_ratio(source, output) < 0.5 and all(w in src_words for w in words):
        return "partial_copy"
    return None


def summary(values: list[float]) -> dict:
    vals = sorted(v for v in values if v == v)  # drop NaN
    if not vals:
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan")}
    return {"mean": statistics.fmean(vals), "median": statistics.median(vals), "p10": vals[int(0.1 * (len(vals) - 1))]}


def score_rows(rows: list[dict]) -> dict:
    """rows: dicts with source, output, sem_sim, len_ratio, fluency, degenerate. Returns aggregate metrics."""
    return {
        "lipogram_rate": statistics.fmean(is_lipogram(r["output"]) for r in rows),
        "degenerate_rate": statistics.fmean(r["degenerate"] is not None for r in rows),
        "len_in_range": statistics.fmean(0.7 <= r["len_ratio"] <= 1.3 for r in rows),
        **{f"{k}_{s}": v for k in ("sem_sim", "len_ratio", "fluency") for s, v in summary([r[k] for r in rows]).items()},
    }


def export_human_csv(rows: list[dict], path, k: int = 30, seed: int = 0) -> None:
    """Random sample for blind 1-5 rating (naturel, fidelite). 'system' is deliberately left out."""
    sample = random.Random(seed).sample(rows, min(k, len(rows)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "sortie", "naturel_1_5", "fidelite_1_5", "commentaire"])
        w.writerows([r["id"], r["source"], r["output"], "", "", ""] for r in sample)

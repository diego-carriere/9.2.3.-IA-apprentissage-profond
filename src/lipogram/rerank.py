"""Best-of-n: keep lipogram, non-degenerate candidates and pick the best composite score."""

import math

from lipogram.rules import is_lipogram

DEFAULT_WEIGHTS = {"sem_sim": 0.5, "fluency": 0.3, "length": 0.2}


def length_score(ratio: float, lo: float = 0.7, hi: float = 1.3) -> float:
    """1 inside [lo, hi], linear decay to 0 at 0.5 outside."""
    return max(0.0, 1.0 - max(lo - ratio, ratio - hi, 0.0) / 0.5)


def composite(c: dict, weights: dict = DEFAULT_WEIGHTS) -> float:
    # exp(mean log-prob) = geometric-mean token probability, already in [0, 1]
    flu = math.exp(c["fluency"]) if c["fluency"] == c["fluency"] else 0.0
    return weights["sem_sim"] * c["sem_sim"] + weights["fluency"] * flu + weights["length"] * length_score(c["len_ratio"])


def best_of_n(candidates: list[dict], weights: dict = DEFAULT_WEIGHTS, strict: bool = True) -> dict:
    """candidates: dicts with output, sem_sim, fluency, len_ratio, degenerate. Never empty-handed: falls back
    to degenerate (or, if not strict, non-lipogram) candidates when nothing better exists.
    strict=True (masked generation): a forbidden char means the mask is broken, so raise."""
    lipo = [c for c in candidates if is_lipogram(c["output"])]
    if strict and len(lipo) < len(candidates):
        bad = next(c["output"] for c in candidates if c not in lipo)
        raise AssertionError(f"masked generation produced a forbidden char: {bad!r}")
    pool = lipo or candidates
    pool = [c for c in pool if c["degenerate"] is None] or pool
    return max(pool, key=lambda c: composite(c, weights))

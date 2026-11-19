"""Evaluate one configuration on the eval set: python scripts/eval.py --config configs/baseline.yaml

Generates max(n_values) candidates per text once; best-of-n for each n uses the first n candidates
(candidates are i.i.d. samples, so this is a fair estimate at a fraction of the cost).
Exit code 1 if a masked run is not 100 % lipogram (hard acceptance criterion).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lipogram.generate import generate, load  # noqa: E402
from lipogram.metrics import (EMBEDDER, degenerate_reason, export_human_csv, fluency, len_ratio,  # noqa: E402
                              load_embedder, score_rows, sem_sim)
from lipogram.rerank import DEFAULT_WEIGHTS, best_of_n  # noqa: E402

METRICS = ["lipogram_rate", "sem_sim_mean", "sem_sim_p10", "fluency_mean", "len_ratio_median", "len_in_range",
           "degenerate_rate"]


def candidate_rows(item, cands, model, tok, embedder):
    outs = [c["text"] for c in cands]
    sims = sem_sim([item["text"]] * len(outs), outs, embedder)
    flus = fluency(outs, model, tok)
    return [{"id": item["id"], "kind": item.get("kind", ""), "source": item["text"], "output": c["text"],
             "sem_sim": s, "fluency": f, "len_ratio": len_ratio(item["text"], c["text"]),
             "degenerate": degenerate_reason(item["text"], c["text"], c["truncated"])}
            for c, s, f in zip(cands, sims, flus)]


def markdown_table(results: dict) -> str:
    lines = ["| n | " + " | ".join(METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for n, r in results.items():
        lines.append(f"| {n} | " + " | ".join(f"{r[m]:.3f}" for m in METRICS) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text(encoding="utf-8"))
    torch.manual_seed(cfg["seed"])
    items = [json.loads(line) for f in cfg["eval_files"] for line in Path(f).read_text(encoding="utf-8").splitlines()
             if line.strip()][:cfg.get("limit")]
    use_mask = cfg.get("use_mask", True)
    model, tok, mask = load(cfg["model"], cfg.get("dtype", "auto"), cfg.get("load_in_4bit", False))
    embedder = load_embedder(cfg.get("embedder", EMBEDDER), "cuda" if torch.cuda.is_available() else "cpu")
    n_values, weights = sorted(cfg["n_values"]), cfg.get("rerank_weights", DEFAULT_WEIGHTS)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    per_item, gen_seconds = [], []
    for k, item in enumerate(items):
        t0 = time.perf_counter()
        cands = generate(model, tok, item["text"], mask if use_mask else None, n=n_values[-1], seed=cfg["seed"] + k,
                         **cfg.get("generation", {}))
        gen_seconds.append(time.perf_counter() - t0)
        per_item.append(candidate_rows(item, cands, model, tok, embedder))
        print(f"[{k + 1}/{len(items)}] {gen_seconds[-1]:.1f}s  {per_item[-1][0]['output'][:80]!r}", flush=True)

    results = {n: score_rows([best_of_n(rows[:n], weights, strict=use_mask) for rows in per_item]) for n in n_values}
    by_kind = {kind: score_rows([best_of_n(rows[:n_values[-1]], weights, strict=use_mask)
                                 for rows in per_item if rows[0]["kind"] == kind])
               for kind in sorted({rows[0]["kind"] for rows in per_item})}
    chosen = [best_of_n(rows[:cfg.get("human_n", n_values[-1])], weights, strict=use_mask) for rows in per_item]
    meta = {"config": cfg, "n_items": len(items), "gen_seconds_per_text": statistics.fmean(gen_seconds),
            "candidates_per_text": n_values[-1],
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}

    with open(out_dir / "candidates.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for rows in per_item for r in rows)
    (out_dir / "metrics.json").write_text(json.dumps({"meta": meta, "best_of_n": results, "by_kind": by_kind},
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "table.md").write_text(markdown_table(results) + "\n", encoding="utf-8")
    export_human_csv(chosen, out_dir / "human_rating.csv", seed=cfg["seed"])
    print(markdown_table(results))
    print(f"{meta['gen_seconds_per_text']:.1f}s/text for {n_values[-1]} candidates; peak VRAM {meta['peak_vram_gb']}")

    if use_mask and any(r["lipogram_rate"] < 1.0 for r in results.values()):
        print("FAIL: masked run is not 100 % lipogram", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

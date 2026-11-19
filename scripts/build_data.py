"""Mode A training data.

python scripts/build_data.py corpus                 # sources -> data/processed/sources.jsonl
python scripts/build_data.py pairs --limit 50       # teacher rewrites -> data/processed/sft/*.jsonl

`pairs` is resumable: every teacher answer is cached on disk, so re-running never re-asks a batch.
It stops cleanly at teacher.max_calls or on a teacher error (e.g. subscription rate limit).
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lipogram.data_build import (BudgetExceeded, Teacher, TeacherError, clean_corpus, gutenberg_sources,  # noqa: E402
                                 rejection_reason, rewrite_batch, wikipedia_sources)
from lipogram.metrics import len_ratio, load_embedder, sem_sim  # noqa: E402
from lipogram.rules import is_lipogram  # noqa: E402

SOURCES = Path("data/processed/sources.jsonl")
SFT = Path("data/processed/sft")


def eval_rows() -> list[dict]:
    return [json.loads(line) for f in sorted(Path("data/eval").glob("*.jsonl")) for line in f.open(encoding="utf-8")]


def build_corpus(cfg: dict) -> None:
    c = cfg["corpus"]
    eval_books = {r["source"]["gutenberg_id"] for r in eval_rows()}
    clash = eval_books & {b["id"] for b in c["gutenberg_books"]}
    assert not clash, f"eval books must never be used for training: {clash}"
    rng = random.Random(c["seed"])
    items = gutenberg_sources(c["gutenberg_books"], rng, c["min_words"], c["max_words"], c["para_share"])
    items += wikipedia_sources(c["wikipedia_n"], rng, c["min_words"], c["max_words"])
    rng.shuffle(items)
    kept, dropped = clean_corpus(items, [r["text"] for r in eval_rows()])
    SOURCES.parent.mkdir(parents=True, exist_ok=True)
    SOURCES.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    print(f"{len(items)} raw -> {len(kept)} kept; dropped {dict(dropped)}; "
          f"splits {dict(Counter(r['split'] for r in kept))}; domains {dict(Counter(r['domain'] for r in kept))}")


def build_pairs(cfg: dict, limit: int | None) -> None:
    t, f = cfg["teacher"], cfg["filter"]
    sources = [json.loads(line) for line in SOURCES.open(encoding="utf-8")][:limit]
    teacher = Teacher(t["model"], Path(t["cache_dir"]), t["max_calls"], t.get("effort", "low"))
    done, stop_reason = [], None
    for i in range(0, len(sources), t["batch_size"]):
        batch = sources[i:i + t["batch_size"]]
        try:
            outs = rewrite_batch(teacher, [s["text"] for s in batch], t["max_repairs"])
        except (BudgetExceeded, TeacherError) as e:
            stop_reason = str(e)
            print(f"stopped at batch {i // t['batch_size']}: {e}", file=sys.stderr)
            break
        done += [{**s, **o} for s, o in zip(batch, outs)]
        print(f"{len(done)}/{len(sources)} rewritten ({teacher.calls} new teacher calls)", flush=True)

    embedder = load_embedder()
    sims = sem_sim([d["source"] for d in done], [d["target"] for d in done], embedder) if done else []
    kept, rejected = {"train": [], "val": [], "test": []}, []
    for d, s in zip(done, sims):
        reason = rejection_reason(d["source"], d["target"], s, f)
        row = {"id": d["id"], "source": d["source"], "target": d["target"],
               "meta": {"origin": "teacher", "sem_sim": round(s, 4), "len_ratio": round(len_ratio(d["source"], d["target"]), 4),
                        "domain": d["domain"], "repairs": d["repairs"], "attribution": d["attribution"]}}
        (rejected.append({**row, "reason": reason}) if reason else kept[d["split"]].append(row))

    SFT.mkdir(parents=True, exist_ok=True)
    for split, rows in kept.items():
        (SFT / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (SFT / "rejected.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected), encoding="utf-8")
    needed = [d for d in done if d["repairs"] > 0]
    report = {
        "rewritten": len(done), "kept": sum(map(len, kept.values())), "kept_by_split": {k: len(v) for k, v in kept.items()},
        "rejected_by_reason": dict(Counter(r["reason"] for r in rejected)),
        "lipogram_first_try": len(done) - len(needed),
        "needed_repair": len(needed), "repair_success": sum(is_lipogram(d["target"]) for d in needed),
        "new_teacher_calls": teacher.calls,
        "indicative_cost_usd_not_billed": round(teacher.indicative_cost, 4),
        "sem_sim": {"min": min(sims, default=None), "mean": sum(sims) / len(sims) if sims else None},
        "stop_reason": stop_reason,
    }
    (SFT / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["corpus", "pairs"])
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    build_corpus(cfg) if args.step == "corpus" else build_pairs(cfg, args.limit)


if __name__ == "__main__":
    main()

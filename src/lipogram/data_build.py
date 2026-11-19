"""Training data (mode A): source corpus -> teacher rewrites (claude -p) -> filter/repair -> SFT pairs.

The teacher runs through the Claude Code CLI with the user's claude.ai subscription login: no API key
is ever used (API-key variables are stripped from the subprocess and user settings are not loaded).
"""

import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from lipogram.gutenberg import SENT_SPLIT, WORD, clean_sentence, fetch, ok_text, paragraphs
from lipogram.metrics import degenerate_reason, len_ratio, unknown_words
from lipogram.rules import forbidden_positions, is_lipogram

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
API_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")
FRENCH_FUNCTION_WORDS = set("le la les de des du et un une est que qui dans pour sur pas il elle ne se au".split())


# ---------------------------------------------------------------- source corpus

def normalize(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def shingles(text: str, n: int = 8) -> set[tuple[str, ...]]:
    w = normalize(text).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)} or {tuple(w)}


def looks_french(text: str) -> bool:
    words = normalize(text).split()
    return len(words) > 0 and sum(w in FRENCH_FUNCTION_WORDS for w in words) / len(words) >= 0.1


def split_of(text: str) -> str:
    """Deterministic 90/5/5 split from the text hash."""
    b = int(hashlib.sha256(normalize(text).encode()).hexdigest(), 16) % 100
    return "train" if b < 90 else "val" if b < 95 else "test"


def gutenberg_sources(books: list[dict], rng: random.Random, min_words: int, max_words: int,
                      para_share: float) -> list[dict]:
    out = []
    for b in books:
        paras = paragraphs(fetch(b["id"], b["file"]))
        sents = [clean_sentence(s) for p in paras for s in SENT_SPLIT.split(p)]
        sents = [s for s in sents if ok_text(s, min_words, 40)]
        paras = [p for p in paras if ok_text(p, 41, max_words)]
        n_para = min(len(paras), int(b["n"] * para_share))
        picked = rng.sample(paras, n_para) + rng.sample(sents, min(len(sents), b["n"] - n_para))
        out += [{"text": t, "domain": f"gutenberg:{b['genre']}",
                 "attribution": f"{b['author']}, {b['title']} (Project Gutenberg #{b['id']}, domaine public)"}
                for t in picked]
    return out


def _get_json(url: str, attempts: int = 6) -> dict:
    """GET with backoff on HTTP 429/5xx (honours Retry-After)."""
    for k in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "lipogram-course-project/0.1"})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or k == attempts - 1:
                raise
            time.sleep(float(e.headers.get("Retry-After") or 5 * 2 ** k))


def wikipedia_sources(n: int, rng: random.Random, min_words: int, max_words: int, pause: float = 1.0) -> list[dict]:
    """Random fr.wikipedia article intros (CC BY-SA 4.0), split into sentences/short paragraphs."""
    out, seen_titles = [], set()
    while len(out) < n:
        q = urllib.parse.urlencode({"action": "query", "format": "json", "generator": "random", "grnnamespace": 0,
                                    "grnlimit": 20, "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": 20})
        pages = _get_json(f"https://fr.wikipedia.org/w/api.php?{q}").get("query", {}).get("pages", {}).values()
        for page in pages:
            if page["title"] in seen_titles or not page.get("extract"):
                continue
            seen_titles.add(page["title"])
            text = re.sub(r"\s+", " ", page["extract"]).strip()
            words = len(WORD.findall(text))
            chunks = [text] if min_words <= words <= max_words else SENT_SPLIT.split(text)
            for c in chunks:
                if ok_text(c, min_words, max_words):
                    out.append({"text": c, "domain": "wikipedia", "attribution":
                                f"Wikipédia, « {page['title']} » (CC BY-SA 4.0)"})
        time.sleep(pause)
    rng.shuffle(out)
    return out[:n]


def clean_corpus(items: list[dict], eval_texts: list[str]) -> tuple[list[dict], Counter]:
    """French filter, near-duplicate removal (shared 8-gram) and eval-overlap removal (shared 8-gram)."""
    eval_index = set().union(*(shingles(t) for t in eval_texts))
    index, kept, dropped = set(), [], Counter()
    for it in items:
        sh = shingles(it["text"])
        if not looks_french(it["text"]):
            dropped["not_french"] += 1
        elif sh & eval_index:
            dropped["eval_overlap"] += 1
        elif sh & index:
            dropped["near_duplicate"] += 1
        else:
            index |= sh
            kept.append({**it, "id": hashlib.sha256(normalize(it["text"]).encode()).hexdigest()[:12],
                         "split": split_of(it["text"])})
    return kept, dropped


# ---------------------------------------------------------------- teacher

class BudgetExceeded(Exception):
    pass


class TeacherError(Exception):
    pass


def claude_cli(system: str, prompt: str, model: str, effort: str = "low", timeout: int = 900) -> dict:
    """One headless Claude Code call on the subscription login (API-key env stripped, no settings/tools/MCP)."""
    env = {k: v for k, v in os.environ.items() if k not in API_ENV_VARS}
    cmd = ["claude", "-p", "--output-format", "json", "--model", model, "--effort", effort, "--system-prompt", system,
           "--tools", "", "--no-session-persistence", "--strict-mcp-config", "--setting-sources", ""]
    with tempfile.TemporaryDirectory() as cwd:  # outside the repo: no CLAUDE.md picked up
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout)
    try:
        out = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise TeacherError(f"claude exited {r.returncode}: {(r.stderr or r.stdout)[:300]}")
    if out.get("is_error") or r.returncode:
        raise TeacherError(f"claude error: {str(out.get('result'))[:300]}")
    return out


class Teacher:
    """Cached, budgeted teacher. `runner(system, prompt, model, effort) -> {"result": str, ...}` is injectable."""

    def __init__(self, model: str, cache_dir: Path, max_calls: int, effort: str = "low", runner=claude_cli):
        self.model, self.effort, self.cache_dir, self.max_calls, self.runner = model, effort, Path(cache_dir), max_calls, runner
        self.calls = 0
        self.indicative_cost = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def ask(self, system: str, prompt: str) -> str:
        key = hashlib.sha256(f"{self.model}\0{self.effort}\0{system}\0{prompt}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["result"]
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"max_calls={self.max_calls} reached")
        self.calls += 1
        out = self.runner(system, prompt, self.model, self.effort)
        self.indicative_cost += out.get("total_cost_usd") or 0.0
        path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")  # written only on success
        return out["result"]


def teacher_system() -> str:
    rules = (PROMPTS_DIR / "system.txt").read_text(encoding="utf-8").strip()
    shots = [json.loads(line) for line in (PROMPTS_DIR / "fewshot.jsonl").read_text(encoding="utf-8").splitlines()]
    examples = "\n".join(f"- {s['source']}\n  → {s['target']}" for s in shots)
    return (f"{rules}\n\nExemples :\n{examples}\n\n"
            'Tu reçois un objet JSON {"1": texte, "2": texte, ...}. Réponds UNIQUEMENT par un objet JSON avec les mêmes '
            "clés, chaque valeur étant la réécriture du texte correspondant. Aucun autre texte. "
            "Vérifie chaque mot lettre par lettre : un seul « e » rend la réécriture inutilisable.")


def parse_json_object(text: str, keys: list[str]) -> dict[str, str]:
    """Answers for the expected keys; missing or malformed entries are simply absent (re-asked later)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        obj = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        obj = None
    if not isinstance(obj, dict):
        raise TeacherError(f"teacher answer is not a JSON object: {text[:200]!r}")
    return {k: obj[k].strip() for k in keys if isinstance(obj.get(k), str) and obj[k].strip()}


def offending_words(text: str) -> list[str]:
    """Words containing a forbidden char, with their character offset, e.g. ['le@0', 'chemin@3']."""
    bad = set(forbidden_positions(text))
    return [f"{m.group(0)}@{m.start()}" for m in WORD.finditer(text) if bad & set(range(m.start(), m.end()))]


def needs_repair(source: str, target: str | None) -> bool:
    return not target or not is_lipogram(target) or bool(unknown_words(target, source))


def capitalize_like(source: str, target: str) -> str:
    """Restore the leading capital the teacher sometimes drops (never adds an « e »)."""
    i = next((k for k, c in enumerate(target) if c.isalpha()), None)
    if i is None or not source.lstrip("«—–- ")[:1].isupper():
        return target
    return target[:i] + target[i].upper() + target[i + 1:]


def repair_prompt(items: dict[str, tuple[str, str | None]]) -> str:
    payload = {k: {"source": s, "ta_version": t, "mots_avec_e": offending_words(t) if t else [],
                   "mots_inexistants": unknown_words(t, s) if t else []}
               for k, (s, t) in items.items()}
    return ("Corrige ces réécritures. Remplace les mots listés dans mots_avec_e (mot@position, ils contiennent un « e ») "
            "et dans mots_inexistants (ils n'existent pas en français : utilise un vrai mot courant, quitte à "
            "reformuler) ; ne change que le nécessaire. Si ta_version est null, réécris la source. Le résultat doit être "
            "sans aucun « e », en vrais mots français, avec les majuscules de la source (noms propres, débuts de "
            "phrase), fidèle à la source. Réponds par un objet JSON avec les mêmes clés.\n"
            + json.dumps(payload, ensure_ascii=False))


def rewrite_batch(teacher: Teacher, sources: list[str], max_repairs: int = 2) -> list[dict]:
    """Teacher rewrites + up to max_repairs repair rounds for outputs that are missing, contain an « e »,
    or use words absent from the French lexicon."""
    system = teacher_system()
    keys = [str(i + 1) for i in range(len(sources))]
    got = parse_json_object(teacher.ask(system, json.dumps(dict(zip(keys, sources)), ensure_ascii=False)), keys)
    targets = [capitalize_like(s, got[k]) if k in got else None for s, k in zip(sources, keys)]
    repairs = [0] * len(sources)
    for _ in range(max_repairs):
        todo = {k: (sources[i], targets[i]) for i, k in enumerate(keys) if needs_repair(sources[i], targets[i])}
        if not todo:
            break
        fixed = parse_json_object(teacher.ask(system, repair_prompt(todo)), list(todo))
        for k, t in fixed.items():
            i = int(k) - 1
            targets[i], repairs[i] = capitalize_like(sources[i], t), repairs[i] + 1
    return [{"source": s, "target": t or "", "repairs": r} for s, t, r in zip(sources, targets, repairs)]


# ---------------------------------------------------------------- filtering

def rejection_reason(source: str, target: str, sim: float, cfg: dict) -> str | None:
    if not target:
        return "missing"
    if not is_lipogram(target):
        return "has_e"
    if len(unknown_words(target, source)) > cfg.get("max_unknown_words", 0):
        return "unknown_words"
    if sim < cfg["min_sem_sim"]:
        return "low_sem_sim"
    if not cfg["len_ratio_min"] <= len_ratio(source, target) <= cfg["len_ratio_max"]:
        return "len_ratio"
    if (reason := degenerate_reason(source, target)) is not None:
        return reason
    return None

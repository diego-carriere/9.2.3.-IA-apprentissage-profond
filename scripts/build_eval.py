"""Build the frozen eval set from public-domain French books (Project Gutenberg mirror).

python scripts/build_eval.py   ->  data/eval/{phrases,paragraphes,pieges}.jsonl + MANIFEST.sha256

Books are hand-picked (authors dead before 1915, public domain in France and the US) and
tagged with a genre. These books must never be used for training (data_build excludes them).
"""

import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lipogram.gutenberg import WORD, SENT_SPLIT, clean_sentence, fetch, ok_text, paragraphs  # noqa: E402
from lipogram.rules import count_forbidden  # noqa: E402

OUT = Path("data/eval")
SEED = 20261119
N_SENT_PER_GENRE, N_PARA_PER_GENRE, N_TRAPS, MAX_TRAPS_PER_BOOK = 40, 10, 30, 3

# (gutenberg id, mirror file, author, title, genre)
BOOKS = [
    (6838, "6838-8.txt", "Victor Hugo", "Le Dernier Jour d'un condamné", "narratif"),
    (5781, "5781-0.txt", "Alain-Fournier", "Le Grand Meaulnes", "narratif"),
    (14155, "14155-0.txt", "Gustave Flaubert", "Madame Bovary", "narratif"),
    (11049, "11049-0.txt", "Honoré de Balzac", "Eugénie Grandet", "narratif"),
    (34204, "34204-8.txt", "George Sand", "La Petite Fadette", "narratif"),
    (8712, "8712-8.txt", "Émile Zola", "La Conquête de Plassans", "narratif"),
    (4791, "4791-0.txt", "Jules Verne", "Voyage au centre de la Terre", "descriptif"),
    (5097, "5097-0.txt", "Jules Verne", "Vingt mille lieues sous les mers", "descriptif"),
    (19657, "19657-8.txt", "Victor Hugo", "Notre-Dame de Paris", "descriptif"),
    (18812, "18812-8.txt", "Arnauld d'Abbadie", "Douze ans de séjour dans la Haute-Éthiopie", "descriptif"),
    (13951, "13951-0.txt", "Alexandre Dumas", "Les Trois Mousquetaires", "dialogue"),
    (7442, "7442-0.txt", "Jules Verne", "Michel Strogoff", "dialogue"),
    (3456, "3456-8.txt", "Jules Verne", "Le Tour du monde en quatre-vingts jours", "dialogue"),
    (13704, "13704-0.txt", "Edmond About", "L'Homme à l'oreille cassée", "dialogue"),
    (30513, "30513-0.txt", "Alexis de Tocqueville", "De la démocratie en Amérique", "argumentatif"),
    (13846, "13846-0.txt", "René Descartes", "Œuvres", "argumentatif"),
    (4650, "4650-0.txt", "Voltaire", "Candide, ou l'Optimisme", "argumentatif"),
    (16234, "16234-8.txt", "Claude Bernard", "Introduction à l'étude de la médecine expérimentale", "technique"),
    (799, "799-0.txt", "Jules Verne", "De la Terre à la Lune", "technique"),
]

NUMBER_WORDS = set("deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize vingt "
                   "trente quarante cinquante soixante cent cents mille million millions".split())
def spoken(p: str) -> bool:
    return p.startswith(("«", "—"))


def dialogue_chunks(paras: list[str], lo: int = 60, hi: int = 180) -> list[str]:
    """Runs of consecutive spoken lines (≥ 2), one line per reply, lo-hi words."""
    chunks, run = [], []
    for p in paras + [""]:
        n = sum(len(WORD.findall(x)) for x in run)
        if spoken(p) and n + len(WORD.findall(p)) <= hi:
            run.append(p)
            continue
        if len(run) >= 2 and n >= lo:
            chunks.append("\n".join(run))
        run = [p] if spoken(p) else []
    return chunks


def trap_score(s: str) -> tuple[int, int]:
    words = WORD.findall(s)
    names = sum(1 for w in words[1:] if w[0].isupper() and count_forbidden(w))
    numbers = sum(1 for w in words if w.lower() in NUMBER_WORDS)
    plurals = sum(1 for w in words if w.lower().endswith("ent") and len(w) > 4)
    quote = int("«" in s)
    feats = [names >= 2, numbers >= 1, plurals >= 2, quote == 1]
    return sum(feats), names + 2 * numbers + plurals + quote


def main() -> None:
    rng = random.Random(SEED)
    pool = {}  # genre -> list of (book, sentences, paragraphs)
    for book in BOOKS:
        paras = paragraphs(fetch(book[0], book[1]))
        if book[4] == "dialogue":  # keep only spoken lines; paragraphs = runs of replies
            sents = [clean_sentence(s) for p in paras if spoken(p) for s in SENT_SPLIT.split(p)]
            paras = dialogue_chunks(paras)
        else:
            sents = [clean_sentence(s) for p in paras for s in SENT_SPLIT.split(p)]
        pool.setdefault(book[4], []).append((book, [s for s in sents if ok_text(s, 8, 40)],
                                             [p for p in paras if ok_text(p, 60, 180)]))
        print(f"{book[0]:>6} {book[3][:40]:<40} sentences={len(pool[book[4]][-1][1]):>5} "
              f"paragraphs={len(pool[book[4]][-1][2]):>4}")

    def take(genre: str, idx: int, k: int) -> list[dict]:
        """k items spread round-robin over the genre's books."""
        books = [(b, rng.sample(items[idx], len(items[idx]))) for b, *items in [(b, s, p) for b, s, p in pool[genre]]]
        picked, i = [], 0
        while len(picked) < k:
            b, items = books[i % len(books)]
            if items:
                picked.append((b, items.pop()))
            elif all(not it for _, it in books):
                raise ValueError(f"not enough texts for {genre}")
            i += 1
        return picked

    seen, rows = set(), {"phrases": [], "paragraphes": [], "pieges": []}

    def add(kind: str, genre: str, book, text: str) -> None:
        seen.add(text)
        rows[kind].append({"id": f"{kind[:4]}-{len(rows[kind]) + 1:03d}", "kind": kind, "genre": genre, "text": text,
                           "source": {"gutenberg_id": book[0], "author": book[2], "title": book[3]},
                           "license": "Domaine public (Project Gutenberg)"})

    for genre in pool:
        for book, s in take(genre, 0, N_SENT_PER_GENRE):
            add("phrases", genre, book, s)
        for book, p in take(genre, 1, N_PARA_PER_GENRE):
            add("paragraphes", genre, book, p)

    traps = [(trap_score(s), book, s) for g in pool for book, sents, _ in pool[g] for s in sents
             if s not in seen and ok_text(s, 10, 50)]
    traps = [t for t in traps if t[0][0] >= 2]
    rng.shuffle(traps)
    traps.sort(key=lambda t: t[0], reverse=True)
    per_book = {}
    for _, book, s in traps:
        if sum(per_book.values()) == N_TRAPS:
            break
        if per_book.get(book[0], 0) < MAX_TRAPS_PER_BOOK:
            per_book[book[0]] = per_book.get(book[0], 0) + 1
            add("pieges", book[4], book, s)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for kind, items in rows.items():
        path = OUT / f"{kind}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")
        manifest.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        print(f"{path}: {len(items)}")
    (OUT / "MANIFEST.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print("eval set hash:", hashlib.sha256("\n".join(manifest).encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()

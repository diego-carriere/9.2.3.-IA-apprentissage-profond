"""Project Gutenberg helpers shared by the eval-set and training-corpus builders."""

import re
import urllib.request
from pathlib import Path

from lipogram.rules import count_forbidden

MIRROR = "http://mirrors.xmission.com/gutenberg"  # www.gutenberg.org was down (503) when this was written
RAW = Path("data/raw/gutenberg")
SENT_SPLIT = re.compile(r"(?<=[.!?…])(?:\s*»)?\s+(?=[«—–-]?\s*[A-ZÀÂÇÉÈÊÎÔÛ])")
WORD = re.compile(r"[\wÀ-ÿ'’-]+")


def mirror_path(book_id: int) -> str:
    s = str(book_id)
    return "/".join(s[:-1]) + f"/{s}/"


def fetch(book_id: int, fname: str) -> str:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{book_id}.txt"
    if not path.exists():
        req = urllib.request.Request(MIRROR + "/" + mirror_path(book_id) + fname,
                                     headers={"User-Agent": "lipogram-course-project/0.1"})
        raw = urllib.request.urlopen(req, timeout=60).read()
        path.write_text(raw.decode("latin-1" if fname.endswith("-8.txt") else "utf-8"), encoding="utf-8")
    return path.read_text(encoding="utf-8")


def paragraphs(text: str) -> list[str]:
    """Body paragraphs: Gutenberg boilerplate, front/back matter, headings and notes removed."""
    text = text.replace("\r", "")
    start = re.search(r"\*\*\* ?START OF.*\n", text)
    end = re.search(r"\*\*\* ?END OF", text)
    body = text[start.end() if start else 0:end.start() if end else len(text)]
    paras = [re.sub(r"\s+", " ", p.replace("_", "").replace("--", "—")).strip() for p in re.split(r"\n\s*\n", body)]
    paras = [p for p in paras if p and not p.isupper() and not re.search(r"\[\d+\]|\[Illustration|http|\*\*\*", p)]
    return paras[len(paras) // 20: len(paras) - len(paras) // 50]  # skip preface / end matter



def clean_sentence(s: str) -> str:
    return s.strip().lstrip("—–- ").strip()


def ok_text(t: str, lo: int, hi: int) -> bool:
    n = len(WORD.findall(t))
    return (lo <= n <= hi and (t[0].isupper() or t[0] in "«—") and t[-1] in ".!?…»" and count_forbidden(t) > 0
            and not re.search(r"[0-9]{3,}|[\[\]{}<>|#*=]|[^\W\d_]\d|\d[^\W\d_'’]|\d['’]", t)
            and t.count("«") == t.count("»"))

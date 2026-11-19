"""Write docs/reports/token_mask.md: mask statistics for the candidate tokenizers."""

import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lipogram.token_mask import FRENCH_LETTERS, load_or_build_mask, token_bytes  # noqa: E402

TOKENIZERS = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-0.6B", "hf-internal-testing/llama-tokenizer"]
WORDS = ("un au du ma mon son nos vos ou mais donc car ainsi dans sur par pour sans qui dont avant aussi "
         "plus tout tous sont ont font vont à où là ça").split()


def section(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    mask = load_or_build_mask(tok)
    tb = token_bytes(tok)
    shown = lambda i: repr(tb[i].decode("utf-8", errors="replace"))
    allowed = [i for i in range(len(tok)) if not mask[i] and tb[i]]
    words = [i for i in allowed if len(tb[i].strip()) >= 2 and tb[i].strip().isalpha()]
    shortest = sorted(words, key=lambda i: (len(tb[i]), i))[:50]
    frequent = words[:50]  # byte-level BPE ids follow merge order, a proxy for corpus frequency
    rows = []
    for w in WORDS:
        for form in (w, " " + w):
            ids = tok.encode(form, add_special_tokens=False)
            rows.append(f"| `{form!r}` | {[tok.convert_ids_to_tokens(i) for i in ids]} | "
                        f"{'oui' if not any(mask[i] for i in ids) else '**NON**'} |")
    letters_ok = all(not mask[i] for c in FRENCH_LETTERS for f in (c, " " + c)
                     for i in tok.encode(f, add_special_tokens=False))
    specials = [tok.convert_ids_to_tokens(i) for i in tok.added_tokens_decoder if not mask[i]]
    text = f"""## `{name}`

- Taille du vocabulaire : {len(tok)} ; tokens interdits : **{int(mask.sum())}** ({mask.float().mean():.1%})
- Tokens spéciaux autorisés : {specials}
- Lettres `{FRENCH_LETTERS}` représentables (seules et précédées d'un espace) : **{'oui' if letters_ok else 'NON'}**

50 tokens autorisés les plus courts (mots ≥ 2 lettres) : {', '.join(shown(i) for i in shortest)}

50 tokens autorisés les plus fréquents (approximation : ordre des fusions BPE) : {', '.join(shown(i) for i in frequent)}

| Mot outil | Tokens | Tous autorisés |
|---|---|---|
""" + "\n".join(rows) + "\n"
    return text, mask


def main() -> None:
    parts, masks = [], {}
    for name in TOKENIZERS:
        text, masks[name] = section(name)
        parts.append(text)
    same = bool((masks[TOKENIZERS[0]] == masks[TOKENIZERS[1]]).all())
    header = (f"# Rapport — masque de tokens\n\nGénéré par `python scripts/mask_report.py`.\n\n"
              f"Masques Qwen3-4B-Instruct-2507 et Qwen3-0.6B identiques : **{'oui' if same else 'NON'}** "
              f"(ce qui justifie le développement sur CPU avec le 0.6B).\n\n")
    out = Path("docs/reports/token_mask.md")
    out.write_text(header + "\n".join(parts), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

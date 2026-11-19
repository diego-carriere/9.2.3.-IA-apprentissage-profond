import math

import pytest
import torch

from lipogram.metrics import degenerate_reason, export_human_csv, fluency, len_ratio, score_rows, summary
from lipogram.rerank import best_of_n, composite, length_score

SRC = "Le chat dort sur le canapé depuis ce matin."


@pytest.mark.parametrize("out,truncated,reason", [
    ("", False, "empty"), ("...", False, "empty"),
    ("Mon chat dort sur un divan.", True, "truncated"),
    ("a. a. a. a. a. a. a. a.", False, "repetitive"),
    ("Voici : mon chat dort sur un divan.", False, "meta"),
    ("chat dort", False, "partial_copy"),
    ("Mon chat dort sur un divan, tout au long du matin.", False, None),
])
def test_degenerate(out, truncated, reason):
    assert degenerate_reason(SRC, out, truncated) == reason


def test_length_score():
    assert length_score(1.0) == 1 and length_score(0.7) == 1 and length_score(1.3) == 1
    assert length_score(0.45) == pytest.approx(0.5) and length_score(2.0) == 0


def test_summary_and_rows():
    assert summary([1, 2, 3, float("nan")])["median"] == 2
    rows = [{"source": SRC, "output": "Mon chat dort.", "sem_sim": 0.8, "len_ratio": 0.3, "fluency": -2.0, "degenerate": None},
            {"source": SRC, "output": "Un matou", "sem_sim": 0.4, "len_ratio": 1.0, "fluency": -3.0, "degenerate": "x"}]
    s = score_rows(rows)
    assert s["lipogram_rate"] == 1 and s["degenerate_rate"] == 0.5 and s["len_in_range"] == 0.5
    assert s["sem_sim_mean"] == pytest.approx(0.6)


def _cand(out, sim, flu=-2.0, ratio=1.0, deg=None):
    return {"output": out, "sem_sim": sim, "fluency": flu, "len_ratio": ratio, "degenerate": deg}


def test_best_of_n_prefers_quality_and_skips_degenerate():
    good, worse, broken = _cand("a", 0.9), _cand("b", 0.5), _cand("c", 0.99, deg="repetitive")
    assert best_of_n([worse, broken, good]) is good
    assert best_of_n([broken]) is broken  # fallback when everything is degenerate
    assert composite(good) > composite(worse)
    assert composite(_cand("d", 0.9, flu=float("nan"))) == pytest.approx(0.5 * 0.9 + 0.2)


def test_best_of_n_rejects_forbidden_char():
    with pytest.raises(AssertionError):
        best_of_n([_cand("le chat", 0.9)])


def test_len_ratio():
    assert len_ratio("abcd", "ab") == 0.5 and len_ratio("", "a") == 1


@pytest.mark.slow
def test_fluency_tiny_lm():
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=1, head_dim=8)).eval()
    texts = ["Un chat dort.", "Mon chat dort sur un divan, tout au long du matin.", "a", ""]
    scores = fluency(texts, model, tok, batch_size=2)
    assert all(s < 0 for s in scores[:2]) and math.isnan(scores[2]) and math.isnan(scores[3])
    # padding must not change a text's score
    assert fluency(texts[:1], model, tok)[0] == pytest.approx(scores[0], abs=1e-4)


def test_export_human_csv(tmp_path):
    rows = [{"id": i, "source": "s", "output": "o"} for i in range(50)]
    export_human_csv(rows, tmp_path / "h.csv", k=30)
    lines = (tmp_path / "h.csv").read_text().splitlines()
    assert len(lines) == 31 and "system" not in lines[0]


def test_best_of_n_non_strict_prefers_lipograms():
    lipo, other = _cand("un chat", 0.5), _cand("le chat", 0.99)
    assert best_of_n([other, lipo], strict=False) is lipo
    assert best_of_n([other], strict=False) is other

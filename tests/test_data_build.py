import json
import subprocess

import pytest

from lipogram import data_build as db
from lipogram.data_build import (BudgetExceeded, Teacher, TeacherError, clean_corpus, offending_words,
                                 parse_json_list, rejection_reason, rewrite_batch, split_of)

FILTER = {"min_sem_sim": 0.8, "len_ratio_min": 0.7, "len_ratio_max": 1.3}


class FakeRunner:
    """Answers batches with canned rewrites; the first answer for 'bad' keeps an « e » to force a repair."""

    def __init__(self):
        self.prompts = []

    def __call__(self, system, prompt, model):
        self.prompts.append(prompt)
        if prompt.startswith("Ces réécritures"):
            payload = json.loads(prompt.split("\n", 1)[1])
            return {"result": json.dumps(["Un chat dort sur un divan." for _ in payload]), "total_cost_usd": 0.01}
        sources = json.loads(prompt)
        return {"result": json.dumps(["Un chat qui dort, je crois." if "bad" in s else "Mon chat dort sur un divan." for s in sources]),
                "total_cost_usd": 0.02}


def test_teacher_caches_and_caps(tmp_path):
    runner = FakeRunner()
    t = Teacher("m", tmp_path, max_calls=1, runner=runner)
    assert t.ask("sys", '["a"]') == t.ask("sys", '["a"]')  # second answer from cache
    assert len(runner.prompts) == 1 and t.indicative_cost == pytest.approx(0.02)
    with pytest.raises(BudgetExceeded):
        t.ask("sys", '["b"]')
    assert Teacher("m", tmp_path, max_calls=0, runner=runner).ask("sys", '["a"]')  # cache survives restarts


def test_rewrite_batch_repairs(tmp_path):
    runner = FakeRunner()
    out = rewrite_batch(Teacher("m", tmp_path, 10, runner), ["Le chat dort.", "bad: le chat dort."], max_repairs=2)
    assert [o["repairs"] for o in out] == [0, 1]
    assert all(db.is_lipogram(o["target"]) for o in out)
    repair_payload = json.loads(runner.prompts[1].split("\n", 1)[1])
    assert repair_payload == [{"source": "bad: le chat dort.", "ta_version": "Un chat qui dort, je crois.", "mots_fautifs": ["je@18"]}]


def test_offending_words():
    assert offending_words("Un chat qui dort là") == []
    assert offending_words("Le chat mange") == ["Le@0", "mange@8"]


def test_parse_json_list():
    assert parse_json_list('Voici:\n["a", "b"]', 2) == ["a", "b"]
    for bad in ('["a"]', "pas de json", '[1, 2]'):
        with pytest.raises(TeacherError):
            parse_json_list(bad, 2)


def test_rejection_reasons():
    src = "Le chat dort sur le canapé depuis ce matin."
    assert rejection_reason(src, "Mon chat dort sur un divan, tout au long du matin.", 0.9, FILTER) is None
    assert rejection_reason(src, "Le chat dort sur un divan, tout au long du matin.", 0.9, FILTER) == "has_e"
    assert rejection_reason(src, "Mon chat dort sur un divan, tout au long du matin.", 0.5, FILTER) == "low_sem_sim"
    assert rejection_reason(src, "Mon chat dort.", 0.9, FILTER) == "len_ratio"
    assert rejection_reason(src, "Voici mon chat qui dort sur un divan au matin.", 0.9, FILTER) == "meta"


def test_clean_corpus_drops_dups_overlap_and_non_french():
    eval_text = "Il pleuvait depuis trois jours et les rues de la ville étaient désertes ce soir-là."
    items = [{"text": "Le chat de la maison dort sur le canapé et il ne se réveille pas avant midi."},
             {"text": "Le chat de la maison dort sur le canapé et il ne se réveille pas avant midi, dit-on."},
             {"text": "Hier encore, il pleuvait depuis trois jours et les rues de la ville étaient désertes."},
             {"text": "The cat of the house sleeps on the sofa and does not wake up before noon."}]
    kept, dropped = clean_corpus(items, [eval_text])
    assert len(kept) == 1 and dropped == {"near_duplicate": 1, "eval_overlap": 1, "not_french": 1}
    assert kept[0]["split"] in {"train", "val", "test"} and len(kept[0]["id"]) == 12


def test_split_is_deterministic_and_roughly_90_5_5():
    splits = [split_of(f"phrase numéro {i} pour tester") for i in range(4000)]
    assert splits == [split_of(f"phrase numéro {i} pour tester") for i in range(4000)]
    assert 0.87 < splits.count("train") / 4000 < 0.93


def test_cli_strips_api_keys(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(env=kw["env"], cmd=cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"result": "[]", "is_error": False}), stderr="")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://x")
    monkeypatch.setattr(db.subprocess, "run", fake_run)
    db.claude_cli("sys", "prompt", "sonnet")
    assert not any(k in seen["env"] for k in db.API_ENV_VARS)
    assert seen["cmd"][seen["cmd"].index("--setting-sources") + 1] == ""

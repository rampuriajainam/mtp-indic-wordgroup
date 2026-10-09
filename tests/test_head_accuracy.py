import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from mtp.config import load_config
from mtp.data.collate import Collator
from mtp.eval.head_accuracy import evaluate_heads
from mtp.model.heads import MTPOutput

ROOT = Path(__file__).resolve().parents[1]
V = 6

# Two sentences, the second padded to length 4. group_id: A = [0,0 | 1,1], B = [0 | 1,1].
EXAMPLES = [
    {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], "group_start": [1, 0, 1, 0], "group_id": [0, 0, 1, 1]},
    {"input_ids": [5, 1, 2], "attention_mask": [1, 1, 1], "group_start": [1, 1, 0], "group_id": [0, 1, 1]},
]
# Forced argmax per head [b][t] (None = position has no valid target). Targets in comments.
PRED = [
    [[2, 5, 4, None], [1, 0, None, None]],     # head 0 targets: A 2,3,4 | B 1,2   -> A1 and B1 wrong
    [[3, 0, None, None], [2, None, None, None]],  # head 1 targets: A 3,4 | B 2      -> A1 wrong
]


class ForcedModel(nn.Module):
    """Logits = small distinct base (-0.1 * token id, so ranks are unambiguous) + 4 on the forced token."""

    num_heads = 2

    def __init__(self):
        super().__init__()
        self.logits = []
        for head in PRED:
            lg = (-0.1 * torch.arange(V, dtype=torch.float)).repeat(2, 4, 1)
            for b, row in enumerate(head):
                for t, p in enumerate(row):
                    if p is not None:
                        lg[b, t, p] += 4.0
            self.logits.append(lg)

    def forward(self, input_ids, attention_mask=None, **kw):
        return MTPOutput(logits=[lg[: input_ids.shape[0], : input_ids.shape[1]] for lg in self.logits],
                         hidden=torch.zeros(input_ids.shape[0], input_ids.shape[1], 1))


def _ce(model, head, positions):
    """Reference CE at (b, t) positions, computed one by one."""
    ids = Collator(0)(EXAMPLES)["input_ids"]
    return [F.cross_entropy(model.logits[head][b, t], ids[b, t + head + 1]).item() for b, t in positions]


def test_forced_logits_known_answer(tmp_path):
    model = ForcedModel()
    cfg = SimpleNamespace(optim=SimpleNamespace(batch_size=2), dtype="fp32")
    tok = SimpleNamespace(convert_ids_to_tokens=lambda i: f"tok{i}")
    dump = tmp_path / "tokens.jsonl"
    h0, h1 = evaluate_heads(model, EXAMPLES, Collator(0), cfg, "cpu", top_k=(1, 3, 5), dump_path=dump, tokenizer=tok)

    # head 0: 5 valid targets, 3 right; wrong ones have rank 5 (A1) and rank 3 (B1)
    assert (h0["head"], h0["n"], h0["top1"], h0["top3"], h0["top5"]) == (0, 5, 0.6, 0.8, 1.0)
    # in-group targets: A0->A1, A2->A3, B1->B2 (2 right); at boundary: A1->A2, B0->B1 (1 right)
    assert (h0["n_in_group"], h0["top1_in_group"], h0["n_at_boundary"], h0["top1_at_boundary"]) == (3, 2 / 3, 2, 0.5)
    all0 = _ce(model, 0, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])
    assert h0["loss"] == pytest.approx(sum(all0) / 5, abs=1e-6)
    assert h0["ppl"] == pytest.approx(math.exp(h0["loss"]))
    assert h0["loss_in_group"] == pytest.approx(sum(_ce(model, 0, [(0, 0), (0, 2), (1, 1)])) / 3, abs=1e-6)
    assert h0["loss_at_boundary"] == pytest.approx(sum(_ce(model, 0, [(0, 1), (1, 0)])) / 2, abs=1e-6)

    # head 1: 3 valid targets, all cross a group boundary
    assert (h1["n"], h1["top1"], h1["top3"], h1["top5"]) == (3, 2 / 3, 2 / 3, 1.0)
    assert (h1["n_in_group"], h1["top1_in_group"], h1["loss_in_group"]) == (0, None, None)
    assert (h1["n_at_boundary"], h1["top1_at_boundary"]) == (3, 2 / 3)

    lines = [json.loads(l) for l in dump.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 5 + 3
    assert set(lines[0]) == {"sent_id", "token_idx", "token", "head", "target_idx", "correct", "rank",
                             "group_start", "group_id", "target_group_id"}
    by_key = {(r["head"], r["sent_id"], r["token_idx"]): r for r in lines}
    assert by_key[(0, 0, 1)] == {"sent_id": 0, "token_idx": 1, "token": "tok2", "head": 0, "target_idx": 2,
                                 "correct": False, "rank": 5, "group_start": 0, "group_id": 0, "target_group_id": 1}
    assert by_key[(0, 1, 1)]["rank"] == 3 and by_key[(1, 1, 0)]["target_idx"] == 2
    assert sum(r["correct"] for r in lines if r["head"] == 0) == 3


def test_without_groups_and_mode_restored():
    model = ForcedModel().train()
    cfg = SimpleNamespace(optim=SimpleNamespace(batch_size=2), dtype="fp32")   # ForcedModel's logits are per batch row
    plain = [{k: e[k] for k in ("input_ids", "attention_mask")} for e in EXAMPLES]
    res = evaluate_heads(model, plain, Collator(0), cfg, "cpu")
    assert model.training
    assert res[0]["top1"] == 0.6 and res[0]["top1_in_group"] is None and res[0]["n_in_group"] == 0
    assert {"top1", "top5"} <= set(res[0])
    with pytest.raises(ValueError):
        evaluate_heads(model, plain, Collator(0), cfg, "cpu", dump_path="x.jsonl")
    with pytest.raises(ValueError):
        evaluate_heads(model, plain, Collator(0), cfg, "cpu", texts=["only one"])


def _load_train():
    spec = importlib.util.spec_from_file_location("train_script_om4", ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("variant", [None, "S23"])
def test_agrees_with_train_evaluate(tmp_path, tiny_model_dir, tiny_tok, variant):
    """On a tiny trained run, evaluate_heads == scripts/train.py's evaluate (to 1e-3, as INTERFACES §10)."""
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.model.build import build_model
    from mtp.model.checkpoint import load_run

    train = _load_train()
    d = yaml.safe_load((ROOT / "configs" / "R2.yaml").read_text(encoding="utf-8"))
    d.update(run_name="t", model_name=str(tiny_model_dir), num_heads=3, dtype="fp32", log_every=1, eval_every=2, save_every=2)
    d["optim"].update(max_steps=3, batch_size=2)
    d["data"].update(cache_dir=None, grouper="hi_rules_v0")
    d["losses"]["structural"].update(enabled=variant is not None, variant=variant)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")

    texts = ["मैं कल बाजार जा रहा था।", "बच्चे पार्क में खेल रहे हैं।", "वह घर से आया था",
             "भारत एक विशाल और विविधतापूर्ण देश है।", "के बारे बात कर"]
    examples = label_batch(texts, tiny_tok, get_grouper("hi_rules_v0"))
    train.seed_everything(0)
    model, tok = build_model(cfg, "cpu")
    train.train(cfg, model, tok, examples, examples, tmp_path / "runs" / "t")

    loaded, tok2, cfg2 = load_run(tmp_path / "runs" / "t", device="cpu")
    collator = Collator(tok2.pad_token_id)
    ref = train.evaluate(loaded, examples, collator, cfg2, "cpu")
    loaded.eval()
    ours = evaluate_heads(loaded, examples, collator, cfg2, "cpu", dump_path=tmp_path / "dump.jsonl",
                          texts=texts, tokenizer=tok2)
    assert len(ours) == len(ref) == 3
    for r in ours:
        exp = ref[r["head"]]
        for key in ("n", "n_in_group", "n_at_boundary"):
            assert r[key] == exp[key], (r["head"], key)
        for key in ("loss", "top1", "top1_in_group", "top1_at_boundary"):
            if r[key] is None:  # empty split: train.py reports 0/max(0, 1) = 0.0, we report None
                assert r["n_" + key.removeprefix("top1_")] == 0, (r["head"], key)
                continue
            assert r[key] == pytest.approx(exp[key], abs=1e-3), (r["head"], key)
        assert r["n_in_group"] + r["n_at_boundary"] > 0 and r["top1"] <= r["top5"]
    lines = (tmp_path / "dump.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == sum(r["n"] for r in ours)
    first = json.loads(lines[0])
    assert first["token"] == tok2.convert_ids_to_tokens(examples[0]["input_ids"][0])

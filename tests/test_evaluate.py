import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from mtp.config import load_config

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ["मैं कल बाजार जा रहा था।", "बच्चे पार्क में खेल रहे हैं।", "वह घर से आया था",
         "भारत एक विशाल और विविधतापूर्ण देश है।", "के बारे बात कर"]
SECTION10_KEYS = {"run_name", "dataset", "step", "grouper", "git_commit", "per_head", "spec_decode"}
PER_HEAD_KEYS = {"head", "loss", "ppl", "top1", "top5", "n", "top1_in_group", "top1_at_boundary",
                 "loss_in_group", "loss_at_boundary", "n_in_group", "n_at_boundary"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def evaluate():
    return _load("evaluate_script", ROOT / "scripts" / "evaluate.py")


@pytest.fixture
def tiny_run(tmp_path, tiny_model_dir, tiny_tok):
    """A tiny R2-style run trained for 2 steps by scripts/train.py (CPU, seconds)."""
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.model.build import build_model

    train = _load("train_script_eval", ROOT / "scripts" / "train.py")
    d = yaml.safe_load((ROOT / "configs" / "R2.yaml").read_text(encoding="utf-8"))
    d.update(run_name="tiny_R2", model_name=str(tiny_model_dir), num_heads=3, dtype="fp32",
             log_every=1, eval_every=2, save_every=2)
    d["optim"].update(max_steps=2, batch_size=2)
    d["data"].update(cache_dir=None, grouper="hi_rules_v0")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")
    examples = label_batch(TEXTS, tiny_tok, get_grouper("hi_rules_v0"))
    train.seed_everything(0)
    model, tok = build_model(cfg, "cpu")
    run_dir = tmp_path / "runs" / "tiny_R2"
    train.train(cfg, model, tok, examples, examples, run_dir)
    return run_dir


def fake_texts(name, cfg, n=None):
    texts = TEXTS if name.startswith("indiccorp") else TEXTS[::-1]
    return texts[:n] if n else texts


def test_writes_section10_json_matching_evaluate_heads(evaluate, tiny_run, tmp_path):
    from mtp.data.collate import Collator
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.eval.head_accuracy import evaluate_heads
    from mtp.model.checkpoint import load_run

    out = tmp_path / "results"
    written = evaluate.evaluate_run(tiny_run, ["indiccorp_eval", "flores_hi"], out_dir=out, device="cpu",
                                    dump_tokens=True, texts_fn=fake_texts)
    assert set(written) == {"indiccorp_eval", "flores_hi"}

    rec = json.loads((out / "tiny_R2" / "eval_indiccorp_eval.json").read_text(encoding="utf-8"))
    assert set(rec) == SECTION10_KEYS
    assert (rec["run_name"], rec["dataset"], rec["step"], rec["grouper"]) == ("tiny_R2", "indiccorp_eval", 2, "hi_rules_v0")
    assert rec["git_commit"] and rec["spec_decode"] is None
    assert [r["head"] for r in rec["per_head"]] == [0, 1, 2]
    assert all(PER_HEAD_KEYS <= set(r) for r in rec["per_head"])

    # Same numbers as calling evaluate_heads directly on the loaded run.
    model, tok, cfg = load_run(tiny_run, device="cpu")
    examples = label_batch(TEXTS, tok, get_grouper("hi_rules_v0"))
    ref = evaluate_heads(model, examples, Collator(tok.pad_token_id), cfg, "cpu")
    for got, exp in zip(rec["per_head"], ref):
        for key in PER_HEAD_KEYS:
            if exp[key] is None or isinstance(exp[key], int):
                assert got[key] == exp[key], key
            else:
                assert got[key] == pytest.approx(exp[key], abs=1e-6), key

    # FLORES got its own texts; the token dump carries sentence text.
    flores = json.loads((out / "tiny_R2" / "eval_flores_hi.json").read_text(encoding="utf-8"))
    assert flores["dataset"] == "flores_hi"
    dump = [json.loads(l) for l in (out / "tiny_R2" / "tokens_indiccorp_eval.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(dump) == sum(r["n"] for r in rec["per_head"])
    assert dump[0]["sent_id"] == 0 and dump[0]["head"] == 0


def test_rerun_keeps_other_sections(evaluate, tiny_run, tmp_path):
    out = tmp_path / "results"
    path = out / "tiny_R2" / "eval_indiccorp_eval.json"
    evaluate.evaluate_run(tiny_run, ["indiccorp_eval"], out_dir=out, device="cpu", texts_fn=fake_texts)
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["spec_decode"] = [{"policy": "fixed_k", "speedup": 1.3}]  # as if OM-6 ran in another session
    path.write_text(json.dumps(rec), encoding="utf-8")

    evaluate.evaluate_run(tiny_run, ["indiccorp_eval"], out_dir=out, device="cpu", texts_fn=fake_texts, n=3)
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again["spec_decode"] == [{"policy": "fixed_k", "speedup": 1.3}]
    assert again["per_head"][0]["n"] < rec["per_head"][0]["n"]  # per_head recomputed on 3 sentences


def test_unknown_grouper_evaluates_without_split(evaluate, tiny_run, tmp_path):
    with pytest.warns(UserWarning, match="not registered"):
        evaluate.evaluate_run(tiny_run, ["indiccorp_eval_small"], grouper_name="mr_rules_v1",
                              out_dir=tmp_path, device="cpu", texts_fn=fake_texts)
    rec = json.loads((tmp_path / "tiny_R2" / "eval_indiccorp_eval_small.json").read_text(encoding="utf-8"))
    assert rec["grouper"] is None
    assert rec["per_head"][0]["top1_in_group"] is None and rec["per_head"][0]["n_in_group"] == 0


def test_hindi_grouper_on_marathi_dataset_evaluates_without_split(evaluate, tiny_run, tmp_path):
    """A Hindi run on flores_mr: hi_rules_v0 refuses lang "mr" (ValueError); that must not crash."""
    with pytest.warns(UserWarning, match="cannot group 'mr'"):
        evaluate.evaluate_run(tiny_run, ["flores_hi", "flores_mr"], grouper_name="hi_rules_v0",
                              out_dir=tmp_path, device="cpu", texts_fn=fake_texts)
    hi = json.loads((tmp_path / "tiny_R2" / "eval_flores_hi.json").read_text(encoding="utf-8"))
    mr = json.loads((tmp_path / "tiny_R2" / "eval_flores_mr.json").read_text(encoding="utf-8"))
    assert hi["grouper"] == "hi_rules_v0" and mr["grouper"] is None
    assert mr["per_head"][0]["n_in_group"] == 0


SPEC_KEYS = {"policy", "mean_accepted_len", "accept_rate_per_head", "tokens_per_sec", "greedy_tokens_per_sec",
             "speedup", "outputs_match_greedy", "group_integrity", "n_prompts", "match_rate", "max_mismatch_margin", "fp32_check"}
LONG = ["मैं कल बाजार जा रहा था। वह घर से आया था भारत एक विशाल और विविधतापूर्ण देश है।",
        "बच्चे पार्क में खेल रहे हैं। के बारे बात कर वह घर से आया था मैं कल बाजार जा रहा था।"]


def long_texts(name, cfg, n=None):
    return LONG + TEXTS


def test_spec_decode_section(evaluate, tiny_run, tmp_path):
    out = tmp_path / "results"
    evaluate.evaluate_run(tiny_run, ["indiccorp_eval"], out_dir=out, device="cpu", spec_decode=True,
                          policies=["fixed_k", "confidence_cut"], tau=0.2, max_new_tokens=8, texts_fn=long_texts)
    rec = json.loads((out / "tiny_R2" / "eval_indiccorp_eval.json").read_text(encoding="utf-8"))
    assert rec["per_head"] is not None and [e["policy"] for e in rec["spec_decode"]] == ["fixed_k", "confidence_cut"]
    for e in rec["spec_decode"]:
        assert set(e) == SPEC_KEYS
        assert e["outputs_match_greedy"] is True and e["n_prompts"] == 2   # only LONG has > 16 words
        assert len(e["accept_rate_per_head"]) == 2 and e["mean_accepted_len"] >= 1.0

    # spec-only session: per_head is kept from the first run, spec_decode replaced
    before = rec["per_head"]
    evaluate.evaluate_run(tiny_run, ["indiccorp_eval"], out_dir=out, device="cpu", heads=False,
                          spec_decode=True, policies=["fixed_k"], max_new_tokens=4, texts_fn=long_texts)
    again = json.loads((out / "tiny_R2" / "eval_indiccorp_eval.json").read_text(encoding="utf-8"))
    assert again["per_head"] == before and [e["policy"] for e in again["spec_decode"]] == ["fixed_k"]


def test_spec_decode_skips_single_head_run(evaluate, tmp_path, tiny_model_dir, tiny_tok):
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.model.build import build_model

    train = _load("train_script_eval_r0", ROOT / "scripts" / "train.py")
    d = yaml.safe_load((ROOT / "configs" / "R0.yaml").read_text(encoding="utf-8"))
    d.update(run_name="tiny_R0", model_name=str(tiny_model_dir), dtype="fp32", log_every=1, eval_every=1, save_every=1)
    d["optim"].update(max_steps=1, batch_size=2)
    d["data"].update(cache_dir=None, grouper="hi_rules_v0")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")
    examples = label_batch(TEXTS, tiny_tok, get_grouper("hi_rules_v0"))
    train.seed_everything(0)
    model, tok = build_model(cfg, "cpu")
    run_dir = tmp_path / "runs" / "tiny_R0"
    train.train(cfg, model, tok, examples, examples, run_dir)

    with pytest.warns(UserWarning, match="one head"):
        evaluate.evaluate_run(run_dir, ["indiccorp_eval"], out_dir=tmp_path / "res", device="cpu",
                              spec_decode=True, texts_fn=long_texts)
    rec = json.loads((tmp_path / "res" / "tiny_R0" / "eval_indiccorp_eval.json").read_text(encoding="utf-8"))
    assert rec["spec_decode"] is None and len(rec["per_head"]) == 1


def test_cli_errors(evaluate, tiny_run):
    with pytest.raises(SystemExit):
        evaluate.main(["--run_dir", str(tiny_run), "--datasets", "nope"])
    with pytest.raises(SystemExit, match="not available"):
        evaluate.main(["--run_dir", str(tiny_run), "--spec_decode", "--policies", "no_such_policy"])
    with pytest.raises(SystemExit, match="nothing to do"):
        evaluate.main(["--run_dir", str(tiny_run), "--no_heads"])


def test_frozen_self_distillation_run_is_lossless(evaluate, tmp_path, tiny_model_dir, tiny_tok):
    """An Rsd-style run (freeze_backbone + data.train_file) goes through train.py -> load_run ->
    evaluate.py --spec_decode, and head 0 of the loaded run IS the untouched base model."""
    import torch
    from transformers import AutoModelForCausalLM

    from mtp.data.collate import Collator
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.model.build import build_model
    from mtp.model.checkpoint import load_run

    train = _load("train_script_eval_rsd", ROOT / "scripts" / "train.py")
    sd_file = tmp_path / "sd.jsonl"            # stands in for gen_selfdistill.py output
    sd_file.write_text("\n".join(json.dumps({"text": t}) for t in TEXTS * 4) + "\n", encoding="utf-8")
    d = yaml.safe_load((ROOT / "configs" / "Rsd.yaml").read_text(encoding="utf-8"))
    d.update(run_name="tiny_Rsd", model_name=str(tiny_model_dir), num_heads=3, dtype="fp32",
             log_every=1, eval_every=2, save_every=2)
    d["optim"].update(max_steps=3, batch_size=2)
    d["data"].update(cache_dir=None, train_file=str(sd_file), eval_n=None)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")
    assert cfg.freeze_backbone is True

    train.seed_everything(0)
    model, tok = build_model(cfg, "cpu")
    examples = label_batch([json.loads(l)["text"] for l in sd_file.read_text(encoding="utf-8").splitlines()],
                           tiny_tok, get_grouper("hi_rules_v0"))
    run_dir = tmp_path / "runs" / "tiny_Rsd"
    train.train(cfg, model, tok, examples, examples, run_dir)

    loaded, tok2, _ = load_run(run_dir, device="cpu")
    pristine = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32).eval()
    batch = Collator(tok2.pad_token_id)(label_batch(TEXTS, tok2, get_grouper("hi_rules_v0")))
    with torch.no_grad():
        head0 = loaded(batch["input_ids"], batch["attention_mask"]).logits[0]
        base = pristine(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        heads = loaded(batch["input_ids"], batch["attention_mask"]).logits[1:]
    torch.testing.assert_close(head0, base, rtol=0, atol=0)          # lossless: exactly the base model
    assert not torch.equal(heads[0], head0)                           # the extra heads did train

    out = tmp_path / "res"
    evaluate.evaluate_run(run_dir, ["indiccorp_eval"], out_dir=out, device="cpu", spec_decode=True,
                          policies=["fixed_k"], max_new_tokens=8, texts_fn=long_texts)
    rec = json.loads((out / "tiny_Rsd" / "eval_indiccorp_eval.json").read_text(encoding="utf-8"))
    assert rec["spec_decode"][0]["outputs_match_greedy"] is True and len(rec["per_head"]) == 3

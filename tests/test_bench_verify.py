"""Verify-pass timing benchmark (issue #33: how many tree nodes per verify pass are affordable)."""

import importlib.util
import json
from pathlib import Path

import yaml

from mtp.config import load_config

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ["मैं कल बाजार जा रहा था।", "बच्चे पार्क में खेल रहे हैं।", "वह घर से आया था",
         "भारत एक विशाल और विविधतापूर्ण देश है।", "के बारे बात कर"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_model(tiny_model_dir, tmp_path, head_type="resblock"):
    from mtp.model.build import build_model

    d = yaml.safe_load((ROOT / "configs" / "R2.yaml").read_text(encoding="utf-8"))
    d.update(model_name=str(tiny_model_dir), num_heads=3, dtype="fp32", head_type=head_type)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    model, tok = build_model(load_config(tmp_path / "c.yaml"), "cpu")
    return model.eval(), tok


def test_bench_verify_pass_shape_and_cache_rollback(tiny_model_dir, tmp_path, monkeypatch):
    from mtp.eval import spec_decode as sd

    model, tok = _tiny_model(tiny_model_dir, tmp_path)
    prefix = tok(" ".join(TEXTS))["input_ids"]
    lengths = []
    real_crop = sd._crop

    def spy_crop(cache, n):
        real_crop(cache, n)
        lengths.append(cache.get_seq_length())

    monkeypatch.setattr(sd, "_crop", spy_crop)
    res = sd.bench_verify_pass(model, prefix, sizes=(1, 4, 9), repeats=3, warmup=1)
    assert res["prefix_len"] == len(prefix) and res["greedy_ms"] > 0
    assert [p["positions"] for p in res["passes"]] == [1, 4, 9]
    assert all(p["ms"] > 0 and p["ms_p90"] >= p["ms"] and p["x_greedy"] > 0 for p in res["passes"])
    assert lengths and set(lengths) == {len(prefix)}      # every timed pass is rolled back to the prefix


def test_evaluate_bench_verify_only(tmp_path, tiny_model_dir, tiny_tok):
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.model.build import build_model

    train = _load("train_script_bench", ROOT / "scripts" / "train.py")
    evaluate = _load("evaluate_script_bench", ROOT / "scripts" / "evaluate.py")
    d = yaml.safe_load((ROOT / "configs" / "R2.yaml").read_text(encoding="utf-8"))
    d.update(run_name="tiny_bench", model_name=str(tiny_model_dir), num_heads=3, dtype="fp32",
             log_every=1, eval_every=2, save_every=2)
    d["optim"].update(max_steps=2, batch_size=2)
    d["data"].update(cache_dir=None, grouper="hi_rules_v0")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")
    examples = label_batch(TEXTS, tiny_tok, get_grouper("hi_rules_v0"))
    train.seed_everything(0)
    model, tok = build_model(cfg, "cpu")
    run_dir = tmp_path / "runs" / "tiny_bench"
    train.train(cfg, model, tok, examples, examples, run_dir)

    out = tmp_path / "res"
    written = evaluate.evaluate_run(run_dir, ["indiccorp_eval"], out_dir=out, device="cpu", heads=False,
                                    bench_verify=True, texts_fn=lambda name, cfg, n=None: TEXTS)
    assert set(written) == {"bench_verify"}
    bench = json.loads((out / "tiny_bench" / "bench_verify.json").read_text(encoding="utf-8"))
    assert bench["run_name"] == "tiny_bench" and bench["fp16_autocast"] is False and bench["gpu"] is None
    assert [p["positions"] for p in bench["passes"]] == [1, 4, 8, 16, 25, 32, 64]
    assert not (out / "tiny_bench" / "eval_indiccorp_eval.json").exists()


def test_bench_verify_seq_heads_verify_with_head0_only(tiny_model_dir, tmp_path, monkeypatch):
    """Sequential heads: verify passes skip the extra heads (as generate() does) and the draft chain
    is timed separately."""
    from mtp.eval import spec_decode as sd

    model, tok = _tiny_model(tiny_model_dir, tmp_path, head_type="seq")
    flags = []
    real_forward = model.forward

    def spy_forward(*a, extra_heads=True, **kw):
        flags.append(extra_heads)
        return real_forward(*a, extra_heads=extra_heads, **kw)

    monkeypatch.setattr(model, "forward", spy_forward)
    res = sd.bench_verify_pass(model, tok(" ".join(TEXTS))["input_ids"], sizes=(1, 4), repeats=2, warmup=1)
    assert flags and not any(flags)
    assert res["head_type"] == "seq" and res["draft_ms"] > 0


def test_bench_verify_parallel_heads_no_draft_ms(tiny_model_dir, tmp_path):
    from mtp.eval import spec_decode as sd

    model, tok = _tiny_model(tiny_model_dir, tmp_path)
    res = sd.bench_verify_pass(model, tok(" ".join(TEXTS))["input_ids"], sizes=(1,), repeats=2, warmup=1)
    assert res["head_type"] == "resblock" and res["draft_ms"] is None

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import MistralConfig, MistralForCausalLM

from mtp.config import load_config
from mtp.data.collate import Collator
from mtp.model.build import uses_probes
from mtp.model.heads import MTPModel

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train_mod = _load("train_script", ROOT / "scripts" / "train.py")

V = 50


def make_cfg(tmp_path, **over):
    d = yaml.safe_load((ROOT / "configs" / "R2.yaml").read_text(encoding="utf-8"))
    d.update(run_name="t", model_name="tiny", num_heads=3, dtype="fp32", log_every=2, eval_every=3, save_every=3)
    d["optim"].update(max_steps=6, batch_size=2)
    d["data"].update(cache_dir=str(tmp_path / "no_cache"))
    for k, v in over.items():
        d[k] = v
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    return load_config(p)


def make_model(cfg):
    torch.manual_seed(cfg.seed)
    base = MistralForCausalLM(MistralConfig(
        vocab_size=V, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=False))
    base = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.1,
                                           target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model = MTPModel(base, num_heads=cfg.num_heads, head_type=cfg.head_type)
    return model, SimpleNamespace(pad_token_id=0)


def make_examples(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        L = int(torch.randint(4, 12, (1,), generator=g))
        out.append({"input_ids": torch.randint(1, V, (L,), generator=g).tolist(), "attention_mask": [1] * L})
    return out


def run(cfg, run_dir, resume="none"):
    train_mod.seed_everything(cfg.seed)
    model, tok = make_model(cfg)
    train_mod.train(cfg, model, tok, make_examples(20), make_examples(5, seed=1), run_dir, resume=resume)
    return model


def trainable_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def test_train_writes_metrics_and_checkpoints(tmp_path):
    cfg = make_cfg(tmp_path)
    run(cfg, tmp_path / "run")
    rows = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    names = {(r["split"], r["name"]) for r in rows}
    for d in range(3):
        assert ("train", f"loss/head{d}") in names
        assert ("train", f"weight/head{d}") in names
        assert ("eval", f"loss/head{d}") in names
        assert ("eval", f"acc/top1/head{d}") in names
    assert all(r["value"] == r["value"] for r in rows)  # no NaN
    w = {r["name"]: r["value"] for r in rows if r["name"].startswith("weight/")}
    assert w["weight/head2"] == pytest.approx(0.8 ** 2)
    step_dir = tmp_path / "run" / "step_6"
    for f in ("lora/adapter_model.safetensors", "heads.pt", "weighting.pt", "optim.pt"):
        assert (step_dir / f).exists(), f
    saved = yaml.safe_load((tmp_path / "run" / "config.yaml").read_text(encoding="utf-8"))
    assert "git_commit" in saved


@pytest.mark.parametrize("scheme", ["fixed", "uncertainty", "dwa"])
def test_resume_matches_uninterrupted_run(tmp_path, scheme):
    def cfg_for(steps):
        cfg = make_cfg(tmp_path)
        cfg.weighting.scheme = scheme
        cfg.weighting.lr = 0.05          # make learned weights move visibly
        cfg.weighting.dwa_window = 1
        cfg.optim.max_steps = steps
        return cfg

    full = run(cfg_for(6), tmp_path / "full")
    run(cfg_for(3), tmp_path / "split")
    resumed = run(cfg_for(6), tmp_path / "split", resume="auto")

    a, b = trainable_state(full), trainable_state(resumed)
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0, msg=name)


def test_resume_skips_interrupted_save(tmp_path, capsys):
    """Issue #13: a session killed between heads.pt and optim.pt leaves a step that resume must skip."""
    full = run(make_cfg(tmp_path), tmp_path / "full")
    cfg = make_cfg(tmp_path)
    cfg.optim.max_steps = 3
    run(cfg, tmp_path / "split")
    partial = tmp_path / "split" / "step_6"  # simulate the kill: weights written, optim.pt not
    partial.mkdir()
    (partial / "heads.pt").write_bytes(b"truncated")
    (tmp_path / "split" / "step_3" / "optim.pt.tmp").write_bytes(b"leftover")

    resumed = run(make_cfg(tmp_path), tmp_path / "split", resume="auto")
    assert "resuming from step_3" in capsys.readouterr().out
    a, b = trainable_state(full), trainable_state(resumed)
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0, msg=name)
    assert (partial / "optim.pt").exists() and not (partial / "optim.pt.tmp").exists()


def test_resume_without_any_complete_save_fails(tmp_path):
    (tmp_path / "run" / "step_3").mkdir(parents=True)
    (tmp_path / "run" / "step_3" / "heads.pt").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="optim.pt"):
        run(make_cfg(tmp_path), tmp_path / "run", resume="auto")


def test_refuses_to_overwrite_existing_run(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.optim.max_steps = 3
    run(cfg, tmp_path / "run")
    with pytest.raises(FileExistsError):
        run(make_cfg(tmp_path), tmp_path / "run")


def test_grad_accum_and_data_wraparound(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.optim.grad_accum = 2
    cfg.optim.max_steps = 8  # 8 * 2 * 2 = 32 examples > 20 available
    run(cfg, tmp_path / "run")
    assert (tmp_path / "run" / "step_8" / "optim.pt").exists()


def test_freeze_backbone_trains_only_the_extra_heads(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.freeze_backbone = True
    train_mod.seed_everything(cfg.seed)
    model, tok = make_model(cfg)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    train_mod.train(cfg, model, tok, make_examples(20), make_examples(5, seed=1), tmp_path / "run")
    changed = {n for n, p in model.named_parameters() if not torch.equal(p.detach(), before[n])}
    assert changed and all(n.startswith("extra_heads.") for n in changed)


def test_train_file_replaces_corpus(tmp_path, monkeypatch):
    """data.train_file (self-distillation text) is read instead of IndicCorp train; eval still uses the corpus."""
    f = tmp_path / "sd.jsonl"
    f.write_text("\n".join(json.dumps({"text": f"generated {i}"}) for i in range(100)) + "\n", encoding="utf-8")
    cfg = make_cfg(tmp_path)
    cfg.data.train_file, cfg.data.grouper = str(f), None
    calls = []
    monkeypatch.setattr(train_mod, "load_split", lambda lang, split, n=None: calls.append(split) or ["eval text"])

    class Tok:
        def __call__(self, texts, truncation, max_length):
            ids = [[1] + [ord(c) for c in t] for t in texts]  # distinct per text
            return {"input_ids": ids, "attention_mask": [[1] * len(i) for i in ids]}

    train, ev = train_mod.build_data(cfg, Tok())
    assert calls == ["eval_small"] and len(train) == min(100, train_mod.num_train_examples(cfg))
    assert train[0]["input_ids"] == [1] + [ord(c) for c in "generated 0"] and len(ev) == 1

    cfg.data.shuffle = True  # seeded order: same seed -> same order, other seed -> other order
    a = [e["input_ids"] for e in train_mod.build_data(cfg, Tok())[0]]
    assert a == [e["input_ids"] for e in train_mod.build_data(cfg, Tok())[0]] and a != [e["input_ids"] for e in train]
    cfg.seed = cfg.seed + 1
    assert a != [e["input_ids"] for e in train_mod.build_data(cfg, Tok())[0]]


def test_group_losses_need_cache(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.losses.structural.enabled = True
    cfg.data.grouper = "not_a_registered_grouper"     # and no cache -> fail before any download
    with pytest.raises(FileNotFoundError):
        train_mod.build_data(cfg, tokenizer=None)
    cfg.losses.structural.variant = "TBD"
    with pytest.raises(ValueError):
        train_mod.build_aux_losses(cfg)
    cfg.losses.structural.enabled = False
    cfg.losses.contrastive.enabled = True
    with pytest.raises(NotImplementedError):
        train_mod.build_aux_losses(cfg)


def with_groups(examples, seed=0):
    g = torch.Generator().manual_seed(seed)
    for e in examples:
        n = len(e["input_ids"])
        start = [1] + (torch.rand(n - 1, generator=g) < 0.6).long().tolist()
        e["group_start"] = start
        e["group_id"] = torch.tensor(start).cumsum(0).sub(1).tolist()
    return examples


def test_mix_terms_get_their_own_lambda(tmp_path):
    from mtp.losses.weighting import LossWeighting

    cfg = make_cfg(tmp_path)
    cfg.losses.structural.lambda_s3, cfg.losses.structural.lambda_s3_all = 0.5, 0.25
    w = LossWeighting("fixed", ["struct/consistency/h1", "struct/consistency_all/h1"], aux_weights=train_mod.aux_weights(cfg))
    assert w._fixed_weight("struct/consistency/h1") == 0.5
    assert w._fixed_weight("struct/consistency_all/h1") == 0.25


@pytest.mark.parametrize("variant", ["S2", "S3_h0", "S3_chain", "S3_all", "S23", "S3_mix", "S23_mix"])
def test_train_with_structural_loss_and_resume(tmp_path, variant):
    cfg = make_cfg(tmp_path)
    cfg.losses.structural.enabled = True
    cfg.losses.structural.variant = variant
    cfg.optim.max_steps = 4

    def go(run_dir, max_steps, resume="none"):
        cfg.optim.max_steps = max_steps
        train_mod.seed_everything(cfg.seed)
        torch.manual_seed(cfg.seed)
        base_model, tok = make_model(cfg)
        model = MTPModel(base_model.base_model, num_heads=cfg.num_heads, head_type=cfg.head_type,
                         boundary_probes=uses_probes(cfg))
        train_mod.train(cfg, model, tok, with_groups(make_examples(20)), with_groups(make_examples(5, seed=1)),
                        run_dir, resume=resume)
        return model

    model = go(tmp_path / "run", 4)
    rows = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    struct = {r["name"] for r in rows if r["name"].startswith("struct/")}
    assert struct, rows[:5]
    assert all(r["value"] == r["value"] for r in rows)
    if variant in ("S2", "S23"):
        assert len(model.boundary_probes) == cfg.num_heads
        assert any(k.startswith("boundary_probes.") for k in torch.load(tmp_path / "run" / "step_4" / "heads.pt", weights_only=True))
        w = {r["name"]: r["value"] for r in rows if r["name"].startswith("weight/struct/boundary")}
        assert w and all(v == pytest.approx(0.1) for v in w.values())

    go(tmp_path / "split", 2)
    resumed = go(tmp_path / "split", 4, resume="auto")
    a, b = trainable_state(model), trainable_state(resumed)
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0, msg=name)


def test_batch_at_is_deterministic():
    collate = Collator(pad_id=0)
    ex = make_examples(5)
    b = train_mod.batch_at(ex, collate, index=2, batch_size=2)   # rows 4, 0
    assert b["input_ids"][0, : len(ex[4]["input_ids"])].tolist() == ex[4]["input_ids"]
    assert b["input_ids"][1, : len(ex[0]["input_ids"])].tolist() == ex[0]["input_ids"]
    assert b["attention_mask"].sum().item() == len(ex[4]["input_ids"]) + len(ex[0]["input_ids"])


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.stem)
def test_configs_load(path):
    cfg = load_config(path)
    for field in ("run_name", "lang", "model_name", "num_heads", "head_type", "seed", "dtype", "eval_every", "save_every"):
        assert hasattr(cfg, field), field
    assert cfg.optim.lr == 2e-4 and cfg.optim.batch_size == 8
    assert cfg.lora.r == 8 and list(cfg.lora.targets) == ["q_proj", "v_proj"]


def test_cache_path_shuffle_matches_raw_order(tmp_path):
    """data.shuffle on a boundary cache gives the same seeded order as on raw text (seed replicates)."""
    import random

    from datasets import Dataset

    cfg = make_cfg(tmp_path)
    cfg.data.cache_dir, cfg.data.grouper, cfg.data.shuffle, cfg.seed = str(tmp_path / "cache"), "g", True, 43
    n = train_mod.num_train_examples(cfg)
    rows = {"input_ids": [[i + 3, 4] for i in range(n + 5)], "attention_mask": [[1, 1]] * (n + 5),
            "group_start": [[1, 0]] * (n + 5), "group_id": [[0, 0]] * (n + 5)}
    for split in ("train", "eval"):
        Dataset.from_dict(rows).save_to_disk(str(tmp_path / "cache" / f"hi_g_{split}"))
    train, _ = train_mod.build_data(cfg, tokenizer=None)
    order = list(range(n))
    random.Random(43).shuffle(order)
    assert [r["input_ids"][0] - 3 for r in train] == order

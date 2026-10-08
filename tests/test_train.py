import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import MistralConfig, MistralForCausalLM

from mtp.losses.weighting import LossWeighting
from mtp.model.heads import MTPModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train_mod = _load("train_script", ROOT / "scripts" / "train.py")
stubs = _load("_stubs_om", ROOT / "scripts" / "_stubs_om.py")

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
    return stubs.load_config(p)


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


def test_resume_matches_uninterrupted_run(tmp_path):
    full = run(make_cfg(tmp_path), tmp_path / "full")

    cfg = make_cfg(tmp_path)
    cfg.optim.max_steps = 3
    run(cfg, tmp_path / "split")
    cfg = make_cfg(tmp_path)
    resumed = run(cfg, tmp_path / "split", resume="auto")

    a, b = trainable_state(full), trainable_state(resumed)
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0, msg=name)


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


def test_group_losses_need_cache(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.losses.structural.enabled = True
    with pytest.raises(FileNotFoundError):
        train_mod.build_data(cfg, tokenizer=None)
    with pytest.raises(NotImplementedError):
        train_mod.build_aux_losses(cfg)


def test_batch_at_is_deterministic():
    collate = stubs.Collator(pad_id=0)
    ex = make_examples(5)
    b = train_mod.batch_at(ex, collate, index=2, batch_size=2)   # rows 4, 0
    assert b["input_ids"][0, : len(ex[4]["input_ids"])].tolist() == ex[4]["input_ids"]
    assert b["input_ids"][1, : len(ex[0]["input_ids"])].tolist() == ex[0]["input_ids"]
    assert b["attention_mask"].sum().item() == len(ex[4]["input_ids"]) + len(ex[0]["input_ids"])


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.stem)
def test_configs_load(path):
    cfg = stubs.load_config(path)
    for field in ("run_name", "lang", "model_name", "num_heads", "head_type", "seed", "dtype", "eval_every", "save_every"):
        assert hasattr(cfg, field), field
    assert cfg.optim.lr == 2e-4 and cfg.optim.batch_size == 8
    assert cfg.lora.r == 8 and list(cfg.lora.targets) == ["q_proj", "v_proj"]


def test_overrides():
    cfg = SimpleNamespace(optim=SimpleNamespace(lr=1.0))
    stubs.apply_overrides(cfg, ["optim.lr=1e-4", "losses.structural.enabled=true"])
    assert cfg.optim.lr == 1e-4 and cfg.losses.structural.enabled is True


def test_fixed_weighting():
    names = ["loss/head0", "loss/head1", "loss/head2", "struct/boundary_bce/h1", "contrastive/supcon"]
    w = LossWeighting("fixed", names, head_decay=0.8, aux_weights={"struct": 0.1, "struct/boundary_bce": 0.3})
    losses = {n: torch.tensor(1.0) for n in names}
    total, weights = w(losses)
    assert weights["loss/head2"] == pytest.approx(0.64)
    assert weights["struct/boundary_bce/h1"] == pytest.approx(0.3)   # longest prefix wins
    assert weights["contrastive/supcon"] == 1.0
    assert total.item() == pytest.approx(1 + 0.8 + 0.64 + 0.3 + 1.0)
    with pytest.raises(KeyError):
        w({"loss/head9": torch.tensor(1.0)})
    with pytest.raises(NotImplementedError):
        LossWeighting("uncertainty", names)

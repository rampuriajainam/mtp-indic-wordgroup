import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from mtp import device as dev
from mtp.config import apply_overrides, cfg_get, load_config, save_config, to_dict
from mtp.data.collate import Collator
from mtp.model.checkpoint import latest_step, load_run, save_run

ROOT = Path(__file__).resolve().parents[1]


def test_config_roundtrip(tmp_path):
    cfg = load_config(ROOT / "configs" / "R2.yaml")
    save_config(cfg, tmp_path / "c.yaml")
    assert to_dict(load_config(tmp_path / "c.yaml")) == to_dict(cfg)
    assert cfg_get(cfg, "optim.lr") == 2e-4
    assert cfg_get(cfg, "optim.missing", 7) == 7 and cfg_get(cfg, "a.b.c") is None


def test_overrides():
    cfg = SimpleNamespace(optim=SimpleNamespace(lr=1.0))
    apply_overrides(cfg, ["optim.lr=1e-4", "losses.structural.enabled=true", "data.grouper=hi_rules_v0"])
    assert cfg.optim.lr == 1e-4 and cfg.losses.structural.enabled is True and cfg.data.grouper == "hi_rules_v0"


def test_collate_pads_per_contract():
    out = Collator(pad_id=0)([
        {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "group_start": [1, 0, 1], "group_id": [0, 0, 1]},
        {"input_ids": [8], "attention_mask": [1], "group_start": [1], "group_id": [0]},
    ])
    assert out["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert out["group_start"].tolist() == [[1, 0, 1], [1, 0, 0]]
    assert out["group_id"].tolist() == [[0, 0, 1], [0, -1, -1]]
    plain = Collator(pad_id=9)([{"input_ids": [1], "attention_mask": [1]}, {"input_ids": [1, 2], "attention_mask": [1, 1]}])
    assert set(plain) == {"input_ids", "attention_mask"} and plain["input_ids"].tolist() == [[1, 9], [1, 2]]


def test_auto_dtype_ignores_emulated_bf16(monkeypatch):
    # T4 (sm_75): torch says bf16 is "supported" (emulated) -> fp32 weights + fp16 autocast
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    cfg = SimpleNamespace(dtype="auto")
    assert dev.pick_dtype("auto") == torch.float32 and dev.uses_fp16_autocast(cfg)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    assert dev.pick_dtype("auto") == torch.bfloat16 and not dev.uses_fp16_autocast(cfg)
    assert dev.pick_dtype("fp16") == torch.float32 and dev.pick_dtype("bf16") == torch.bfloat16


def _load_train():
    spec = importlib.util.spec_from_file_location("train_script_infra", ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("variant", [None, "S23"])
def test_build_train_save_load_run(tmp_path, tiny_model_dir, tiny_tok, variant):
    """Real path end to end: build_model -> train -> save_run -> load_run gives the same logits."""
    from mtp.data.grouping.align import label_batch
    from mtp.data.grouping.base import get_grouper
    from mtp.model.build import build_model

    train = _load_train()
    d = yaml.safe_load((ROOT / "configs" / "R2.yaml").read_text(encoding="utf-8"))
    d.update(run_name="t", model_name=str(tiny_model_dir), num_heads=3, dtype="fp32", log_every=1, eval_every=2, save_every=2)
    d["optim"].update(max_steps=2, batch_size=2)
    d["data"].update(cache_dir=None, grouper="hi_rules_v0")
    d["losses"]["structural"].update(enabled=variant is not None, variant=variant)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml")

    texts = ["मैं कल बाजार जा रहा था।", "बच्चे पार्क में खेल रहे हैं।", "वह घर से आया था", "भारत एक विशाल देश है।"]
    examples = label_batch(texts, tiny_tok, get_grouper("hi_rules_v0"))
    train.seed_everything(0)
    model, tok = build_model(cfg, "cpu")
    assert len(model.boundary_probes) == (3 if variant else 0)
    run_dir = tmp_path / "runs" / "t"
    train.train(cfg, model, tok, examples, examples, run_dir)
    assert latest_step(run_dir) == 2

    loaded, tok2, cfg2 = load_run(run_dir, device="cpu")
    assert loaded.loaded_step == 2 and not loaded.training and cfg2.git_commit == cfg.git_commit
    batch = Collator(tok.pad_token_id)(examples)
    model.eval()
    with torch.no_grad():
        a = model(batch["input_ids"], batch["attention_mask"])
        b = loaded(batch["input_ids"], batch["attention_mask"])
    for x, y in zip(a.logits, b.logits):
        torch.testing.assert_close(x, y)
    if variant:
        for x, y in zip(a.aux["boundary_logits"], b.aux["boundary_logits"]):
            torch.testing.assert_close(x, y)


def test_save_run_keeps_last_two(tmp_path, tiny_model_dir):
    from mtp.model.build import build_model

    cfg = load_config(ROOT / "configs" / "R2.yaml")
    cfg.model_name, cfg.num_heads, cfg.dtype = str(tiny_model_dir), 2, "fp32"
    cfg.keep_steps = [1]
    model, _ = build_model(cfg, "cpu")
    for step in (1, 2, 3, 4):
        save_run(tmp_path, model, cfg, step)
    assert sorted(p.name for p in tmp_path.glob("step_*")) == ["step_1", "step_3", "step_4"]
    assert latest_step(tmp_path) == 4 and latest_step(tmp_path / "nope") is None

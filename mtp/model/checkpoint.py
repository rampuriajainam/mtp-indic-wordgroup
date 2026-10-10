"""
Run folders (INTERFACES §8):

runs/{run_name}/
  config.yaml            written once (includes git_commit)
  metrics.jsonl
  step_{N}/lora/         PEFT save_pretrained
  step_{N}/heads.pt      model.head_state_dict(): extra heads + boundary probes
  step_{N}/weighting.pt  LossWeighting state (learned weights / DWA buffers)
  step_{N}/optim.pt      written by scripts/train.py: optimizer, scaler, RNG, step

Only the last `keep_last` step folders are kept, plus any listed in cfg.keep_steps.
"""

import re
import shutil
from pathlib import Path

import torch

from mtp.config import cfg_get, load_config, save_config


def save_run(run_dir, model, cfg, step, optimizer=None, weighting=None, keep_last=2):
    if keep_last < 1:  # [:-0] would keep everything
        raise ValueError(f"keep_last must be >= 1, got {keep_last}")
    run_dir = Path(run_dir)
    step_dir = run_dir / f"step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "config.yaml").exists():
        save_config(cfg, run_dir / "config.yaml")
    model.base_model.save_pretrained(step_dir / "lora")
    torch.save(model.head_state_dict(), step_dir / "heads.pt")
    if weighting is not None:
        torch.save(weighting.state_dict(), step_dir / "weighting.pt")
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "step": step}, step_dir / "optim.pt")
    keep = set(cfg_get(cfg, "keep_steps", None) or [])
    for s in sorted(_step_dirs(run_dir))[:-keep_last]:
        if s not in keep:
            shutil.rmtree(run_dir / f"step_{s}", ignore_errors=True)


def _step_dirs(run_dir):
    for p in Path(run_dir).glob("step_*"):
        m = re.fullmatch(r"step_(\d+)", p.name)
        if m and (p / "heads.pt").exists():
            yield int(m.group(1))


def latest_step(run_dir):
    return max(_step_dirs(run_dir), default=None) if Path(run_dir).exists() else None


def load_lora(model, step_dir):
    """LoRA adapter only, from a step folder into an already built MTPModel."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    step_dir = Path(step_dir)
    result = set_peft_model_state_dict(model.base_model, load_file(step_dir / "lora" / "adapter_model.safetensors"))
    missing = [k for k in result.missing_keys if "lora_" in k]
    if missing:
        raise RuntimeError(f"LoRA weights missing from {step_dir}: {missing[:3]}")


def load_weights(model, step_dir):
    """LoRA adapter + heads/probes from a step folder into an already built MTPModel."""
    step_dir = Path(step_dir)
    load_lora(model, step_dir)
    model.load_head_state_dict(torch.load(step_dir / "heads.pt", weights_only=True, map_location="cpu"))


def load_run(run_dir, device="auto", step=None):
    """Rebuild a trained run for evaluation: (model in eval mode, tokenizer, cfg).
    step=None loads the latest step. model.loaded_step tells which one."""
    from mtp.device import pick_device
    from mtp.model.build import build_model

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    device = pick_device() if device == "auto" else device
    step = latest_step(run_dir) if step is None else step
    if step is None:
        raise FileNotFoundError(f"no step_* folder with heads.pt in {run_dir}")
    model, tokenizer = build_model(cfg, device)
    load_weights(model, run_dir / f"step_{step}")
    model.loaded_step = step
    return model.eval(), tokenizer, cfg

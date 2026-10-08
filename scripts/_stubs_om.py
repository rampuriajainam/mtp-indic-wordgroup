"""
TEMPORARY stand-ins for Om's modules (OM-1, OM-2), so scripts/train.py runs
before H1/H2 land. train.py imports each name from mtp.* first and only
falls back to this file on ImportError. Delete once OM-1 and OM-2 are merged.

Implements just enough of INTERFACES §3, §4, §6, §8, §9:
  config:     load_config, save_config, apply_overrides
  device:     pick_device, pick_dtype, autocast_ctx, make_scaler
  logging:    MetricLogger
  data:       load_split, Collator
  checkpoint: save_run, latest_step
"""

import contextlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


# --------------------------------------------------------------------- config

def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return _to_ns(yaml.safe_load(f))


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_to_dict(cfg), f, sort_keys=False, allow_unicode=True)


def apply_overrides(cfg, overrides):
    """--set optim.lr=1e-4 --set losses.structural.enabled=true"""
    for item in overrides or []:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            if not hasattr(node, p):
                setattr(node, p, SimpleNamespace())
            node = getattr(node, p)
        parsed = yaml.safe_load(value)
        if isinstance(parsed, str):
            try:  # YAML 1.1 reads "1e-4" (no dot) as a string
                parsed = float(parsed)
            except ValueError:
                pass
        setattr(node, leaf, parsed)
    return cfg


# --------------------------------------------------------------------- device

def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(name="auto"):
    """dtype for the model WEIGHTS. fp16 means fp32 weights + fp16 autocast."""
    if name == "bf16":
        return torch.bfloat16
    if name in ("fp32", "fp16"):
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _uses_fp16_autocast(cfg):
    if not torch.cuda.is_available():
        return False
    name = getattr(cfg, "dtype", "auto")
    return name == "fp16" or (name == "auto" and not torch.cuda.is_bf16_supported())


def autocast_ctx(cfg):
    if _uses_fp16_autocast(cfg):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_scaler(cfg):
    return torch.amp.GradScaler("cuda", enabled=_uses_fp16_autocast(cfg))


# --------------------------------------------------------------------- logging

class MetricLogger:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step, split, name, value):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "split": split, "name": name, "value": float(value)}) + "\n")


# --------------------------------------------------------------------- data

_INDICCORP = {"hi": "hin_Deva", "mr": "mar_Deva"}


def load_split(lang, split, n=None):
    """INTERFACES §4. eval / eval_small = raw IndicCorp rows 0-999 / 0-99 with the
    blank separator rows dropped (so ~500 / ~50 sentences; open question for OM-1,
    see design_structural_loss.md §4). train = the first n NON-BLANK rows from row 1000."""
    from datasets import load_dataset
    if split == "flores":
        ds = load_dataset("facebook/flores", _INDICCORP[lang], split="devtest")
        texts = [r["sentence"] for r in ds]
        return texts[:n] if n else texts
    raw = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split=_INDICCORP[lang], streaming=True)
    if split in ("eval", "eval_small"):
        texts = [r["text"].strip() for r in raw.take(1000 if split == "eval" else 100)]
        return [t for t in texts if t]
    if split != "train":
        raise ValueError(split)
    texts = []
    for r in raw.skip(1000):
        t = r["text"].strip()
        if t:
            texts.append(t)
            if n and len(texts) >= n:
                break
    return texts


class Collator:
    """INTERFACES §3: right-pad; group_* columns optional."""

    PAD = {"attention_mask": 0, "group_start": 0, "group_id": -1}

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        T = max(len(e["input_ids"]) for e in examples)
        out = {}
        for key in ("input_ids", "attention_mask", "group_start", "group_id"):
            if key not in examples[0]:
                continue
            pad = self.pad_id if key == "input_ids" else self.PAD[key]
            out[key] = torch.tensor([list(e[key]) + [pad] * (T - len(e[key])) for e in examples])
        return out


# --------------------------------------------------------------------- checkpoint

def save_run(run_dir, model, cfg, step, optimizer=None, weighting=None, keep_last=2):
    run_dir = Path(run_dir)
    step_dir = run_dir / f"step_{step}"
    step_dir.mkdir(parents=True, exist_ok=True)
    if not (run_dir / "config.yaml").exists():
        save_config(cfg, run_dir / "config.yaml")
    model.base_model.save_pretrained(step_dir / "lora")
    torch.save(model.head_state_dict(), step_dir / "heads.pt")  # extra heads + boundary probes
    if weighting is not None:
        torch.save(weighting.state_dict(), step_dir / "weighting.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), step_dir / "optim.pt")
    keep = set(getattr(cfg, "keep_steps", None) or [])
    steps = sorted(_step_dirs(run_dir))
    for s in steps[:-keep_last]:
        if s not in keep:
            shutil.rmtree(run_dir / f"step_{s}", ignore_errors=True)


def _step_dirs(run_dir):
    for p in Path(run_dir).glob("step_*"):
        m = re.fullmatch(r"step_(\d+)", p.name)
        if m and (p / "heads.pt").exists():
            yield int(m.group(1))


def latest_step(run_dir):
    steps = list(_step_dirs(run_dir)) if Path(run_dir).exists() else []
    return max(steps) if steps else None

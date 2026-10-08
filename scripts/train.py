"""
Single training entry point (JN-3).

  python scripts/train.py --config configs/R2.yaml [--resume auto] [--set optim.lr=1e-4] [--run_root runs]

Builds tokenizer -> base model (mtp.device.pick_dtype) -> LoRA -> MTPModel ->
loss terms -> LossWeighting -> AdamW, then trains for optim.max_steps.

Data: the boundary cache (INTERFACES §3) if data.cache_dir/{lang}_{grouper}_{train,eval}
exists; otherwise, for runs without group losses (R0-R2, R8-R9), raw IndicCorp text
tokenized here. Batch i always holds the same examples, so resuming at step N just
starts at batch N.

Run folder (INTERFACES §8): {run_root}/{run_name}/config.yaml, metrics.jsonl,
step_{N}/lora, heads.pt, weighting.pt, optim.pt. optim.pt holds everything resume
needs beyond weights: optimizer, GradScaler, RNG states, step.

--resume auto   continue from the latest step_N if there is one, else start fresh.
--resume none   (default) start fresh; refuses to overwrite a run that has checkpoints.
--stop_after_min M   save and exit cleanly after M minutes (Kaggle session limits).
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Om's modules (OM-1, OM-2); stand-ins in scripts/_stubs_om.py until H1/H2 land.
try:
    from mtp.config import load_config, save_config
except ImportError:
    from _stubs_om import load_config, save_config
try:
    from mtp.config import apply_overrides
except ImportError:
    from _stubs_om import apply_overrides
try:
    from mtp.device import autocast_ctx, make_scaler, pick_device, pick_dtype
except ImportError:
    from _stubs_om import autocast_ctx, make_scaler, pick_device, pick_dtype
try:
    from mtp.utils.logging import MetricLogger
except ImportError:
    from _stubs_om import MetricLogger
try:
    from mtp.data.collate import Collator
except ImportError:
    from _stubs_om import Collator
try:
    from mtp.data.corpus import load_split
except ImportError:
    from _stubs_om import load_split
try:
    from mtp.model.checkpoint import latest_step, save_run
except ImportError:
    from _stubs_om import latest_step, save_run

from mtp.losses.mtp_ce import per_head_ce, shift_targets
from mtp.losses.weighting import LossWeighting
from mtp.model.heads import MTPModel


def cfg_get(cfg, dotted, default=None):
    """Optional config field with a default, so configs and RunConfig versions
    that predate a field still load."""
    node = cfg
    for part in dotted.split("."):
        if node is None or not hasattr(node, part):
            return default
        node = getattr(node, part)
    return node


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


# --------------------------------------------------------------------- build

def build_model(cfg, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=pick_dtype(cfg.dtype)).to(device)
    base = get_peft_model(base, LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.targets), bias="none", task_type="CAUSAL_LM",
    ))
    model = MTPModel(base, num_heads=cfg.num_heads, head_type=cfg.head_type,
                     n_layers=cfg_get(cfg, "head_layers", 1),
                     backbone_grad=cfg_get(cfg, "head_backbone_grad", 1.0))
    return model, tokenizer


def build_aux_losses(cfg):
    """Callables (mtp_output, batch) -> dict[name, scalar], each with .term_names."""
    if cfg_get(cfg, "losses.structural.enabled", False):
        raise NotImplementedError("structural loss lands with JN-5")
    if cfg_get(cfg, "losses.contrastive.enabled", False):
        raise NotImplementedError("contrastive loss lands with JN-6")
    return []


def needs_groups(cfg):
    return bool(cfg_get(cfg, "losses.structural.enabled", False) or cfg_get(cfg, "losses.contrastive.enabled", False))


def num_train_examples(cfg):
    return cfg.optim.max_steps * cfg.optim.batch_size * cfg_get(cfg, "optim.grad_accum", 1)


def build_data(cfg, tokenizer):
    """Returns (train_examples, eval_examples): sequences of unpadded dicts."""
    lang, grouper = cfg.lang, cfg_get(cfg, "data.grouper")
    cache_root = Path(cfg_get(cfg, "data.cache_dir", "") or "__missing__")
    train_dir = cache_root / f"{lang}_{grouper}_train"
    eval_dir = cache_root / f"{lang}_{grouper}_eval"
    eval_n = cfg_get(cfg, "data.eval_n", 100)

    if train_dir.exists() and eval_dir.exists():
        from datasets import load_from_disk
        meta = train_dir.with_name(train_dir.name + "_meta.json")
        meta = meta if meta.exists() else train_dir / "meta.json"
        if meta.exists():
            info = json.loads(meta.read_text(encoding="utf-8"))
            tok_name = info.get("tokenizer") or info.get("tokenizer_name")
            if tok_name and tok_name != cfg.model_name:
                raise ValueError(f"cache {train_dir} was built with tokenizer {tok_name}, run uses {cfg.model_name}")
        cols = ["input_ids", "attention_mask"] + (["group_start", "group_id"] if needs_groups(cfg) else [])
        train = load_from_disk(str(train_dir)).select_columns(cols)
        train = train.select(range(min(len(train), num_train_examples(cfg))))
        ev = load_from_disk(str(eval_dir)).select_columns(cols)
        ev = ev.select(range(min(len(ev), eval_n)))
        print(f"data: boundary cache {train_dir.name} ({len(train)} train) / {eval_dir.name} ({len(ev)} eval)")
        return train, ev

    if needs_groups(cfg):
        raise FileNotFoundError(f"group losses need the boundary cache (H3); not found: {train_dir}")

    max_len = cfg.data.max_length

    def tokenize(texts):
        enc = tokenizer(texts, truncation=True, max_length=max_len)
        return [{"input_ids": i, "attention_mask": m} for i, m in zip(enc["input_ids"], enc["attention_mask"])]

    train_texts = load_split(lang, "train", num_train_examples(cfg))
    eval_texts = load_split(lang, cfg_get(cfg, "data.eval_split", "eval_small"))[:eval_n]
    print(f"data: raw IndicCorp text, {len(train_texts)} train / {len(eval_texts)} eval sentences (no cache)")
    return tokenize(train_texts), tokenize(eval_texts)


def batch_at(examples, collator, index, batch_size):
    """The index-th batch of a fixed example order (wraps around if data runs out)."""
    n = len(examples)
    rows = [examples[(index * batch_size + j) % n] for j in range(batch_size)]
    return collator(rows)


# --------------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(model, examples, collator, cfg, device):
    """Token-weighted per-head CE and top-1 over the eval examples."""
    model.eval()
    k = model.num_heads
    ce_sum, correct, count = [0.0] * k, [0] * k, [0] * k
    bs = cfg.optim.batch_size
    for i in range((len(examples) + bs - 1) // bs):
        rows = [examples[j] for j in range(i * bs, min((i + 1) * bs, len(examples)))]
        batch = {key: v.to(device) for key, v in collator(rows).items()}
        ids, mask = batch["input_ids"], batch["attention_mask"]
        with autocast_ctx(cfg):
            out = model(ids, attention_mask=mask)
        sums = per_head_ce(out.logits, ids, mask, reduction="sum")
        for d, logits in enumerate(out.logits):
            targets, valid = shift_targets(ids, mask, d + 1)
            pred = logits[:, : targets.shape[1]].argmax(-1)
            ce_sum[d] += sums[d].item()
            correct[d] += (pred == targets)[valid].sum().item()
            count[d] += valid.sum().item()
    model.train()
    return {d: {"loss": ce_sum[d] / max(count[d], 1), "top1": correct[d] / max(count[d], 1), "n": count[d]}
            for d in range(k)}


# --------------------------------------------------------------------- checkpoints

def save_checkpoint(run_dir, model, cfg, step, optimizer, scaler, weighting):
    save_run(str(run_dir), model, cfg, step, optimizer=None, weighting=weighting)
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(state, Path(run_dir) / f"step_{step}" / "optim.pt")


def restore_checkpoint(run_dir, step, model, optimizer, scaler, weighting):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    step_dir = Path(run_dir) / f"step_{step}"
    if not (step_dir / "optim.pt").exists():
        raise FileNotFoundError(f"{step_dir} has no optim.pt (interrupted save?); delete that folder and resume again")
    result = set_peft_model_state_dict(model.base_model, load_file(step_dir / "lora" / "adapter_model.safetensors"))
    missing = [k for k in result.missing_keys if "lora_" in k]
    if missing:
        raise RuntimeError(f"LoRA weights missing from {step_dir}: {missing[:3]}...")
    model.extra_heads.load_state_dict(torch.load(step_dir / "heads.pt", weights_only=True))
    if (step_dir / "weighting.pt").exists():
        weighting.load_state_dict(torch.load(step_dir / "weighting.pt", weights_only=True))
    # Our own file; RNG states need full unpickling.
    state = torch.load(step_dir / "optim.pt", weights_only=False)
    assert state["step"] == step, (state["step"], step)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    if state["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])


# --------------------------------------------------------------------- train

def train(cfg, model, tokenizer, train_examples, eval_examples, run_dir, resume="none", stop_after_min=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    k = model.num_heads
    log_every = cfg_get(cfg, "log_every", 20)
    accum = cfg_get(cfg, "optim.grad_accum", 1)
    max_steps = cfg.optim.max_steps

    aux_losses = build_aux_losses(cfg)
    term_names = [f"loss/head{d}" for d in range(k)] + [n for loss in aux_losses for n in loss.term_names]
    weighting = LossWeighting(
        cfg_get(cfg, "weighting.scheme", "fixed"), term_names,
        head_decay=cfg_get(cfg, "weighting.head_decay", 0.8),
        aux_weights={"struct": cfg_get(cfg, "losses.structural.weight", 1.0),
                     "contrastive": cfg_get(cfg, "losses.contrastive.weight", 1.0)},
    ).to(device)

    model_params = [p for p in model.parameters() if p.requires_grad]
    weighting_params = [p for p in weighting.parameters() if p.requires_grad]
    groups = [{"params": model_params, "weight_decay": cfg_get(cfg, "optim.weight_decay", 0.01)}]
    if weighting_params:
        groups.append({"params": weighting_params, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(groups, lr=cfg.optim.lr)
    scaler = make_scaler(cfg)
    collator = Collator(tokenizer.pad_token_id)
    logger = MetricLogger(str(run_dir))

    start = 0
    last = latest_step(str(run_dir))
    if last is not None:
        if resume != "auto":
            raise FileExistsError(f"{run_dir} already has step_{last}; pass --resume auto or pick a new run_name")
        restore_checkpoint(run_dir, last, model, optimizer, scaler, weighting)
        start = last
        print(f"resumed from step {start}")
    cfg.git_commit = git_commit()
    if not (run_dir / "config.yaml").exists():
        save_config(cfg, str(run_dir / "config.yaml"))

    n_trainable = sum(p.numel() for p in model_params)
    print(f"device {device} | weights {next(model.parameters()).dtype} | fp16 scaler {scaler.is_enabled()} | "
          f"trainable {n_trainable:,} | heads {k} ({model.head_type}) | steps {start}->{max_steps}")

    model.train()
    t0 = time.time()
    t_window = time.time()
    running = {}
    step = start
    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        for micro in range(accum):
            batch = {key: v.to(device) for key, v in
                     batch_at(train_examples, collator, step * accum + micro, cfg.optim.batch_size).items()}
            with autocast_ctx(cfg):
                out = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            losses = {f"loss/head{d}": l for d, l in
                      enumerate(per_head_ce(out.logits, batch["input_ids"], batch["attention_mask"], reduction="mean"))}
            for loss in aux_losses:
                losses.update(loss(out, batch))
            total, weights = weighting(losses)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: { {n: v.item() for n, v in losses.items()} }")
            scaler.scale(total / accum).backward()
            for name, value in list(losses.items()) + [("loss/total", total)]:
                running[name] = running.get(name, 0.0) + value.item() / accum

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model_params + weighting_params, cfg.optim.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step += 1

        if step % log_every == 0:
            sec = (time.time() - t_window) / log_every
            for name, value in running.items():
                logger.log(step, "train", name, value / log_every)
            for name, w in weights.items():
                logger.log(step, "train", f"weight/{name.removeprefix('loss/')}", w)
            logger.log(step, "train", "lr", optimizer.param_groups[0]["lr"])
            logger.log(step, "train", "grad_norm", grad_norm.item())
            logger.log(step, "train", "time/sec_per_step", sec)
            heads = " | ".join(f"h{d} {running[f'loss/head{d}'] / log_every:.4f}" for d in range(k))
            print(f"step {step} | {heads} | {sec:.2f}s/step ({3600 / max(sec, 1e-9):.0f} steps/h)")
            running, t_window = {}, time.time()

        out_of_time = stop_after_min is not None and (time.time() - t0) / 60 >= stop_after_min
        if step % cfg.eval_every == 0 or step == max_steps:
            res = evaluate(model, eval_examples, collator, cfg, device)
            for d, r in res.items():
                logger.log(step, "eval", f"loss/head{d}", r["loss"])
                logger.log(step, "eval", f"acc/top1/head{d}", r["top1"])
            print(f"  >> eval step {step}: " + " | ".join(f"h{d} {r['loss']:.4f} ({100 * r['top1']:.1f}%)" for d, r in res.items()))
        if step % cfg.save_every == 0 or step == max_steps or out_of_time:
            save_checkpoint(run_dir, model, cfg, step, optimizer, scaler, weighting)
            print(f"  -> saved step_{step}")
        if out_of_time and step < max_steps:
            print(f"stopping after {stop_after_min} min at step {step}; rerun with --resume auto")
            break
    return step


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", choices=["none", "auto"], default="none")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--run_root", default=str(ROOT / "runs"))
    ap.add_argument("--stop_after_min", type=float, default=None)
    args = ap.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.set)
    seed_everything(cfg.seed)
    device = pick_device()
    if device == "cuda":
        print(f"GPU {torch.cuda.get_device_name(0)} | bf16 supported {torch.cuda.is_bf16_supported()}")
    model, tokenizer = build_model(cfg, device)
    train_examples, eval_examples = build_data(cfg, tokenizer)
    train(cfg, model, tokenizer, train_examples, eval_examples, Path(args.run_root) / cfg.run_name,
          resume=args.resume, stop_after_min=args.stop_after_min)


if __name__ == "__main__":
    main()

"""
Single training entry point (JN-3).

  python scripts/train.py --config configs/R2.yaml [--resume auto] [--set optim.lr=1e-4] [--run_root runs]

Builds tokenizer -> base model (mtp.device.pick_dtype) -> LoRA -> MTPModel ->
loss terms -> LossWeighting -> AdamW, then trains for optim.max_steps.

Data: the boundary cache (INTERFACES §3) if data.cache_dir/{lang}_{grouper}_{train,eval}
exists; otherwise raw IndicCorp text (mtp.data.corpus), labelled on the fly with the
data.grouper grouper (mtp.data.grouping) when the run has group losses. The eval set is
labelled whenever the grouper exists, so every run logs in-group vs at-boundary top-1.
Batch i always holds the same examples, so resuming at step N just starts at batch N.

Run folder (INTERFACES §8): {run_root}/{run_name}/config.yaml, metrics.jsonl,
step_{N}/lora, heads.pt, weighting.pt, optim.pt. optim.pt holds everything resume
needs beyond weights: optimizer, GradScaler, RNG states, step.

--resume auto   continue from the latest step_N if there is one, else start fresh.
--resume none   (default) start fresh; refuses to overwrite a run that has checkpoints.
--stop_after_min M   save and exit cleanly after M minutes (Kaggle session limits).
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mtp.config import apply_overrides, cfg_get, load_config, save_config
from mtp.data.collate import Collator
from mtp.data.corpus import load_split
from mtp.data.grouping.align import label_batch
from mtp.data.grouping.base import REGISTRY, get_grouper
from mtp.device import autocast_ctx, make_scaler, pick_device
from mtp.losses.mtp_ce import per_head_ce, shift_targets
from mtp.losses.structural import StructuralLoss
from mtp.losses.weighting import LossWeighting
from mtp.model.build import build_model
from mtp.model.checkpoint import latest_step, load_weights, save_run
from mtp.utils.logging import MetricLogger


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

def build_aux_losses(cfg):
    """Callables (mtp_output, batch) -> dict[name, scalar], each with .term_names."""
    losses = []
    if cfg_get(cfg, "losses.structural.enabled", False):
        losses.append(StructuralLoss(cfg_get(cfg, "losses.structural.variant"), cfg.num_heads,
                                     s3_teacher=cfg_get(cfg, "losses.structural.s3_teacher", None)))
    if cfg_get(cfg, "losses.contrastive.enabled", False):
        raise NotImplementedError("contrastive loss lands with JN-6")
    return losses


def aux_weights(cfg):
    """Fixed weights by term-name prefix (LossWeighting picks the longest match)."""
    w = cfg_get(cfg, "losses.structural.weight", 1.0)
    return {
        "struct": w,
        "struct/boundary_bce": w * cfg_get(cfg, "losses.structural.lambda_s2", 0.1),
        "struct/consistency": w * cfg_get(cfg, "losses.structural.lambda_s3", 0.5),
        "struct/consistency_all": w * cfg_get(cfg, "losses.structural.lambda_s3_all", 0.25),
        "contrastive": cfg_get(cfg, "losses.contrastive.weight", 1.0),
    }


def needs_groups(cfg):
    return bool(cfg_get(cfg, "losses.structural.enabled", False) or cfg_get(cfg, "losses.contrastive.enabled", False))


def num_train_examples(cfg):
    return cfg.optim.max_steps * cfg.optim.batch_size * cfg_get(cfg, "optim.grad_accum", 1)


def build_data(cfg, tokenizer):
    """Returns (train_examples, eval_examples): sequences of unpadded dicts."""
    lang, grouper_name = cfg.lang, cfg_get(cfg, "data.grouper")
    cache_root = Path(cfg_get(cfg, "data.cache_dir", "") or "__missing__")
    train_dir = cache_root / f"{lang}_{grouper_name}_train"
    eval_dir = cache_root / f"{lang}_{grouper_name}_eval"
    eval_n = cfg_get(cfg, "data.eval_n", 100)

    if train_dir.exists() and eval_dir.exists():
        from datasets import load_from_disk
        for meta in (train_dir / "meta.json", train_dir.with_name(train_dir.name + "_meta.json")):
            if meta.exists():
                info = json.loads(meta.read_text(encoding="utf-8"))
                tok_name = info.get("tokenizer") or info.get("tokenizer_name")
                if tok_name and tok_name != cfg.model_name:
                    raise ValueError(f"cache {train_dir} was built with tokenizer {tok_name}, run uses {cfg.model_name}")
        cols = ["input_ids", "attention_mask", "group_start", "group_id"]
        train = load_from_disk(str(train_dir)).select_columns(cols if needs_groups(cfg) else cols[:2])
        train = train.select(range(min(len(train), num_train_examples(cfg))))
        ev = load_from_disk(str(eval_dir)).select_columns(cols)
        ev = ev.select(range(min(len(ev), eval_n)))
        print(f"data: boundary cache {train_dir.name} ({len(train)} train) / {eval_dir.name} ({len(ev)} eval)")
        return train, ev

    grouper = get_grouper(grouper_name, lang) if grouper_name in REGISTRY else None
    if needs_groups(cfg) and grouper is None:
        raise FileNotFoundError(f"group losses need data.grouper to be a registered grouper or a boundary cache; "
                                f"{grouper_name!r} is neither (registered: {sorted(REGISTRY)}; cache: {train_dir})")
    max_len = cfg.data.max_length

    def prepare(texts, with_groups):
        if with_groups:
            return label_batch(texts, tokenizer, grouper, max_len)
        enc = tokenizer(texts, truncation=True, max_length=max_len)
        return [{"input_ids": i, "attention_mask": m} for i, m in zip(enc["input_ids"], enc["attention_mask"])]

    train_file = cfg_get(cfg, "data.train_file")
    if train_file:  # e.g. self-distillation text from scripts/gen_selfdistill.py (one {"text"} per line)
        with open(train_file, encoding="utf-8") as f:
            train_texts = [json.loads(line)["text"] for line in f if line.strip()][:num_train_examples(cfg)]
        print(f"data: train text from {train_file}")
    else:
        train_texts = load_split(lang, "train", num_train_examples(cfg))
    if cfg_get(cfg, "data.shuffle", False):
        # A seeded order. Without it the seed may change nothing at all: zero-init heads and a frozen
        # backbone make training deterministic in the data order, so a seed replicate needs a new order.
        random.Random(cfg.seed).shuffle(train_texts)
    eval_texts = load_split(lang, cfg_get(cfg, "data.eval_split", "eval_small"))[:eval_n]
    labelled = f"labelled with {grouper_name}" if grouper else f"no grouper ({grouper_name!r} not registered)"
    print(f"data: raw IndicCorp text, {len(train_texts)} train / {len(eval_texts)} eval sentences, {labelled}")
    return prepare(train_texts, needs_groups(cfg)), prepare(eval_texts, grouper is not None)


def batch_at(examples, collator, index, batch_size):
    """The index-th batch of a fixed example order (wraps around if data runs out)."""
    n = len(examples)
    rows = [examples[(index * batch_size + j) % n] for j in range(batch_size)]
    return collator(rows)


# --------------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(model, examples, collator, cfg, device):
    """Token-weighted per-head CE and top-1. If the examples carry group_id, also top-1 split by
    whether the target is in the SOURCE token's group (in_group) or not (at_boundary), as in OM-4."""
    model.eval()
    k = model.num_heads
    stats = {d: {"ce": 0.0, "n": 0, "ok": 0, "n_in": 0, "ok_in": 0, "n_bd": 0, "ok_bd": 0} for d in range(k)}
    bs = cfg.optim.batch_size
    for i in range((len(examples) + bs - 1) // bs):
        rows = [examples[j] for j in range(i * bs, min((i + 1) * bs, len(examples)))]
        batch = {key: v.to(device) for key, v in collator(rows).items()}
        ids, mask = batch["input_ids"], batch["attention_mask"]
        with autocast_ctx(cfg):
            out = model(ids, attention_mask=mask)
        sums = per_head_ce(out.logits, ids, mask, reduction="sum")
        for d, logits in enumerate(out.logits):
            shift = d + 1
            targets, valid = shift_targets(ids, mask, shift)
            hit = logits[:, : targets.shape[1]].argmax(-1) == targets
            st = stats[d]
            st["ce"] += sums[d].item()
            st["ok"] += hit[valid].sum().item()
            st["n"] += valid.sum().item()
            if "group_id" in batch and targets.shape[1] > 0:
                gid = batch["group_id"]
                known = valid & (gid[:, :-shift] >= 0) & (gid[:, shift:] >= 0)
                same = gid[:, :-shift] == gid[:, shift:]
                for tag, m in (("in", known & same), ("bd", known & ~same)):
                    st[f"n_{tag}"] += m.sum().item()
                    st[f"ok_{tag}"] += hit[m].sum().item()
    model.train()
    res = {}
    for d, st in stats.items():
        r = {"loss": st["ce"] / max(st["n"], 1), "top1": st["ok"] / max(st["n"], 1), "n": st["n"]}
        if st["n_in"] + st["n_bd"]:
            r.update(top1_in_group=st["ok_in"] / max(st["n_in"], 1), n_in_group=st["n_in"],
                     top1_at_boundary=st["ok_bd"] / max(st["n_bd"], 1), n_at_boundary=st["n_bd"])
        res[d] = r
    return res


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
    # Written last and atomically: a step folder with optim.pt is complete (issue #13).
    path = Path(run_dir) / f"step_{step}" / "optim.pt"
    torch.save(state, path.with_suffix(".pt.tmp"))
    os.replace(path.with_suffix(".pt.tmp"), path)


def resumable_step(run_dir):
    """Newest step whose save finished (has optim.pt). A session killed mid-save leaves a newer
    step_N without it; that one is skipped with a warning and overwritten when training reaches N."""
    last = latest_step(str(run_dir))
    steps = sorted((int(p.parent.name[len("step_"):]) for p in Path(run_dir).glob("step_*/optim.pt")
                    if p.parent.name[len("step_"):].isdigit()), reverse=True)
    if not steps:
        raise FileNotFoundError(f"no step_* folder in {run_dir} has optim.pt (first save interrupted?); "
                                "delete the run folder and start again")
    if steps[0] != last:
        print(f"WARNING: step_{last} is an incomplete save (no optim.pt); resuming from step_{steps[0]}")
    return steps[0]


def restore_checkpoint(run_dir, step, model, optimizer, scaler, weighting):
    step_dir = Path(run_dir) / f"step_{step}"
    load_weights(model, step_dir)
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
    cuda_rng = state["rng"]["cuda"]
    if cuda_rng is not None and torch.cuda.is_available() and len(cuda_rng) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda_rng)


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
        aux_weights=aux_weights(cfg),
        fix_head0=cfg_get(cfg, "weighting.fix_head0", True),
        dwa_window=cfg_get(cfg, "weighting.dwa_window", 20),
        dwa_temperature=cfg_get(cfg, "weighting.dwa_temperature", 2.0),
    ).to(device)

    if cfg_get(cfg, "freeze_backbone", False):
        # Medusa-1 style: only the extra heads (and probes) train; head 0 stays the base model exactly,
        # so heads can learn from self-distillation text without the verifier drifting toward it.
        for p in model.base_model.parameters():
            p.requires_grad_(False)
    model_params = [p for p in model.parameters() if p.requires_grad]
    weighting_params = [p for p in weighting.parameters() if p.requires_grad]
    groups = [{"params": model_params, "weight_decay": cfg_get(cfg, "optim.weight_decay", 0.01)}]
    if weighting_params:
        groups.append({"params": weighting_params, "weight_decay": 0.0,
                       "lr": cfg_get(cfg, "weighting.lr", cfg.optim.lr)})
    optimizer = torch.optim.AdamW(groups, lr=cfg.optim.lr)
    scaler = make_scaler(cfg)
    collator = Collator(tokenizer.pad_token_id)
    logger = MetricLogger(str(run_dir))

    start = 0
    last = latest_step(str(run_dir))
    if last is not None:
        if resume != "auto":
            raise FileExistsError(f"{run_dir} already has step_{last}; pass --resume auto or pick a new run_name")
        start = resumable_step(run_dir)
        restore_checkpoint(run_dir, start, model, optimizer, scaler, weighting)
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
                if "top1_in_group" in r:
                    logger.log(step, "eval", f"acc/top1_in_group/head{d}", r["top1_in_group"])
                    logger.log(step, "eval", f"acc/top1_at_boundary/head{d}", r["top1_at_boundary"])
            print(f"  >> eval step {step}: " + " | ".join(f"h{d} {r['loss']:.4f} ({100 * r['top1']:.1f}%)" for d, r in res.items()))
            if "top1_in_group" in res[0]:
                print("     in-group / boundary top-1: " + " | ".join(
                    f"h{d} {100 * r['top1_in_group']:.1f}/{100 * r['top1_at_boundary']:.1f}%" for d, r in res.items()))
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
        cap = torch.cuda.get_device_capability()
        print(f"GPU {torch.cuda.get_device_name(0)} | compute capability {cap[0]}.{cap[1]} | native bf16 {cap[0] >= 8}")
    model, tokenizer = build_model(cfg, device)
    train_examples, eval_examples = build_data(cfg, tokenizer)
    train(cfg, model, tokenizer, train_examples, eval_examples, Path(args.run_root) / cfg.run_name,
          resume=args.resume, stop_after_min=args.stop_after_min)


if __name__ == "__main__":
    main()

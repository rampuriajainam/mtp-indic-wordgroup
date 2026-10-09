"""
Per-head evaluation (OM-4, INTERFACES §10 "per_head").

Head d at position t predicts token t+d+1. For every head: token-weighted CE (loss, ppl),
top-k accuracy, and both split by whether the target is in the SOURCE token's word group
(in_group) or not (at_boundary). The math mirrors scripts/train.py's evaluate() (same
batching, autocast, per_head_ce, argmax top-1, group masks), so the two agree.
"""

import json
import math
from pathlib import Path

import torch

from mtp.device import autocast_ctx
from mtp.losses.mtp_ce import per_head_ce, shift_targets


def _new_stats(top_k):
    s = {"ce": 0.0, "n": 0, **{f"ok{k}": 0 for k in top_k}}
    for tag in ("in", "bd"):
        s.update({f"n_{tag}": 0, f"ok1_{tag}": 0, f"ce_{tag}": 0.0})
    return s


def _finish(d, st, top_k, has_groups):
    n = st["n"]
    loss = st["ce"] / max(n, 1)
    r = {"head": d, "loss": loss, "ppl": math.exp(min(loss, 700.0)), **{f"top{k}": st[f"ok{k}"] / max(n, 1) for k in top_k},
         "n": n}
    for tag, name in (("in", "in_group"), ("bd", "at_boundary")):
        m = st[f"n_{tag}"]
        r[f"top1_{name}"] = st[f"ok1_{tag}"] / m if has_groups and m else None
        r[f"loss_{name}"] = st[f"ce_{tag}"] / m if has_groups and m else None
        r[f"n_{name}"] = m
    return r


@torch.no_grad()
def evaluate_heads(model, examples, collator, cfg, device, top_k=(1, 5), dump_path=None, texts=None, tokenizer=None):
    """One dict per head with every field of INTERFACES §10 per_head (top{k} for each k in top_k).

    examples: unpadded dicts, e.g. label_batch(texts, tokenizer, grouper). Without group_id the
    in-group / at-boundary fields are None (n_* = 0).
    dump_path: write the §10 per-token dump (one line per sentence, position, head); needs tokenizer
    for the token strings. texts, if given, must line up with examples (sent_id indexes both).
    """
    top_k = tuple(sorted(set(top_k) | {1}))
    if texts is not None and len(texts) != len(examples):
        raise ValueError(f"texts ({len(texts)}) and examples ({len(examples)}) must line up")
    if dump_path is not None and tokenizer is None:
        raise ValueError("dump_path needs tokenizer (for the token strings)")
    was_training = model.training
    model.eval()
    k = model.num_heads
    has_groups = bool(examples) and "group_id" in examples[0]
    stats = [_new_stats(top_k) for _ in range(k)]
    dump = None
    if dump_path is not None:
        Path(dump_path).parent.mkdir(parents=True, exist_ok=True)
        dump = open(dump_path, "w", encoding="utf-8")
    bs = cfg.optim.batch_size
    try:
        for start in range(0, len(examples), bs):
            rows = [examples[j] for j in range(start, min(start + bs, len(examples)))]
            batch = {key: v.to(device) for key, v in collator(rows).items()}
            ids, mask = batch["input_ids"], batch["attention_mask"]
            with autocast_ctx(cfg):
                out = model(ids, attention_mask=mask)
            ce = per_head_ce(out.logits, ids, mask, reduction="none")
            gid = batch.get("group_id")
            for d, logits in enumerate(out.logits):
                shift = d + 1
                targets, valid = shift_targets(ids, mask, shift)
                if targets.shape[1] == 0:
                    continue
                lg = logits[:, : targets.shape[1]]
                hit1 = lg.argmax(-1) == targets                     # as train.py: argmax, not topk(1)
                kmax = min(max(top_k), lg.shape[-1])
                in_topk = lg.topk(kmax, dim=-1).indices == targets.unsqueeze(-1)
                st = stats[d]
                st["n"] += valid.sum().item()
                st["ce"] += ce[d][valid].sum().item()
                for kk in top_k:
                    hit = hit1 if kk == 1 else in_topk[..., :kk].any(-1)
                    st[f"ok{kk}"] += hit[valid].sum().item()
                if has_groups:
                    known = valid & (gid[:, :-shift] >= 0) & (gid[:, shift:] >= 0)
                    same = gid[:, :-shift] == gid[:, shift:]
                    for tag, m in (("in", known & same), ("bd", known & ~same)):
                        st[f"n_{tag}"] += m.sum().item()
                        st[f"ok1_{tag}"] += hit1[m].sum().item()
                        st[f"ce_{tag}"] += ce[d][m].sum().item()
                if dump is not None:
                    target_logit = lg.float().gather(-1, targets.unsqueeze(-1))
                    rank = (lg.float() > target_logit).sum(-1) + 1
                    _write_dump(dump, tokenizer, start, d, ids, valid, hit1, rank, batch)
    finally:
        if dump is not None:
            dump.close()
        model.train(was_training)
    return [_finish(d, st, top_k, has_groups) for d, st in enumerate(stats)]


def _write_dump(f, tokenizer, first_sent, head, ids, valid, hit1, rank, batch):
    shift = head + 1
    gs, gid = batch.get("group_start"), batch.get("group_id")
    for b, t in valid.nonzero().tolist():
        f.write(json.dumps({
            "sent_id": first_sent + b, "token_idx": t,
            "token": tokenizer.convert_ids_to_tokens(int(ids[b, t])),
            "head": head, "target_idx": t + shift,
            "correct": bool(hit1[b, t]), "rank": int(rank[b, t]),
            "group_start": int(gs[b, t]) if gs is not None else None,
            "group_id": int(gid[b, t]) if gid is not None else None,
            "target_group_id": int(gid[b, t + shift]) if gid is not None else None,
        }, ensure_ascii=False) + "\n")

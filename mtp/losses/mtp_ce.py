"""
Per-head shifted cross-entropy (ported from compute_head_loss in
train_mtp_baseline.py).

Head d (d = 0..k-1) at position t predicts the token at t+d+1, so its
predictions are logits[d][:, :T-s] against input_ids[:, s:] with shift
s = d+1. A position counts only if both the source token t and the target
token t+s are real (attention_mask == 1). With right padding that is the
same as the old target-only mask.
"""

import torch
import torch.nn.functional as F


def shift_targets(input_ids, attention_mask, shift: int):
    """Targets and validity mask for a head that predicts `shift` tokens ahead.

    Returns (targets [B, T-shift] long, valid [B, T-shift] bool). Both are
    empty along dim 1 when T <= shift.
    """
    if input_ids.shape[1] <= shift:
        empty = input_ids[:, :0]
        return empty, empty.bool()
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    targets = input_ids[:, shift:]
    valid = (attention_mask[:, shift:] != 0) & (attention_mask[:, :-shift] != 0)
    return targets, valid


def per_head_ce(logits_list, input_ids, attention_mask=None, reduction: str = "none"):
    """
    reduction="none": list of k tensors, element d is [B, T-d-1] fp32 per-position
                      CE, 0 at invalid positions (use shift_targets for the mask).
    reduction="mean": list of k scalars, each the mean over that head's valid
                      positions. 0 (still attached to the graph) if a head has
                      no valid position, so a short batch never yields NaN.
    reduction="sum":  list of k scalars, sum over valid positions.
    """
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"reduction must be 'none', 'mean' or 'sum', got {reduction!r}")

    out = []
    for d, logits in enumerate(logits_list):
        shift = d + 1
        targets, valid = shift_targets(input_ids, attention_mask, shift)
        B, T, V = logits.shape
        n = max(T - shift, 0)
        # CE in fp32 even under fp16/bf16 autocast.
        pred = logits[:, :n, :].float()
        losses = F.cross_entropy(
            pred.reshape(-1, V),
            targets.masked_fill(~valid, -100).reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(B, n)

        if reduction == "none":
            out.append(losses)
        elif reduction == "sum":
            out.append(losses.sum())
        else:
            out.append(losses.sum() / valid.sum().clamp(min=1))
    return out

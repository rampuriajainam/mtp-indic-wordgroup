"""
Word-group structural losses (JN-5). Design and evidence: docs/design_structural_loss.md.

Notation: head d at position t predicts u = t+d+1; g = group_id (-1 = special/pad),
s = group_start. A position is valid if t and u are real tokens (attention_mask)
with g >= 0.

S2  boundary probe, every head d:
      struct/boundary_bce/h{d} = mean_valid BCE(boundary_logits[d][t], s[u])
    The probes live in MTPModel (boundary_probes=True); this only scores them.

S3  consistency, every extra head d >= 1:
      struct/consistency/h{d} = mean_mask KL( sg[p_teacher] || p_d(.|t) )
    teacher "h0":    head 0 at u-1 (the LM, which has seen every token up to u-1)
    teacher "chain": head d-1 at t+1
    mask "in_group": g(t+1) == g(u)  (the extra context the teacher saw lies in u's group)
    mask "all":      every valid pair (control: plain self-distillation)

Variants: S2 | S3_h0 | S3_chain | S3_all | S23 (S2 + S3 with s3_teacher, default h0).
Terms are returned unweighted; train.py passes lambda_s2 / lambda_s3 to LossWeighting
under the prefixes "struct/boundary_bce" and "struct/consistency".
"""

import torch.nn.functional as F

VARIANTS = {
    "S2": {"s2": True, "s3": None},
    "S3_h0": {"s2": False, "s3": ("h0", "in_group")},
    "S3_chain": {"s2": False, "s3": ("chain", "in_group")},
    "S3_all": {"s2": False, "s3": ("h0", "all")},
    "S23": {"s2": True, "s3": ("h0", "in_group")},
}


def needs_boundary_probes(variant) -> bool:
    return variant in VARIANTS and VARIANTS[variant]["s2"]


def _zero(like):
    """0 that stays attached to the graph, so a batch with no valid position never gives NaN."""
    return like.sum() * 0.0


class StructuralLoss:
    def __init__(self, variant: str, num_heads: int, s3_teacher: str = "h0", **kw):
        if variant not in VARIANTS:
            raise ValueError(f"unknown structural variant {variant!r}; one of {sorted(VARIANTS)}")
        spec = VARIANTS[variant]
        self.variant = variant
        self.num_heads = num_heads
        self.use_s2 = spec["s2"]
        self.s3 = spec["s3"]
        if variant == "S23":
            if s3_teacher not in ("h0", "chain"):
                raise ValueError(f"s3_teacher must be 'h0' or 'chain', got {s3_teacher!r}")
            self.s3 = (s3_teacher, "in_group")
        self.term_names = (
            [f"struct/boundary_bce/h{d}" for d in range(num_heads)] if self.use_s2 else []
        ) + (
            [f"struct/consistency/h{d}" for d in range(1, num_heads)] if self.s3 else []
        )

    def __call__(self, out, batch) -> dict:
        ids = batch["input_ids"]
        real = batch["attention_mask"] != 0
        gid = batch["group_id"]
        ok = real & (gid >= 0)
        T = ids.shape[1]
        terms = {}

        if self.use_s2:
            if "boundary_logits" not in out.aux:
                raise KeyError("S2 needs MTPModel(boundary_probes=True)")
            start = batch["group_start"].float()
            for d, logit in enumerate(out.aux["boundary_logits"]):
                s = d + 1
                if T <= s:
                    terms[f"struct/boundary_bce/h{d}"] = _zero(logit)
                    continue
                valid = ok[:, :-s] & ok[:, s:]
                bce = F.binary_cross_entropy_with_logits(logit[:, :-s].float(), start[:, s:], reduction="none")
                terms[f"struct/boundary_bce/h{d}"] = (bce * valid).sum() / valid.sum().clamp(min=1)

        if self.s3:
            teacher, mask_kind = self.s3
            for d in range(1, self.num_heads):
                n = T - d - 1                       # student positions t = 0..T-d-2
                name = f"struct/consistency/h{d}"
                if n <= 0:
                    terms[name] = _zero(out.logits[d])
                    continue
                stu = out.logits[d][:, :n]
                tea = out.logits[0][:, d:T - 1] if teacher == "h0" else out.logits[d - 1][:, 1:T - d]
                ok_t, ok_t1, ok_u = ok[:, :n], ok[:, 1:n + 1], ok[:, d + 1:]
                mask = ok_t & ok_t1 & ok_u
                if mask_kind == "in_group":
                    mask = mask & (gid[:, 1:n + 1] == gid[:, d + 1:])
                if not mask.any():
                    terms[name] = _zero(stu)
                    continue
                log_s = F.log_softmax(stu[mask].float(), dim=-1)
                log_t = F.log_softmax(tea[mask].detach().float(), dim=-1)
                terms[name] = (log_t.exp() * (log_t - log_s)).sum(-1).mean()
        return terms

"""
Contrastive word-group loss (JN-6): supervised contrastive (SupCon, Khosla et al. 2020) over tokens.

MTPModel(contrastive_dim=128) projects the last hidden state with a 2-layer MLP and
L2-normalises it (out.aux["contrastive_z"], [B, T, D]). Anchors and candidates are the valid
tokens of the batch (attention_mask and group_id >= 0), subsampled to at most max_tokens.
Positives of an anchor = the other sampled tokens in the same (sentence, group).

    contrastive/supcon = mean over anchors with >= 1 positive of
        - mean_{p in P(i)} [ z_i.z_p / tau - log sum_{a != i} exp(z_i.z_a / tau) ]

Groups are short (~1.05-1.5 words, a few tokens), so many anchors have no positive; they are
skipped. The term is returned unweighted; train.py weights it with losses.contrastive.weight.
It reaches the backbone (through LoRA), so it can move head 0: watch h0 against R3.
"""

import torch


class SupConLoss:
    term_names = ["contrastive/supcon"]

    def __init__(self, temperature: float = 0.1, max_tokens: int = 512):
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, out, batch) -> dict:
        z = out.aux.get("contrastive_z")
        if z is None:
            raise KeyError("contrastive loss needs MTPModel(contrastive_dim > 0)")
        gid = batch["group_id"]
        ok = (batch["attention_mask"] != 0) & (gid >= 0)
        b_idx, t_idx = ok.nonzero(as_tuple=True)
        if b_idx.numel() > self.max_tokens:
            keep = torch.randperm(b_idx.numel(), device=b_idx.device)[: self.max_tokens]
            b_idx, t_idx = b_idx[keep], t_idx[keep]
        if b_idx.numel() < 2:
            return {"contrastive/supcon": z.sum() * 0.0}
        feats = z[b_idx, t_idx].float()                               # [N, D], unit norm
        label = b_idx * (int(gid.max()) + 1) + gid[b_idx, t_idx]      # (sentence, group) key
        n = feats.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=feats.device)
        sim = (feats @ feats.T / self.temperature).masked_fill(eye, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos = (label[:, None] == label[None, :]) & ~eye
        n_pos = pos.sum(1)
        has = n_pos > 0
        if not has.any():
            return {"contrastive/supcon": z.sum() * 0.0}
        mean_log_prob_pos = (log_prob.masked_fill(~pos, 0.0).sum(1)[has]) / n_pos[has]
        return {"contrastive/supcon": -mean_log_prob_pos.mean()}

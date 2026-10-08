"""
Loss weighting (JN-7). Only "fixed" exists so far; "uncertainty" and "dwa"
come with JN-7.

Term names follow the metrics contract: per-head CE terms are "loss/head{d}",
auxiliary terms are anything else (e.g. "struct/boundary_bce/h1").

fixed: CE term for head d gets head_decay**d (Medusa uses 0.8; 1.0 reproduces
the laptop runs, which summed the heads). Aux terms get aux_weights[name], or
aux_weights[prefix] for the longest matching "/"-prefix, else 1.0.
"""

import re

import torch
import torch.nn as nn

_HEAD_TERM = re.compile(r"loss/head(\d+)$")


class LossWeighting(nn.Module):
    def __init__(self, scheme: str, term_names, head_decay: float = 0.8, aux_weights: dict | None = None):
        super().__init__()
        if scheme != "fixed":
            raise NotImplementedError(f"weighting scheme {scheme!r} lands with JN-7; only 'fixed' exists")
        self.scheme = scheme
        self.term_names = list(term_names)
        self.head_decay = head_decay
        self.aux_weights = dict(aux_weights or {})

    def _fixed_weight(self, name):
        m = _HEAD_TERM.match(name)
        if m:
            return self.head_decay ** int(m.group(1))
        parts = name.split("/")
        for i in range(len(parts), 0, -1):
            key = "/".join(parts[:i])
            if key in self.aux_weights:
                return float(self.aux_weights[key])
        return 1.0

    def forward(self, losses: dict):
        unknown = set(losses) - set(self.term_names)
        if unknown:
            raise KeyError(f"unregistered loss terms: {sorted(unknown)}")
        weights = {name: self._fixed_weight(name) for name in losses}
        total = sum(weights[n] * losses[n] for n in losses)
        if not torch.is_tensor(total):
            raise ValueError("no loss terms given")
        return total, weights

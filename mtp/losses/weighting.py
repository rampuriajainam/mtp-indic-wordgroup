"""
Loss weighting (JN-7): LossWeighting(scheme, term_names)(losses) -> (total, effective_weights).

Term names follow the metrics contract: per-head CE terms are "loss/head{d}",
auxiliary terms are anything else (e.g. "struct/boundary_bce/h1").

Every scheme starts from the FIXED weight of each term:
  CE term of head d: head_decay**d (Medusa uses 0.8; 1.0 = plain sum, as on the laptop)
  aux term: aux_weights[longest matching "/"-prefix of its name], else 1.0

fixed        total = sum_i w_i L_i
uncertainty  Kendall et al. 2018: learnable s_i = log sigma_i^2, clamped to [-4, 4],
             total = sum_i 0.5*exp(-s_i) L_i + 0.5*s_i. s_i starts at -log(2 w_i), so the
             effective weight 0.5*exp(-s_i) starts at the fixed weight.
dwa          Dynamic Weight Average (Liu et al. 2019), rescaling the fixed weights:
             w_i = fixed_i * n * softmax_i(r / T), r_i = Lbar_i(window-1) / Lbar_i(window-2),
             Lbar = mean loss over a window of `dwa_window` calls. Plain fixed weights
             until two windows have been seen. State is in buffers, so it checkpoints.

fix_head0 (uncertainty, dwa): head 0 keeps weight 1 and is left out of the scheme, so
the LM is never de-prioritised.
"""

import math
import re

import torch
import torch.nn as nn

_HEAD_TERM = re.compile(r"loss/head(\d+)$")
SCHEMES = ("fixed", "uncertainty", "dwa")


class LossWeighting(nn.Module):
    def __init__(self, scheme: str, term_names, head_decay: float = 0.8, aux_weights: dict | None = None,
                 fix_head0: bool = True, dwa_window: int = 20, dwa_temperature: float = 2.0, **kw):
        super().__init__()
        if scheme not in SCHEMES:
            raise ValueError(f"weighting scheme must be one of {SCHEMES}, got {scheme!r}")
        self.scheme = scheme
        self.term_names = list(term_names)
        self.head_decay = head_decay
        self.aux_weights = dict(aux_weights or {})
        self.fix_head0 = fix_head0
        self.dwa_window = dwa_window
        self.dwa_temperature = dwa_temperature

        self.fixed = {n: self._fixed_weight(n) for n in self.term_names}
        self.adaptive = [n for n in self.term_names if not (fix_head0 and n == "loss/head0")]
        self._idx = {n: i for i, n in enumerate(self.adaptive)}
        n = len(self.adaptive)
        if scheme == "uncertainty":
            init = [-math.log(2 * self.fixed[t]) for t in self.adaptive]
            self.log_var = nn.Parameter(torch.tensor(init, dtype=torch.float32).clamp(-4, 4))
        elif scheme == "dwa":
            for name in ("win_sum", "prev", "prev2"):
                self.register_buffer(name, torch.zeros(n))
            self.register_buffer("win_count", torch.zeros((), dtype=torch.long))
            self.register_buffer("windows_done", torch.zeros((), dtype=torch.long))
            self.register_buffer("scale", torch.ones(n))

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
        if not losses:
            raise ValueError("no loss terms given")
        if self.scheme == "uncertainty":
            return self._uncertainty(losses)
        if self.scheme == "dwa":
            return self._dwa(losses)
        weights = {n: self.fixed[n] for n in losses}
        return sum(weights[n] * losses[n] for n in losses), weights

    def _uncertainty(self, losses):
        s = self.log_var.clamp(-4, 4)
        total, weights = 0.0, {}
        for name, value in losses.items():
            if name in self._idx:
                si = s[self._idx[name]]
                total = total + 0.5 * torch.exp(-si) * value + 0.5 * si
                weights[name] = 0.5 * math.exp(-si.item())
            else:
                total = total + value
                weights[name] = 1.0
        return total, weights

    def _dwa(self, losses):
        if self.training:
            with torch.no_grad():
                for name, value in losses.items():
                    if name in self._idx:
                        self.win_sum[self._idx[name]] += value.detach().float()
                self.win_count += 1
                if self.win_count >= self.dwa_window:
                    mean = self.win_sum / self.win_count
                    self.prev2.copy_(self.prev)
                    self.prev.copy_(mean)
                    self.win_sum.zero_()
                    self.win_count.zero_()
                    self.windows_done += 1
                    if self.windows_done >= 2:
                        r = self.prev / self.prev2.clamp(min=1e-8)
                        self.scale.copy_(len(self.adaptive) * torch.softmax(r / self.dwa_temperature, dim=0))
        weights = {n: self.fixed[n] * (self.scale[self._idx[n]].item() if n in self._idx else 1.0) for n in losses}
        return sum(weights[n] * losses[n] for n in losses), weights

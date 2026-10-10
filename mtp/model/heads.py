"""
Multi-token prediction heads on top of a causal LM (INTERFACES §5).

Head 0 is the base model's own LM head (next token). Heads 1..k-1 predict
tokens further ahead; head d at position t predicts the token at t+d+1.
All heads read the same last hidden state h (after the final norm).

head_type="linear"   : head d = fresh nn.Linear(hidden, vocab), random init.
                       Same as the old MedusaWrapper; used to reproduce R1.
head_type="resblock" : Medusa-1. h_d = ResBlock^n(h), logits_d = lm_head(h_d),
                       with ResBlock(x) = x + SiLU(W x + b) and W, b zero-init,
                       so every head starts as an exact copy of head 0. The
                       base lm_head is shared (frozen under LoRA), so each
                       extra head costs only n_layers * (hidden^2 + hidden).
head_type="seq"      : sequential (Hydra / EAGLE-style) heads. Head d also reads the
                       tokens it drafts on top of: s_0 = h_t,
                       s_d = s_{d-1} + SiLU(W_d [s_{d-1}; e(x_{t+d})] + b_d), logits_d = lm_head(s_d),
                       e = the base input embeddings (detached), W_d, b_d zero-init (so it
                       starts as head 0, like resblock). In forward() x_{t+d} is the input
                       token at t+d (teacher forcing; zeros past the end), so at the last
                       positions the extra-head logits are not drafts: decoding calls
                       draft_chain(), which feeds head 0's token and then each draft.

backbone_grad scales the gradient that heads 1..k-1 send back into the shared
hidden state (and so into LoRA): 1.0 = full joint training, 0.0 = the extra
heads train on a detached copy (LoRA learns from head 0 only, Medusa-1 style).
The forward pass is identical for every value.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_TYPES = ("linear", "resblock", "seq")


@dataclass
class MTPOutput:
    logits: list[torch.Tensor]   # length k; logits[i] is [B, T, V]; head i predicts token t+i+1
    hidden: torch.Tensor         # [B, T, H] last hidden state (for contrastive loss)
    aux: dict = field(default_factory=dict)
    # aux["head_hidden"]: list of k tensors [B, T, H], the state each head's
    #   output layer reads (h_0 = hidden). Structural-loss probes sit on these.
    # aux["boundary_logits"]: list of k tensors [B, T], present when the model has
    #   boundary probes (S2): logit that token t+d+1 starts a new word group.
    # aux["past_key_values"]: present only when forward(use_cache=True).


class ResBlock(nn.Module):
    """x + SiLU(W x + b), zero-initialised so it starts as the identity."""

    def __init__(self, hidden_size, dtype=None, device=None):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size, dtype=dtype, device=device)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return x + F.silu(self.linear(x))


class SeqBlock(nn.Module):
    """s + SiLU(W [s; e] + b): one sequential head step, zero-initialised (starts as the identity in s)."""

    def __init__(self, hidden_size, dtype=None, device=None):
        super().__init__()
        self.linear = nn.Linear(2 * hidden_size, hidden_size, dtype=dtype, device=device)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, s, e):
        return s + F.silu(self.linear(torch.cat([s, e], dim=-1)))


class MTPModel(nn.Module):
    """
    Wraps a HF causal LM (optionally PEFT/LoRA-wrapped) with k prediction heads.

    num_heads counts ALL heads including head 0 (num_heads = num_extra_heads + 1).
    Extra heads live in self.extra_heads (len num_heads - 1). With
    boundary_probes=True every head d also gets a linear probe on h_d
    (self.boundary_probes, len num_heads) for the S2 structural loss and the
    GroupAware draft policy. Checkpoints save head_state_dict(): everything
    except the base model.
    """

    def __init__(self, base_model, num_heads: int, head_type: str = "linear", n_layers: int = 1,
                 backbone_grad: float = 1.0, boundary_probes: bool = False, contrastive_dim: int = 0, **kw):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if head_type not in HEAD_TYPES:
            raise ValueError(f"head_type must be one of {HEAD_TYPES}, got {head_type!r}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if not 0.0 <= backbone_grad <= 1.0:
            raise ValueError(f"backbone_grad must be in [0, 1], got {backbone_grad}")

        self.base_model = base_model
        self.num_heads = num_heads
        self.head_type = head_type
        self.n_layers = n_layers
        self.backbone_grad = backbone_grad

        lm_head = self._lm_head()
        hidden_size = lm_head.in_features
        vocab_size = lm_head.out_features
        # Match the base weights' dtype/device; mtp.device decides what that is.
        factory = {"dtype": lm_head.weight.dtype, "device": lm_head.weight.device}

        if head_type == "linear":
            self.extra_heads = nn.ModuleList([
                nn.Linear(hidden_size, vocab_size, bias=False, **factory)
                for _ in range(num_heads - 1)
            ])
        elif head_type == "seq":
            self.extra_heads = nn.ModuleList([SeqBlock(hidden_size, **factory) for _ in range(num_heads - 1)])
        else:
            self.extra_heads = nn.ModuleList([
                nn.Sequential(*[ResBlock(hidden_size, **factory) for _ in range(n_layers)])
                for _ in range(num_heads - 1)
            ])
        self.boundary_probes = nn.ModuleList(
            [nn.Linear(hidden_size, 1, **factory) for _ in range(num_heads)] if boundary_probes else []
        )
        # JN-6: 2-layer projection of the last hidden state for the contrastive loss (None = off)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, **factory), nn.GELU(),
            nn.Linear(hidden_size, contrastive_dim, **factory),
        ) if contrastive_dim > 0 else None

    def head_state_dict(self):
        """Trainable non-base parameters (extra heads + probes), for heads.pt."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("base_model.")}

    def load_head_state_dict(self, state):
        """Inverse of head_state_dict. Also accepts the legacy MedusaWrapper /
        extra_heads-only format ("0.weight", ...)."""
        if state and not any(k.startswith(("extra_heads.", "boundary_probes.", "contrastive_proj.")) for k in state):
            state = {f"extra_heads.{k}": v for k, v in state.items()}
        own = self.head_state_dict()
        missing, unexpected = own.keys() - state.keys(), state.keys() - own.keys()
        if missing or unexpected:
            raise KeyError(f"head state mismatch: missing {sorted(missing)[:3]}, unexpected {sorted(unexpected)[:3]}")
        self.load_state_dict(state, strict=False)

    # The decoder and lm_head are looked up through base_model on every call
    # rather than stored, so they are not registered twice in state_dict().
    # PEFT forwards attribute access to the wrapped model, and LoRA layers are
    # injected in place, so calling the inner decoder still applies LoRA.
    def _decoder(self):
        return self.base_model.get_decoder()

    def _lm_head(self):
        return self.base_model.get_output_embeddings()

    def forward(self, input_ids, attention_mask=None, use_cache: bool = False, extra_heads: bool = True,
                **kw) -> MTPOutput:
        out = self._decoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kw,
        )
        hidden = out.last_hidden_state  # after the final norm, same input lm_head sees
        lm_head = self._lm_head()

        logits = [lm_head(hidden)]
        head_hidden = [hidden]
        a = self.backbone_grad
        h_in = hidden if a == 1.0 else (hidden.detach() if a == 0.0 else a * hidden + (1 - a) * hidden.detach())
        # extra_heads=False: head 0 only (decoding with seq heads, whose drafts come from draft_chain)
        if extra_heads and self.head_type == "seq":
            emb = self.base_model.get_input_embeddings()(input_ids).detach()
            s = h_in
            for d, head in enumerate(self.extra_heads, start=1):
                e = torch.zeros_like(emb)
                if emb.shape[1] > d:
                    e[:, : emb.shape[1] - d] = emb[:, d:]      # x_{t+d} at position t
                s = head(s, e.to(s.dtype))
                logits.append(lm_head(s))
                head_hidden.append(s)
        elif extra_heads:
            for head in self.extra_heads:
                if self.head_type == "linear":
                    logits.append(head(h_in))
                    head_hidden.append(h_in)
                else:
                    h_d = head(h_in)
                    logits.append(lm_head(h_d))
                    head_hidden.append(h_d)

        aux = {"head_hidden": head_hidden}
        if len(self.boundary_probes):
            aux["boundary_logits"] = [probe(h).squeeze(-1) for probe, h in zip(self.boundary_probes, head_hidden)]
        if self.contrastive_proj is not None:
            aux["contrastive_z"] = F.normalize(self.contrastive_proj(hidden).float(), dim=-1)
        if use_cache:
            aux["past_key_values"] = out.past_key_values
        return MTPOutput(logits=logits, hidden=hidden, aux=aux)

    @torch.no_grad()
    def draft_chain(self, h, t0: int, n: int, return_logits: bool = False):
        """Greedy chain drafts for head_type "seq": h = last hidden state [H] at the position whose
        head-0 token is t0. Returns the n drafted tokens for t+2 .. t+n+1 (and, with return_logits,
        the logits [V] each draft was taken from)."""
        embed, lm_head = self.base_model.get_input_embeddings(), self._lm_head()
        s, tok, out, logits = h, t0, [], []
        for head in self.extra_heads[:n]:
            e = embed(torch.tensor([tok], device=h.device))[0]
            s = head(s, e.to(s.dtype))
            lg = lm_head(s)
            tok = int(lg.argmax())
            out.append(tok)
            logits.append(lg)
        return (out, logits) if return_logits else out

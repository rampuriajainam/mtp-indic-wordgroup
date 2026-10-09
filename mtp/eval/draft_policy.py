"""
Draft-length policies for self-speculative decoding (INTERFACES §11).

Each step the engine (mtp.eval.spec_decode) shows the policy every head's logits at the last
position and asks how many of the k-1 drafts (heads 1..k-1, in order) to propose. Whatever a
policy returns, the output stays identical to greedy decoding: head 0 verifies every draft.
"""

from typing import Protocol

import torch


class DraftPolicy(Protocol):
    name: str

    def num_draft_tokens(self, head_logits: list[torch.Tensor], context_ids: torch.Tensor, step_state: dict) -> int:
        """head_logits: k tensors [V] at the last position (index 0 = head 0). context_ids: [T] tokens so far.
        Return how many of the k-1 drafts to propose, 0..k-1."""


class FixedK:
    """Always propose all k-1 drafts (the Medusa default)."""

    name = "fixed_k"

    def num_draft_tokens(self, head_logits, context_ids, step_state):
        return len(head_logits) - 1


class ConfidenceCut:
    """Propose drafts while each head is confident: stop at the first head d >= 1 whose max softmax
    probability is below tau."""

    name = "confidence_cut"

    def __init__(self, tau: float = 0.5):
        self.tau = tau

    def num_draft_tokens(self, head_logits, context_ids, step_state):
        n = 0
        for logits in head_logits[1:]:
            if torch.softmax(logits.float(), dim=-1).max().item() < self.tau:
                break
            n += 1
        return n


POLICIES = {"fixed_k": FixedK, "confidence_cut": ConfidenceCut}
try:  # Jainam's policy (JN-9) lives in its own file so we never edit the same one
    from mtp.eval.group_aware import GroupAware

    POLICIES["group_aware"] = GroupAware
except ImportError:
    pass


def get_policy(name: str, **kwargs) -> DraftPolicy:
    if name not in POLICIES:
        raise KeyError(f"unknown draft policy {name!r}; available: {sorted(POLICIES)}")
    return POLICIES[name](**kwargs)

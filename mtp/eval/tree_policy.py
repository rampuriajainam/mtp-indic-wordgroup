"""
Tree draft policies (JN-9, issue #33). Om's engine (generate_tree) verifies the tree; this file
only decides its shape.

A tree is a list of nodes (parent, d, rank), in an order where every parent comes before its
children:
  parent  index of the parent node in the list, -1 = the root (head 0's token t0, always kept)
  d       the head that drafts this node, 1..k-1; always parent's d + 1 (the root has d = 0)
  rank    0-based rank in head d's logits at the last position (0 = argmax)
So node (p, d, r) drafts head d's r-th most likely token as the continuation of node p. Every
head drafts from the same last position (Medusa), so a node's token does not depend on its path.

Policies (issue #33, step 1: word groups do not help, entropy does a little):
  StaticTree   the same tree every step: the max_nodes most frequent rank prefixes on
               calibration text (Medusa's sparse tree), or a product of widths like (3, 2, 1)
  EntropyTree  one static tree per entropy cell: bit d = head d's entropy above its calibration
               median, so 2^(k-1) trees, each with max_nodes nodes
Fit them with fit_static / fit_entropy on ranks from collect_ranks, or scripts/fit_tree.py.
"""

import json
from collections import Counter
from pathlib import Path

import torch


def nodes_from_prefixes(prefixes):
    """Rank tuples (prefix-closed) -> node list. (r1,) is head 1's r1-th token after t0, etc."""
    order = sorted(set(prefixes), key=lambda p: (len(p), p))
    index, nodes = {}, []
    for p in order:
        parent = -1 if len(p) == 1 else index.get(p[:-1])
        if parent is None:
            raise ValueError(f"rank tuples are not prefix-closed: {p[:-1]} missing for {p}")
        index[p] = len(nodes)
        nodes.append((parent, len(p), p[-1]))
    return nodes


def prefixes_from_nodes(nodes):
    out = []
    for parent, d, rank in nodes:
        out.append(((out[parent] if parent >= 0 else ()) + (rank,)))
    return out


def check_tree(nodes, num_heads=None):
    for i, (parent, d, rank) in enumerate(nodes):
        if not -1 <= parent < i:
            raise ValueError(f"node {i}: parent {parent} must come before it")
        if d != (nodes[parent][1] + 1 if parent >= 0 else 1):
            raise ValueError(f"node {i}: head {d} is not its parent's head + 1")
        if rank < 0 or (num_heads is not None and d >= num_heads):
            raise ValueError(f"node {i}: bad head {d} / rank {rank}")
    if len(set(prefixes_from_nodes(nodes))) != len(nodes):
        raise ValueError("duplicate nodes")


def draft_tokens(nodes, head_logits):
    """Token id of every node: head d's rank-th token. head_logits: k tensors [V] (index 0 = head 0)."""
    need = {}
    for _, d, rank in nodes:
        need[d] = max(need.get(d, 0), rank + 1)
    top = {d: head_logits[d].topk(r).indices.tolist() for d, r in need.items()}
    return [top[d][rank] for _, d, rank in nodes]


def accepted_length(nodes, ranks):
    """Drafts accepted when the greedy continuation has ranks (r1, r2, ...) under heads 1, 2, ...:
    the longest prefix of ranks that is a node (the oracle; equals the engine's count for a
    lossless verifier)."""
    have = set(prefixes_from_nodes(nodes))
    j = 0
    while j < len(ranks) and tuple(ranks[: j + 1]) in have:
        j += 1
    return j


def fit_prefixes(ranks, max_nodes):
    """The max_nodes most frequent rank prefixes of `ranks` ([n, k-1] ints, 0-based; a negative
    or >= cap value ends the prefix). Frequencies only fall along a path, so the top set is
    prefix-closed (ties: shorter first)."""
    c = Counter()
    for r in ranks:
        for j in range(len(r)):
            if r[j] < 0:
                break
            c[tuple(int(x) for x in r[: j + 1])] += 1
    return sorted(c, key=lambda p: (-c[p], len(p), p))[:max_nodes]


def widths_tree(widths):
    """Full product tree, e.g. (3, 2, 1): 3 + 6 + 6 = 15 nodes. (1, 1, 1) = the FixedK chain."""
    prefixes = [()]
    out = []
    for w in widths:
        prefixes = [p + (r,) for p in prefixes for r in range(w)]
        out += prefixes
    return nodes_from_prefixes(out)


def entropy(logits):
    lp = torch.log_softmax(logits.float(), -1)
    return float(-(lp.exp() * lp).sum())


class StaticTree:
    name = "static_tree"

    def __init__(self, nodes, num_heads=None):
        check_tree(nodes, num_heads)
        self.nodes = [tuple(n) for n in nodes]
        self.max_nodes = len(self.nodes)

    def tree(self, head_logits, context_ids, step_state):
        return self.nodes

    @classmethod
    def from_widths(cls, widths):
        return cls(widths_tree(widths))

    def to_json(self):
        return {"policy": self.name, "nodes": self.nodes}


class EntropyTree:
    name = "entropy_tree"

    def __init__(self, medians, trees, fallback):
        """medians: k-1 entropy thresholds (heads 1..k-1); trees: {cell: nodes}, cell = sum of
        2^(d-1) over heads d above their median; fallback: nodes for cells without a tree."""
        self.medians = [float(m) for m in medians]
        self.trees = {int(c): [tuple(n) for n in t] for c, t in trees.items()}
        self.fallback = [tuple(n) for n in fallback]
        for t in list(self.trees.values()) + [self.fallback]:
            check_tree(t, len(self.medians) + 1)
        self.max_nodes = max(len(t) for t in list(self.trees.values()) + [self.fallback])

    def cell(self, head_logits):
        return sum(1 << (d - 1) for d in range(1, len(self.medians) + 1)
                   if entropy(head_logits[d]) > self.medians[d - 1])

    def tree(self, head_logits, context_ids, step_state):
        return self.trees.get(self.cell(head_logits), self.fallback)

    def to_json(self):
        return {"policy": self.name, "medians": self.medians,
                "trees": {str(c): t for c, t in self.trees.items()}, "fallback": self.fallback}


TREE_POLICIES = {"static_tree": StaticTree, "entropy_tree": EntropyTree}


def fit_static(ranks, max_nodes):
    return StaticTree(nodes_from_prefixes(fit_prefixes(ranks, max_nodes)))


def fit_entropy(ranks, entropies, max_nodes, min_cell=100):
    """ranks, entropies: [n, k-1]. Cells with fewer than min_cell positions use the static tree."""
    medians = [sorted(e)[len(e) // 2] for e in zip(*entropies)]
    cells = [sum(1 << d for d, (e, m) in enumerate(zip(row, medians)) if e > m) for row in entropies]
    fallback = nodes_from_prefixes(fit_prefixes(ranks, max_nodes))
    trees = {}
    for c in set(cells):
        rows = [r for r, cc in zip(ranks, cells) if cc == c]
        if len(rows) >= min_cell:
            trees[c] = nodes_from_prefixes(fit_prefixes(rows, max_nodes))
    return EntropyTree(medians, trees, fallback)


def save_policy(policy, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(policy.to_json()), encoding="utf-8")


def load_policy(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d["policy"] == "static_tree":
        return StaticTree(d["nodes"])
    if d["policy"] == "entropy_tree":
        return EntropyTree(d["medians"], d["trees"], d["fallback"])
    raise KeyError(f"unknown tree policy {d['policy']!r}; known: {sorted(TREE_POLICIES)}")


@torch.no_grad()
def collect_ranks(model, tokenizer, prompts, max_new_tokens=64, cap=50, amp=None):
    """Calibration data: greedy-generate from each prompt, then at every generated position the
    0-based rank of the greedy token t+d+1 under head d (-1 if >= cap) and head d's entropy.
    Returns (ranks, entropies), lists of [k-1] rows."""
    from mtp.eval.spec_decode import greedy_generate
    import contextlib

    amp = amp or contextlib.nullcontext
    device = next(model.parameters()).device
    k = model.num_heads
    ranks, ents = [], []
    for p in prompts:
        new, _ = greedy_generate(model, tokenizer, p, max_new_tokens, amp=amp)
        full = list(p) + new
        with amp():
            out = model(torch.tensor([full], device=device))
        for t in range(len(p) - 1, len(full) - k):
            r, e = [], []
            for d in range(1, k):
                lg = out.logits[d][0, t].float()
                rank = int((lg > lg[full[t + d + 1]]).sum())
                r.append(rank if rank < cap else -1)
                e.append(entropy(lg))
            ranks.append(r)
            ents.append(e)
    return ranks, ents

"""Shaped-tree oracle (issue #33, step 1). CPU only, reads tree_collect.py output.

python scripts/analysis/tree_shape.py <run.npz> [budgets...]   (issue #33: budgets 3 7 15 25)
A tree = a prefix-closed set of rank tuples (a1,..,aj): node = head 1's a1-th token, then head 2's
a2-th, ... A position accepts the longest prefix of its true ranks (r1, r2, r3) that is a node.
For each condition (a few bits known per position) the best tree with N nodes is the N most frequent
rank prefixes on the calibration half (Medusa's sparse tree); every condition gets the same N, as
with a per-pass max_nodes cap. 2-fold cross-fit over prompts; paired bootstrap over prompts.

Signals (3 bits each, one per drafted token):
  static     no signal (one tree for every position)
  entropy    head d's entropy above its median (realistic: the policy has it)
  word       drafted token starts a new word           } ORACLE: true labels of
  hi_rules   drafted token starts a new hi_rules group } future tokens = ceilings
  random     ... a new RandomGrouper group (R6a control)}
  ent+X      entropy bits + X bits (falls back to the entropy-only tree when a cell is rare)
"""
import sys
from collections import Counter

import numpy as np

z = np.load(sys.argv[1])
budgets = [int(b) for b in sys.argv[2:]] or [3, 7, 15, 25]
rank, ent, prompt = z["rank"], z["ent"], z["prompt"]
n, D = rank.shape
CAP = rank.max()
MIN_CELL = 100
B = 2000


def bits(m):  # bool (n, D) -> int code
    return (m.astype(int) * (1 << np.arange(m.shape[1]))).sum(1)


def build(idx, N):
    c = Counter()
    for r in rank[idx]:
        for j in range(1, D + 1):
            if r[j - 1] >= CAP:
                break
            c[tuple(r[:j])] += 1
    return set(sorted(c, key=lambda k: (-c[k], len(k), k))[:N])


def accepted(idx, tree):
    out = np.zeros(len(idx))
    for i, r in enumerate(rank[idx]):
        j = 0
        while j < D and tuple(r[:j + 1]) in tree:
            j += 1
        out[i] = j
    return out


def evaluate(cond_fn, N):
    """Per-position accepted drafts, cross-fitted. cond_fn(cal_idx) -> (codes for all, fallback codes)."""
    res = np.zeros(n)
    folds = prompt % 2
    for f in (0, 1):
        cal, test = np.where(folds != f)[0], np.where(folds == f)[0]
        codes, fallback = cond_fn(cal)
        static = build(cal, N)
        for code in np.unique(codes[test]):
            tc = test[codes[test] == code]
            cc = cal[codes[cal] == code]
            if len(cc) >= MIN_CELL:
                tree = build(cc, N)
            elif fallback is not None:
                fc = cal[fallback[cal] == fallback[tc[0]]]
                tree = build(fc, N) if len(fc) >= MIN_CELL else static
            else:
                tree = static
            res[tc] = accepted(tc, tree)
    return res


def ent_bits(cal):
    return bits(ent > np.median(ent[cal], 0))


signals = {
    "static": lambda cal: (np.zeros(n, int), None),
    "entropy": lambda cal: (ent_bits(cal), None),
    "word": lambda cal: (bits(z["gs_word"]), None),
    "hi_rules": lambda cal: (bits(z["gs_hi_rules"]), None),
    "random": lambda cal: (bits(z["gs_random"]), None),
    "ent+word": lambda cal: (ent_bits(cal) * 8 + bits(z["gs_word"]), ent_bits(cal)),
    "ent+hi_rules": lambda cal: (ent_bits(cal) * 8 + bits(z["gs_hi_rules"]), ent_bits(cal)),
    "ent+random": lambda cal: (ent_bits(cal) * 8 + bits(z["gs_random"]), ent_bits(cal)),
}
rng = np.random.default_rng(0)
P = np.unique(prompt)
boot = [rng.choice(P, len(P)) for _ in range(B)]
pos_of = {p: np.where(prompt == p)[0] for p in P}


def per_prompt(x):
    return np.array([x[pos_of[p]].sum() for p in P]), np.array([len(pos_of[p]) for p in P])


def diff_ci(a, b):
    sa, cnt = per_prompt(a)
    sb, _ = per_prompt(b)
    ix = {p: i for i, p in enumerate(P)}
    ds = []
    for bs in boot:
        k = [ix[p] for p in bs]
        ds.append((sa[k].sum() - sb[k].sum()) / cnt[k].sum())
    return a.mean() - b.mean(), np.percentile(ds, 2.5), np.percentile(ds, 97.5)


print(f"{sys.argv[1]}: {n} positions, {len(P)} prompts; accepted drafts/step (+1 = tokens/step)")
print("group-start rate per depth: " + ", ".join(f"{g} {z['gs_' + g].mean(0).round(2).tolist()}" for g in ("word", "hi_rules", "random")))
for N in budgets:
    acc = {s: evaluate(fn, N) for s, fn in signals.items()}
    print(f"\nN = {N} nodes")
    for s, a in acc.items():
        print(f"  {s:13s} {a.mean():.3f}")
    for a, b in [("entropy", "static"), ("hi_rules", "static"), ("hi_rules", "random"), ("hi_rules", "word"),
                 ("ent+hi_rules", "entropy"), ("ent+hi_rules", "ent+random"), ("ent+hi_rules", "ent+word")]:
        d, lo, hi = diff_ci(acc[a], acc[b])
        print(f"  {a} - {b}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if lo > 0 or hi < 0 else ''}")

"""Collect per-position draft ranks + signals for the shaped-tree oracle (issue #33, step 1).

python scripts/analysis/tree_collect.py <run_dir> <n_prompts> <out.npz>
(issue #33; the paper used 300 prompts, Rsd and R2 at step 12500)
Greedy-generate 64 tokens per prompt (IndicCorp eval, make_prompts seed=1 as accept_boot), then at
every generated position t (teacher-forced on the greedy text) and head d=1..K-1 store:
  rank[d]     rank of the greedy token t+d+1 in head d's logits (capped at CAP)
  ent[d]      entropy of head d's distribution at t (the signal a real policy has)
  gs[g][d]    token t+d+1 starts a new group under grouper g: hi_rules_v0, word (every word is a
              group = word start), random (RandomGrouper, R6a's control)
  prompt      prompt index (for cross-fitting / bootstrap over prompts)
"""
import sys
from pathlib import Path

import numpy as np
import torch

run_dir, n_prompts, out_path = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mtp.data.corpus import load_split
from mtp.data.grouping.base import get_grouper
from mtp.device import autocast_ctx
from mtp.eval.spec_decode import greedy_generate, make_prompts, token_group_ids
from mtp.model.checkpoint import load_run

CAP = 50


class WordGrouper:
    name, lang = "word", "hi"

    def group_words(self, s):
        return [[w] for w in s.split()]

    def group_types(self, s):
        return ["single"] * len(s.split())


model, tok, cfg = load_run(run_dir, device="cuda")
groupers = {"hi_rules": get_grouper("hi_rules_v0", "hi"), "word": WordGrouper(), "random": get_grouper("random", "hi")}
K = model.num_heads
prompts = make_prompts(load_split("hi", "eval"), tok, n=n_prompts, seed=1)
rank, ent, pidx = [], [], []
gs = {g: [] for g in groupers}
with torch.no_grad():
    for i, p in enumerate(prompts):
        new, _ = greedy_generate(model, tok, p, 64, amp=lambda: autocast_ctx(cfg))
        full = list(p) + new
        gids = {g: token_group_ids(full, tok, gr) for g, gr in groupers.items()}
        with autocast_ctx(cfg):
            out = model(torch.tensor([full], device="cuda"))
        L, T = len(p), len(full)
        for t in range(L - 1, T - K):
            r, e = [], []
            for d in range(1, K):
                lg = out.logits[d][0, t].float()
                r.append(min(int((lg > lg[full[t + d + 1]]).sum()) + 1, CAP))
                lp = torch.log_softmax(lg, -1)
                e.append(float(-(lp.exp() * lp).sum()))
            rank.append(r)
            ent.append(e)
            pidx.append(i)
            for g, gid in gids.items():
                gs[g].append([gid[u] != gid[u - 1] and gid[u] >= 0 for u in range(t + 2, t + K + 1)])
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(prompts)}", flush=True)
np.savez(out_path, rank=np.array(rank), ent=np.array(ent), prompt=np.array(pidx),
         **{f"gs_{g}": np.array(v, dtype=bool) for g, v in gs.items()})
print(f"wrote {out_path}: {len(rank)} positions, run {cfg.run_name} step {model.loaded_step}")

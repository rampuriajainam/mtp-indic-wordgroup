"""
Fit tree draft policies for a run (JN-9) and score them offline.

python scripts/fit_tree.py --run_dir runs/Rsd_hi_k4_frozen_sd [--nodes 3 7 15 25] [--n_calib 300] [--n_eval 200]

Calibration: greedy continuations of IndicCorp *train* prompts (rows >= 1000, so never eval).
Scoring: the same on IndicCorp eval prompts (make_prompts seed 1). For each node budget it fits
StaticTree and EntropyTree and reports the oracle accepted drafts/step on the eval prompts, next
to the chain (FixedK) and the (3,2,1)-style product tree. Writes
results/{run_name}/tree_policy_{policy}_n{N}.json (load with tree_policy.load_policy) and
results/{run_name}/tree_fit.json (the scores).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtp.data.corpus import load_split
from mtp.device import autocast_ctx
from mtp.eval.spec_decode import make_prompts
from mtp.eval.tree_policy import (StaticTree, accepted_length, collect_ranks, fit_entropy, fit_static, save_policy,
                                  widths_tree)
from mtp.model.checkpoint import load_run

PRODUCT = {3: (1, 1, 1), 7: (2, 2, 1), 15: (3, 2, 1), 25: (5, 2, 1)}


def score(policy_tree, ranks, ents=None, policy=None):
    if policy is None:
        return sum(accepted_length(policy_tree, r) for r in ranks) / len(ranks)
    tot = 0
    for r, e in zip(ranks, ents):
        cell = sum(1 << d for d, (x, m) in enumerate(zip(e, policy.medians)) if x > m)
        tot += accepted_length(policy.trees.get(cell, policy.fallback), r)
    return tot / len(ranks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--step", default=None)
    ap.add_argument("--nodes", type=int, nargs="+", default=[3, 7, 15, 25])
    ap.add_argument("--n_calib", type=int, default=300)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    model, tok, cfg = load_run(args.run_dir, step=int(args.step) if args.step else None)
    amp = lambda: autocast_ctx(cfg)
    calib = make_prompts(load_split(cfg.lang, "train", 4 * args.n_calib), tok, n=args.n_calib, seed=0)
    evalp = make_prompts(load_split(cfg.lang, "eval"), tok, n=args.n_eval, seed=1)
    print(f"{cfg.run_name} step {model.loaded_step}: {len(calib)} calibration / {len(evalp)} eval prompts", flush=True)
    cr, ce = collect_ranks(model, tok, calib, args.max_new_tokens, amp=amp)
    er, ee = collect_ranks(model, tok, evalp, args.max_new_tokens, amp=amp)

    out = Path(args.out_dir) / cfg.run_name
    rows = []
    for n in args.nodes:
        st, et = fit_static(cr, n), fit_entropy(cr, ce, n)
        row = {"max_nodes": n, "static": score(st.nodes, er), "entropy": score(None, er, ee, et)}
        if n in PRODUCT:
            row[f"product{PRODUCT[n]}"] = score(widths_tree(PRODUCT[n]), er)
        rows.append(row)
        save_policy(st, out / f"tree_policy_static_n{n}.json")
        save_policy(et, out / f"tree_policy_entropy_n{n}.json")
        print("  " + " | ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()),
              flush=True)
    chain = score(StaticTree.from_widths((1,) * (model.num_heads - 1)).nodes, er)
    print(f"  chain (FixedK) {chain:.3f}   (accepted drafts/step; +1 = tokens/step)")
    (out / "tree_fit.json").write_text(json.dumps({
        "run_name": cfg.run_name, "step": model.loaded_step, "n_calib_positions": len(cr),
        "n_eval_positions": len(er), "chain": chain, "rows": rows,
        "note": "oracle accepted drafts per step on IndicCorp eval greedy text (teacher-forced)"}, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()

"""
Official evaluation of a trained run (OM-7): per-head loss / ppl / top-k with the in-group /
at-boundary split (mtp.eval.head_accuracy), written per INTERFACES §10.

    python scripts/evaluate.py --run_dir /kaggle/input/mtp-run-r2 \
        --datasets indiccorp_eval flores_hi [--step N] [--grouper hi_rules_v0] [--dump_tokens]

--run_dir is a run folder (config.yaml + step_N/), e.g. a published mtp-run-<ID> dataset.
Writes results/{run_name}/eval_{dataset}.json (commit these) and, with --dump_tokens,
results/{run_name}/tokens_{dataset}.jsonl (large: keep out of git).

Datasets (INTERFACES §4): indiccorp_eval, indiccorp_eval_small (in the run's language),
flores_hi, flores_mr, and flores = FLORES in the run's language (written as flores_{lang}).
FLORES is gated: accept the terms on huggingface.co/datasets/facebook/flores and log in (HF_TOKEN).

--grouper takes one name or one per run language, e.g. "hi:hi_rules_v0,mr:mr_rules_v1", so Hindi
and Marathi runs can share one command line (a language not listed uses the run's data.grouper).

--spec_decode adds the §10 spec_decode section (mtp.eval.spec_decode): prompts are the first 8-16
words of the first --spec_n sentences of each dataset, batch size 1, outputs checked against greedy.
--ignore_eos auto (default) masks EOS for head 0 when the run has a frozen backbone on an *instruct*
base model (frozen Misal-1B stops after one sentence, #39); on / off force it. Recorded per entry.
Re-running with the same run, step and grouper updates only the sections it computes, so per-head
and spec-decode numbers can come from separate sessions (--no_heads skips the per-head part).
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `import mtp` without `pip install -e .`

from mtp.config import cfg_get  # noqa: E402

DATASETS = {
    # name -> (lang or None for the run's lang, load_split split)
    "indiccorp_eval": (None, "eval"),
    "indiccorp_eval_small": (None, "eval_small"),
    "flores_hi": ("hi", "flores"),
    "flores_mr": ("mr", "flores"),
    "flores": (None, "flores"),   # the run's language; renamed flores_{lang} in evaluate_run
}


def dataset_lang(name, cfg):
    lang, _ = DATASETS[name]
    return lang or cfg.lang


def load_texts(name, cfg, n=None):
    from mtp.data.corpus import load_split

    _, split = DATASETS[name]
    return load_split(dataset_lang(name, cfg), split, n)


def grouper_for_lang(spec, lang):
    """Grouper name for a run in `lang`: `spec` is one name, or "lang:name,lang:name" (None if
    `lang` is not listed, i.e. use the run's config)."""
    if not spec or ":" not in spec:
        return spec or None
    by_lang = dict(part.strip().split(":", 1) for part in spec.split(",") if part.strip())
    return by_lang.get(lang) or None


def resolve_ignore_eos(value, cfg):
    """auto: on for a frozen backbone on an instruct base model (it keeps stopping at EOS after one
    sentence); LoRA-tuned runs (R9), frozen runs whose LoRA comes from a tuned run (init_lora_from,
    e.g. Rsd_r8 from R8) and base models (ganga-1b) run to max_new_tokens anyway."""
    if value in (True, "on"):
        return True
    if value in (False, "off", None):
        return False
    return (bool(cfg_get(cfg, "freeze_backbone", False)) and not cfg_get(cfg, "init_lora_from")
            and "instruct" in str(cfg.model_name).lower())


def resolve_grouper(name, lang):
    """(grouper, name) or (None, None) if `name` is unset, not registered yet (e.g. mr_rules_v1
    before JI-7) or does not support `lang` (hi_rules_v0 on flores_mr): the run is then evaluated
    without the in-group / at-boundary split."""
    from mtp.data.grouping.base import get_grouper

    if not name:
        return None, None
    try:
        return get_grouper(name, lang), name
    except KeyError:
        warnings.warn(f"grouper {name!r} is not registered yet; evaluating without the in-group split")
    except ValueError as e:  # language not supported by this grouper
        warnings.warn(f"grouper {name!r} cannot group {lang!r} ({e}); evaluating without the in-group split")
    return None, None


def make_examples(texts, tokenizer, grouper, max_length):
    if grouper is not None:
        from mtp.data.grouping.align import label_batch

        return label_batch(texts, tokenizer, grouper, max_length=max_length)
    enc = tokenizer(list(texts), truncation=True, max_length=max_length)
    return [{"input_ids": list(i), "attention_mask": list(m)} for i, m in zip(enc["input_ids"], enc["attention_mask"])]


def write_result(path, record):
    """Write record, keeping sections of an existing file for the same run / step / grouper
    that this call did not compute (e.g. spec_decode from another session)."""
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if all(old.get(k) == record[k] for k in ("run_name", "dataset", "step", "grouper")):
            for key in ("per_head", "spec_decode"):
                if record.get(key) is None and old.get(key) is not None:
                    record[key] = old[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_table(name, per_head):
    print(f"\n{name}")
    print(f"  {'head':>4} {'n':>7} {'loss':>7} {'ppl':>8} {'top1':>6} {'top5':>6} {'in-grp':>7} {'bound':>7}")
    pct = lambda v: f"{100 * v:6.1f}" if v is not None else "     -"  # noqa: E731
    for r in per_head:
        print(f"  {r['head']:>4} {r['n']:>7} {r['loss']:>7.3f} {r['ppl']:>8.2f} {pct(r['top1'])} {pct(r['top5'])}"
              f" {pct(r['top1_in_group'])} {pct(r['top1_at_boundary'])}")


def print_spec(name, entries):
    print(f"\n{name}: speculative decoding")
    for e in entries:
        acc = " / ".join(f"{100 * a:.0f}%" if a is not None else "-" for a in e["accept_rate_per_head"])
        gi = f"{100 * e['group_integrity']:.0f}%" if e["group_integrity"] is not None else "-"
        print(f"  {e['policy']:<15} accepted len {e['mean_accepted_len']:.2f} | accept/head {acc} | "
              f"{e['tokens_per_sec']:.1f} vs greedy {e['greedy_tokens_per_sec']:.1f} tok/s = x{e['speedup']:.2f} | "
              f"match greedy {e['outputs_match_greedy']} ({100 * e['match_rate']:.0f}% of prompts"
              + (f", max margin at a divergence {e['max_mismatch_margin']:.3f}" if e["max_mismatch_margin"] is not None else "")
              + ")" + (f" | fp32 re-check {e['fp32_check']['outputs_match_greedy']} on {e['fp32_check']['n_prompts']}"
                       + (f" (margin {e['fp32_check']['max_mismatch_margin']:.2e})"
                          if e["fp32_check"].get("max_mismatch_margin") is not None else "")
                       if e["fp32_check"] else "")
              + f" | group integrity {gi} | {e['n_prompts']} prompts")


def make_policies(names, tau):
    from mtp.eval.draft_policy import POLICIES, get_policy

    missing = [p for p in names if p not in POLICIES]
    if missing:
        raise SystemExit(f"draft polic(ies) {missing} not available; have {sorted(POLICIES)}")
    return [get_policy(p, tau=tau) if p == "confidence_cut" else get_policy(p) for p in names]


def evaluate_run(run_dir, datasets, step=None, grouper_name=None, out_dir=ROOT / "results", dump_tokens=False,
                 spec_decode=False, policies=("fixed_k", "confidence_cut"), tau=0.5, spec_n=200, max_new_tokens=64,
                 fp32_check_n=20, ignore_eos="auto",
                 heads=True, bench_verify=False, device="auto", batch_size=None, n=None, texts_fn=load_texts):
    """Evaluate one run on each dataset; returns {dataset: path of eval_{dataset}.json}.
    heads=False skips the per-head section (e.g. a session that only adds spec_decode)."""
    import torch

    from mtp.data.collate import Collator
    from mtp.device import autocast_ctx, uses_fp16_autocast
    from mtp.eval.head_accuracy import evaluate_heads
    from mtp.eval.spec_decode import bench_verify_pass, evaluate_spec_decode, make_prompts
    from mtp.model.checkpoint import load_run

    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; choose from {sorted(DATASETS)}")
    if not heads and not spec_decode and not bench_verify:
        raise SystemExit("nothing to do: --no_heads without --spec_decode or --bench_verify")
    draft_policies = make_policies(policies, tau) if spec_decode else []

    model, tokenizer, cfg = load_run(run_dir, device=device, step=step)
    device = next(model.parameters()).device
    if batch_size:
        cfg.optim.batch_size = batch_size
    max_length = cfg_get(cfg, "data.max_length", 128)
    grouper_name = grouper_for_lang(grouper_name, cfg.lang) or cfg_get(cfg, "data.grouper")
    datasets = [f"flores_{cfg.lang}" if d == "flores" else d for d in datasets]
    collator = Collator(tokenizer.pad_token_id)
    ignore_eos = resolve_ignore_eos(ignore_eos, cfg)
    print(f"run {cfg.run_name} | step {model.loaded_step} | heads {model.num_heads} | device {device}"
          + (" | ignore_eos" if ignore_eos and spec_decode else ""))
    if spec_decode and model.num_heads < 2:
        warnings.warn(f"{cfg.run_name} has one head (no drafts): skipping speculative decoding")
        draft_policies = []

    written = {}
    if bench_verify:   # once per run: verify-pass cost vs positions scored (chain drafts / tree nodes)
        texts = texts_fn(datasets[0], cfg, n)
        prefix = [t for text in texts for t in tokenizer(text)["input_ids"]][:64]
        amp_b = (lambda: autocast_ctx(cfg)) if uses_fp16_autocast(cfg) else None
        bench = {"run_name": cfg.run_name, "step": model.loaded_step, "device": str(device),
                 "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                 "fp16_autocast": amp_b is not None, **bench_verify_pass(model, prefix, amp=amp_b)}
        path = Path(out_dir) / cfg.run_name / "bench_verify.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bench, indent=2) + "\n", encoding="utf-8")
        written["bench_verify"] = path
        print(f"\nverify-pass cost ({bench['gpu'] or device}, prefix {bench['prefix_len']} tokens): "
              f"greedy 1-token pass {bench['greedy_ms']:.1f} ms")
        for b in bench["passes"]:
            print(f"  {b['positions']:>3} positions: {b['ms']:7.1f} ms  (x{b['x_greedy']:.2f} greedy, p90 {b['ms_p90']:.1f})")
    if not heads and not spec_decode:
        return written
    for name in datasets:
        t0 = time.perf_counter()
        texts = texts_fn(name, cfg, n)
        grouper, used_grouper = resolve_grouper(grouper_name, dataset_lang(name, cfg))
        out = Path(out_dir) / cfg.run_name
        per_head = spec = None
        if heads:
            examples = make_examples(texts, tokenizer, grouper, max_length)
            dump = out / f"tokens_{name}.jsonl" if dump_tokens else None
            per_head = evaluate_heads(model, examples, collator, cfg, device, top_k=(1, 5), dump_path=dump,
                                      texts=texts, tokenizer=tokenizer)
        if draft_policies:
            prompts = make_prompts(texts, tokenizer, n=spec_n)
            # fp16 autocast only where the run uses it (T4); the fp32 re-check then turns it off
            amp = (lambda: autocast_ctx(cfg)) if uses_fp16_autocast(cfg) else None
            spec = evaluate_spec_decode(model, tokenizer, prompts, draft_policies, max_new_tokens=max_new_tokens,
                                        grouper=grouper, amp=amp, log_every=25, fp32_check_n=fp32_check_n,
                                        ignore_eos=ignore_eos)
        record = {
            "run_name": cfg.run_name, "dataset": name, "step": model.loaded_step, "grouper": used_grouper,
            "git_commit": cfg_get(cfg, "git_commit"), "per_head": per_head, "spec_decode": spec,
        }
        path = out / f"eval_{name}.json"
        write_result(path, record)
        written[name] = path
        took = f"{time.perf_counter() - t0:.0f}s"
        if per_head is not None:
            print_table(f"{name}: {len(texts)} sentences, grouper {used_grouper}, {took} -> {path}", per_head)
        if spec is not None:
            print_spec(f"{name} ({took} -> {path})", spec)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="run folder with config.yaml and step_N/")
    ap.add_argument("--datasets", nargs="+", default=["indiccorp_eval", "flores_hi"], choices=sorted(DATASETS))
    ap.add_argument("--step", type=int, default=None, help="default: latest step")
    ap.add_argument("--grouper", default=None, help='one name or "hi:hi_rules_v0,mr:mr_rules_v1"; '
                    "default: data.grouper from the run's config")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--dump_tokens", action="store_true", help="also write results/{run}/tokens_{dataset}.jsonl")
    ap.add_argument("--no_heads", action="store_true", help="skip the per-head section (keeps an existing one)")
    ap.add_argument("--spec_decode", action="store_true", help="self-speculative decoding (OM-6)")
    ap.add_argument("--policies", nargs="+", default=["fixed_k", "confidence_cut"])
    ap.add_argument("--tau", type=float, default=0.5, help="confidence_cut threshold")
    ap.add_argument("--spec_n", type=int, default=200, help="prompts per dataset (first 8-16 words of a sentence)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--bench_verify", action="store_true",
                    help="time one verify pass vs positions scored (1..64) -> results/{run}/bench_verify.json")
    ap.add_argument("--fp32_check_n", type=int, default=20,
                    help="under fp16 autocast, re-run this many prompts in fp32: outputs must equal greedy")
    ap.add_argument("--ignore_eos", choices=["auto", "on", "off"], default="auto",
                    help="mask EOS for head 0 in spec decode; auto = frozen backbone on an instruct model, "
                         "no init_lora_from")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=None, help="default: optim.batch_size of the run")
    ap.add_argument("--n", type=int, default=None, help="cap sentences per dataset (smoke tests)")
    args = ap.parse_args(argv)

    evaluate_run(args.run_dir, args.datasets, step=args.step, grouper_name=args.grouper, out_dir=args.out,
                 dump_tokens=args.dump_tokens, spec_decode=args.spec_decode, policies=args.policies, tau=args.tau,
                 spec_n=args.spec_n, max_new_tokens=args.max_new_tokens, fp32_check_n=args.fp32_check_n,
                 ignore_eos=args.ignore_eos, heads=not args.no_heads, bench_verify=args.bench_verify,
                 device=args.device, batch_size=args.batch_size, n=args.n)


if __name__ == "__main__":
    main()

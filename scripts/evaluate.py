"""
Official evaluation of a trained run (OM-7): per-head loss / ppl / top-k with the in-group /
at-boundary split (mtp.eval.head_accuracy), written per INTERFACES §10.

    python scripts/evaluate.py --run_dir /kaggle/input/mtp-run-r2 \
        --datasets indiccorp_eval flores_hi [--step N] [--grouper hi_rules_v0] [--dump_tokens]

--run_dir is a run folder (config.yaml + step_N/), e.g. a published mtp-run-<ID> dataset.
Writes results/{run_name}/eval_{dataset}.json (commit these) and, with --dump_tokens,
results/{run_name}/tokens_{dataset}.jsonl (large: keep out of git).

Datasets (INTERFACES §4): indiccorp_eval, indiccorp_eval_small (in the run's language),
flores_hi, flores_mr. FLORES is gated: accept the terms on huggingface.co/datasets/facebook/flores
and log in (HF_TOKEN).

Re-running with the same run, step and grouper updates only the sections it computes, so per-head
numbers and --spec_decode numbers (OM-6) can be produced in separate sessions.
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
}


def dataset_lang(name, cfg):
    lang, _ = DATASETS[name]
    return lang or cfg.lang


def load_texts(name, cfg, n=None):
    from mtp.data.corpus import load_split

    _, split = DATASETS[name]
    return load_split(dataset_lang(name, cfg), split, n)


def resolve_grouper(name, lang):
    """(grouper, name) or (None, None) if `name` is unset or not registered yet (e.g. mr_rules_v1
    before JI-7): the run is then evaluated without the in-group / at-boundary split."""
    from mtp.data.grouping.base import get_grouper

    if not name:
        return None, None
    try:
        return get_grouper(name, lang), name
    except KeyError:
        warnings.warn(f"grouper {name!r} is not registered yet; evaluating without the in-group split")
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


def evaluate_run(run_dir, datasets, step=None, grouper_name=None, out_dir=ROOT / "results", dump_tokens=False,
                 spec_decode=False, device="auto", batch_size=None, n=None, texts_fn=load_texts):
    """Evaluate one run on each dataset; returns {dataset: path of eval_{dataset}.json}."""
    from mtp.data.collate import Collator
    from mtp.eval.head_accuracy import evaluate_heads
    from mtp.model.checkpoint import load_run

    if spec_decode:
        raise SystemExit("--spec_decode needs mtp/eval/spec_decode.py (OM-6), which is not on main yet")
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset(s) {unknown}; choose from {sorted(DATASETS)}")

    model, tokenizer, cfg = load_run(run_dir, device=device, step=step)
    device = next(model.parameters()).device
    if batch_size:
        cfg.optim.batch_size = batch_size
    max_length = cfg_get(cfg, "data.max_length", 128)
    grouper_name = grouper_name or cfg_get(cfg, "data.grouper")
    collator = Collator(tokenizer.pad_token_id)
    print(f"run {cfg.run_name} | step {model.loaded_step} | heads {model.num_heads} | device {device}")

    written = {}
    for name in datasets:
        t0 = time.perf_counter()
        texts = texts_fn(name, cfg, n)
        grouper, used_grouper = resolve_grouper(grouper_name, dataset_lang(name, cfg))
        examples = make_examples(texts, tokenizer, grouper, max_length)
        out = Path(out_dir) / cfg.run_name
        dump = out / f"tokens_{name}.jsonl" if dump_tokens else None
        per_head = evaluate_heads(model, examples, collator, cfg, device, top_k=(1, 5), dump_path=dump,
                                  texts=texts, tokenizer=tokenizer)
        record = {
            "run_name": cfg.run_name, "dataset": name, "step": model.loaded_step, "grouper": used_grouper,
            "git_commit": cfg_get(cfg, "git_commit"), "per_head": per_head, "spec_decode": None,
        }
        path = out / f"eval_{name}.json"
        write_result(path, record)
        written[name] = path
        print_table(f"{name}: {len(texts)} sentences, grouper {used_grouper}, {time.perf_counter() - t0:.0f}s"
                    f" -> {path}", per_head)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="run folder with config.yaml and step_N/")
    ap.add_argument("--datasets", nargs="+", default=["indiccorp_eval", "flores_hi"], choices=sorted(DATASETS))
    ap.add_argument("--step", type=int, default=None, help="default: latest step")
    ap.add_argument("--grouper", default=None, help="default: data.grouper from the run's config")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--dump_tokens", action="store_true", help="also write results/{run}/tokens_{dataset}.jsonl")
    ap.add_argument("--spec_decode", action="store_true", help="speculative decoding (OM-6, not available yet)")
    ap.add_argument("--policies", nargs="+", default=["fixed_k", "confidence_cut"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=None, help="default: optim.batch_size of the run")
    ap.add_argument("--n", type=int, default=None, help="cap sentences per dataset (smoke tests)")
    args = ap.parse_args(argv)

    evaluate_run(args.run_dir, args.datasets, step=args.step, grouper_name=args.grouper, out_dir=args.out,
                 dump_tokens=args.dump_tokens, spec_decode=args.spec_decode, device=args.device,
                 batch_size=args.batch_size, n=args.n)


if __name__ == "__main__":
    main()

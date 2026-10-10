"""JI-9 token-dump analysis, target group type and target position in group.

Usage: --dump R2=results/R2/tokens_flores_hi.jsonl --dump R5=... . The same
sentences and token-index pairs must be present across runs. Type labels are
assigned with one shared grouper and tokenizer, independent of each run's own
cache, so comparison remains meaningful.
"""

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtp.data.grouping.base import get_grouper
from mtp.data.grouping.align import label_tokens


def annotate_dump(rows, tokenizer, grouper, max_length=128):
    labels = {}
    counts = {}
    for row in rows:
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError("dump requires original text")
        if text not in labels:
            lab = label_tokens(text, tokenizer, grouper, max_length)
            types = grouper.group_types(text)
            positions = {}
            for i, g in enumerate(lab["group_id"]):
                if g >= 0:
                    positions.setdefault(g, []).append(i)
            labels[text] = (lab, types, positions)
        lab, types, positions = labels[text]
        target = row["target_idx"]
        if target < 0 or target >= len(lab["group_id"]):
            raise ValueError("target_idx outside tokenizer labels")
        gid = lab["group_id"][target]
        if gid < 0:
            continue
        span = positions[gid]
        position = (
            "single"
            if len(span) == 1
            else (
                "first"
                if target == span[0]
                else "last" if target == span[-1] else "middle"
            )
        )
        key = (row["head"], types[gid], position)
        d = counts.setdefault(key, Counter())
        d["n"] += 1
        d["correct"] += bool(row["correct"])
    return [
        {
            "head": h,
            "group_type": g,
            "position": p,
            "n": d["n"],
            "top1": d["correct"] / d["n"],
        }
        for (h, g, p), d in sorted(counts.items())
    ]


def compare_examples(a, b):
    key = lambda r: (r["sent_id"], r["token_idx"], r["head"])
    aa = {key(r): r for r in a}
    bb = {key(r): r for r in b}
    if len(aa) != len(a) or len(bb) != len(b):
        raise ValueError("duplicate token dump keys")
    if aa.keys() != bb.keys():
        raise ValueError("token dump keys differ")
    improved = []
    worse = []
    for k in aa:
        x, y = aa[k], bb[k]
        if x["text"] != y["text"] or x["target_idx"] != y["target_idx"]:
            raise ValueError("dump tokenization mismatch")
        if x["head"] < 1:
            continue
        if not x["correct"] and y["correct"]:
            improved.append(y)
        elif x["correct"] and not y["correct"]:
            worse.append(y)
    return {"improved": improved[:10], "worse": worse[:5]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", action="append", required=True)
    p.add_argument("--lang", default="hi", choices=["hi", "mr"])
    p.add_argument("--grouper", default="hi_rules_v1")
    p.add_argument("--tokenizer", default="LingoIITGN/ganga-1b")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument(
        "--out", type=Path, default=Path("results/grouping/error_analysis.json")
    )
    a = p.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer, use_fast=True)
    g = get_grouper(a.grouper, a.lang)
    raw = {}
    reports = {}
    for spec in a.dump:
        name, path = spec.split("=", 1)
        raw[name] = [
            json.loads(s)
            for s in Path(path).read_text(encoding="utf-8").splitlines()
            if s.strip()
        ]
        reports[name] = annotate_dump(raw[name], tok, g, a.max_length)
    examples = (
        compare_examples(raw["R2"], raw["R5"]) if {"R2", "R5"} <= raw.keys() else None
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            {
                "grouper": g.name,
                "tokenizer": a.tokenizer,
                "runs": reports,
                "examples": examples,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

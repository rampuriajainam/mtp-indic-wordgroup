"""JI-5 gold evaluation; rule drafts cannot select the final grouper."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtp.data.annotation import read_jsonl, groups_to_indices
from mtp.data.grouping.base import get_grouper, check_partition
from mtp.eval.group_metrics import score_records, agreement


def reviewed(rows):
    return all(
        r.get("annotation_status") != "unreviewed_rule_draft"
        and r.get("annotator") not in {"rules", "rule_draft", "ai_draft"}
        for r in rows
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--lang", choices=["hi", "mr"], default="hi")
    p.add_argument("--groupers", nargs="+")
    p.add_argument("--agreement", type=Path)
    p.add_argument("--allow-unreviewed", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    gold = read_jsonl(a.gold, True)
    if not gold:
        p.error("empty gold set")
    official = reviewed(gold)
    if not official and not a.allow_unreviewed:
        p.error(
            "unreviewed drafts are not gold; use --allow-unreviewed only for smoke tests"
        )
    names = a.groupers or (
        ["hi_rules_v0", "hi_rules_v1", "trankit", "random_hi_v1"]
        if a.lang == "hi"
        else ["mr_rules_v1", "trankit", "random_mr_v1"]
    )
    scores = {}
    for name in names:
        grouper = get_grouper(name, a.lang)
        pred = []
        for row in gold:
            groups = grouper.group_words(row["text"])
            check_partition(row["text"], groups)
            pred.append(
                {
                    **row,
                    "groups": groups_to_indices(groups),
                    "types": grouper.group_types(row["text"]),
                }
            )
        scores[name] = score_records(gold, pred)
    complete_gold = len(gold) == (200 if a.lang == "hi" else 150)
    result = {
        "official_gold": official,
        "complete_gold": complete_gold,
        "scores": scores,
        "agreement": None,
        "best_by_boundary_f1": (
            max(names, key=lambda n: scores[n]["boundary"]["f1"])
            if official and complete_gold
            else None
        ),
    }
    if a.agreement:
        other = read_jsonl(a.agreement, True)
        if len(other) != 50:
            p.error(
                "complete the 50-sentence blind annotation set before computing agreement"
            )
        if not reviewed(other):
            p.error("agreement input must be human reviewed")
        subset = [r for r in gold if r["id"] in {x["id"] for x in other}]
        result["agreement"] = agreement(subset, other) if official else None
    out = a.out or Path(f"results/grouping/{a.lang}_scores.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    table = [
        "| Grouper | Attachment P | R | F1 | Exact groups |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, score in scores.items():
        b = score["boundary"]
        table.append(
            f'| {name} | {b["precision"]:.4f} | {b["recall"]:.4f} | {b["f1"]:.4f} | {score["exact_group_accuracy"]:.4f} |'
        )
    out.with_suffix(".md").write_text(
        (
            "Human-reviewed gold\n\n"
            if official
            else "UNREVIEWED DRAFT SMOKE TEST: no linguistic accuracy claim\n\n"
        )
        + "\n".join(table)
        + "\n",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()

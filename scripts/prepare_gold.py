"""JI-2/JI-7: reproducible raw-row selection, drafts and a text-only blind packet.

Human annotations are never manufactured. Outputs go to data/annotation/;
reviewers use annotate.py to write the actual data/gold/*.jsonl.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtp.data.annotation import write_jsonl, groups_to_indices
from mtp.data.grouping.base import get_grouper


def select_candidates(candidates, n, seed=42, exclude=None):
    seen = set(exclude or [])
    eligible = []
    for source, source_id, text in candidates:
        text = text.strip()
        if 6 <= len(text.split()) <= 25 and text not in seen:
            seen.add(text)
            eligible.append((source, source_id, text))
    if len(eligible) < n:
        raise ValueError(f"need {n} candidates, only {len(eligible)} eligible")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:n]


def candidates_from_corpus(lang):
    from datasets import load_dataset

    code = {"hi": "hin_Deva", "mr": "mar_Deva"}[lang]
    from mtp.data.flores import load_flores_text

    flores = load_flores_text(lang)
    clean = [("flores_devtest", str(i), text) for i, text in enumerate(flores)]
    raw = load_dataset(
        "ai4bharat/IndicCorpV2", "indiccorp_v2", split=code, streaming=True
    )
    train = [
        ("indiccorp_train", str(i + 1000), row["text"])
        for i, row in enumerate(raw.skip(1000).take(4001))
    ]
    return clean, train


def main():
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", choices=["hi", "mr"], default="hi")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--flores", type=Path)
    p.add_argument("--indiccorp", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/annotation"))
    a = p.parse_args()
    if bool(a.flores) != bool(a.indiccorp):
        p.error("provide both --flores and --indiccorp")
    if a.flores:
        clean = [
            ("flores_devtest", str(i), s)
            for i, s in enumerate(a.flores.read_text(encoding="utf-8").splitlines())
        ]
        train = [
            ("indiccorp_train_local", str(i), s)
            for i, s in enumerate(a.indiccorp.read_text(encoding="utf-8").splitlines())
        ]
    else:
        clean, train = candidates_from_corpus(a.lang)
    n = 100 if a.lang == "hi" else 75
    selected = select_candidates(clean, n, a.seed)
    selected += select_candidates(train, n, a.seed, exclude=[t for _, _, t in selected])
    rows = [
        {"id": f"{a.lang}_{i+1:04d}", "source": src, "source_id": sid, "text": text}
        for i, (src, sid, text) in enumerate(selected)
    ]
    a.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out / f"{a.lang}_sentences.jsonl", rows)
    grouper = get_grouper(f"{a.lang}_rules_v1", a.lang)
    drafts = [
        {
            **r,
            "groups": groups_to_indices(grouper.group_words(r["text"])),
            "types": grouper.group_types(r["text"]),
            "annotator": "rule_draft",
            "annotation_status": "unreviewed_rule_draft",
            "notes": "Review without treating this as gold.",
        }
        for r in rows
    ]
    write_jsonl(a.out / f"{a.lang}_draft.jsonl", drafts)
    if a.lang == "hi":
        # 25 from each source, shuffled; never expose source_id, groups or types.
        blind = random.Random(a.seed).sample(rows[:n], 25) + random.Random(
            a.seed + 1
        ).sample(rows[n:], 25)
        random.Random(a.seed + 2).shuffle(blind)
        write_jsonl(
            a.out / "hi_om50_text_only.jsonl",
            [{"id": r["id"], "text": r["text"]} for r in blind],
        )
    (a.out / f"{a.lang}_selection_meta.json").write_text(
        json.dumps(
            {
                "seed": a.seed,
                "n": len(rows),
                "sources": {"flores_devtest": n, "indiccorp_train": n},
                "human_annotations_completed": 0,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"{len(rows)} sentences prepared; human gold annotations remain to be completed"
    )


if __name__ == "__main__":
    main()

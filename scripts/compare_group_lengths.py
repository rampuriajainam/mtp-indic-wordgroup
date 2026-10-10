"""JI-8: compare full-text Hindi/Marathi group lengths on shared tokenizers."""

import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hi", type=Path, default=Path("results/grouping/stats_hi.json"))
    p.add_argument("--mr", type=Path, default=Path("results/grouping/stats_mr.json"))
    p.add_argument(
        "--out", type=Path, default=Path("results/figures/group_lengths_hi_mr.pdf")
    )
    a = p.parse_args()
    hi = json.loads(a.hi.read_text())
    mr = json.loads(a.mr.read_text())
    shared = {r["tokenizer"] for r in hi} & {r["tokenizer"] for r in mr}
    rows = [
        (lang, r)
        for lang, data in [("hi", hi), ("mr", mr)]
        for r in data
        if r["tokenizer"] in shared and r["grouper"] in {"hi_rules_v1", "mr_rules_v1"}
    ]
    if not rows:
        p.error("no shared tokenizers/rule reports")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [lang + " / " + r["tokenizer"].split("/")[-1] for lang, r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(rows))
    ax.bar(
        [i - 0.2 for i in x],
        [r["words_per_group"] for _, r in rows],
        width=0.4,
        label="Words/group",
    )
    ax.bar(
        [i + 0.2 for i in x],
        [r["tokens_per_group"] for _, r in rows],
        width=0.4,
        label="Tokens/group",
    )
    ax.set_xticks(list(x), labels, rotation=25, ha="right")
    ax.set_ylabel("Mean length (full sentences)")
    ax.legend()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out)
    plt.close(fig)


if __name__ == "__main__":
    main()

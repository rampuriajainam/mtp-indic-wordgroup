"""JI-8 real-data reach study. Full sentences by default; --max-length for training."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtp.data.grouping.base import get_grouper
from mtp.eval.group_statistics import measure_grouping


def plot_reports(reports, lang, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groupers = list(dict.fromkeys(r["grouper"] for r in reports))
    cols = 2 if len(groupers) > 1 else 1
    rows = (len(groupers) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.7 * rows), squeeze=False)
    for ax, name in zip(axes.flat, groupers):
        for row in reports:
            if row["grouper"] != name:
                continue
            ax.plot(
                range(1, 5),
                [row["same_group_rate"][str(d)] for d in range(1, 5)],
                marker="o",
                label=row["tokenizer"].split("/")[-1],
            )
        ax.set(
            title=name,
            xlabel="Lookahead (tokens)",
            ylabel="Same-group rate",
            xticks=range(1, 5),
            ylim=(0, 1),
        )
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(groupers) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"reach_{lang}.pdf")
    plt.close(fig)
    labels = [r["grouper"] + " / " + r["tokenizer"].split("/")[-1] for r in reports]
    x = range(len(reports))
    fig, ax = plt.subplots(figsize=(max(8, len(reports) * 0.8), 5))
    ax.bar(
        [i - 0.2 for i in x],
        [r["words_per_group"] for r in reports],
        width=0.4,
        label="Words/group",
    )
    ax.bar(
        [i + 0.2 for i in x],
        [r["tokens_per_group"] for r in reports],
        width=0.4,
        label="Tokens/group",
    )
    ax.set_xticks(list(x), labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Mean length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"group_lengths_{lang}.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", choices=["hi", "mr"], default="hi")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--input", type=Path)
    p.add_argument("--groupers", nargs="+")
    p.add_argument(
        "--tokenizers",
        nargs="+",
        default=[
            "LingoIITGN/ganga-1b",
            "smallstepai/Misal-1B-instruct-v0.1",
            "google/mt5-small",
            "bigscience/bloom-560m",
            "xlm-roberta-base",
        ],
    )
    p.add_argument("--max-length", type=int)
    p.add_argument("--out", type=Path)
    p.add_argument("--figures", type=Path, default=Path("results/figures"))
    a = p.parse_args()
    if a.n <= 0 or (a.max_length is not None and a.max_length <= 0):
        p.error("sample size and max length must be positive")
    if a.input:
        texts = [
            s.strip()
            for s in a.input.read_text(encoding="utf-8").splitlines()
            if s.strip()
        ][: a.n]
    else:
        from mtp.data.corpus import load_split

        texts = load_split(a.lang, "train", a.n)
    if not texts:
        p.error("empty input")
    from transformers import AutoTokenizer

    names = a.groupers or (
        ["words", "hi_rules_v0", "hi_rules_v1", "random_hi_v1"]
        if a.lang == "hi"
        else ["words", "mr_rules_v1", "random_mr_v1"]
    )
    out = a.out or Path(f"results/grouping/stats_{a.lang}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    for tok_name in a.tokenizers:
        tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
        if not tok.is_fast:
            raise ValueError("fast tokenizer with offsets required")
        for name in names:
            reports.append(
                measure_grouping(texts, tok, get_grouper(name, a.lang), a.max_length)
            )
            print(f"Measured {name} / {tok_name}", flush=True)
            # Checkpoint completed pairs so a later tokenizer error loses no work.
            out.write_text(
                json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            out.with_suffix(".status.json").write_text(
                json.dumps(
                    {
                        "requested_pairs": len(a.tokenizers) * len(names),
                        "completed_pairs": len(reports),
                        "complete": len(reports) == len(a.tokenizers) * len(names),
                        "requested_sentences": a.n,
                        "actual_sentences": len(texts),
                        "tokenizers": a.tokenizers,
                        "groupers": names,
                    },
                    indent=2,
                )
                + "\n"
            )
    out = a.out or Path(f"results/grouping/stats_{a.lang}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_reports(reports, a.lang, a.figures)


if __name__ == "__main__":
    main()

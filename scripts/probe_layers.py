"""JI-10 frozen-model forward passes plus CPU logistic probes; no LM training."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtp.data.grouping.base import get_grouper
from mtp.eval.layer_probes import run_multi_layer_probes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", choices=["hi", "mr"], default="hi")
    p.add_argument("--model")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument(
        "--groupers",
        nargs="+",
        default=None,
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-positions", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("results/probing"))
    a = p.parse_args()
    if a.n <= 0:
        p.error("--n must be positive")
    a.model = a.model or (
        "LingoIITGN/ganga-1b"
        if a.lang == "hi"
        else "smallstepai/Misal-1B-instruct-v0.1"
    )
    a.groupers = a.groupers or (
        ["hi_rules_v1", "hi_rules_v0", "random_hi_v1", "words"]
        if a.lang == "hi"
        else ["mr_rules_v1", "random_mr_v1", "words"]
    )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mtp.data.corpus import load_split

    # Use platform device policy without constructing LoRA or prediction heads.
    from mtp.device import pick_device, pick_dtype

    device = pick_device() if a.device == "auto" else str(torch.device(a.device))
    dtype = torch.float32 if device == "cpu" else pick_dtype("auto")
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    train = load_split(a.lang, "train", a.n)
    evaluation = load_split(a.lang, "eval")
    # Exact duplicates across raw splits would leak evaluation. Exclude from train.
    eval_set = set(evaluation)
    train = [t for t in train if t not in eval_set]
    a.out.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    reports = run_multi_layer_probes(
        model,
        tok,
        [get_grouper(name, a.lang) for name in a.groupers],
        train,
        evaluation,
        str(device),
        a.max_length,
        a.max_positions,
        a.seed,
    )
    for name, report in reports.items():
        report["model"] = a.model
        (a.out / f"{a.lang}_{name}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        ax.plot(report["layer"], report["f1"], label=name)
    ax.set(
        xlabel="Frozen-model layer (0 = embeddings)",
        ylabel="Next-token boundary F1",
        ylim=(0, 1),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out / f"{a.lang}_layer_probes.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()

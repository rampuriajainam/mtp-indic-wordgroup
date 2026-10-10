"""JI-6 cache builder; output naming matches the train.py cache loader."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtp.data.grouping.base import get_grouper
from mtp.data.boundary_cache import build_cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", choices=["hi", "mr"], required=True)
    p.add_argument("--grouper", required=True)
    p.add_argument(
        "--split", choices=["train", "eval", "eval_small", "flores"], required=True
    )
    p.add_argument("--n", type=int)
    p.add_argument("--out", type=Path, default=Path("boundary_cache"))
    p.add_argument("--tokenizer")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--input", type=Path)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    n = a.n if a.n is not None else (200000 if a.split == "train" else None)
    if n is not None and n <= 0:
        p.error("--n must be positive")
    if a.input:
        texts = [
            s.strip()
            for s in a.input.read_text(encoding="utf-8").splitlines()
            if s.strip()
        ]
        if n is not None:
            texts = texts[:n]
    elif a.split == "flores":
        from mtp.data.flores import load_flores_text

        texts = load_flores_text(a.lang)
        if n is not None:
            texts = texts[:n]
    else:
        from mtp.data.corpus import load_split

        texts = load_split(a.lang, a.split, n)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        a.tokenizer
        or (
            "LingoIITGN/ganga-1b"
            if a.lang == "hi"
            else "smallstepai/Misal-1B-instruct-v0.1"
        ),
        use_fast=True,
    )
    meta = build_cache(
        texts,
        tok,
        get_grouper(a.grouper, a.lang),
        a.out / f"{a.lang}_{a.grouper}_{a.split}",
        a.max_length,
        a.batch_size,
        a.overwrite,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

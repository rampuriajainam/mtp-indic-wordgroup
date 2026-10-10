"""JI-1 real-text coverage. Imports never download data or a tokenizer.

python scripts/ji1_coverage.py --input hindi_sentences.txt --n 5000
python scripts/ji1_coverage.py --corpus --n 5000 --tokenizer LingoIITGN/ganga-1b
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtp.data.grouping.base import check_partition, get_grouper


def measure(sentences, grouper, tokenizer=None, max_length=128):
    counts = {"sentences": 0, "words": 0, "groups": 0, "attached_words": 0}
    digest = hashlib.sha256()
    types_count = {}
    token_count = token_groups = 0
    for text in sentences:
        groups = grouper.group_words(text)
        check_partition(text, groups)
        types = grouper.group_types(text)
        if len(types) != len(groups):
            raise ValueError("group_types and group_words disagree")
        counts["sentences"] += 1
        counts["words"] += len(text.split())
        counts["groups"] += len(groups)
        counts["attached_words"] += len(text.split()) - len(groups)
        digest.update((text + "\n").encode("utf-8"))
        for kind in types:
            types_count[kind] = types_count.get(kind, 0) + 1
        if tokenizer is not None:
            from mtp.data.grouping.align import label_tokens

            lab = label_tokens(text, tokenizer, grouper, max_length)
            ids = [g for g in lab["group_id"] if g >= 0]
            token_count += len(ids)
            token_groups += len(set(ids))
    return {
        **counts,
        "grouper": grouper.name,
        "sentence_sha256": digest.hexdigest(),
        "percent_words_attached": (
            100 * counts["attached_words"] / counts["words"] if counts["words"] else 0
        ),
        "words_per_group": (
            counts["words"] / counts["groups"] if counts["groups"] else 0
        ),
        "tokens_per_group": token_count / token_groups if token_groups else None,
        "tokenizer": getattr(tokenizer, "name_or_path", None),
        "max_length": max_length if tokenizer is not None else None,
        "group_types": types_count,
        "partition_check": "passed",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--corpus", action="store_true")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--tokenizer")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.n <= 0 or args.max_length <= 0:
        p.error("--n and --max-length must be positive")
    if args.corpus:
        from mtp.data.corpus import load_split

        texts = load_split("hi", "train", args.n)
        source_name = "IndicCorpV2/indiccorp_v2/hin_Deva train (raw rows >= 1000)"
    else:
        texts = []
        with args.input.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    texts.append(line.strip())
                    if len(texts) == args.n:
                        break
        source_name = str(args.input)
    if not texts:
        p.error("no non-blank sentences found")
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    reports = [
        measure(texts, get_grouper(name), tok, args.max_length)
        for name in ("hi_rules_v0", "hi_rules_v1")
    ]
    result = {
        "source": source_name,
        "requested_sentences": args.n,
        "complete_sample": len(texts) == args.n,
        "reports": reports,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

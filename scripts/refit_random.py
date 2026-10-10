"""Measure v1 histograms; leave teammate random defaults and R6a untouched."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtp.data.grouping.base import get_grouper
from mtp.data.grouping.random_grouper import RandomGrouper
from mtp.eval.group_statistics import sentence_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lang", choices=["hi", "mr"], default="hi")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--input", type=Path)
    p.add_argument(
        "--out", type=Path, default=Path("mtp/data/grouping/histograms_v1.json")
    )
    a = p.parse_args()
    if a.n <= 0:
        p.error("--n must be positive")
    if a.input:
        texts = [
            s.strip()
            for s in a.input.read_text(encoding="utf-8").splitlines()
            if s.strip()
        ][: a.n]
    else:
        from mtp.data.corpus import load_split

        texts = load_split(a.lang, "train", a.n)
    grouper = get_grouper(f"{a.lang}_rules_v1", a.lang)
    random = RandomGrouper.fit_from(grouper, texts)
    payload = json.loads(a.out.read_text()) if a.out.exists() else {}
    payload[a.lang] = {
        "grouper": grouper.name,
        "n_sentences": len(texts),
        "requested_sentences": a.n,
        "source": (
            str(a.input) if a.input else f"IndicCorpV2 {a.lang} train raw rows >= 1000"
        ),
        "sentence_sha256": sentence_digest(texts),
        "histogram": random.histogram,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload[a.lang], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

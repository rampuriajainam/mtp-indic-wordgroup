"""
Fixed text splits (INTERFACES §4). Nothing is downloaded at import time.

IndicCorpV2 separates documents with blank rows (every other row is blank).
Splits are defined on RAW row numbers and blank rows are then dropped:
  eval        raw rows 0-999   (~500 sentences)
  eval_small  raw rows 0-99    (~50 sentences; the laptop runs' held-out set)
  train       raw rows >= 1000, the first n NON-BLANK ones
  flores      FLORES-200 devtest (1,012 sentences)
"""

INDICCORP_SPLIT = {"hi": "hin_Deva", "mr": "mar_Deva"}


def load_split(lang, split, n=None):
    from datasets import load_dataset

    if split == "flores":
        ds = load_dataset("facebook/flores", INDICCORP_SPLIT[lang], split="devtest")
        texts = [r["sentence"] for r in ds]
        return texts[:n] if n else texts
    raw = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split=INDICCORP_SPLIT[lang], streaming=True)
    if split in ("eval", "eval_small"):
        texts = [r["text"].strip() for r in raw.take(1000 if split == "eval" else 100)]
        texts = [t for t in texts if t]
        return texts[:n] if n else texts
    if split != "train":
        raise ValueError(f"unknown split {split!r}")
    texts = []
    for r in raw.skip(1000):
        t = r["text"].strip()
        if t:
            texts.append(t)
            if n and len(texts) >= n:
                break
    return texts

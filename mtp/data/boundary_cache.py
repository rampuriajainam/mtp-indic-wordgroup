"""JI-6: strict, unpadded HuggingFace boundary caches and loader metadata."""

import json
import shutil
import tempfile
from pathlib import Path
from .grouping.align import label_batch
from .grouping.base import check_partition


def build_cache(
    texts, tokenizer, grouper, out_dir, max_length=128, batch_size=64, overwrite=False
):
    from datasets import Dataset

    if max_length <= 0 or batch_size <= 0:
        raise ValueError("max length and batch size must be positive")
    out = Path(out_dir)
    if out.exists() and not overwrite:
        raise FileExistsError(out)
    rows = []
    nw = ng = nt = observed = starts = 0
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        labels = label_batch(chunk, tokenizer, grouper, max_length)
        for text, label in zip(chunk, labels):
            groups = grouper.group_words(text)
            check_partition(text, groups)
            if len({len(v) for v in label.values()}) != 1:
                raise ValueError("label lengths differ")
            nw += len(text.split())
            ng += len(groups)
            ids = [g for g in label["group_id"] if g >= 0]
            nt += len(ids)
            observed += len(set(ids))
            starts += sum(label["group_start"])
            rows.append({"text": text, **label})
    if not rows:
        raise ValueError("empty cache")
    meta = {
        "grouper": grouper.name,
        "lang": grouper.lang,
        "tokenizer": tokenizer.name_or_path,
        "max_length": max_length,
        "n_rows": len(rows),
        "words_per_group": nw / ng if ng else None,
        "tokens_per_group": nt / observed if observed else None,
        "pct_tokens_group_start": 100 * starts / nt if nt else None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="boundary-", dir=out.parent))
    try:
        Dataset.from_list(rows).save_to_disk(str(tmp))
        (tmp / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    return meta


def load_cache(path, tokenizer_name=None, max_length=None):
    from datasets import load_from_disk

    path = Path(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if tokenizer_name is not None and meta["tokenizer"] != tokenizer_name:
        raise ValueError("tokenizer mismatch")
    if max_length is not None and meta["max_length"] != max_length:
        raise ValueError("max_length mismatch")
    ds = load_from_disk(str(path))
    if len(ds) != meta["n_rows"]:
        raise ValueError("cache count mismatch")
    if set(ds.column_names) != {
        "text",
        "input_ids",
        "attention_mask",
        "group_start",
        "group_id",
    }:
        raise ValueError("unexpected cache schema")
    return ds, meta

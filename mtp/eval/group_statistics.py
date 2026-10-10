"""JI-8 statistics with explicit denominators and per-sentence pair counting."""

from collections import Counter
import hashlib
from mtp.data.grouping.align import _labels, _group_start_chars
from mtp.data.grouping.base import check_partition


def sentence_digest(texts):
    return hashlib.sha256("".join(t + "\n" for t in texts).encode("utf-8")).hexdigest()


def measure_grouping(texts, tokenizer, grouper, max_length=None):
    nw = ng = nt = observed_groups = 0
    counts = Counter()
    pairs = {d: [0, 0] for d in range(1, 5)}
    for text in texts:
        groups = grouper.group_words(text)
        check_partition(text, groups)
        nw += len(text.split())
        ng += len(groups)
        counts.update(" ".join(g) for g in groups if len(g) > 1)
        kw = {"return_offsets_mapping": True, "truncation": max_length is not None}
        if max_length is not None:
            kw["max_length"] = max_length
        enc = tokenizer(text, **kw)
        _, ids = _labels(enc["offset_mapping"], _group_start_chars(text, grouper))
        real = [g for g in ids if g >= 0]
        nt += len(real)
        observed_groups += len(set(real))
        for d, stat in pairs.items():
            for i in range(max(0, len(real) - d)):
                stat[0] += real[i] == real[i + d]
                stat[1] += 1
    return {
        "grouper": grouper.name,
        "tokenizer": getattr(tokenizer, "name_or_path", "injected"),
        "n_sentences": len(texts),
        "n_words": nw,
        "n_groups": ng,
        "n_tokens": nt,
        "words_per_group": nw / ng if ng else None,
        "tokens_per_group": nt / observed_groups if observed_groups else None,
        "tokens_per_word": nt / nw if nw else None,
        "pct_words_attached": 100 * (nw - ng) / nw if nw else None,
        "same_group_rate": {
            str(d): s / n if n else None for d, (s, n) in pairs.items()
        },
        "same_group_counts": {
            str(d): {"same": s, "total": n} for d, (s, n) in pairs.items()
        },
        "sentence_sha256": sentence_digest(texts),
        "max_length": max_length,
        "top_multiword_groups": [
            {"text": g, "count": n} for g, n in counts.most_common(30)
        ],
    }

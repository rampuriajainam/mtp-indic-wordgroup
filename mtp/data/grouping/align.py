"""
Word groups -> token labels (INTERFACES §2), via the tokenizer's offset_mapping
on the FULL sentence (same tokenization as training).

A token starts group g if group g's first character falls in [tok_start, tok_end),
which also covers SentencePiece tokens whose span includes the leading space.
Special tokens (empty span) and padding get group_start 0, group_id -1.
"""

from bisect import bisect_left


def _group_start_chars(sentence, grouper):
    groups = grouper.group_words(sentence)
    starts, pos = [], 0
    for group in groups:
        first = True
        for word in group:
            idx = sentence.index(word, pos)
            if first:
                starts.append(idx)
                first = False
            pos = idx + len(word)
    return starts


def _labels(offsets, starts):
    group_start, group_id, prev = [], [], -1
    for a, b in offsets:
        if a == b:
            group_start.append(0)
            group_id.append(-1)
            continue
        g = max(bisect_left(starts, b) - 1, 0)   # last group starting before this token ends
        group_start.append(1 if g != prev else 0)
        group_id.append(g)
        prev = g
    return group_start, group_id


def label_tokens(sentence, tokenizer, grouper, max_length=128):
    """dict(input_ids, attention_mask, group_start, group_id), all the same length."""
    return label_batch([sentence], tokenizer, grouper, max_length)[0]


def label_batch(sentences, tokenizer, grouper, max_length=128):
    """label_tokens for many sentences with one (fast) tokenizer call."""
    enc = tokenizer(list(sentences), return_offsets_mapping=True, truncation=True, max_length=max_length)
    out = []
    for i, sentence in enumerate(sentences):
        start, gid = _labels(enc["offset_mapping"][i], _group_start_chars(sentence, grouper))
        out.append({"input_ids": list(enc["input_ids"][i]), "attention_mask": list(enc["attention_mask"][i]),
                    "group_start": start, "group_id": gid})
    return out

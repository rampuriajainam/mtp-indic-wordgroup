"""Collator edge cases (INTERFACES §3). Ported from Om's om/OM-1-skeleton tests;
the basic padding contract is in test_infra.py."""

import torch

from mtp.data.collate import Collator

PAD = 2


def test_long_tensors_and_extra_keys_dropped():
    rows = [
        {"input_ids": [1, 10, 11, 12], "attention_mask": [1, 1, 1, 1],
         "group_start": [0, 1, 0, 1], "group_id": [-1, 0, 0, 1], "text": "a b"},
        {"input_ids": [1, 20], "attention_mask": [1, 1],
         "group_start": [0, 1], "group_id": [-1, 0], "text": "c"},
    ]
    batch = Collator(PAD)(rows)
    assert set(batch) == {"input_ids", "attention_mask", "group_start", "group_id"}
    assert batch["input_ids"].tolist() == [[1, 10, 11, 12], [1, 20, PAD, PAD]]
    assert batch["group_id"].tolist() == [[-1, 0, 0, 1], [-1, 0, -1, -1]]
    assert all(t.dtype == torch.long for t in batch.values())


def test_equal_lengths_need_no_padding():
    batch = Collator(PAD)([{"input_ids": [5, 6], "attention_mask": [1, 1]}] * 3)
    assert batch["input_ids"].shape == (3, 2) and batch["attention_mask"].all()

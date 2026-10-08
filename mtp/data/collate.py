"""Right-padding collator (INTERFACES §3). group_* columns are optional."""

import torch


class Collator:
    PAD = {"attention_mask": 0, "group_start": 0, "group_id": -1}
    KEYS = ("input_ids", "attention_mask", "group_start", "group_id")

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, examples):
        T = max(len(e["input_ids"]) for e in examples)
        out = {}
        for key in self.KEYS:
            if key not in examples[0]:
                continue
            pad = self.pad_id if key == "input_ids" else self.PAD[key]
            out[key] = torch.tensor([list(e[key]) + [pad] * (T - len(e[key])) for e in examples])
        return out

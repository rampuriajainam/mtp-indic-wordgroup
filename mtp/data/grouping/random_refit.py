"""Measured v1 controls without changing teammate-owned random/R6a defaults."""

import json
from pathlib import Path
from .random_grouper import RandomGrouper


class HindiV1RandomGrouper(RandomGrouper):
    name = "random_hi_v1"
    source_lang = "hi"

    def __init__(self, lang="hi", histogram=None, seed=0, **kwargs):
        if lang != self.source_lang:
            raise ValueError(f"{self.name} requires {self.source_lang}")
        if histogram is None:
            path = Path(__file__).with_name("histograms_v1.json")
            if not path.exists():
                raise FileNotFoundError("Run scripts/refit_random.py first")
            payload = json.loads(path.read_text(encoding="utf-8"))[lang]
            histogram = {int(k): v for k, v in payload["histogram"].items()}
        super().__init__(lang=lang, histogram=histogram, seed=seed, **kwargs)


class MarathiV1RandomGrouper(HindiV1RandomGrouper):
    name = "random_mr_v1"
    source_lang = "mr"

    def __init__(self, lang="mr", **kwargs):
        super().__init__(lang=lang, **kwargs)

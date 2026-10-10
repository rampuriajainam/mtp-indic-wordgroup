"""
Random word grouper (JI-4): the control for every word-group experiment (R6a, R6).

Groups have the same length distribution (in words) as a real grouper, but are placed at
random, so a model trained with them sees "group" structure with no linguistics in it. If a
gain survives with `random`, it does not come from the word groups.

group_words samples group lengths from the histogram until the sentence is covered (the last
group is cut to fit). It is deterministic per (seed, sentence) through a SHA-256 of both, not
Python's hash(), which changes between processes. group_types is all "other".

DEFAULT_HISTOGRAM = hi_rules_v0 on 5,000 IndicCorp hi train sentences (211,477 groups, 1.33
words/group), measured 2026-10-10 because hi_rules_v1 did not exist yet. When JI-1 lands, re-fit
with RandomGrouper.fit_from(get_grouper("hi_rules_v1"), load_split("hi", "train", 5000)) and
replace the constant.
"""

import hashlib
import random
from collections import Counter

DEFAULT_HISTOGRAM = {1: 0.6974, 2: 0.2799, 3: 0.0198, 4: 0.0028, 5: 0.0001}


class RandomGrouper:
    name = "random"

    def __init__(self, lang="hi", histogram=None, seed=0, **kw):
        hist = dict(DEFAULT_HISTOGRAM if histogram is None else histogram)
        if not hist or any(not isinstance(k, int) or k < 1 for k in hist) or any(v < 0 for v in hist.values()):
            raise ValueError(f"histogram must map group lengths >= 1 to non-negative weights, got {histogram!r}")
        total = sum(hist.values())
        if total <= 0:
            raise ValueError("histogram weights sum to 0")
        self.lang = lang
        self.seed = seed
        self.lengths = sorted(hist)
        self.weights = [hist[k] / total for k in self.lengths]

    @property
    def histogram(self):
        return dict(zip(self.lengths, self.weights))

    def _rng(self, sentence):
        digest = hashlib.sha256(f"{self.seed}\x00{sentence}".encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def group_words(self, sentence):
        words = sentence.split()
        rng = self._rng(sentence)
        groups, i = [], 0
        while i < len(words):
            n = rng.choices(self.lengths, self.weights)[0]
            groups.append(words[i:i + n])
            i += n
        return groups

    def group_types(self, sentence):
        return ["other"] * len(self.group_words(sentence))

    @classmethod
    def fit_from(cls, grouper, sentences, seed=0, lang=None):
        """A RandomGrouper whose group-length histogram is measured from `grouper` on `sentences`."""
        counts = Counter(len(g) for s in sentences for g in grouper.group_words(s))
        if not counts:
            raise ValueError("no groups in `sentences`: cannot fit a histogram")
        n = sum(counts.values())
        return cls(lang=lang or grouper.lang, histogram={k: v / n for k, v in counts.items()}, seed=seed)

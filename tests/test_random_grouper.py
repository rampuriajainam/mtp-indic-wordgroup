"""JI-4: the random grouper (control for R6a / R6)."""

import random
from collections import Counter

import pytest

from mtp.data.grouping.align import label_batch
from mtp.data.grouping.base import check_partition, get_grouper
from mtp.data.grouping.random_grouper import DEFAULT_HISTOGRAM, RandomGrouper

SENT = "भारत एक विशाल और विविधतापूर्ण देश है जहाँ कई भाषाएँ बोली जाती हैं।"


def synthetic(n, seed=0):
    rng = random.Random(seed)
    return [" ".join(f"w{rng.randrange(10_000)}" for _ in range(rng.randint(5, 60))) for _ in range(n)]


def test_registered_and_needs_no_arguments():
    g = get_grouper("random")
    assert g.name == "random" and g.lang == "hi"
    assert g.histogram == pytest.approx({k: v / sum(DEFAULT_HISTOGRAM.values()) for k, v in DEFAULT_HISTOGRAM.items()})


def test_deterministic_across_instances_and_stable_across_runs():
    a, b = get_grouper("random"), RandomGrouper()
    assert a.group_words(SENT) == b.group_words(SENT)
    # golden: SHA-256 seeding, not hash(), so this holds in every process / Python version
    assert [len(g) for g in a.group_words(SENT)] == [1, 1, 1, 2, 2, 1, 2, 1, 1, 1]
    assert [len(g) for g in RandomGrouper(seed=1).group_words(SENT)] == [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]


def test_partition_holds_and_types_are_other():
    g = RandomGrouper()
    for s in synthetic(300) + [SENT, "एक", "", "  spaced   out  words "]:
        groups = g.group_words(s)
        check_partition(s, groups)
        assert g.group_types(s) == ["other"] * len(groups)


def test_length_histogram_matches_input():
    hist = {1: 0.5, 2: 0.3, 3: 0.15, 4: 0.05}
    g = RandomGrouper(histogram=hist, seed=3)
    counts = Counter()
    for s in synthetic(2000, seed=1):
        groups = g.group_words(s)
        counts.update(len(x) for x in groups[:-1])  # the last group may be cut to fit
    n = sum(counts.values())
    for k, p in hist.items():
        assert counts[k] / n == pytest.approx(p, abs=0.02)


def test_fit_from_another_grouper():
    class Pairs:  # groups of 2 words, the last one possibly 1
        lang = "hi"

        def group_words(self, s):
            w = s.split()
            return [w[i:i + 2] for i in range(0, len(w), 2)]

    sents = ["a b c d", "a b c"]  # groups: [2, 2] and [2, 1]
    g = RandomGrouper.fit_from(Pairs(), sents, seed=5)
    assert g.histogram == pytest.approx({1: 0.25, 2: 0.75}) and g.seed == 5 and g.lang == "hi"
    with pytest.raises(ValueError):
        RandomGrouper.fit_from(Pairs(), [""])


@pytest.mark.parametrize("bad", [{}, {0: 1.0}, {1: -0.5}, {1: 0.0}, {"2": 1.0}])
def test_bad_histogram(bad):
    with pytest.raises(ValueError):
        RandomGrouper(histogram=bad)


def test_usable_for_training_labels(tiny_tok):
    """Registered groupers are used through label_batch (train.py / evaluate.py)."""
    texts = ["मैं कल बाजार जा रहा था।", SENT]
    rows = label_batch(texts, tiny_tok, get_grouper("random"))
    for row, text in zip(rows, texts):
        n_groups = len(RandomGrouper().group_words(text))
        assert sum(row["group_start"]) == n_groups
        assert max(row["group_id"]) == n_groups - 1

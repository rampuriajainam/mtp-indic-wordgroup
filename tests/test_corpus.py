"""Split boundaries (INTERFACES §4) against a fake IndicCorp stream, no network.
Ported from Om's om/OM-1-skeleton tests; the fake stream has a blank row after every
text row, like hin_Deva."""

import datasets
import pytest

from mtp.data import corpus


class FakeStream:
    """Raw row i is "row{i}" for even i and blank for odd i."""

    def __init__(self, start=0, stop=10_000):
        self.start, self.stop = start, stop

    def skip(self, n):
        return FakeStream(self.start + n, self.stop)

    def take(self, n):
        return FakeStream(self.start, min(self.stop, self.start + n))

    def __iter__(self):
        return ({"text": f"row{i}" if i % 2 == 0 else "  "} for i in range(self.start, self.stop))


@pytest.fixture
def fake_hub(monkeypatch):
    calls = []

    def load_dataset(name, config=None, split=None, streaming=False):
        calls.append((name, config, split))
        if name == "facebook/flores":
            return [{"sentence": f"flores_{config}_{i}"} for i in range(1012)]
        return FakeStream()

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    return calls


def test_eval_splits_use_raw_rows_and_drop_blanks(fake_hub):
    ev = corpus.load_split("hi", "eval")
    assert len(ev) == 500 and ev[0] == "row0" and ev[-1] == "row998"
    small = corpus.load_split("hi", "eval_small")
    assert small == ev[:50]
    assert corpus.load_split("hi", "eval", n=10) == ev[:10]
    assert fake_hub[0][2] == "hin_Deva"


def test_train_starts_at_raw_row_1000(fake_hub):
    assert corpus.load_split("hi", "train", n=3) == ["row1000", "row1002", "row1004"]
    assert not set(corpus.load_split("hi", "train", n=100)) & set(corpus.load_split("hi", "eval"))


def test_lang_codes_and_flores(fake_hub):
    corpus.load_split("mr", "eval_small")
    assert fake_hub[-1][2] == "mar_Deva"
    fl = corpus.load_split("hi", "flores")
    assert len(fl) == 1012 and fl[0] == "flores_hin_Deva_0"
    assert len(corpus.load_split("mr", "flores", n=3)) == 3


def test_bad_args(fake_hub):
    with pytest.raises(KeyError):
        corpus.load_split("ta", "eval")
    with pytest.raises(ValueError):
        corpus.load_split("hi", "test")

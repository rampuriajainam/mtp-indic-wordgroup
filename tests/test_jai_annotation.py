import pytest
from mtp.data.annotation import parse_spans, validate_record, read_jsonl, write_jsonl
from scripts.prepare_gold import select_candidates
from scripts.annotate import annotate_row


def test_parse_spans_completes_singletons():
    assert parse_spans("3-5 7-8", 9) == [[0], [1], [2], [3, 4, 5], [6], [7, 8]]
    assert parse_spans("", 2) == [[0], [1]]


@pytest.mark.parametrize("bad", ["2-1", "0-3", "0-1 1-2", "x", "-1"])
def test_invalid_spans(bad):
    with pytest.raises(ValueError):
        parse_spans(bad, 3)


def test_reviewed_annotation_and_roundtrip(tmp_path):
    row = {"id": "hi_1", "source": "test", "text": "वह जा रहा था"}
    responses = iter(["1-3", "aux_chain", "checked"])
    result = annotate_row(row, "jai", lambda _: next(responses))
    assert (
        result["groups"] == [[0], [1, 2, 3]]
        and result["annotation_status"] == "human_reviewed"
    )
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [result])
    assert read_jsonl(p, True) == [result]


def test_validator_rejects_reordering_and_bool():
    row = {
        "id": "x",
        "text": "a b",
        "groups": [[0], [1]],
        "types": ["single", "single"],
    }
    validate_record(row)
    for groups in [[[1], [0]], [[False], [1]], [[0, 1], []]]:
        with pytest.raises(ValueError):
            validate_record({**row, "groups": groups})


def test_selection_is_deterministic_disjoint_and_length_filtered():
    candidates = [
        ("test", str(i), " ".join([str(i)] + ["word"] * 8)) for i in range(30)
    ]
    first = select_candidates(candidates, 10, 42)
    assert first == select_candidates(candidates, 10, 42)
    second = select_candidates(candidates, 10, 43, exclude=[r[2] for r in first])
    assert not {r[2] for r in first} & {r[2] for r in second}
    with pytest.raises(ValueError):
        select_candidates(candidates, 31)

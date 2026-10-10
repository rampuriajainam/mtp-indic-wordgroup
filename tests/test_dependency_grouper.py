import pytest
from mtp.data.grouping.trankit_grouper import (
    partition_dependencies,
    TrankitGrouper,
    trankit_tokens,
)


def tokens(text, links):
    import re

    spans = [m.span() for m in re.finditer(r"\S+", text)]
    return [
        {"id": i + 1, "head": head, "deprel": rel, "dspan": span, "upos": "VERB"}
        for i, (span, (head, rel)) in enumerate(zip(spans, links))
    ]


def test_auxiliary_chain_and_noncontiguous_gap():
    text = "वह जा रहा था"
    t = tokens(text, [(2, "nsubj"), (0, "root"), (2, "aux"), (2, "aux")])
    assert partition_dependencies(text, [t]) == (
        [["वह"], ["जा", "रहा", "था"]],
        ["single", "aux_chain"],
    )
    text = "जा कल था"
    t = tokens(text, [(0, "root"), (1, "advmod"), (1, "aux")])
    assert partition_dependencies(text, [t])[0] == [["जा"], ["कल"], ["था"]]


def test_local_ids_do_not_cross_sentence_boundaries():
    text = "घर में बात कर"
    t = tokens(text, [(0, "root"), (1, "case"), (0, "root"), (3, "compound:lv")])
    first = t[:2]
    second = [
        {**x, "id": x["id"] - 2, "head": x["head"] - 2 if x["head"] else 0}
        for x in t[2:]
    ]
    assert partition_dependencies(text, [first, second])[0] == [
        ["घर", "में"],
        ["बात", "कर"],
    ]


def test_trankit_adapter_lazy_and_empty():
    class Mock:
        def posdep(self, text):
            return {"sentences": [{"tokens": tokens(text, [(0, "root"), (1, "case")])}]}

    g = TrankitGrouper(pipeline=Mock())
    assert g.group_words("घर में") == [["घर", "में"]]
    assert g.batch_group_words(["घर में", ""]) == [[["घर", "में"]], []]
    assert TrankitGrouper().group_words("") == []


def test_expanded_tokens_use_parent_offsets():
    payload = {
        "tokens": [
            {
                "dspan": (0, 4),
                "expanded": [
                    {"id": 1, "head": 0, "deprel": "root"},
                    {"id": 2, "head": 1, "deprel": "case"},
                ],
            }
        ]
    }
    mapped = trankit_tokens(payload)
    assert mapped[0][1]["dspan"] == (0, 4)


def test_ambiguous_offsets_fail():
    with pytest.raises(ValueError):
        partition_dependencies("a b", [[{"id": 1, "head": 0, "dspan": (0, 3)}]])


def test_types_reuse_last_parse_and_caller_cannot_mutate_cache():
    calls = []

    class Mock:
        def posdep(self, text):
            calls.append(text)
            return {"tokens": tokens(text, [(0, "root"), (1, "case")])}

    g = TrankitGrouper(pipeline=Mock())
    groups = g.group_words("घर में")
    groups[0].append("bad")
    assert g.group_types("घर में") == ["postposition"] and g.group_words("घर में") == [
        ["घर", "में"]
    ]
    assert calls == ["घर में"]

import random
import pytest
from mtp.data.grouping.base import get_grouper, check_partition, GROUP_TYPES


@pytest.mark.parametrize(
    ("text", "groups", "types"),
    [
        (
            "मी घरी जात आहे।",
            [["मी"], ["घरी"], ["जात", "आहे।"]],
            ["single", "single", "aux_chain"],
        ),
        ("मुलं खेळत आहेत", [["मुलं"], ["खेळत", "आहेत"]], ["single", "aux_chain"]),
        ("घर च्या मध्ये", [["घर", "च्या", "मध्ये"]], ["compound_postposition"]),
        (
            "त्याने काम करायला लागला",
            [["त्याने"], ["काम"], ["करायला", "लागला"]],
            ["single", "single", "light_verb"],
        ),
        ("तो खेळू शकतो", [["तो"], ["खेळू", "शकतो"]], ["single", "aux_chain"]),
        ("तो करत नाही", [["तो"], ["करत", "नाही"]], ["single", "aux_chain"]),
        ("तो नाही जात", [["तो"], ["नाही", "जात"]], ["single", "aux_chain"]),
        ("घरात घराला घरासाठी", [["घरात"], ["घराला"], ["घरासाठी"]], ["single"] * 3),
        ("मी। आहे", [["मी।"], ["आहे"]], ["single"] * 2),
        ("", [], []),
    ],
)
def test_marathi(text, groups, types):
    g = get_grouper("mr_rules_v1", "mr")
    assert g.group_words(text) == groups
    assert g.group_types(text) == types
    check_partition(text, groups)


def test_partition_randomised():
    g = get_grouper("mr_rules_v1", "mr")
    rng = random.Random(2)
    for _ in range(1000):
        text = " ".join(
            rng.choices(
                "मी घरी जात नाही आहे च्या मध्ये शिकतो शकतो घरात ।".split(),
                k=rng.randrange(30),
            )
        )
        groups = g.group_words(text)
        check_partition(text, groups)
        assert len(g.group_types(text)) == len(groups) and set(
            g.group_types(text)
        ) <= set(GROUP_TYPES)


def test_language():
    with pytest.raises(ValueError):
        get_grouper("mr_rules_v1", "hi")

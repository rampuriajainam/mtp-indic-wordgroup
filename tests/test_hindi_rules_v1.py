"""JI-1 rule and contract checks; corpus coverage is a separate real-data step."""

import random
import subprocess
import sys
import pytest
from mtp.data.grouping.base import GROUP_TYPES, check_partition, get_grouper
from mtp.data.grouping.hindi_rules import HindiRuleGrouper, WORD_LISTS


@pytest.mark.parametrize(
    ("sentence", "groups", "types"),
    [
        (
            "मैं कल बाजार जा रहा था।",
            [["मैं"], ["कल"], ["बाजार"], ["जा", "रहा", "था।"]],
            ["single"] * 3 + ["aux_chain"],
        ),
        (
            "वह घर के बारे में बात कर रहा है",
            [["वह"], ["घर", "के", "बारे", "में"], ["बात"], ["कर", "रहा", "है"]],
            ["single", "compound_postposition", "single", "aux_chain"],
        ),
        (
            "उसने काम कर दिया है।",
            [["उसने"], ["काम"], ["कर", "दिया", "है।"]],
            ["single", "single", "light_verb"],
        ),
        ("उसने पैसे दिए", [["उसने"], ["पैसे"], ["दिए"]], ["single"] * 3),
        ("उसने पढ़ लिया", [["उसने"], ["पढ़", "लिया"]], ["single", "light_verb"]),
        ("उसने पढ़ लिए", [["उसने"], ["पढ़", "लिए"]], ["single", "light_verb"]),
        ("घर के लिए", [["घर", "के", "लिए"]], ["compound_postposition"]),
        ("उसके लिए", [["उसके", "लिए"]], ["postposition"]),
        (
            "वह नहीं जा रहा है",
            [["वह"], ["नहीं", "जा", "रहा", "है"]],
            ["single", "aux_chain"],
        ),
        ("वह जाता नहीं है", [["वह"], ["जाता", "नहीं", "है"]], ["single", "aux_chain"]),
        (
            "वह काम करता है",
            [["वह"], ["काम"], ["करता", "है"]],
            ["single", "single", "aux_chain"],
        ),
        ("किया जाता है", [["किया", "जाता", "है"]], ["aux_chain"]),
        ("जाने वाला भी", [["जाने", "वाला", "भी"]], ["postposition"]),
        (
            '"घर" के बारे में।',
            [['"घर"', "के", "बारे", "में।"]],
            ["compound_postposition"],
        ),
        ("घर। में", [["घर।"], ["में"]], ["single"] * 2),
        (
            "घर, के बारे में",
            [["घर,"], ["के", "बारे", "में"]],
            ["single", "compound_postposition"],
        ),
        (
            "घर के बारे, में",
            [["घर", "के", "बारे,"], ["में"]],
            ["postposition", "single"],
        ),
        ("घर । में", [["घर"], ["।"], ["में"]], ["single"] * 3),
        ('"घर।" में', [['"घर।"'], ["में"]], ["single"] * 2),
        ("के बारे में", [["के", "बारे", "में"]], ["compound_postposition"]),
        ("", [], []),
        (" \t\n ", [], []),
        (
            "है वह घर से",
            [["है"], ["वह"], ["घर", "से"]],
            ["single", "single", "postposition"],
        ),
    ],
)
def test_examples(sentence, groups, types):
    g = get_grouper("hi_rules_v1")
    assert g.group_words(sentence) == groups
    assert g.group_types(sentence) == types
    check_partition(sentence, groups)


@pytest.mark.parametrize("phrase", WORD_LISTS["v1"]["compound_postposition"])
def test_every_compound(phrase):
    sentence = "घर " + " ".join(phrase)
    g = get_grouper("hi_rules_v1")
    assert g.group_words(sentence) == [sentence.split()]
    assert g.group_types(sentence) == ["compound_postposition"]


def test_longest_compound_first():
    g = get_grouper("hi_rules_v1")
    assert g.group_words("बारिश की वजह से") == [["बारिश", "की", "वजह", "से"]]
    assert g.group_types("बारिश की वजह से") == ["compound_postposition"]


@pytest.mark.parametrize("word", WORD_LISTS["v1"]["light_verb"])
def test_every_light_verb(word):
    g = get_grouper("hi_rules_v1")
    assert g.group_words("कर " + word) == [["कर", word]]
    assert g.group_types("कर " + word) == ["light_verb"]


def test_randomised_partition_1000_synthetic_sentences():
    rng = random.Random(42)
    words = [
        "वह",
        "घर",
        "कर",
        "काम",
        "नहीं",
        "के",
        "बारे",
        "में",
        "लिए",
        "ही",
        "पैसे",
        "दिए",
        "जाता",
        "रहा",
        "था।",
        "हूँ",
        "वाली",
        "॥",
        "(जा)",
        "घर,",
    ]
    g = get_grouper("hi_rules_v1")
    for _ in range(1000):
        sentence = rng.choice([" ", "  ", "\t", "\n"]).join(
            rng.choices(words, k=rng.randrange(40))
        )
        groups = g.group_words(sentence)
        check_partition(sentence, groups)
        types = g.group_types(sentence)
        assert len(types) == len(groups) and set(types) <= set(GROUP_TYPES)
        assert groups == g.group_words(sentence)


def test_alias_and_language():
    assert HindiRuleGrouper is type(get_grouper("hi_rules_v1"))
    with pytest.raises(ValueError):
        get_grouper("hi_rules_v1", "mr")


def test_import_is_offline():
    code = """
import sys
class Block:
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in {'torch','transformers','datasets','huggingface_hub'}:
   raise AssertionError('Unexpected heavy import: '+fullname)
sys.meta_path.insert(0,Block())
from mtp.data.grouping.base import get_grouper
assert get_grouper('hi_rules_v1').group_words('घर के लिए')==[['घर','के','लिए']]
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_v1_align_integration(tiny_tok):
    from mtp.data.grouping.align import label_tokens

    lab = label_tokens(
        "वह घर के बारे में बात कर रहा है", tiny_tok, get_grouper("hi_rules_v1")
    )
    assert lab["group_id"] == [0, 1, 1, 1, 1, 2, 3, 3, 3]
    assert lab["group_start"] == [1, 1, 0, 0, 0, 1, 1, 0, 0]
    assert len({len(v) for v in lab.values()}) == 1

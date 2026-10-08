import pytest

from mtp.data.grouping.align import label_batch, label_tokens
from mtp.data.grouping.base import GROUP_TYPES, REGISTRY, check_partition, get_grouper

SENTENCES = [
    "मैं कल बाजार जा रहा था।",
    "बच्चे पार्क में खेल रहे हैं।",
    "वह घर से आया था",
    "भारत एक विशाल और विविधतापूर्ण देश है।",
    "  है वह  घर से  ",          # leading attaching word, extra spaces
]


def test_registry_and_unknown():
    assert "hi_rules_v0" in REGISTRY
    with pytest.raises(KeyError):
        get_grouper("nope")


def test_v0_groups_match_legacy_rules():
    g = get_grouper("hi_rules_v0", "hi")
    assert g.name == "hi_rules_v0" and g.lang == "hi"
    assert g.group_words("मैं कल बाजार जा रहा था।") == [["मैं"], ["कल"], ["बाजार"], ["जा", "रहा", "था।"]]
    assert g.group_words("बच्चे पार्क में खेल रहे हैं।") == [["बच्चे"], ["पार्क", "में"], ["खेल", "रहे", "हैं।"]]
    assert g.group_types("बच्चे पार्क में खेल रहे हैं।") == ["single", "postposition", "aux_chain"]
    assert g.group_words("  है वह  घर से  ") == [["है"], ["वह"], ["घर", "से"]]


@pytest.mark.parametrize("s", SENTENCES)
def test_partition_and_types(s):
    g = get_grouper("hi_rules_v0")
    groups = g.group_words(s)
    check_partition(s, groups)
    types = g.group_types(s)
    assert len(types) == len(groups) and set(types) <= set(GROUP_TYPES)


def test_check_partition_rejects():
    with pytest.raises(ValueError):
        check_partition("a b c", [["a"], ["c"]])
    with pytest.raises(ValueError):
        check_partition("a b", [["a", "b"], []])


def test_label_tokens_sentencepiece_offsets(tiny_tok):
    g = get_grouper("hi_rules_v0")
    lab = label_tokens("मैं कल बाजार जा रहा था।", tiny_tok, g)
    # tokens: ▁मैं ▁कल ▁बाजार ▁जा ▁रहा ▁था।   groups: [मैं] [कल] [बाजार] [जा रहा था।]
    assert lab["group_start"] == [1, 1, 1, 1, 0, 0]
    assert lab["group_id"] == [0, 1, 2, 3, 3, 3]
    assert lab["attention_mask"] == [1] * 6


@pytest.mark.parametrize("s", SENTENCES)
def test_label_invariants(tiny_tok, s):
    g = get_grouper("hi_rules_v0")
    lab = label_tokens(s, tiny_tok, g, max_length=128)
    assert len({len(v) for v in lab.values()}) == 1
    real = [x for x in lab["group_id"] if x >= 0]
    assert real == sorted(real) and real[0] == 0
    assert sum(lab["group_start"]) == len(set(real))


def test_special_tokens_and_truncation(tiny_tok):
    g = get_grouper("hi_rules_v0")
    tiny_tok._tokenizer.post_processor = __import__("tokenizers").processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 2), ("</s>", 3)])
    lab = label_tokens("वह घर से आया था", tiny_tok, g)
    assert lab["group_id"][0] == -1 and lab["group_id"][-1] == -1
    assert lab["group_start"][0] == 0 and lab["group_start"][-1] == 0
    short = label_tokens("वह घर से आया था", tiny_tok, g, max_length=3)
    assert len(short["input_ids"]) == 3


def test_label_batch_equals_single(tiny_tok):
    g = get_grouper("hi_rules_v0")
    batch = label_batch(SENTENCES, tiny_tok, g)
    assert batch == [label_tokens(s, tiny_tok, g) for s in SENTENCES]

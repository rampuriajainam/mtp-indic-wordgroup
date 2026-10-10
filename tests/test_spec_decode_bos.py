"""Spec decode with a BOS-adding tokenizer (Misal-1B for Marathi adds <s>; ganga-1b adds nothing)."""

import pytest

from tests.conftest import make_tiny_tokenizer

TEXTS = ["मैं कल बाजार जा रहा था। वह घर से आया था", "बच्चे पार्क में खेल रहे हैं। के बारे बात कर"]


def bos_tokenizer():
    from tokenizers import processors

    tok = make_tiny_tokenizer()
    tok.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", tok.bos_token_id)])
    return tok


def test_bos_tokenizer_adds_bos():
    tok = bos_tokenizer()
    assert tok(TEXTS[0])["input_ids"][0] == tok.bos_token_id


@pytest.mark.parametrize("text", TEXTS)
def test_token_group_ids_ignores_bos(text):
    from mtp.data.grouping.base import get_grouper
    from mtp.eval.spec_decode import token_group_ids

    grouper = get_grouper("hi_rules_v0")
    plain, bos = make_tiny_tokenizer(), bos_tokenizer()
    ids_plain = plain(text)["input_ids"]
    ids_bos = bos(text)["input_ids"]
    assert ids_bos[1:] == ids_plain
    g_plain = token_group_ids(ids_plain, plain, grouper)
    g_bos = token_group_ids(ids_bos, bos, grouper)
    assert g_bos[0] == -1 and g_bos[1:] == g_plain


def test_make_prompts_keep_bos():
    from mtp.eval.spec_decode import make_prompts

    tok = bos_tokenizer()
    prompts = make_prompts(TEXTS, tok, n=2, min_words=3, max_words=4)
    assert prompts and all(p[0] == tok.bos_token_id and tok.bos_token_id not in p[1:] for p in prompts)

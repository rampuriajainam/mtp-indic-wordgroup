import sys
from pathlib import Path

import pytest

# `import mtp` works without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WORDS = ("मैं कल बाजार जा रहा था। वह घर से आया था भारत एक विशाल और विविधतापूर्ण देश है। "
         "बच्चे पार्क में खेल रहे हैं। के बारे बात कर").split()


def make_tiny_tokenizer():
    """Word-level tokenizer with SentencePiece-style '▁' prefixes: offsets include the leading
    space, exactly like ganga-1b's tokenizer. No BOS/EOS added. Unknown words -> <unk>."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
    for w in WORDS:
        vocab.setdefault("▁" + w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Metaspace()
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", unk_token="<unk>",
                                   bos_token="<s>", eos_token="</s>")


@pytest.fixture
def tiny_tok():
    return make_tiny_tokenizer()


@pytest.fixture
def tiny_model_dir(tmp_path):
    """A tiny random Mistral + tokenizer saved like a HF repo, usable as cfg.model_name."""
    import torch
    from transformers import MistralConfig, MistralForCausalLM

    torch.manual_seed(0)
    tok = make_tiny_tokenizer()
    model = MistralForCausalLM(MistralConfig(
        vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=False))
    path = tmp_path / "tiny_base"
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return path

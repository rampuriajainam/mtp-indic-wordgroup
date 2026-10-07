"""
Step: properly align word-group boundaries to the REAL full-sentence
tokenization, using the tokenizer's character offset mapping.

Why this matters: token_level_boundaries.py tokenized each word-group
SEPARATELY and concatenated results. That can disagree with how the
tokenizer splits things when given the whole sentence at once. For a
loss function, labels must match the exact tokenization used in
training -- this script fixes that by using offset_mapping instead.
"""

import torch
from transformers import AutoTokenizer
from word_group_boundaries import find_word_groups

MODEL_NAME = "LingoIITGN/ganga-1b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def get_word_char_spans(sentence: str):
    """
    Returns a list of (word, start_char, end_char) for each
    whitespace-separated word in the sentence.
    """
    spans = []
    pos = 0
    for word in sentence.split():
        start = sentence.index(word, pos)
        end = start + len(word)
        spans.append((word, start, end))
        pos = end
    return spans


def get_group_char_spans(sentence: str):
    """
    Returns a list of (group_start_char, group_end_char) for each
    word-group, by combining the char spans of the words in each group.
    """
    word_spans = get_word_char_spans(sentence)
    groups = find_word_groups(sentence)

    group_spans = []
    word_idx = 0
    for group in groups:
        start_char = word_spans[word_idx][1]
        end_char = word_spans[word_idx + len(group) - 1][2]
        group_spans.append((start_char, end_char))
        word_idx += len(group)

    return group_spans


def get_aligned_boundary_labels(sentence: str, max_length=128):
    """
    Tokenizes the FULL sentence once (matching real training behavior),
    then uses offset_mapping to assign each token a boundary label:
    1 = this token is the FIRST token of a new word-group
    0 = this token continues the previous group
    """
    encoding = tokenizer(
        sentence, return_offsets_mapping=True,
        truncation=True, max_length=max_length,
    )
    offsets = encoding["offset_mapping"]
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    group_spans = get_group_char_spans(sentence)
    group_starts = sorted(start for start, end in group_spans)

    labels = []
    for (tok_start, tok_end) in offsets:
        if tok_start == tok_end:  # special tokens (e.g. BOS) have empty span
            labels.append(0)
            continue
        # SentencePiece tokens often include the LEADING SPACE in their
        # span, so a group's true start char can fall anywhere inside
        # [tok_start, tok_end), not exactly at tok_start. Check for that.
        is_group_start = any(tok_start <= gs < tok_end for gs in group_starts)
        labels.append(1 if is_group_start else 0)

    return tokens, labels


def print_check(sentence: str):
    tokens, labels = get_aligned_boundary_labels(sentence)
    print(f"\nSentence: {sentence}")
    for tok, lbl in zip(tokens, labels):
        marker = "| NEW GROUP ->" if lbl == 1 else "  (same group)"
        print(f"  {marker} {tok}")


if __name__ == "__main__":
    test_sentences = [
        "मैं कल बाजार जा रहा था।",
        "वह स्कूल जा चुका है।",
        "बच्चे पार्क में खेल रहे हैं।",
    ]

    for sent in test_sentences:
        print_check(sent)

    print("\n--- Compare this output to token_level_boundaries.py's output ---")
    print("If they match, the earlier approximation was fine.")
    print("If they differ, THIS version is the one to trust going forward --")
    print("it uses the tokenizer's real offsets on the full sentence, not")
    print("per-group tokenization glued together.")
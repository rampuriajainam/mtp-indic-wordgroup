"""Whitespace-word baseline, separating subword from word-group signal."""


class WordGrouper:
    name = "words"

    def __init__(self, lang="hi", **kwargs):
        if lang not in {"hi", "mr"}:
            raise ValueError("supported languages: hi, mr")
        self.lang = lang

    def group_words(self, sentence):
        return [[w] for w in sentence.split()]

    def group_types(self, sentence):
        return ["single"] * len(sentence.split())

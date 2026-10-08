"""
Rule-based Hindi word grouper.

A word joins the PREVIOUS group if, after stripping punctuation, it is in one of
the attaching lists; otherwise it starts a new group.

hi_rules_v0 = the original laptop rules (legacy/word_group_boundaries.py):
21 auxiliaries + 8 postpositions. On IndicCorp: 24.6% of words attach,
1.33 words / group, ~1.5 tokens / group (ganga tokenizer).

JI-1 (Jai) adds hi_rules_v1 here: full lists, light verbs, compound
postpositions (multi-word, longest match first). Word lists live in WORD_LISTS
so they can be cited in the report.
"""

WORD_LISTS = {
    "v0": {
        "aux_chain": [
            "है", "हैं", "था", "थी", "थे", "हो", "होगा", "होगी", "होंगे",
            "रहा", "रही", "रहे", "गया", "गई", "गए", "सकता", "सकती", "सकते",
            "चुका", "चुकी", "चुके",
        ],
        "postposition": ["का", "की", "के", "को", "से", "में", "पर", "ने"],
    },
}
PUNCT_V0 = "।,.!?"


class HindiRuleGrouperV0:
    name = "hi_rules_v0"

    def __init__(self, lang="hi", **kw):
        if lang != "hi":
            raise ValueError("hi_rules_v0 is Hindi-only")
        self.lang = lang
        lists = WORD_LISTS["v0"]
        self.attach = {w: kind for kind, words in lists.items() for w in words}

    def _clean(self, word):
        return word.strip(PUNCT_V0)

    def group_words(self, sentence):
        groups = []
        for word in sentence.split():
            if groups and self._clean(word) in self.attach:
                groups[-1].append(word)
            else:
                groups.append([word])
        return groups

    def group_types(self, sentence):
        types = []
        for group in self.group_words(sentence):
            kinds = {self.attach.get(self._clean(w)) for w in group[1:]} - {None}
            if len(group) == 1:
                types.append("single")
            elif "aux_chain" in kinds:
                types.append("aux_chain")
            else:
                types.append("postposition")
        return types

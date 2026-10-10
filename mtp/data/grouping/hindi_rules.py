"""
Rule-based Hindi word grouper.

v0 attaches listed words to the preceding group. v1 additionally matches
compound windows and uses documented context guards for ambiguous verbs and
negation. Both retain the original whitespace words.

hi_rules_v0 = the original laptop rules (legacy/word_group_boundaries.py):
21 auxiliaries + 8 postpositions. On IndicCorp: 24.6% of words attach,
1.33 words / group, ~1.5 tokens / group (ganga tokenizer).

JI-1 adds hi_rules_v1 below: full lists, light verbs, compound
postpositions (multi-word, longest match first). Word lists live in WORD_LISTS
so they can be cited in the report.
"""

WORD_LISTS = {
    "v0": {
        "aux_chain": [
            "है",
            "हैं",
            "था",
            "थी",
            "थे",
            "हो",
            "होगा",
            "होगी",
            "होंगे",
            "रहा",
            "रही",
            "रहे",
            "गया",
            "गई",
            "गए",
            "सकता",
            "सकती",
            "सकते",
            "चुका",
            "चुकी",
            "चुके",
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


# Lists are deliberately explicit and report-citable. v0 remains unchanged.
WORD_LISTS["v1"] = {
    "aux_chain": "है हैं था थी थे हो होगा होगी होंगे हूँ रहा रही रहे गया गई गए गये सकता सकती सकते चुका चुकी चुके पाया पाई पाए".split(),
    "light_verb": "लगा लगी लगे पड़ा पड़ी पड़े दिया दी दिए लिया ली लिए डाला डाली डाले".split(),
    "postposition": "का की के को से में पर ने तक भी ही लिए साथ बारे पास नहीं वाला वाली वाले".split(),
    "habitual": "जाता जाती जाते करता करती करते".split(),
    "compound_postposition": [
        phrase.split()
        for phrase in (
            "के लिए",
            "के साथ",
            "के बारे में",
            "के बाद",
            "के पास",
            "के अंदर",
            "के बाहर",
            "के ऊपर",
            "के नीचे",
            "के दौरान",
            "के अनुसार",
            "की ओर",
            "की तरह",
            "के कारण",
            "की वजह से",
            "की वजह",
        )
    ],
    # A conservative, editable heuristic, not a POS tagger. An unrecognised
    # verb may miss a light-verb attachment; gold scoring belongs to JI-5.
    "verb_bases": "जा आ कर हो पढ़ लिख बोल कह सुन देख खा पी दे ले रख चल बैठ उठ सो रो हँस हंस खेल खोल बंद समझ सीख सिखा बना पा भेज लौट मिल खरीद बेच मार काट धो पहन निकल पहुंच पहुँच रह गिर फेंक डाल लग पड़ पकड़ बदल रोक छोड़ छोड भर चुन बचा ढूंढ ढूँढ खोज मांग माँग भूल सोच चाह दौड़ नाच गा जीत हार टूट तोड़ जोड़ उड़ बता चुका सक".split(),
    "verb_irregular": "किया किये किए की गया गई गये गए आया आई आए आये हुआ हुई हुए दिया दी दिए लिया ली लिए पाया पाई पाए पड़ा पड़ी पड़े रहा रही रहे".split(),
}
PUNCT_V1 = "।॥,.!?;:\"'()“”‘’[]{}"
CLAUSE_PUNCT = "।॥,.!?;:"


class HindiRuleGrouperV1:
    """Deterministic Hindi rules; original whitespace words are never changed.

    Compounds consume the longest window before single-word rules. In
    'वह घर के बारे में बात कर रहा है', groups are [वह], [घर के बारे में],
    [बात], [कर रहा है]: noun + main verb is not itself a rule.

    Light verbs attach only after a recognised verbal word/chain: [कर दिया है],
    but [उसने], [पैसे], [दिए]. लिए wins as a compound in [घर के लिए]; alone
    it is a light verb after a recognised verb, otherwise a postposition per
    the plan. Habitual जाता/करता forms use the same verb-context guard.
    Preverbal नहीं starts [नहीं जा रहा है]; postverbal negation attaches to
    the verb, e.g. [जाता नहीं है]. Else नहीं is kept separate. वाला-forms
    and focus particles attach left. Auxiliary rules retain v0's lexical
    heuristic, including copulas attaching to nouns. This is not parsing.

    Punctuation is stripped for lookup only. Clause punctuation and standalone
    punctuation block cross-clause attachments and compound windows. Closing
    quotes/brackets are ignored when detecting trailing clause punctuation.
    Type precedence: compound_postposition > light_verb > aux_chain >
    postposition > single. Labels describe the strongest rule used, not POS.
    """

    name = "hi_rules_v1"
    negation = "नहीं"

    def __init__(self, lang="hi", **kw):
        if lang != "hi":
            raise ValueError("hi_rules_v1 is Hindi-only")
        self.lang = lang
        lists = WORD_LISTS["v1"]
        self.aux = set(lists["aux_chain"])
        self.light = set(lists["light_verb"])
        self.post = set(lists["postposition"])
        self.habitual = set(lists["habitual"])
        self.verbs = set(lists["verb_bases"]) | set(lists["verb_irregular"])
        # Common inflections of the explicit bases; no unrestricted suffix test.
        self.verbs.update(
            base + ending
            for base in lists["verb_bases"]
            for ending in (
                "ना",
                "नी",
                "ने",
                "ता",
                "ती",
                "ते",
                "ा",
                "ी",
                "े",
                "ूँ",
                "ें",
            )
        )
        self.compounds = sorted(
            (tuple(p) for p in lists["compound_postposition"]),
            key=lambda p: (-len(p), p),
        )

    @staticmethod
    def _clean(word):
        return word.strip(PUNCT_V1)

    @staticmethod
    def _ends_clause(word):
        return (
            bool(word.rstrip("\"'()“”‘’[]{}")[-1:] in CLAUSE_PUNCT) if word else False
        )

    def _partition(self, sentence):
        words = sentence.split()
        clean = [self._clean(w) for w in words]
        groups, types = [], []
        rank = {
            "single": 0,
            "postposition": 1,
            "aux_chain": 2,
            "light_verb": 3,
            "compound_postposition": 4,
        }

        def append(span, kind, attach):
            if attach:
                groups[-1].extend(span)
                if rank[kind] > rank[types[-1]]:
                    types[-1] = kind
            else:
                groups.append(list(span))
                types.append(kind if len(span) > 1 else "single")

        i = 0
        while i < len(words):
            can_attach = bool(
                groups
                and clean[i]
                and self._clean(words[i - 1])
                and not self._ends_clause(words[i - 1])
            )
            match = next(
                (
                    p
                    for p in self.compounds
                    if tuple(clean[i : i + len(p)]) == p
                    and not any(self._ends_clause(w) for w in words[i : i + len(p) - 1])
                ),
                None,
            )
            if match:
                append(words[i : i + len(match)], "compound_postposition", can_attach)
                i += len(match)
                continue
            word = clean[i]
            verbal_left = can_attach and (
                clean[i - 1] in self.verbs or types[-1] in {"aux_chain", "light_verb"}
            )
            if word == self.negation:
                # Prefer forward negation when there is no verbal host on the left.
                forward = (
                    not verbal_left
                    and i + 1 < len(words)
                    and not self._ends_clause(words[i])
                    and clean[i + 1] in self.verbs
                )
                if forward:
                    append(words[i : i + 2], "aux_chain", False)
                    i += 2
                    continue
                append([words[i]], "aux_chain", verbal_left)
            elif word in self.light and verbal_left:
                append([words[i]], "light_verb", True)
            elif word in self.aux or (word in self.habitual and verbal_left):
                append([words[i]], "aux_chain", can_attach)
            elif word in self.post:
                append([words[i]], "postposition", can_attach)
            else:
                append([words[i]], "single", False)
            i += 1
        return groups, types

    def group_words(self, sentence):
        return self._partition(sentence)[0]

    def group_types(self, sentence):
        return self._partition(sentence)[1]


# The PDF names HindiRuleGrouper; existing package convention uses V0/V1.
HindiRuleGrouper = HindiRuleGrouperV1

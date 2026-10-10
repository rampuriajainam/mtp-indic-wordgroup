"""JI-3 dependency groupers with lazy parsers and strict whitespace alignment.

Trankit uses token dspan (document offsets); span is sentence-local. Expanded
MWT words inherit the parent token span. Stanza exposes token start/end chars.
Dependency IDs are sentence-local. Non-contiguous components are split into
contiguous runs, never filling a gap with an unrelated word. Pipeline injection
supports offline tests. Missing/ambiguous offsets raise, rather than invent labels.
"""

import re
from .base import check_partition

REL_TYPES = {
    "aux": "aux_chain",
    "aux:pass": "aux_chain",
    "cop": "aux_chain",
    "case": "postposition",
    "mark": "other",
    "compound:lv": "light_verb",
}
RANK = {
    "single": 0,
    "other": 1,
    "postposition": 2,
    "aux_chain": 3,
    "light_verb": 4,
    "compound_postposition": 5,
}


def partition_dependencies(text, sentences):
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    words = text.split()
    parent = list(range(len(words)))
    kinds = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(b)] = find(a)

    for tokens in sentences:
        ids = {}
        mapped = {}
        for t in tokens:
            if not isinstance(t.get("id"), int):
                continue
            a, b = t["dspan"]
            matches = [i for i, (s, e) in enumerate(spans) if a < e and b > s]
            if len(matches) != 1:
                raise ValueError(
                    f"parser span {(a,b)} does not map to one whitespace word"
                )
            ids[t["id"]] = t
            mapped[t["id"]] = matches[0]
        for tid, t in ids.items():
            head = t.get("head", 0)
            if not head or head not in mapped:
                continue
            a, b = mapped[tid], mapped[head]
            neg = t.get("upos") == "PART" and "Polarity=Neg" in (t.get("feats") or "")
            kind = REL_TYPES.get(t.get("deprel")) or ("aux_chain" if neg else None)
            if not kind or t.get("upos") == "PUNCT":
                continue
            union(a, b)
            kinds.setdefault(a, set()).add(kind)
            kinds.setdefault(b, set()).add(kind)
    groups = []
    types = []
    i = 0
    while i < len(words):
        j = i + 1
        while j < len(words) and find(j) == find(i):
            j += 1
        groups.append(words[i:j])
        used = set().union(*(kinds.get(k, set()) for k in range(i, j)))
        types.append(
            max(used, key=lambda k: RANK[k]) if j - i > 1 and used else "single"
        )
        i = j
    check_partition(text, groups)
    return groups, types


def trankit_tokens(payload):
    result = []
    for sentence in payload.get("sentences", [payload]):
        offset = sentence.get("dspan", (0, 0))[0]
        tokens = []
        for token in sentence.get("tokens", []):
            span = token.get("dspan")
            if span is None and token.get("span") is not None:
                a, b = token["span"]
                span = (a + offset, b + offset)
            for word in token.get("expanded", [token]):
                item = dict(word)
                item["dspan"] = word.get("dspan", span)
                if item["dspan"] is None:
                    raise ValueError("parser token missing offsets")
                tokens.append(item)
        result.append(tokens)
    return result


class TrankitGrouper:
    name = "trankit"

    def __init__(self, lang="hi", pipeline=None, gpu=False, cache_dir=None, **kwargs):
        if lang not in {"hi", "mr"}:
            raise ValueError("supported languages: hi, mr")
        self.lang = lang
        self.pipeline = pipeline
        self.gpu = gpu
        self.cache_dir = cache_dir
        self._last = None

    def _load(self):
        if self.pipeline is None:
            try:
                from trankit import Pipeline
            except ImportError as e:
                raise ImportError(
                    "Install trankit or select the stanza grouper; no silent fallback"
                ) from e
            kw = {"lang": {"hi": "hindi", "mr": "marathi"}[self.lang], "gpu": self.gpu}
            if self.cache_dir is not None:
                kw["cache_dir"] = self.cache_dir
            self.pipeline = Pipeline(**kw)
        return self.pipeline

    def _parse(self, sentence):
        if not sentence.strip():
            return [], []
        return partition_dependencies(
            sentence, trankit_tokens(self._load().posdep(sentence))
        )

    def _partition(self, sentence):
        if self._last is None or self._last[0] != sentence:
            self._last = (sentence, self._parse(sentence))
        groups, types = self._last[1]
        return [list(g) for g in groups], list(types)

    def group_words(self, sentence):
        return self._partition(sentence)[0]

    def group_types(self, sentence):
        return self._partition(sentence)[1]

    def batch_group_words(self, sentences):
        return [self.group_words(s) for s in sentences]


class StanzaGrouper(TrankitGrouper):
    name = "stanza"

    def _load(self):
        if self.pipeline is None:
            try:
                import stanza
            except ImportError as e:
                raise ImportError("Install stanza for the stanza grouper") from e
            kw = {
                "lang": self.lang,
                "processors": "tokenize,pos,lemma,depparse",
                "use_gpu": self.gpu,
            }
            if self.cache_dir is not None:
                kw["dir"] = self.cache_dir
            self.pipeline = stanza.Pipeline(**kw)
        return self.pipeline

    def _parse(self, sentence):
        if not sentence.strip():
            return [], []
        document = self._load()(sentence)
        parsed = []
        for sent in document.sentences:
            words = []
            for token in sent.tokens:
                for word in token.words:
                    words.append(
                        {
                            "id": word.id,
                            "head": word.head,
                            "deprel": word.deprel,
                            "upos": word.upos,
                            "feats": word.feats,
                            "dspan": (token.start_char, token.end_char),
                        }
                    )
            parsed.append(words)
        return partition_dependencies(sentence, parsed)

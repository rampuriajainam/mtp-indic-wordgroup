"""
Grouper contract (INTERFACES §2) + registry.

A grouper partitions the whitespace words of a sentence into contiguous word
groups; concatenating the groups must give exactly sentence.split().
Register a new grouper by adding one line to REGISTRY ("name": "module:Class");
the module is only imported when that grouper is requested.
"""

import importlib
from typing import Protocol

GROUP_TYPES = ("single", "aux_chain", "postposition", "compound_postposition", "light_verb", "other")

REGISTRY = {
    "hi_rules_v0": "mtp.data.grouping.hindi_rules:HindiRuleGrouperV0",
    "random": "mtp.data.grouping.random_grouper:RandomGrouper",
    "hi_rules_v1": "mtp.data.grouping.hindi_rules:HindiRuleGrouperV1",
    "mr_rules_v1": "mtp.data.grouping.marathi_rules:MarathiRuleGrouper",
    "trankit": "mtp.data.grouping.trankit_grouper:TrankitGrouper",
    "stanza": "mtp.data.grouping.trankit_grouper:StanzaGrouper",
    "words": "mtp.data.grouping.words:WordGrouper",
    "random_hi_v1": "mtp.data.grouping.random_refit:HindiV1RandomGrouper",
    "random_mr_v1": "mtp.data.grouping.random_refit:MarathiV1RandomGrouper",
}


class Grouper(Protocol):
    name: str            # e.g. "hi_rules_v1", "trankit", "random"
    lang: str            # "hi" or "mr"

    def group_words(self, sentence: str) -> list[list[str]]:
        """Whitespace words of `sentence`, partitioned into contiguous groups."""

    def group_types(self, sentence: str) -> list[str]:
        """One entry of GROUP_TYPES per group, same order as group_words()."""


def get_grouper(name: str, lang: str = "hi", **kwargs) -> Grouper:
    if name not in REGISTRY:
        raise KeyError(f"unknown grouper {name!r}; registered: {sorted(REGISTRY)}")
    module, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), cls)(lang=lang, **kwargs)


def check_partition(sentence: str, groups: list[list[str]]) -> None:
    """Raise if groups are not a contiguous partition of sentence.split()."""
    flat = [w for g in groups for w in g]
    if flat != sentence.split() or any(len(g) == 0 for g in groups):
        raise ValueError(f"groups do not partition the sentence: {groups!r}")

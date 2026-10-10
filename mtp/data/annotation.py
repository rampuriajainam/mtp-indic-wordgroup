"""Validated gold JSONL and annotation selections; no automatic gold creation."""

import json
import re
from pathlib import Path
from mtp.data.grouping.base import GROUP_TYPES


def validate_record(row, require_groups=True):
    if (
        not isinstance(row.get("id"), str)
        or not row["id"]
        or not isinstance(row.get("text"), str)
    ):
        raise ValueError("record needs a non-empty string id and string text")
    if not require_groups:
        return row
    groups = row.get("groups")
    n = len(row["text"].split())
    if not isinstance(groups, list) or any(
        not isinstance(g, list) or not g for g in groups
    ):
        raise ValueError("groups must contain non-empty lists")
    flat = [i for g in groups for i in g]
    if any(type(i) is not int for i in flat) or flat != list(range(n)):
        raise ValueError("groups must partition all word indices contiguously in order")
    if len(row.get("types", [])) != len(groups) or any(
        t not in GROUP_TYPES for t in row["types"]
    ):
        raise ValueError("types must be contract labels, one per group")
    return row


def read_jsonl(path, require_groups=False):
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = validate_record(json.loads(line), require_groups)
            if row["id"] in seen:
                raise ValueError(f'duplicate id: {row["id"]}')
            seen.add(row["id"])
            rows.append(row)
    return rows


def write_jsonl(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def parse_spans(value, n_words):
    """Selected spans merge words; all unselected words stay singletons."""
    spans = []
    for part in value.split():
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            raise ValueError(f"bad span: {part}")
        a = int(match[1])
        b = int(match[2] or match[1])
        if a > b or a < 0 or b >= n_words:
            raise ValueError(f"out of bounds span: {part}")
        spans.append((a, b))
    spans.sort()
    for (_, b), (c, _) in zip(spans, spans[1:]):
        if c <= b:
            raise ValueError("overlapping spans")
    starts = dict(spans)
    groups = []
    i = 0
    while i < n_words:
        end = starts.get(i, i)
        groups.append(list(range(i, end + 1)))
        i = end + 1
    return groups


def groups_to_indices(groups):
    out = []
    i = 0
    for group in groups:
        out.append(list(range(i, i + len(group))))
        i += len(group)
    return out

"""JI-5: attachment-edge P/R/F1, exact spans, per-type scores and agreement."""

from collections import Counter
from mtp.data.annotation import validate_record


def attachments(groups):
    return {i for group in groups for i in group[:-1]}


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if not fp else 0.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def score_records(gold, predictions):
    byid = {r["id"]: validate_record(r) for r in predictions}
    if len(byid) != len(predictions):
        raise ValueError("duplicate prediction IDs")
    if len({r["id"] for r in gold}) != len(gold):
        raise ValueError("duplicate gold IDs")
    if {r["id"] for r in gold} != set(byid):
        raise ValueError("prediction/gold ID sets differ")
    tp = fp = fn = exact = total = sentence_exact = 0
    per_type = {}
    for row in gold:
        validate_record(row)
        pred = byid[row["id"]]
        if pred["text"] != row["text"]:
            raise ValueError("text differs for matching ID")
        truth = attachments(row["groups"])
        guess = attachments(pred["groups"])
        tp += len(truth & guess)
        fp += len(guess - truth)
        fn += len(truth - guess)
        truth_spans = {tuple(g): t for g, t in zip(row["groups"], row["types"])}
        guess_spans = {tuple(g): t for g, t in zip(pred["groups"], pred["types"])}
        common = truth_spans.keys() & guess_spans.keys()
        exact += len(common)
        total += len(truth_spans)
        sentence_exact += row["groups"] == pred["groups"]
        for kind in set(truth_spans.values()) | set(guess_spans.values()):
            t = {g for g, k in truth_spans.items() if k == kind}
            p = {g for g, k in guess_spans.items() if k == kind}
            d = per_type.setdefault(kind, Counter())
            d["tp"] += len(t & p)
            d["fp"] += len(p - t)
            d["fn"] += len(t - p)
    return {
        "boundary": prf(tp, fp, fn),
        "exact_group_accuracy": exact / total if total else None,
        "sentence_exact_accuracy": sentence_exact / len(gold) if gold else None,
        "n_sentences": len(gold),
        "n_gold_groups": total,
        "per_type": {k: prf(v["tp"], v["fp"], v["fn"]) for k, v in per_type.items()},
    }


def agreement(a, b):
    first = {r["id"]: validate_record(r) for r in a}
    second = {r["id"]: validate_record(r) for r in b}
    if len(first) != len(a) or len(second) != len(b):
        raise ValueError("duplicate annotation IDs")
    if set(first) != set(second):
        raise ValueError("agreement requires exactly matching ID sets")
    table = Counter()
    for sid, row in first.items():
        other = second[sid]
        if row["text"] != other["text"]:
            raise ValueError("agreement text mismatch")
        x = attachments(row["groups"])
        y = attachments(other["groups"])
        for i in range(len(row["text"].split()) - 1):
            table[(int(i in x), int(i in y))] += 1
    n = sum(table.values())
    if not n:
        return {
            "kappa": None,
            "observed_agreement": None,
            "n_boundaries": 0,
            "n_sentences": len(a),
        }
    observed = (table[(0, 0)] + table[(1, 1)]) / n
    p1 = sum(v for (x, y), v in table.items() if x == 1) / n
    p2 = sum(v for (x, y), v in table.items() if y == 1) / n
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    return {
        "kappa": (observed - expected) / (1 - expected) if expected < 1 else None,
        "observed_agreement": observed,
        "n_boundaries": n,
        "n_sentences": len(a),
    }

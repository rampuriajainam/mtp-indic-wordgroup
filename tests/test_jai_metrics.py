import pytest
from mtp.eval.group_metrics import score_records, agreement
from mtp.eval.group_statistics import measure_grouping
from mtp.data.grouping.base import get_grouper
from scripts.score_groupers import reviewed


def record(groups, types=None):
    return {
        "id": "x",
        "text": "a b c d",
        "groups": groups,
        "types": types or ["single" if len(g) == 1 else "aux_chain" for g in groups],
        "annotator": "jai",
    }


def test_known_boundary_scores_and_exact_groups():
    gold = record([[0, 1], [2, 3]])
    pred = record([[0, 1, 2], [3]])
    score = score_records([gold], [pred])
    b = score["boundary"]
    assert b["tp"] == 1 and b["fp"] == 1 and b["fn"] == 1
    assert b["precision"] == b["recall"] == b["f1"] == 0.5
    assert score["exact_group_accuracy"] == 0
    assert score_records([gold], [gold])["exact_group_accuracy"] == 1


def test_agreement_includes_nonattachments():
    a = record([[0, 1], [2], [3]])
    b = record([[0], [1], [2, 3]])
    result = agreement([a], [b])
    assert result["n_boundaries"] == 3
    assert result["observed_agreement"] == pytest.approx(1 / 3)
    assert result["kappa"] == pytest.approx(-0.5)
    with pytest.raises(ValueError):
        agreement([a], [])


def test_drafts_are_never_official_gold():
    assert not reviewed([{**record([[0], [1], [2], [3]]), "annotator": "rule_draft"}])
    assert not reviewed(
        [{**record([[0], [1], [2], [3]]), "annotation_status": "unreviewed_rule_draft"}]
    )


def test_pair_denominators_are_within_sentences(tiny_tok):
    stats = measure_grouping(["जा रहा था।", "घर"], tiny_tok, get_grouper("hi_rules_v1"))
    assert stats["n_words"] == 4 and stats["n_groups"] == 2
    assert stats["same_group_counts"]["1"] == {"same": 2, "total": 2}
    assert stats["same_group_counts"]["2"] == {"same": 1, "total": 1}
    assert stats["same_group_rate"]["3"] is None
    assert stats["words_per_group"] == stats["tokens_per_group"] == 2

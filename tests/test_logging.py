"""metrics.jsonl format (INTERFACES §9). Ported from Om's om/OM-1-skeleton tests."""

import json

import torch

from mtp.utils.logging import MetricLogger, read_metrics


def test_jsonl_format(tmp_path):
    logger = MetricLogger(tmp_path / "run")
    logger.log(500, "train", "loss/head1", torch.tensor(6.12))
    logger.log(500, "eval", "loss/head0", 3.07)

    rows = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert set(rows[0]) == {"step", "split", "name", "value"}
    assert rows[0]["step"] == 500 and rows[0]["name"] == "loss/head1" and isinstance(rows[0]["value"], float)
    assert abs(rows[0]["value"] - 6.12) < 1e-5
    assert rows[1] == {"step": 500, "split": "eval", "name": "loss/head0", "value": 3.07}


def test_appends_across_sessions(tmp_path):
    run = tmp_path / "run"
    MetricLogger(run).log(20, "train", "loss/head0", 5.0)
    MetricLogger(run).log(40, "train", "loss/head0", 4.0)  # resumed session
    assert [(r["step"], r["value"]) for r in read_metrics(run)] == [(20, 5.0), (40, 4.0)]

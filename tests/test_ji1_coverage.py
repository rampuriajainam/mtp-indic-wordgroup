import json
import subprocess
import sys
import pytest
from mtp.data.grouping.base import get_grouper
from scripts.ji1_coverage import measure


def test_word_metrics_and_absent_tokenizer():
    report = measure(["घर के लिए", "जा रहा था"], get_grouper("hi_rules_v1"))
    assert report["sentences"] == 2
    assert report["words"] == 6 and report["groups"] == 2
    assert report["attached_words"] == 4
    assert report["percent_words_attached"] == pytest.approx(100 * 4 / 6)
    assert report["words_per_group"] == 3
    assert report["tokens_per_group"] is None
    assert report["partition_check"] == "passed"


def test_actual_token_metrics(tiny_tok):
    report = measure(["घर के बारे में"], get_grouper("hi_rules_v1"), tiny_tok)
    assert report["tokens_per_group"] == 4
    assert report["max_length"] == 128


def test_empty_metrics():
    report = measure([], get_grouper("hi_rules_v1"))
    assert report["words_per_group"] == 0
    assert report["percent_words_attached"] == 0


def test_cli_local_input(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("घर के लिए\n\nजा रहा था\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/ji1_coverage.py",
            "--input",
            str(path),
            "--n",
            "5000",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["complete_sample"] is False
    assert len(payload["reports"]) == 2
    assert {r["sentences"] for r in payload["reports"]} == {2}
    assert len({r["sentence_sha256"] for r in payload["reports"]}) == 1

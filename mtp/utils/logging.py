"""metrics.jsonl writer (INTERFACES §9): {"step", "split", "name", "value"} per line."""

import json
from pathlib import Path


class MetricLogger:
    def __init__(self, run_dir):
        self.path = Path(run_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step, split, name, value):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "split": split, "name": name, "value": float(value)}) + "\n")


def read_metrics(run_dir):
    path = Path(run_dir) / "metrics.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

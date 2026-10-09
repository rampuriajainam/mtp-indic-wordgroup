"""Run-folder edge cases (INTERFACES §8). Ported from Om's om/OM-2-checkpoint tests;
save -> load_run round trip and retention are in test_infra.py."""

import pytest

from mtp.model.checkpoint import latest_step, load_run, save_run


def test_save_run_rejects_keep_last_zero(tmp_path):
    with pytest.raises(ValueError, match="keep_last"):
        save_run(tmp_path, model=None, cfg=None, step=1, keep_last=0)


def test_latest_step_ignores_partial_saves(tmp_path):
    for s in (500, 1000):
        (tmp_path / f"step_{s}").mkdir()
        (tmp_path / f"step_{s}" / "heads.pt").write_bytes(b"x")
    (tmp_path / "step_1500").mkdir()  # session died before heads.pt was written
    (tmp_path / "step_x").mkdir()
    (tmp_path / "step_x" / "heads.pt").write_bytes(b"x")
    assert latest_step(tmp_path) == 1000


def test_load_run_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "missing", device="cpu")
    (tmp_path / "config.yaml").write_text("run_name: t\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no step_"):
        load_run(tmp_path, device="cpu")

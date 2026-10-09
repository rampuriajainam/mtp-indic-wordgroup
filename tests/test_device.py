"""Device / dtype policy on CPU and fake GPUs. Ported from Om's om/OM-1-skeleton tests."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from mtp import device


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_cpu_is_fp32_without_autocast_or_scaler(no_gpu):
    cfg = SimpleNamespace(dtype="auto")
    assert device.pick_device() == "cpu"
    assert device.pick_dtype("auto") == torch.float32
    assert isinstance(device.autocast_ctx(cfg), contextlib.nullcontext)
    assert not device.make_scaler(cfg).is_enabled()
    with device.autocast_ctx(cfg):
        y = torch.ones(2, 2) @ torch.ones(2, 2)
    assert y.dtype == torch.float32


def test_fp16_request_on_cpu_is_a_noop(no_gpu):
    cfg = SimpleNamespace(dtype="fp16")
    assert not device.uses_fp16_autocast(cfg) and not device.make_scaler(cfg).is_enabled()


def test_explicit_modes_on_native_bf16_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    assert device.pick_dtype("fp32") == torch.float32
    assert not device.uses_fp16_autocast(SimpleNamespace(dtype="fp32"))
    assert device.uses_fp16_autocast(SimpleNamespace(dtype="fp16"))
    assert not device.uses_fp16_autocast(SimpleNamespace(dtype="bf16"))
    # a cfg without a dtype field means auto
    assert not device.uses_fp16_autocast(SimpleNamespace())

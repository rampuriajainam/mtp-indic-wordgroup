"""
Device / dtype policy. train.py and evaluate.py never branch on dtype themselves.

dtype: auto  -> bf16 weights on GPUs with NATIVE bf16 (compute capability >= 8);
                otherwise (Kaggle T4/P100) fp32 weights + fp16 autocast + GradScaler.
       bf16 | fp16 (= fp32 weights + fp16 autocast) | fp32
"""

import contextlib

import torch


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def native_bf16():
    """torch.cuda.is_bf16_supported() is also True when bf16 is only EMULATED
    (T4, sm_75), which trains ~6x slower (measured: 2.1 vs 0.34 s/step)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def pick_dtype(name="auto"):
    """dtype for the model WEIGHTS."""
    if name == "bf16":
        return torch.bfloat16
    if name in ("fp32", "fp16"):
        return torch.float32
    return torch.bfloat16 if native_bf16() else torch.float32


def uses_fp16_autocast(cfg):
    if not torch.cuda.is_available():
        return False
    name = getattr(cfg, "dtype", "auto")
    return name == "fp16" or (name == "auto" and not native_bf16())


def autocast_ctx(cfg):
    if uses_fp16_autocast(cfg):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_scaler(cfg):
    return torch.amp.GradScaler("cuda", enabled=uses_fp16_autocast(cfg))

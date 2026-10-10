"""
One way to build the model, shared by training (scripts/train.py) and evaluation
(load_run): tokenizer -> base LM (mtp.device.pick_dtype) -> LoRA -> MTPModel.
"""

from mtp.config import cfg_get
from mtp.device import pick_dtype
from mtp.losses.structural import needs_boundary_probes
from mtp.model.heads import MTPModel


def load_tokenizer(name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def uses_probes(cfg):
    return bool(cfg_get(cfg, "losses.structural.enabled", False)
                and needs_boundary_probes(cfg_get(cfg, "losses.structural.variant")))


def build_model(cfg, device):
    """Returns (MTPModel with fresh LoRA + heads, tokenizer). Seed before calling for reproducible head init."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(cfg.model_name)
    base = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=pick_dtype(cfg.dtype)).to(device)
    base = get_peft_model(base, LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.targets), bias="none", task_type="CAUSAL_LM",
    ))
    model = MTPModel(base, num_heads=cfg.num_heads, head_type=cfg.head_type,
                     n_layers=cfg_get(cfg, "head_layers", 1),
                     backbone_grad=cfg_get(cfg, "head_backbone_grad", 1.0),
                     boundary_probes=uses_probes(cfg),
                     contrastive_dim=128 if cfg_get(cfg, "losses.contrastive.enabled", False) else 0)
    return model, tokenizer

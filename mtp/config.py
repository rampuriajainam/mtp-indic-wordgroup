"""
Run configs (INTERFACES §6): YAML -> nested attribute namespace (cfg.optim.lr).

A namespace rather than a fixed dataclass on purpose: optional fields can be
added to configs without touching this file. Read optional fields with
cfg_get(cfg, "optim.grad_accum", 1).
"""

from types import SimpleNamespace

import yaml


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    return obj


def from_dict(d):
    return _to_ns(d)


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return _to_ns(yaml.safe_load(f))


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False, allow_unicode=True)


def apply_overrides(cfg, overrides):
    """CLI overrides: ["optim.lr=1e-4", "losses.structural.enabled=true"]. Creates missing levels."""
    for item in overrides or []:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            if not hasattr(node, p) or getattr(node, p) is None:
                setattr(node, p, SimpleNamespace())
            node = getattr(node, p)
        parsed = yaml.safe_load(value)
        if isinstance(parsed, str):
            try:  # YAML 1.1 reads "1e-4" (no dot) as a string
                parsed = float(parsed)
            except ValueError:
                pass
        setattr(node, leaf, parsed)
    return cfg


def cfg_get(cfg, dotted, default=None):
    """Optional field with a default: cfg_get(cfg, "losses.structural.lambda_s2", 0.1)."""
    node = cfg
    for part in dotted.split("."):
        if node is None or not hasattr(node, part):
            return default
        node = getattr(node, part)
    return node

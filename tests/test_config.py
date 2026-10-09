"""Config contract (INTERFACES §6). Ported from Om's om/OM-1-skeleton tests to main's namespace API."""

import pytest

from mtp.config import apply_overrides, cfg_get, from_dict, load_config, save_config, to_dict

# The example from INTERFACES §6, verbatim.
SECTION6_YAML = """\
run_name: hi_mtp_k4_struct_adaptive
lang: hi
model_name: LingoIITGN/ganga-1b
num_heads: 4
head_type: resblock
lora: {r: 8, alpha: 16, dropout: 0.05, targets: [q_proj, v_proj]}
optim: {lr: 2.0e-4, batch_size: 8, grad_clip: 1.0, max_steps: 12500}
data: {cache_dir: /kaggle/input/mtp-boundary-cache, grouper: hi_rules_v1, max_length: 128}
losses:
  structural: {enabled: true, variant: S2, weight: 0.1}
  contrastive: {enabled: false, weight: 0.05, temperature: 0.1}
weighting: {scheme: uncertainty}     # fixed | uncertainty | dwa
eval_every: 250
save_every: 500
seed: 42
dtype: auto                          # auto | bf16 | fp16 | fp32
"""


@pytest.fixture
def section6(tmp_path):
    path = tmp_path / "R5.yaml"
    path.write_text(SECTION6_YAML, encoding="utf-8")
    return path


def test_loads_section6_example(section6):
    cfg = load_config(section6)
    assert cfg.run_name == "hi_mtp_k4_struct_adaptive" and cfg.num_heads == 4
    assert cfg.lora.targets == ["q_proj", "v_proj"]
    assert cfg.optim.lr == pytest.approx(2e-4)
    assert cfg.data.grouper == "hi_rules_v1"
    assert cfg.losses.structural.enabled is True
    assert cfg.losses.contrastive.temperature == pytest.approx(0.1)
    assert cfg.weighting.scheme == "uncertainty"
    # optional fields are read with defaults, never required
    assert cfg_get(cfg, "keep_steps", []) == [] and cfg_get(cfg, "optim.grad_accum", 1) == 1


def test_round_trip_devanagari_and_none(tmp_path):
    cfg = from_dict({"run_name": "हिंदी_run", "git_commit": None, "optim": {"lr": 2e-4}})
    save_config(cfg, tmp_path / "c.yaml")
    assert "हिंदी_run" in (tmp_path / "c.yaml").read_text(encoding="utf-8")
    assert to_dict(load_config(tmp_path / "c.yaml")) == to_dict(cfg)


def test_overrides_coerce_types(section6):
    cfg = apply_overrides(load_config(section6), [
        "optim.lr=1e-4",  # PyYAML alone reads this as a string
        "num_heads=2",
        "losses.contrastive.enabled=true",
        "lora.targets=[q_proj,k_proj,v_proj]",
        "data.grouper=null",
        "keep_steps=[500, 1000]",
    ])
    assert cfg.optim.lr == 1e-4 and isinstance(cfg.optim.lr, float)
    assert cfg.num_heads == 2 and cfg.losses.contrastive.enabled is True
    assert cfg.lora.targets == ["q_proj", "k_proj", "v_proj"]
    assert cfg.data.grouper is None and cfg.keep_steps == [500, 1000]


def test_override_value_may_contain_equals():
    cfg = apply_overrides(from_dict({}), ["run_name=a=b"])
    assert cfg.run_name == "a=b"

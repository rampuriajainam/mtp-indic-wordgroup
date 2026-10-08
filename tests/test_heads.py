import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import MistralConfig, MistralForCausalLM

from mtp.losses.mtp_ce import per_head_ce
from mtp.model.heads import MTPModel, MTPOutput

V, H = 100, 32


def tiny_base(seed=0):
    torch.manual_seed(seed)
    cfg = MistralConfig(
        vocab_size=V, hidden_size=H, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=False,  # ganga-1b has an untied lm_head
    )
    return MistralForCausalLM(cfg).eval()


def lora(model):
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                     lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, cfg)
    # Non-zero LoRA B so the adapter actually changes the output.
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.5)
    return peft_model


def batch():
    torch.manual_seed(1)
    ids = torch.randint(3, V, (2, 7))
    mask = torch.ones_like(ids)
    mask[1, 5:] = 0          # second row right-padded
    ids[1, 5:] = 0
    return ids, mask


@pytest.mark.parametrize("use_lora", [False, True])
@pytest.mark.parametrize("head_type", ["linear", "resblock"])
def test_head0_equals_base_logits(head_type, use_lora):
    base = tiny_base()
    if use_lora:
        base = lora(base)
    ids, mask = batch()
    model = MTPModel(base, num_heads=3, head_type=head_type).eval()
    with torch.no_grad():
        ref = base(input_ids=ids, attention_mask=mask).logits
        out = model(ids, attention_mask=mask)
    assert isinstance(out, MTPOutput)
    assert len(out.logits) == 3
    assert out.hidden.shape == (2, 7, H)
    for lg in out.logits:
        assert lg.shape == (2, 7, V)
    torch.testing.assert_close(out.logits[0], ref)


def test_lora_is_applied_through_decoder():
    ids, mask = batch()
    plain = tiny_base()
    with torch.no_grad():
        plain_logits = MTPModel(plain, num_heads=1)(ids, mask).logits[0]
    lora_model = MTPModel(lora(tiny_base()), num_heads=1)
    with torch.no_grad():
        lora_logits = lora_model(ids, mask).logits[0]
    assert not torch.allclose(plain_logits, lora_logits)


@pytest.mark.parametrize("n_layers", [1, 2])
def test_resblock_zero_init_copies_head0(n_layers):
    ids, mask = batch()
    model = MTPModel(lora(tiny_base()), num_heads=4, head_type="resblock", n_layers=n_layers)
    with torch.no_grad():
        out = model(ids, mask)
    for d in range(1, 4):
        assert torch.equal(out.logits[d], out.logits[0])
        assert torch.equal(out.aux["head_hidden"][d], out.hidden)


def test_param_counts():
    lin = MTPModel(tiny_base(), num_heads=4, head_type="linear")
    res = MTPModel(tiny_base(), num_heads=4, head_type="resblock", n_layers=2)
    ntp = MTPModel(tiny_base(), num_heads=1, head_type="resblock")
    count = lambda m: sum(p.numel() for p in m.extra_heads.parameters())
    assert count(lin) == 3 * H * V
    assert count(res) == 3 * 2 * (H * H + H)
    assert count(ntp) == 0
    # decoder / lm_head must not be registered a second time outside base_model
    assert all(k.startswith(("base_model.", "extra_heads.")) for k in res.state_dict())


def test_resblock_learns_and_lm_head_stays_frozen():
    ids, mask = batch()
    model = MTPModel(lora(tiny_base()), num_heads=3, head_type="resblock")
    model.train()
    out = model(ids, mask)
    sum(per_head_ce(out.logits, ids, mask, reduction="mean")).backward()
    for head in model.extra_heads:
        assert head[0].linear.weight.grad is not None
        assert head[0].linear.weight.grad.abs().sum() > 0
    assert model.base_model.get_output_embeddings().weight.grad is None


def test_heads_follow_base_dtype():
    base = tiny_base().to(torch.bfloat16)
    for head_type in ("linear", "resblock"):
        model = MTPModel(base, num_heads=2, head_type=head_type)
        assert all(p.dtype == torch.bfloat16 for p in model.extra_heads.parameters())


def test_bad_args():
    with pytest.raises(ValueError):
        MTPModel(tiny_base(), num_heads=0)
    with pytest.raises(ValueError):
        MTPModel(tiny_base(), num_heads=2, head_type="mlp")


def test_use_cache_returns_past_key_values():
    ids, mask = batch()
    model = MTPModel(tiny_base(), num_heads=2, head_type="resblock")
    with torch.no_grad():
        assert "past_key_values" not in model(ids, mask).aux
        assert model(ids, mask, use_cache=True).aux["past_key_values"] is not None


@pytest.mark.parametrize("head_type", ["linear", "resblock"])
def test_backbone_grad_scaling(head_type):
    ids, mask = batch()

    def lora_grad(a):
        model = MTPModel(lora(tiny_base()), num_heads=3, head_type=head_type, backbone_grad=a)
        if head_type == "resblock":  # non-zero W so the head path has a gradient to scale
            for head in model.extra_heads:
                torch.nn.init.normal_(head[0].linear.weight, std=0.1)
        out = model(ids, mask)
        sum(per_head_ce(out.logits[1:], ids, mask, reduction="mean")).backward()
        return out, torch.cat([p.grad.flatten() for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
                              or [torch.zeros(1)])

    out1, g1 = lora_grad(1.0)
    out0, g0 = lora_grad(0.0)
    out_h, g_h = lora_grad(0.25)
    for d in range(3):
        torch.testing.assert_close(out0.logits[d], out1.logits[d])   # forward unchanged
    assert g1.abs().sum() > 0
    assert g0.abs().sum() == 0                                     # heads 1..k-1 don't reach LoRA
    torch.testing.assert_close(g_h, 0.25 * g1)

import math

import pytest
import torch
import torch.nn.functional as F

from mtp.losses.contrastive import SupConLoss
from mtp.model.heads import MTPOutput


def _out(z):
    return MTPOutput(logits=[], hidden=z, aux={"contrastive_z": F.normalize(z, dim=-1)})


def test_supcon_matches_hand_computation():
    torch.manual_seed(0)
    z = torch.randn(1, 4, 8)
    batch = {"group_id": torch.tensor([[0, 0, 1, -1]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    got = SupConLoss(temperature=0.5)(_out(z), batch)["contrastive/supcon"]
    f = F.normalize(z[0, :3], dim=-1)
    sim = f @ f.T / 0.5
    # anchors 0 and 1 have one positive each (each other); anchor 2 has none
    l0 = -(sim[0, 1] - torch.logsumexp(sim[0, [1, 2]], 0))
    l1 = -(sim[1, 0] - torch.logsumexp(sim[1, [0, 2]], 0))
    torch.testing.assert_close(got, (l0 + l1) / 2)


def test_supcon_separates_groups_across_sentences():
    """Same group_id in different sentences is not a positive pair."""
    z = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    batch = {"group_id": torch.tensor([[0], [0]]), "attention_mask": torch.ones(2, 1, dtype=torch.long)}
    out = SupConLoss()(_out(z), batch)["contrastive/supcon"]
    assert out.item() == 0.0      # no anchor has a positive


def test_supcon_lower_when_groups_cluster():
    good = torch.tensor([[[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [0.1, 0.99]]])
    bad = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.99, 0.1], [0.1, 0.99]]])
    batch = {"group_id": torch.tensor([[0, 0, 1, 1]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    loss = SupConLoss(temperature=0.1)
    assert loss(_out(good), batch)["contrastive/supcon"] < loss(_out(bad), batch)["contrastive/supcon"]


def test_supcon_subsamples_and_has_grad():
    z = torch.randn(2, 400, 16, requires_grad=True)
    gid = torch.arange(400).div(2, rounding_mode="floor").repeat(2, 1)
    batch = {"group_id": gid, "attention_mask": torch.ones(2, 400, dtype=torch.long)}
    out = SupConLoss(max_tokens=512)(_out(z), batch)["contrastive/supcon"]
    out.backward()
    assert math.isfinite(out.item()) and z.grad is not None


def test_model_projection_and_checkpoint_keys():
    from tests.test_heads import tiny_base
    from mtp.model.heads import MTPModel

    m = MTPModel(tiny_base(), num_heads=2, head_type="resblock", contrastive_dim=8)
    out = m(torch.randint(3, 100, (1, 5)))
    assert out.aux["contrastive_z"].shape == (1, 5, 8)
    torch.testing.assert_close(out.aux["contrastive_z"].norm(dim=-1), torch.ones(1, 5))
    assert any(k.startswith("contrastive_proj.") for k in m.head_state_dict())
    m2 = MTPModel(tiny_base(), num_heads=2, head_type="resblock", contrastive_dim=8)
    m2.load_head_state_dict(m.head_state_dict())
    with pytest.raises(KeyError):
        MTPModel(tiny_base(), num_heads=2, head_type="resblock").load_head_state_dict(m.head_state_dict())

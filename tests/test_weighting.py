import math

import pytest
import torch

from mtp.losses.weighting import LossWeighting

NAMES = ["loss/head0", "loss/head1", "loss/head2", "struct/boundary_bce/h1", "contrastive/supcon"]
AUX = {"struct": 0.1, "struct/boundary_bce": 0.3}


def ones(names=NAMES):
    return {n: torch.tensor(1.0) for n in names}


def test_fixed():
    w = LossWeighting("fixed", NAMES, head_decay=0.8, aux_weights=AUX)
    total, weights = w(ones())
    assert weights["loss/head2"] == pytest.approx(0.64)
    assert weights["struct/boundary_bce/h1"] == pytest.approx(0.3)   # longest prefix wins
    assert weights["contrastive/supcon"] == 1.0
    assert total.item() == pytest.approx(1 + 0.8 + 0.64 + 0.3 + 1.0)
    assert list(w.parameters()) == []
    with pytest.raises(KeyError):
        w({"loss/head9": torch.tensor(1.0)})
    with pytest.raises(ValueError):
        LossWeighting("magic", NAMES)


@pytest.mark.parametrize("fix_head0", [True, False])
def test_uncertainty_starts_at_fixed_weights(fix_head0):
    fixed = LossWeighting("fixed", NAMES, aux_weights=AUX)
    unc = LossWeighting("uncertainty", NAMES, aux_weights=AUX, fix_head0=fix_head0)
    _, wf = fixed(ones())
    _, wu = unc(ones())
    for n in NAMES:
        assert wu[n] == pytest.approx(wf[n], rel=1e-5)
    assert unc.log_var.numel() == len(NAMES) - (1 if fix_head0 else 0)


def test_uncertainty_formula_and_learning():
    unc = LossWeighting("uncertainty", ["loss/head0", "loss/head1"], fix_head0=True)
    with torch.no_grad():
        unc.log_var.fill_(0.7)
    L0, L1 = torch.tensor(3.0), torch.tensor(6.0)
    total, w = unc({"loss/head0": L0, "loss/head1": L1})
    assert total.item() == pytest.approx(3.0 + 0.5 * math.exp(-0.7) * 6.0 + 0.5 * 0.7)
    assert w["loss/head0"] == 1.0
    # optimum of 0.5 e^{-s} L + 0.5 s is s = log L: gradient descent should head there
    opt = torch.optim.SGD(unc.parameters(), lr=0.5)
    for _ in range(300):
        opt.zero_grad()
        unc({"loss/head1": L1})[0].backward()
        opt.step()
    assert unc.log_var.item() == pytest.approx(math.log(6.0), abs=1e-2)


def test_uncertainty_clamps():
    unc = LossWeighting("uncertainty", ["loss/head1"], fix_head0=False)
    with torch.no_grad():
        unc.log_var.fill_(10.0)
    _, w = unc({"loss/head1": torch.tensor(1.0)})
    assert w["loss/head1"] == pytest.approx(0.5 * math.exp(-4))


def test_dwa_windows_and_state_dict():
    names = ["loss/head0", "loss/head1", "loss/head2"]
    dwa = LossWeighting("dwa", names, head_decay=0.8, dwa_window=2, dwa_temperature=1.0)
    seq = [  # head1 falls fast (ratio 0.5), head2 stays flat (ratio 1.0)
        {"loss/head0": 3.0, "loss/head1": 8.0, "loss/head2": 9.0},
        {"loss/head0": 3.0, "loss/head1": 8.0, "loss/head2": 9.0},
        {"loss/head0": 3.0, "loss/head1": 4.0, "loss/head2": 9.0},
        {"loss/head0": 3.0, "loss/head1": 4.0, "loss/head2": 9.0},
    ]
    for i, step in enumerate(seq):
        _, w = dwa({k: torch.tensor(v) for k, v in step.items()})
        if i < 3:   # fewer than two finished windows -> fixed weights
            assert w["loss/head1"] == pytest.approx(0.8) and w["loss/head2"] == pytest.approx(0.64)
    sm = torch.softmax(torch.tensor([0.5, 1.0]), 0) * 2
    assert w["loss/head0"] == 1.0
    assert w["loss/head1"] == pytest.approx(0.8 * sm[0].item(), rel=1e-5)
    assert w["loss/head2"] == pytest.approx(0.64 * sm[1].item(), rel=1e-5)
    # buffers carry the state across a checkpoint
    fresh = LossWeighting("dwa", names, head_decay=0.8, dwa_window=2, dwa_temperature=1.0)
    fresh.load_state_dict(dwa.state_dict())
    fresh.eval()
    _, w2 = fresh({k: torch.tensor(v) for k, v in seq[-1].items()})
    assert w2 == pytest.approx(w)

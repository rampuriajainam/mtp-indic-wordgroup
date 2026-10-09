import math

import pytest
import torch
import torch.nn.functional as F

from mtp.losses.structural import StructuralLoss
from mtp.model.heads import MTPOutput

# One sentence, groups:  [0 1 2] [3] [4 5]  pad
#   position:  0  1  2  3  4  5  6
GID = torch.tensor([[0, 0, 0, 1, 2, 2, -1]])
START = torch.tensor([[1, 0, 0, 1, 1, 0, 0]])
MASK = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
IDS = torch.tensor([[1, 2, 3, 4, 5, 6, 0]])
BATCH = {"input_ids": IDS, "attention_mask": MASK, "group_id": GID, "group_start": START}
T, V, K = 7, 5, 3


def out_with(logits=None, boundary=None):
    g = torch.Generator().manual_seed(0)
    logits = logits or [torch.randn(1, T, V, generator=g, requires_grad=True) for _ in range(K)]
    aux = {} if boundary is None else {"boundary_logits": boundary}
    return MTPOutput(logits=logits, hidden=torch.zeros(1, T, 4), aux=aux)


def kl(teacher_logits, student_logits):
    lt, ls = F.log_softmax(teacher_logits, -1), F.log_softmax(student_logits, -1)
    return (lt.exp() * (lt - ls)).sum().item()


def test_term_names():
    assert StructuralLoss("S2", 3).term_names == [f"struct/boundary_bce/h{d}" for d in range(3)]
    assert StructuralLoss("S3_h0", 3).term_names == ["struct/consistency/h1", "struct/consistency/h2"]
    assert len(StructuralLoss("S23", 4).term_names) == 4 + 3
    with pytest.raises(ValueError):
        StructuralLoss("TBD", 3)
    with pytest.raises(ValueError):
        StructuralLoss("S23", 3, s3_teacher="oracle")


def test_s2_uniform_probe_is_ln2():
    boundary = [torch.zeros(1, T, requires_grad=True) for _ in range(K)]
    terms = StructuralLoss("S2", K)(out_with(boundary=boundary), BATCH)
    for d in range(K):
        assert terms[f"struct/boundary_bce/h{d}"].item() == pytest.approx(math.log(2))


def test_s2_hand_computed_and_masking():
    # Head 0 (shift 1): valid pairs t=0..4 (t=5 -> u=6 is pad); targets s[1..5] = 0,0,1,1,0.
    logit0 = torch.tensor([[2.0, -1.0, 0.5, 3.0, -2.0, 9.0, 9.0]])   # t=5,6 must be ignored
    boundary = [logit0] + [torch.zeros(1, T) for _ in range(K - 1)]
    terms = StructuralLoss("S2", K)(out_with(boundary=boundary), BATCH)
    sp = lambda x: math.log1p(math.exp(x))                           # softplus
    expected = (sp(2.0) + sp(-1.0) + sp(-0.5) + sp(-3.0) + sp(-2.0)) / 5   # BCE: y=0 -> sp(x), y=1 -> sp(-x)
    assert terms["struct/boundary_bce/h0"].item() == pytest.approx(expected, rel=1e-6)


def test_s2_needs_probes():
    with pytest.raises(KeyError):
        StructuralLoss("S2", K)(out_with(), BATCH)


def test_s3_h0_pairs_and_mask():
    out = out_with()
    terms = StructuralLoss("S3_h0", K)(out, BATCH)
    L = out.logits
    # head 1: t, u=t+2, teacher head 0 at t+1. In-group needs g(t+1)==g(u), all real:
    #   t=0: g1=0,g2=0 yes | t=1: g2=0,g3=1 no | t=2: g3=1,g4=2 no | t=3: g4=2,g5=2 yes | t=4: u=6 pad
    exp1 = (kl(L[0][0, 1], L[1][0, 0]) + kl(L[0][0, 4], L[1][0, 3])) / 2
    assert terms["struct/consistency/h1"].item() == pytest.approx(exp1, rel=1e-5)
    # head 2: u=t+3, teacher head 0 at u-1=t+2; in-group g(t+1)==g(u): t=0: g1=0,g3=1 no |
    #   t=1: g2=0,g4=2 no | t=2: g3=1,g5=2 no | t=3: u=6 pad  -> no pairs -> 0
    assert terms["struct/consistency/h2"].item() == 0.0


def test_s3_chain_uses_previous_head_as_teacher():
    out = out_with()
    terms = StructuralLoss("S3_chain", K)(out, BATCH)
    L = out.logits
    h1 = (kl(L[0][0, 1], L[1][0, 0]) + kl(L[0][0, 4], L[1][0, 3])) / 2   # d=1: chain teacher is head 0 too
    assert terms["struct/consistency/h1"].item() == pytest.approx(h1, rel=1e-5)


def test_s3_all_ignores_groups():
    out = out_with()
    terms = StructuralLoss("S3_all", K)(out, BATCH)
    L = out.logits
    # head 2: every valid t (u=t+3 real): t=0,1,2; teacher head 0 at t+2
    exp2 = sum(kl(L[0][0, t + 2], L[2][0, t]) for t in range(3)) / 3
    assert terms["struct/consistency/h2"].item() == pytest.approx(exp2, rel=1e-5)


def test_s3_teacher_gets_no_gradient():
    out = out_with()
    terms = StructuralLoss("S3_all", K)(out, BATCH)
    sum(terms.values()).backward()
    L = out.logits
    assert L[1].grad.abs().sum() > 0 and L[2].grad.abs().sum() > 0
    # head 0 is only ever a teacher -> no gradient at all
    assert L[0].grad is None or L[0].grad.abs().sum() == 0


def test_short_and_empty_batches_give_zero_not_nan():
    short = {k: v[:, :2] for k, v in BATCH.items()}
    out = MTPOutput(logits=[torch.randn(1, 2, V, requires_grad=True) for _ in range(K)],
                    hidden=torch.zeros(1, 2, 4), aux={"boundary_logits": [torch.zeros(1, 2, requires_grad=True)] * K})
    terms = StructuralLoss("S23", K)(out, short)
    assert all(torch.isfinite(v) for v in terms.values())
    sum(terms.values()).backward()

    no_groups = dict(BATCH, group_id=torch.full_like(GID, -1))
    terms = StructuralLoss("S23", K)(out_with(boundary=[torch.zeros(1, T)] * K), no_groups)
    assert all(v.item() == 0.0 for v in terms.values())


@pytest.mark.parametrize("variant", ["S3_h0", "S3_chain", "S23"])
def test_no_in_group_pair_under_fp16_is_zero_not_nan(variant):
    """Kaggle pilots, step 134: a batch with no in-group pair for the last head. The zero term was
    logits.sum() * 0, and the sum of fp16 logits overflows to inf -> NaN."""
    every_token_its_own_group = dict(BATCH, group_id=torch.tensor([[0, 1, 2, 3, 4, 5, -1]]))
    logits = [torch.full((1, T, V), 1e4, dtype=torch.float16, requires_grad=True) for _ in range(K)]
    assert torch.isinf(logits[0].sum())  # the trap
    boundary = [torch.zeros(1, T, dtype=torch.float16, requires_grad=True) for _ in range(K)]
    terms = StructuralLoss(variant, K)(out_with(logits, boundary), every_token_its_own_group)
    for name, v in terms.items():
        assert torch.isfinite(v), name
    sum(v for n, v in terms.items() if n.startswith("struct/consistency")).backward()
    assert all(torch.isfinite(l.grad).all() for l in logits if l.grad is not None)

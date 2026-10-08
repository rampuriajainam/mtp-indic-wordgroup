import math

import pytest
import torch
import torch.nn.functional as F

from mtp.losses.mtp_ce import per_head_ce, shift_targets


def legacy_compute_head_loss(logits, input_ids, attention_mask, shift):
    """Verbatim logic of compute_head_loss in train_mtp_baseline.py."""
    if logits.shape[1] <= shift:
        return None
    pred_logits = logits[:, :-shift, :]
    targets = input_ids[:, shift:].clone()
    targets[attention_mask[:, shift:] == 0] = -100
    return F.cross_entropy(pred_logits.reshape(-1, pred_logits.shape[-1]), targets.reshape(-1), ignore_index=-100)


def make(B=3, T=9, V=17, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = [torch.randn(B, T, V, generator=g) for _ in range(k)]
    ids = torch.randint(0, V, (B, T), generator=g)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[1, 6:] = 0
    mask[2, 3:] = 0
    return logits, ids, mask


def test_mean_matches_legacy_right_padding():
    logits, ids, mask = make()
    ours = per_head_ce(logits, ids, mask, reduction="mean")
    for d, lg in enumerate(logits):
        torch.testing.assert_close(ours[d], legacy_compute_head_loss(lg, ids, mask, d + 1))


def test_none_shapes_and_masking():
    logits, ids, mask = make()
    per_pos = per_head_ce(logits, ids, mask, reduction="none")
    for d, losses in enumerate(per_pos):
        s = d + 1
        assert losses.shape == (3, 9 - s)
        assert losses.dtype == torch.float32
        _, valid = shift_targets(ids, mask, s)
        assert torch.all(losses[~valid] == 0)
        assert torch.all(losses[valid] > 0)
        sums = per_head_ce(logits, ids, mask, reduction="sum")
        torch.testing.assert_close(losses.sum(), sums[d])


def test_hand_computed_value():
    # One sentence, V=2, uniform logits -> every valid position costs ln 2.
    logits = [torch.zeros(1, 4, 2), torch.zeros(1, 4, 2)]
    ids = torch.tensor([[0, 1, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    none = per_head_ce(logits, ids, mask, reduction="none")
    ln2 = math.log(2)
    # head 0 (shift 1): pairs (0->1),(1->2) valid, (2->3) target is pad
    torch.testing.assert_close(none[0], torch.tensor([[ln2, ln2, 0.0]]))
    # head 1 (shift 2): (0->2) valid, (1->3) pad
    torch.testing.assert_close(none[1], torch.tensor([[ln2, 0.0]]))
    mean = per_head_ce(logits, ids, mask, reduction="mean")
    torch.testing.assert_close(mean[0], torch.tensor(ln2))
    torch.testing.assert_close(mean[1], torch.tensor(ln2))


def test_source_padding_is_masked_too():
    # Left padding: source pad -> real target must not count (legacy would count it).
    logits = [torch.randn(1, 4, 5)]
    ids = torch.tensor([[0, 2, 3, 4]])
    mask = torch.tensor([[0, 1, 1, 1]])
    _, valid = shift_targets(ids, mask, 1)
    assert valid.tolist() == [[False, True, True]]
    assert per_head_ce(logits, ids, mask)[0][0, 0] == 0


def test_short_sequence_gives_zero_not_nan():
    lg = torch.randn(2, 3, 5, requires_grad=True)
    ids = torch.randint(0, 5, (2, 3))
    mask = torch.ones_like(ids)
    out = per_head_ce([lg, lg, lg, lg], ids, mask, reduction="mean")
    assert out[3].item() == 0 and out[2].item() == 0      # T <= shift
    assert per_head_ce([lg] * 4, ids, mask, reduction="none")[3].shape == (2, 0)
    all_pad = per_head_ce([lg], ids, torch.zeros_like(ids), reduction="mean")[0]
    assert all_pad.item() == 0 and not torch.isnan(all_pad)
    sum(out).backward()                                    # graph stays intact
    assert lg.grad is not None


def test_no_mask_means_all_valid():
    logits, ids, _ = make()
    torch.testing.assert_close(
        per_head_ce(logits, ids, None, reduction="mean"),
        per_head_ce(logits, ids, torch.ones_like(ids), reduction="mean"),
    )


def test_half_precision_logits_upcast():
    logits, ids, mask = make()
    half = [lg.to(torch.bfloat16) for lg in logits]
    for x in per_head_ce(half, ids, mask, reduction="none"):
        assert x.dtype == torch.float32


def test_bad_reduction():
    logits, ids, mask = make()
    with pytest.raises(ValueError):
        per_head_ce(logits, ids, mask, reduction="avg")

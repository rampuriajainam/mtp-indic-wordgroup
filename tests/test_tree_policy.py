"""JN-9: tree draft policies (mtp/eval/tree_policy.py)."""

import random

import pytest
import torch

from mtp.eval.tree_policy import (EntropyTree, StaticTree, accepted_length, check_tree, collect_ranks,
                                  draft_tokens, fit_entropy, fit_prefixes, fit_static, load_policy,
                                  nodes_from_prefixes, prefixes_from_nodes, save_policy, widths_tree)
from test_spec_decode import _tiny_model


def test_widths_tree_shape_and_chain():
    t = widths_tree((3, 2, 1))
    assert len(t) == 3 + 6 + 6
    check_tree(t, num_heads=4)
    assert widths_tree((1, 1, 1)) == [(-1, 1, 0), (0, 2, 0), (1, 3, 0)]  # = FixedK
    assert prefixes_from_nodes(nodes_from_prefixes(prefixes_from_nodes(t))) == prefixes_from_nodes(t)


@pytest.mark.parametrize("bad", [[(0, 1, 0)], [(-1, 2, 0)], [(-1, 1, 0), (0, 3, 0)], [(-1, 1, 0), (-1, 1, 0)],
                                 [(-1, 1, -1)]])
def test_check_tree_rejects(bad):
    with pytest.raises(ValueError):
        check_tree(bad)


def test_not_prefix_closed():
    with pytest.raises(ValueError):
        nodes_from_prefixes([(0,), (1, 0)])


def test_accepted_length():
    t = widths_tree((2, 1, 1))
    assert accepted_length(t, (0, 0, 0)) == 3
    assert accepted_length(t, (1, 0, 5)) == 2
    assert accepted_length(t, (2, 0, 0)) == 0
    assert accepted_length(t, (0, -1, 0)) == 1


def test_fit_prefixes_is_prefix_closed_and_capped():
    rng = random.Random(0)
    ranks = [[min(int(rng.expovariate(0.7)), 49) for _ in range(3)] for _ in range(2000)]
    for n in (1, 3, 7, 15, 25):
        p = fit_prefixes(ranks, n)
        assert len(p) == n
        nodes_from_prefixes(p)  # raises if not prefix-closed
    assert fit_prefixes(ranks, 1) == [(0,)]


def test_fit_beats_product_tree_at_equal_size():
    """Medusa's point: on skewed data the frequency-fitted tree accepts more than a product tree."""
    rng = random.Random(1)
    ranks = [[0 if rng.random() < 0.6 else rng.randrange(1, 4) for _ in range(3)] for _ in range(3000)]
    fixed, fitted = widths_tree((3, 2, 1)), fit_static(ranks, 15).nodes
    acc = lambda t: sum(accepted_length(t, r) for r in ranks)
    assert acc(fitted) >= acc(fixed)


def test_draft_tokens():
    logits = [torch.zeros(10)] + [torch.arange(10.0) * (d + 1) for d in range(1, 4)]  # argmax 9, then 8, ...
    t = widths_tree((2, 1, 1))
    assert draft_tokens(t, logits) == [9, 8, 9, 9, 9, 9]


def test_entropy_tree_picks_cell_and_roundtrips(tmp_path):
    rng = random.Random(2)
    ranks, ents = [], []
    for _ in range(4000):
        sharp = rng.random() < 0.5
        ranks.append([0 if sharp or rng.random() < 0.3 else rng.randrange(1, 6) for _ in range(3)])
        ents.append([(0.05 if sharp else 1.0) + 0.01 * rng.random()] * 3)  # same per row: cells 0 or 7
    pol = fit_entropy(ranks, ents, max_nodes=7)
    assert pol.max_nodes <= 7 and set(pol.trees) == {0, 7}
    sharp = [torch.zeros(10)] + [torch.tensor([20.0] + [0.0] * 9)] * 3
    flat = [torch.zeros(10)] * 4
    assert pol.cell(sharp) == 0 and pol.cell(flat) == 7
    assert pol.tree(sharp, None, {}) == pol.trees[0]
    # the confident cell's tree is a long chain, the unsure cell's is wide
    assert max(d for _, d, _ in pol.trees[0]) == 3
    assert sum(1 for _, d, _ in pol.trees[7] if d == 1) > 1
    save_policy(pol, tmp_path / "p.json")
    back = load_policy(tmp_path / "p.json")
    assert isinstance(back, EntropyTree) and back.trees == pol.trees and back.medians == pol.medians
    st = StaticTree.from_widths((3, 2, 1))
    save_policy(st, tmp_path / "s.json")
    assert load_policy(tmp_path / "s.json").nodes == st.nodes


def test_collect_ranks_matches_tree_verification(tiny_model_dir, tmp_path):
    """accepted_length on collected ranks == verifying the tree's actual tokens against greedy."""
    model, tok = _tiny_model(tiny_model_dir, tmp_path, "resblock")
    prompts = [tok(s)["input_ids"] for s in ["मैं कल बाजार", "भारत एक विशाल", "यह एक परीक्षण"]]
    ranks, ents = collect_ranks(model, tok, prompts, max_new_tokens=12)
    assert ranks and len(ranks) == len(ents) and all(len(r) == 3 for r in ranks)
    tree = widths_tree((4, 3, 2))
    from mtp.eval.spec_decode import greedy_generate
    i = 0
    with torch.no_grad():
        for p in prompts:
            new, _ = greedy_generate(model, tok, p, 12)
            full = list(p) + new
            out = model(torch.tensor([full]))
            for t in range(len(p) - 1, len(full) - 4):
                tokens = draft_tokens(tree, [lg[0, t] for lg in out.logits])
                path = {}
                best = 0
                for (parent, d, _), tk in zip(tree, tokens):
                    ok = (parent < 0 or path[parent]) and tk == full[t + d + 1]
                    path[len(path)] = ok
                    best = max(best, d if ok else 0)
                assert accepted_length(tree, ranks[i]) == best
                i += 1
    assert i == len(ranks)

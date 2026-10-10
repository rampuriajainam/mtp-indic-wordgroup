import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml

from mtp.config import load_config
from mtp.eval.draft_policy import POLICIES, ConfidenceCut, FixedK, get_policy
from mtp.eval.spec_decode import (evaluate_spec_decode, generate, greedy_generate, group_integrity_counts,
                                  make_prompts, token_group_ids)
from mtp.model.heads import MTPOutput

ROOT = Path(__file__).resolve().parents[1]
V = 50
EOS = None


# ----------------------------------------------------------------------------- fakes with known answers

class FakeCache:
    def __init__(self):
        self.n = 0

    def crop(self, k):
        assert k < 0, "engine must crop with a negative 'tokens to remove'"
        self.n += k

    def get_seq_length(self):
        return self.n


class CountingModel(nn.Module):
    """Greedy continuation of token x is x+1 (mod V). Head d proposes x+d+1 (right) or x+d+2 (wrong)."""

    def __init__(self, num_heads=3, right=True):
        super().__init__()
        self.num_heads, self.right = num_heads, right
        self.dummy = nn.Parameter(torch.zeros(1))
        self.base_model = self._base

    def _logits(self, ids, shift):
        return torch.nn.functional.one_hot((ids + shift) % V, V).float()[None] * 5

    def _base(self, input_ids, past_key_values=None, use_cache=False):
        cache = past_key_values or FakeCache()
        cache.n += input_ids.shape[1]
        return SimpleNamespace(logits=self._logits(input_ids[0], 1), past_key_values=cache)

    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        x = input_ids[0]
        logits = [self._logits(x, 1)] + [self._logits(x, d + 1 + (0 if self.right else 1)) for d in range(1, self.num_heads)]
        aux = {}
        if use_cache:
            cache = past_key_values or FakeCache()
            cache.n += input_ids.shape[1]
            aux["past_key_values"] = cache
        return MTPOutput(logits=logits, hidden=torch.zeros(1, len(x), 1), aux=aux)


TOK = SimpleNamespace(eos_token_id=None)


@pytest.mark.parametrize("use_cache", [True, False])
def test_perfect_drafts_all_accepted(use_cache):
    model = CountingModel(num_heads=3, right=True)
    ids, st = generate(model, TOK, [7], 12, FixedK(), use_cache=use_cache)
    ref, gst = greedy_generate(model, TOK, [7], 12, use_cache=use_cache)
    assert ids == ref == [(7 + i) % V for i in range(1, 13)]
    assert st["steps"] == 4 and st["mean_accepted_len"] == 3.0 and st["spans"] == [3, 3, 3, 3]
    assert st["accepted"] == st["proposed"] == [4, 4]
    assert st["forward_passes"] == 5 and gst["forward_passes"] == 12  # 1 prompt pass + 1 per step


@pytest.mark.parametrize("use_cache", [True, False])
def test_wrong_drafts_all_rejected_output_unchanged(use_cache):
    model = CountingModel(num_heads=3, right=False)
    ids, st = generate(model, TOK, [7], 10, FixedK(), use_cache=use_cache)
    assert ids == greedy_generate(model, TOK, [7], 10, use_cache=use_cache)[0]
    assert st["mean_accepted_len"] == 1.0 and st["accepted"] == [0, 0] and st["proposed"][0] > 0


def test_cache_rolled_back_to_sequence_length():
    model = CountingModel(num_heads=4, right=False)
    seen = []
    orig = model.forward

    def spy(input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        if past_key_values is not None:
            seen.append(past_key_values.get_seq_length())
        return orig(input_ids, use_cache=use_cache, past_key_values=past_key_values)

    model.forward = spy
    prompt = [1, 2, 3]
    ids, st = generate(model, TOK, prompt, 6, FixedK(), use_cache=True)
    # before each verification pass the cache holds exactly the tokens emitted so far
    assert seen == [len(prompt) + i for i in range(len(seen))]


def test_max_new_tokens_and_eos():
    model = CountingModel(num_heads=4, right=True)
    for n in (1, 2, 5, 7):
        ids, _ = generate(model, TOK, [0], n, FixedK())
        assert ids == list(range(1, n + 1))
    tok = SimpleNamespace(eos_token_id=4)  # greedy emits 1, 2, 3, 4=eos and stops
    ids, _ = generate(model, tok, [0], 20, FixedK())
    assert ids == [1, 2, 3, 4] == greedy_generate(model, tok, [0], 20)[0]


# ----------------------------------------------------------------------------- policies

def test_policies():
    logits = [torch.zeros(5), torch.tensor([9.0, 0, 0, 0, 0]), torch.zeros(5), torch.tensor([9.0, 0, 0, 0, 0])]
    assert FixedK().num_draft_tokens(logits, None, {}) == 3
    assert ConfidenceCut(0.5).num_draft_tokens(logits, None, {}) == 1   # head 2 is uniform (p = 0.2)
    assert ConfidenceCut(0.1).num_draft_tokens(logits, None, {}) == 3
    assert {"fixed_k", "confidence_cut"} <= set(POLICIES)
    assert get_policy("confidence_cut", tau=0.3).tau == 0.3
    with pytest.raises(KeyError):
        get_policy("nope")


class GarbagePolicy:
    """Returns any number, including out of range: the engine must clamp and stay exact."""
    name = "garbage"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def num_draft_tokens(self, head_logits, context_ids, step_state):
        assert set(step_state) == {"boundary_logits", "tokenizer", "grouper", "step"}
        return self.rng.choice([-5, 0, 1, 2, 3, 99])


# ----------------------------------------------------------------------------- real tiny MTPModel

def _tiny_model(tiny_model_dir, tmp_path, head_type, structural=None):
    from mtp.model.build import build_model

    d = yaml.safe_load((ROOT / "configs" / "R2.yaml").read_text(encoding="utf-8"))
    d.update(model_name=str(tiny_model_dir), num_heads=4, head_type=head_type, dtype="fp32")
    d["losses"]["structural"].update(enabled=structural is not None, variant=structural)
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(d), encoding="utf-8")
    torch.manual_seed(0)
    model, tok = build_model(load_config(tmp_path / "c.yaml"), "cpu")
    with torch.no_grad():  # make the heads differ from head 0 (resblocks start as exact copies)
        for p in list(model.extra_heads.parameters()) + list(model.boundary_probes.parameters()):
            p.add_(torch.randn_like(p) * 0.5)
    return model.eval(), tok


class OracleDrafts(nn.Module):
    """Real model, but head d's draft at absolute position t is the true greedy token t+d+1,
    replaced by a random token with probability p_wrong: exercises partial acceptance + cache rollback."""

    def __init__(self, model, full, p_wrong, seed=0):
        super().__init__()
        self.model, self.full, self.p_wrong = model, full, p_wrong
        self.num_heads, self.base_model = model.num_heads, model.base_model
        self.rng = random.Random(seed)

    def forward(self, input_ids, attention_mask=None, use_cache=False, past_key_values=None):
        start = past_key_values.get_seq_length() if past_key_values is not None else 0
        out = self.model(input_ids, use_cache=use_cache, past_key_values=past_key_values)
        vocab = out.logits[0].shape[-1]
        for d in range(1, self.num_heads):
            lg = torch.full_like(out.logits[d], -10.0)
            for i in range(input_ids.shape[1]):
                tgt = start + i + d + 1
                ok = tgt < len(self.full) and self.rng.random() >= self.p_wrong
                lg[0, i, self.full[tgt] if ok else self.rng.randrange(vocab)] = 10.0
            out.logits[d] = lg
        return out


@pytest.mark.parametrize("head_type,structural", [("linear", None), ("resblock", None), ("resblock", "S2")])
@pytest.mark.parametrize("use_cache", [True, False])
def test_outputs_identical_to_greedy_on_20_prompts(tiny_model_dir, tmp_path, head_type, structural, use_cache):
    model, tok = _tiny_model(tiny_model_dir, tmp_path, head_type, structural)
    rng = random.Random(1)
    policies = [FixedK(), ConfidenceCut(0.0), ConfidenceCut(0.3), GarbagePolicy()]
    for _ in range(20):
        prompt = [rng.randrange(4, len(tok)) for _ in range(rng.randint(1, 8))]
        ref, _ = greedy_generate(model, tok, prompt, 16, use_cache=use_cache)
        for policy in policies:
            ids, st = generate(model, tok, prompt, 16, policy, use_cache=use_cache)
            assert ids == ref, (policy.name, prompt)
            assert st["tokens"] == len(ref) and st["forward_passes"] <= len(ref) + 1


@pytest.mark.parametrize("p_wrong,use_cache", [(0.0, True), (0.3, True), (0.3, False), (0.7, True)])
def test_partial_acceptance_with_real_cache(tiny_model_dir, tmp_path, p_wrong, use_cache):
    model, tok = _tiny_model(tiny_model_dir, tmp_path, "resblock")
    rng = random.Random(2)
    total_steps = total_tokens = 0
    for i in range(10):
        prompt = [rng.randrange(4, len(tok)) for _ in range(rng.randint(2, 6))]
        ref, _ = greedy_generate(model, tok, prompt, 24, use_cache=True)
        oracle = OracleDrafts(model, prompt + ref, p_wrong, seed=i)
        ids, st = generate(oracle, tok, prompt, 24, FixedK(), use_cache=use_cache)
        assert ids == ref
        total_steps += st["steps"]
        total_tokens += st["tokens"]
    mean_len = total_tokens / total_steps
    if p_wrong == 0.0:
        assert mean_len > 3.0     # all drafts right: close to k = 4 tokens per step
    else:
        assert 1.0 < mean_len < 4.0


# ----------------------------------------------------------------------------- Group Integrity

def test_group_integrity_counts(tiny_tok):
    from mtp.data.grouping.base import get_grouper

    g = get_grouper("hi_rules_v0")
    ids = tiny_tok("मैं कल बाजार जा रहा था।")["input_ids"]       # groups: मैं | कल | बाजार | जा रहा था।
    # prompt "मैं"; spans: [कल बाजार] -> next token जा starts a group: hit; [जा रहा था।] ends the text: not judged
    assert group_integrity_counts(ids[:1], ids[1:], [2, 3], tiny_tok, g) == (1, 1)
    # prompt "मैं कल"; spans: [बाजार जा] -> next रहा is in जा's group: miss; [रहा था।] ends the text
    assert group_integrity_counts(ids[:2], ids[2:], [2, 2], tiny_tok, g) == (0, 1)
    assert group_integrity_counts(ids[:2], ids[2:], [1, 1, 1, 1], tiny_tok, g) == (0, 0)  # no multi-token spans
    assert group_integrity_counts(ids[:2], ids[2:], [2, 2], tiny_tok, None) == (0, 0)


def test_token_group_ids_match_training_labels(tiny_tok):
    from mtp.data.grouping.align import label_tokens
    from mtp.data.grouping.base import get_grouper

    g = get_grouper("hi_rules_v0")
    for s in ["मैं कल बाजार जा रहा था।", "बच्चे पार्क में खेल रहे हैं।", "वह घर से आया था"]:
        lab = label_tokens(s, tiny_tok, g)
        assert token_group_ids(lab["input_ids"], tiny_tok, g) == lab["group_id"]


# ----------------------------------------------------------------------------- aggregation

def test_evaluate_spec_decode_section10(tiny_model_dir, tmp_path, tiny_tok):
    from mtp.data.grouping.base import get_grouper

    model, tok = _tiny_model(tiny_model_dir, tmp_path, "resblock")
    texts = ["मैं कल बाजार जा रहा था। वह घर से आया था भारत एक विशाल और विविधतापूर्ण देश है।"] * 3 + ["short one"]
    prompts = make_prompts(texts, tok, n=200, min_words=8, max_words=10)
    assert len(prompts) == 3
    res = evaluate_spec_decode(model, tok, prompts, [FixedK(), ConfidenceCut(0.3)], max_new_tokens=12,
                               grouper=get_grouper("hi_rules_v0"))
    assert [r["policy"] for r in res] == ["fixed_k", "confidence_cut"]
    for r in res:
        assert set(r) == {"policy", "mean_accepted_len", "accept_rate_per_head", "tokens_per_sec",
                          "greedy_tokens_per_sec", "speedup", "outputs_match_greedy", "group_integrity", "n_prompts", "match_rate", "max_mismatch_margin", "fp32_check"}
        assert r["outputs_match_greedy"] is True and r["n_prompts"] == 3
        assert len(r["accept_rate_per_head"]) == 3 and r["mean_accepted_len"] >= 1.0


def test_make_prompts_deterministic(tiny_tok):
    texts = [" ".join(f"w{i}" for i in range(n)) for n in (5, 12, 20, 30)]
    a = make_prompts(texts, tiny_tok, n=10, seed=0)
    assert a == make_prompts(texts, tiny_tok, n=10, seed=0)
    assert all(8 <= len(p) <= 16 for p in a) and len(a) >= 2   # the 5-word sentence is skipped


def test_mismatch_accounting_and_fp32_check(tiny_model_dir, tmp_path, monkeypatch):
    """A divergence is counted per prompt, with greedy's top-2 margin at the divergence; with amp the
    first fp32_check_n prompts are re-run without it."""
    import contextlib
    from mtp.eval import spec_decode as sd

    model, tok = _tiny_model(tiny_model_dir, tmp_path, "resblock")
    prompts = [[5, 6, 7], [8, 9], [10, 11, 12, 13]]
    real_generate = sd.generate
    calls = {"n": 0}

    def flaky_generate(model_, tok_, prompt, *a, amp=None, **kw):
        ids, st = real_generate(model_, tok_, prompt, *a, amp=amp, **kw)
        if prompt == prompts[1] and amp is not None:        # corrupt token 2 of prompt 1, only "under amp"
            calls["n"] += 1
            ids = ids[:2] + [(ids[2] + 1) % len(tok_)] + ids[3:]
        return ids, st

    monkeypatch.setattr(sd, "generate", flaky_generate)
    amp = contextlib.nullcontext
    res = sd.evaluate_spec_decode(model, tok, prompts, [FixedK()], max_new_tokens=6, amp=amp, fp32_check_n=2)[0]
    _, gst = sd.greedy_generate(model, tok, prompts[1], 6)
    assert res["outputs_match_greedy"] is False and res["match_rate"] == pytest.approx(2 / 3)
    assert res["max_mismatch_margin"] == pytest.approx(gst["margins"][2], abs=1e-4)
    assert res["fp32_check"] == {"n_prompts": 2, "match_rate": 1.0, "outputs_match_greedy": True,
                                 "max_mismatch_margin": None}
    assert calls["n"] == 1

    clean = sd.evaluate_spec_decode(model, tok, prompts, [FixedK()], max_new_tokens=6)[0]   # no amp
    assert clean["outputs_match_greedy"] is True and clean["match_rate"] == 1.0
    assert clean["max_mismatch_margin"] is None and clean["fp32_check"] is None


def test_first_divergence():
    from mtp.eval.spec_decode import first_divergence

    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 5, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2


def test_greedy_margins(tiny_model_dir, tmp_path):
    model, tok = _tiny_model(tiny_model_dir, tmp_path, "resblock")
    ids, st = greedy_generate(model, tok, [5, 6], 5)
    assert len(st["margins"]) == len(ids) and all(m >= 0 for m in st["margins"])


def test_fp32_check_records_divergence_margin(tiny_model_dir, tmp_path, monkeypatch):
    """A divergence in the fp32 re-check is counted, with greedy's (fp32) top-2 margin at that token."""
    import contextlib
    from mtp.eval import spec_decode as sd

    model, tok = _tiny_model(tiny_model_dir, tmp_path, "resblock")
    prompts = [[5, 6, 7], [8, 9]]
    real_generate = sd.generate

    def fp32_flaky(model_, tok_, prompt, *a, amp=None, **kw):
        ids, st = real_generate(model_, tok_, prompt, *a, amp=amp, **kw)
        if prompt == prompts[0] and amp is None:              # corrupt token 1, only in the fp32 re-check
            ids = ids[:1] + [(ids[1] + 1) % len(tok_)] + ids[2:]
        return ids, st

    monkeypatch.setattr(sd, "generate", fp32_flaky)
    res = sd.evaluate_spec_decode(model, tok, prompts, [FixedK()], max_new_tokens=5,
                                  amp=contextlib.nullcontext, fp32_check_n=2)[0]
    _, gst = sd.greedy_generate(model, tok, prompts[0], 5)
    assert res["outputs_match_greedy"] is True                  # timed run untouched
    fc = res["fp32_check"]
    assert (fc["n_prompts"], fc["match_rate"], fc["outputs_match_greedy"]) == (2, 0.5, False)
    assert fc["max_mismatch_margin"] == pytest.approx(gst["margins"][1], abs=1e-4)

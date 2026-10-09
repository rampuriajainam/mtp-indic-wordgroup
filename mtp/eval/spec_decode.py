"""
Self-speculative decoding with the MTP heads (OM-6, INTERFACES §11).

Every step, at the last position of the sequence so far:
  1. head 0's argmax is the next greedy token t0 (always kept);
  2. heads 1..n propose drafts d1..dn (greedy argmax), n chosen by the draft policy;
  3. ONE forward pass over [t0, d1..dn] (with the KV cache: only those tokens);
  4. head 0 verifies left to right: d_i is accepted while head 0's argmax after the previous
     token equals it. Accepted tokens are emitted, the cache is rolled back past the rejected
     ones, and the logits at the last accepted token seed the next step.
Every emitted token is head 0's greedy choice, so the output is identical to greedy decoding
with head 0 (tested exactly); only the number of forward passes changes.

Greedy reference = the base LM alone (model.base_model) with a KV cache, so its timing does not
pay for the extra heads.

Group Integrity: of the accepted spans with >= 2 tokens, the share that end on a word-group
boundary (the next token starts a new group). Spans that end the generation are not counted.
"""

import contextlib
import random
import time

import torch

from mtp.data.grouping.align import _group_start_chars, _labels


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _crop(cache, n_remove):
    """Drop the last n_remove positions. Negative = "remove n" in every transformers version
    (in 5.18 a positive value is the deprecated "final length")."""
    if n_remove > 0:
        cache.crop(-n_remove)


def _step_state(out, pos, tokenizer, grouper, step):
    bl = out.aux.get("boundary_logits")
    return {"boundary_logits": [float(b[0, pos]) for b in bl] if bl is not None else None,
            "tokenizer": tokenizer, "grouper": grouper, "step": step}


@torch.no_grad()
def generate(model, tokenizer, prompt_ids, max_new_tokens, policy, grouper=None, *, use_cache=True, amp=None):
    """Returns (generated ids, stats). Output == greedy decoding with head 0.

    stats: tokens, forward_passes (incl. the prompt pass), steps, mean_accepted_len (tokens/step),
    proposed / accepted per extra head, seconds, tokens_per_sec, gi_hits / gi_spans (Group
    Integrity counts; 0/0 without a grouper), spans (tokens emitted per step)."""
    device = next(model.parameters()).device
    amp = amp or contextlib.nullcontext
    k = model.num_heads
    eos = tokenizer.eos_token_id
    seq = [int(t) for t in prompt_ids]
    new, spans = [], []
    proposed, accepted = [0] * (k - 1), [0] * (k - 1)
    passes = steps = 0

    _sync(device)
    t_start = time.perf_counter()
    with amp():
        out = model(torch.tensor([seq], device=device), use_cache=use_cache)
        passes += 1
        cache = out.aux.get("past_key_values")
        pos = -1                                   # index (in out's positions) of the last emitted token
        while len(new) < max_new_tokens:
            last = [lg[0, pos] for lg in out.logits]
            t0 = int(last[0].argmax())
            n = policy.num_draft_tokens(last, torch.tensor(seq), _step_state(out, pos, tokenizer, grouper, steps))
            n = max(0, min(int(n), k - 1, max_new_tokens - len(new) - 1))
            if t0 == eos:
                n = 0
            drafts = [int(last[d].argmax()) for d in range(1, n + 1)]
            block = [t0] + drafts

            if n == 0 and (t0 == eos or len(new) + 1 >= max_new_tokens):
                new.append(t0)                     # last token: no verification pass needed
                spans.append(1)
                steps += 1
                break

            inp = block if use_cache else seq + block
            out = model(torch.tensor([inp], device=device), past_key_values=cache, use_cache=use_cache)
            passes += 1
            head0 = out.logits[0][0, -len(block):]  # row i predicts the token after block[i]
            j = 0
            while j < n and int(head0[j].argmax()) == drafts[j]:
                j += 1
            for d in range(n):
                proposed[d] += 1
                accepted[d] += d < j
            emitted = block[: 1 + j]
            if eos in emitted:
                emitted = emitted[: emitted.index(eos) + 1]
            seq += emitted
            new += emitted
            spans.append(len(emitted))
            steps += 1
            pos = -len(block) + j
            if use_cache:
                cache = out.aux["past_key_values"]
                _crop(cache, n - j)
            if new[-1] == eos:
                break
    _sync(device)
    seconds = time.perf_counter() - t_start

    gi_hits, gi_spans = group_integrity_counts(prompt_ids, new, spans, tokenizer, grouper)
    return new, {
        "tokens": len(new), "forward_passes": passes, "steps": steps,
        "mean_accepted_len": len(new) / max(steps, 1),
        "proposed": proposed, "accepted": accepted,
        "seconds": seconds, "tokens_per_sec": len(new) / seconds if seconds > 0 else 0.0,
        "gi_hits": gi_hits, "gi_spans": gi_spans, "spans": spans,
    }


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt_ids, max_new_tokens, *, use_cache=True, amp=None):
    """Plain greedy decoding with head 0 = the base LM (model.base_model). Returns (ids, stats)."""
    device = next(model.parameters()).device
    amp = amp or contextlib.nullcontext
    base = model.base_model
    eos = tokenizer.eos_token_id
    seq = [int(t) for t in prompt_ids]
    new, passes = [], 0
    _sync(device)
    t_start = time.perf_counter()
    with amp():
        out = base(input_ids=torch.tensor([seq], device=device), use_cache=use_cache)
        passes += 1
        while True:
            t = int(out.logits[0, -1].argmax())
            new.append(t)
            seq.append(t)
            if t == eos or len(new) >= max_new_tokens:
                break
            inp = [t] if use_cache else seq
            out = base(input_ids=torch.tensor([inp], device=device),
                       past_key_values=out.past_key_values if use_cache else None, use_cache=use_cache)
            passes += 1
    _sync(device)
    seconds = time.perf_counter() - t_start
    return new, {"tokens": len(new), "forward_passes": passes, "seconds": seconds,
                 "tokens_per_sec": len(new) / seconds if seconds > 0 else 0.0}


def token_group_ids(ids, tokenizer, grouper):
    """Word-group id per token of `ids`, grouping the decoded text exactly as align.py does.

    Token char spans come from re-tokenizing the decoded text when that gives back the same ids
    (the training path: offset_mapping); otherwise from incremental decoding (a '▁word' span then
    also includes its leading space)."""
    ids = [int(t) for t in ids]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    if not text.split():
        return [-1] * len(ids)
    enc = tokenizer(text, return_offsets_mapping=True)
    if list(enc["input_ids"]) == ids:
        offsets = [tuple(o) for o in enc["offset_mapping"]]
    else:
        ends = [len(tokenizer.decode(ids[: i + 1], skip_special_tokens=True)) for i in range(len(ids))]
        offsets = [(ends[i - 1] if i else 0, ends[i]) for i in range(len(ids))]
    return _labels(offsets, _group_start_chars(text, grouper))[1]


def group_integrity_counts(prompt_ids, new, spans, tokenizer, grouper):
    """(spans of >= 2 accepted tokens that end on a group boundary, such spans counted)."""
    if grouper is None:
        return 0, 0
    full = [int(t) for t in prompt_ids] + list(new)
    gid = token_group_ids(full, tokenizer, grouper)
    hits = total = 0
    i = len(prompt_ids)
    for s in spans:
        end = i + s                        # index of the token after the span
        if s >= 2 and end < len(full):     # spans that end the generation are not judged
            last, nxt = gid[end - 1], gid[end]
            if last >= 0 and nxt >= 0:
                total += 1
                hits += nxt != last
        i = end
    return hits, total


def make_prompts(texts, tokenizer, n=200, min_words=8, max_words=16, seed=0):
    """Prompts = the first 8-16 words (seeded per sentence) of the first n sentences that are long
    enough to leave a continuation. Returns token id lists."""
    rng = random.Random(seed)
    prompts = []
    for text in texts:
        words = text.split()
        w = rng.randint(min_words, max_words)
        if len(words) <= w:
            continue
        prompts.append(list(tokenizer(" ".join(words[:w]))["input_ids"]))
        if len(prompts) >= n:
            break
    return prompts


def evaluate_spec_decode(model, tokenizer, prompts, policies, max_new_tokens=64, grouper=None,
                         use_cache=True, amp=None, warmup=1):
    """One INTERFACES §10 spec_decode entry per policy, over the same prompts. Greedy runs once."""
    model.eval()
    for p in prompts[:warmup]:                      # CUDA kernels / allocator warm-up, not timed
        greedy_generate(model, tokenizer, p, 8, use_cache=use_cache, amp=amp)
    greedy_ids, greedy_tokens, greedy_secs = [], 0, 0.0
    for p in prompts:
        ids, st = greedy_generate(model, tokenizer, p, max_new_tokens, use_cache=use_cache, amp=amp)
        greedy_ids.append(ids)
        greedy_tokens += st["tokens"]
        greedy_secs += st["seconds"]
    greedy_tps = greedy_tokens / greedy_secs if greedy_secs > 0 else 0.0

    results = []
    k = model.num_heads
    for policy in policies:
        tokens = steps = hits = spans = 0
        secs = 0.0
        prop, acc = [0] * (k - 1), [0] * (k - 1)
        match = True
        for p, ref in zip(prompts, greedy_ids):
            ids, st = generate(model, tokenizer, p, max_new_tokens, policy, grouper, use_cache=use_cache, amp=amp)
            match &= ids == ref
            tokens += st["tokens"]
            steps += st["steps"]
            secs += st["seconds"]
            hits += st["gi_hits"]
            spans += st["gi_spans"]
            prop = [a + b for a, b in zip(prop, st["proposed"])]
            acc = [a + b for a, b in zip(acc, st["accepted"])]
        tps = tokens / secs if secs > 0 else 0.0
        results.append({
            "policy": policy.name,
            "mean_accepted_len": tokens / max(steps, 1),
            "accept_rate_per_head": [a / q if q else None for a, q in zip(acc, prop)],
            "tokens_per_sec": tps, "greedy_tokens_per_sec": greedy_tps,
            "speedup": tps / greedy_tps if greedy_tps > 0 else None,
            "outputs_match_greedy": bool(match),
            "group_integrity": hits / spans if spans else None,
            "n_prompts": len(prompts),
        })
    return results

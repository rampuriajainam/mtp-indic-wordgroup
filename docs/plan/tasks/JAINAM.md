# Jainam: Modeling & Training

> **How to use this file with Claude:** paste the "Context" block, then `INTERFACES.md`, then the task ID. For JN-2 / JN-5 / JN-6 / JN-7, have Claude argue the design with you before writing code. Branch: `jainam/<task>`.

## Context (paste this to Claude)

We are building *Word-Group Guided Multi-Token Prediction with Adaptive Loss Weighting for Hindi and Marathi*. Base: `LingoIITGN/ganga-1b` (1B, SentencePiece, vocab 30k), LoRA r=8 on `q_proj`/`v_proj`, lr 2e-4, grad clip 1.0, batch 8, max_len 128, IndicCorpV2 `hin_Deva`.

Results so far (laptop, 8 GB GPU):
- NTP baseline: held-out loss 3.05, flat. A sweep showed lr 2e-3 with a larger LoRA diverges, and lr 5e-4 with r=32 slowly degrades. The base model is already well fit to this distribution.
- Token-level MTP, k=2, `MedusaWrapper` with a **randomly initialised** `nn.Linear(hidden, vocab)` extra head on the last hidden state, losses summed: 12,500 batches, head 0 3.06 → 3.09, head 1 8.10 → 5.84.
- Word groups: rule-based grouper (aux + postposition lists), aligned to tokens via `offset_mapping` → per-token `group_start` and `group_id`.

Code lives in the `mtp/` package (INTERFACES.md). Jai provides groupers + a pre-labelled boundary cache. Om provides config, device/dtype handling, logging, checkpointing, Kaggle notebooks and the evaluation suite including the speculative-decoding engine. I own the model, every loss, the weighting scheme, `scripts/train.py`, all training runs, and the group-aware draft policy.

---

> **Waiting points:** JN-1 → JN-3 need nobody (code against `INTERFACES.md`, stub what isn't merged). JN-4 needs H2 (Om: checkpoint + Kaggle notebook). JN-5/JN-6 real training needs H3 (Jai: Hindi cache), and the variant choice needs H4 (Om: in-group eval). R5–R7 need H5. Marathi needs H7. JN-9 needs H6.

## JN-1 · Heads module
**Files:** `mtp/model/heads.py`, `mtp/losses/mtp_ce.py`
- Port `MedusaWrapper` → `MTPModel` (INTERFACES §5): `num_heads` counts head 0. Return `MTPOutput(logits, hidden, aux)`.
- `head_type="linear"`: current behaviour (for reproducing R1).
- `head_type="resblock"` (Medusa-1): `h_d = h + SiLU(W_d h)`, logits `= lm_head(h_d)`. **Init `W_d` to zero** so every head starts as a copy of the LM head, and reuse the base model's (frozen) `lm_head`. Two reasons: (1) it removes the 30k×hidden randomly-initialised matrix that made head 1 start at loss 8.1; (2) each extra head costs only hidden² params, which matters on a 16 GB Kaggle GPU with k=4. Option `n_layers` per head (default 1).
- `mtp_ce.py`: `per_head_ce(logits_list, input_ids, attention_mask, reduction="none")` returns per-position losses `[B, T-d]` for each head, not just the mean, so the structural loss can re-weight them. Port the masking from `compute_head_loss`.
- Sanity test: with `resblock` and zero init, head d's logits equal head 0's at every position.

## JN-2 · Structural-loss design note
**File:** `docs/design_structural_loss.md` (1–2 pages, decide before coding JN-5)

The proposal says the auxiliary loss should penalise heads that *cross word-group boundaries inconsistently*. Turn that into math. Notation: head d (d = 0…k-1) at position t predicts the token at u = t+d+1. Candidate formulations to evaluate:

- **S1 · Boundary-aware re-weighting.** For head d at position t, let `same = [group_id[t] == group_id[u]]`. Weight the per-position CE by `1 + α·same`, or the opposite (down-weight cross-group targets as near-unpredictable noise). Cheap, but it only re-weights existing CE, so it can't add new signal.
- **S2 · Boundary-prediction auxiliary head.** Each extra head d gets a tiny linear probe on `h_d` predicting `group_start[u]` (binary CE). This teaches each head *where the next unit starts*. Bonus: at inference this probe is exactly what the `GroupAware` draft policy needs (JN-9).
- **S3 · In-group self-consistency.** Head d at t and head d-1 at t+1 both predict the **same** token u. When t+1 and u are in the same group, the far head should agree with the nearer head that has seen one more token: `KL( sg[p_{d-1}(·|t+1)] ‖ p_d(·|t) )`, masked to in-group pairs (sg = stop-gradient). This is the most literal reading of "inconsistently crossing boundaries" and the most novel, but also the most expensive (extra logits alignment, so use top-k-sparse KL to save memory).

Recommended path: pilot S2 and S3 (and S2+S3) on 2,000-step runs, pick the best by head-1..3 in-group top-1 on `indiccorp_eval`, write down the choice and the pilot numbers in the note. Those pilot numbers also go into the paper's ablation.

## JN-3 · Unified training script
**File:** `scripts/train.py`
- `python scripts/train.py --config configs/R5.yaml [--resume auto] [--set optim.lr=...]`
- Builds: tokenizer → base model (`mtp.device.pick_dtype`) → LoRA → `MTPModel` → losses from config → weighting module → AdamW over LoRA + heads + probes + weighting params (the weighting params get their own param group, no weight decay).
- Data: boundary cache via `datasets.load_from_disk` + Om's `Collator`. NTP/MTP runs without group losses ignore the group columns.
- Loop: `autocast_ctx` + `GradScaler` from `mtp.device`, optional grad accumulation, clip 1.0, `MetricLogger` every 20 steps (each loss term, each weight), eval on `eval_small` every `eval_every`, `save_run` every `save_every`. On `--resume auto`: `latest_step` → restore model, optimizer, scaler, RNG, data position.
- Deterministic: seed everything, and record git commit hash in `config.yaml`.
- `configs/`: write `R0.yaml` … `R10.yaml` and `pilot_S2.yaml`, `pilot_S3.yaml`, `pilot_S23.yaml`.

## JN-4 · Reproduce baselines on Kaggle
- R0 (NTP) and R1 (MTP k=2 linear) through `train.py` on Kaggle. **Gate:** within ~0.05 of the laptop numbers (3.05 / 3.09 / 5.84). If not, the difference is in dtype handling (fp16 vs bf16), data order or masking, so find it before moving on.
- R2 (k=4 resblock). This is our real token-level baseline. Note: zero-init makes head d start as a *copy* of head 0, so it still predicts t+1 for a t+d+1 target. Measured at init on `eval_small` (JN-1): h0 3.08, h1 9.37, h2 10.59, h3 10.95 (random linear heads: ~10.5 each). Expect head 1 to fall faster than in R1 (12.6M vs 184M new params, shared `lm_head`), not to start near head 0.
- Publish each finished run as a Kaggle Dataset `mtp-run-<ID>` and post the name for Om.

## JN-5 · Structural loss
**File:** `mtp/losses/structural.py`
- Implement the variant(s) chosen in JN-2 behind one interface: `StructuralLoss(variant, **kw)(mtp_output, batch) -> dict[str, Tensor]` returning named scalar terms (e.g. `{"struct/boundary_bce/h1": ..., "struct/consistency/h2": ...}`), so the weighting module can treat each term separately.
- Handle `group_id == -1` (special/pad) and positions where t+d runs off the sequence.
- Unit test: hand-built 1-sentence batch where you can compute the expected loss by hand.
- Pilot runs → pick → R3.

## JN-6 · Contrastive loss
**File:** `mtp/losses/contrastive.py`
- Supervised contrastive (SupCon) over tokens: project `hidden` with a 2-layer MLP to 128-d, L2-normalise. Positives = other tokens with the same `(sentence, group_id)`, negatives = tokens from other groups in the batch. Temperature from config (start 0.1). Subsample ≤ 512 tokens per batch to bound memory.
- Variant to try if SupCon is flat: group-level InfoNCE, with mean-pooled group embedding vs the head-d hidden state `h_d` at the token just before the group.
- Watch for it hurting head 0 perplexity, since pulling token representations together can blur next-token information. Log head-0 loss vs R2 closely. If it hurts, apply the loss on the projection only (already the case) and lower the weight.
- When stable, this becomes run R4.

## JN-7 · Adaptive loss weighting
**File:** `mtp/losses/weighting.py`
- `LossWeighting(scheme, term_names)` with `forward(dict_of_losses) -> (total, dict_of_effective_weights)`.
- `fixed`: per-head weights `0.8^d` (Medusa) for CE terms, and config weights for aux terms.
- `uncertainty` (Kendall et al. 2018): learnable `s_i = log σ_i²`, total `= Σ ½·exp(-s_i)·L_i + ½·s_i`. Clamp `s_i ∈ [-4, 4]`. Head 0's weight can be fixed at 1 so the LM doesn't get de-prioritised; make that a flag and test both.
- `dwa` (Dynamic Weight Average): `w_i(t) ∝ exp(r_i(t-1)/T)`, where `r_i = L_i(t-1)/L_i(t-2)` over epoch-like windows of N steps.
- Log effective weights every 20 steps (they become Om's figure (c)).
- Pilot all three on top of R4's losses (2,000 steps), pick one → R5. Keep the others for the ablation table.

## JN-8 · Run matrix
- Execute R3 → R10 from `README.md §6` in that order (R5 before R6/R7 because R6/R7 copy R5's config). For R6, ask Jai for the `random` cache built from the default grouper's length histogram.
- Marathi (R8–R10): base `smallstepai/Misal-1B-instruct-v0.1`. Check its tokenizer, `lm_head` name and LoRA target module names before launching. Fallback: ganga-1b.
- Keep `docs/runs.md`: run ID, commit, config, Kaggle dataset name, final numbers, notes.

## JN-9 · Group-aware draft policy
**File:** `GroupAware` in `mtp/eval/draft_policy.py` (engine and contract from Om, INTERFACES §11)
- Use the S2 boundary probes: propose drafts from heads d = 1, 2, … while (a) head d's max prob ≥ τ and (b) the probe says its target token is **not** a new group start, plus include the token that closes the group. Intuition: draft the rest of the current word group in one go, then re-verify.
- Fallback when a run has no probe: use Jai's rule grouper on the decoded text so far to tell whether we're mid-group.
- Tune τ on 50 held-out prompts, report on FLORES. Compare against `FixedK` and `ConfidenceCut` in Om's Table 4.

## Write-up (Phase E)
- **Method:** model + heads, structural loss (final variant + the pilot comparison), contrastive loss, weighting, group-aware decoding.
- **Results discussion:** main table, R5 vs R6 (does linguistics matter), in-group vs boundary split, decoding speedups, Marathi vs Hindi, limitations (flat NTP baseline, single seed, rule-based groups).
- **Conclusion + future work.**

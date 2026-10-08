# Structural loss: design note (JN-2)

**Status:** design decided (2026-10-09); the pilot results at the end are to be filled in after Phase C.
**Owner:** Jainam. **Feeds:** JN-5 (`mtp/losses/structural.py`), JN-9 (`GroupAware`), configs `pilot_*.yaml` (S2, S3, S3chain, S23, S3all).

## 0. Notation

- Head d ∈ {0..k-1} at position t predicts token u = t+d+1. h_d(t) is the state its output layer reads (`MTPOutput.aux["head_hidden"][d]`); p_d(·|t) is its distribution.
- g(t) = `group_id[t]` (−1 for special and pad tokens). s(u) = `group_start[u]`.
- A pair (t, u) is *valid* if both tokens are real and g(t), g(u) ≥ 0.
- `same_L` = the fraction of valid pairs (t, t+L) where g(t) = g(t+L).

## 1. Evidence

From `scripts/jn2_structural_stats.py`, using the ganga-1b tokenizer.

**A. How much same-group signal exists (2,500 IndicCorp train sentences, no truncation).**

| grouper | words/group | tokens/group | % tokens = group start | same_1 | same_2 | same_3 | same_4 |
|---|---|---|---|---|---|---|---|
| words (each word its own group) | 1.00 | 1.12 | 89.1 | 11.0% | 3.9% | 1.4% | 0.5% |
| v0 short lists | 1.33 | 1.49 | 67.2 | 33.4% | 7.8% | 2.5% | 0.8% |
| v0 expanded lists | 1.42 | 1.59 | 62.8 | 37.8% | 10.5% | 3.1% | 1.0% |

- ganga's tokenizer averages **1.12 tokens per word**, so a word group spans only about **1.5 tokens**.
- The `words` row is what subword continuation alone produces. Only the part above that line comes from linguistic grouping.
- v0 short reproduces the earlier coverage figures (24.6% of words attached, 1.33 words/group).

**B. The laptop R1 checkpoint (k=2 linear, 12.5k batches) on the fixed eval set (rows 0–999, which is 500 non-blank sentences), v0 short / v0 expanded.**

| | in-group top-1 | at-boundary top-1 | in-group CE | at-boundary CE | in-group share |
|---|---|---|---|---|---|
| head 0 | 57.5 / 57.9% | 34.2 / 32.3% | 1.60 / 1.61 | 3.51 / 3.64 | 33 / 37% |
| head 1 | 28.0 / 29.6% | 14.4 / 13.8% | 3.97 / 3.79 | 5.15 / 5.20 | 7 / 10% |

S3 pair (student = head 1 at t, teacher = head 0 at t+1, both predicting u = t+2), v0 short:

| mask | share of pairs | teacher top-1 | student top-1 | top-1 agreement | KL(teacher ‖ student) |
|---|---|---|---|---|---|
| in-group: g(t+1) = g(u) | 33% | 58.4% | 22.7% | 28.2% | 2.85 |
| at boundary | 67% | 34.4% | 11.9% | 17.9% | 2.67 |

**What follows from this:**
1. In-group targets are about twice as easy as boundary targets for both heads. The structure is real and the heads already exploit it.
2. **For heads 2 and 3 the signal is thin.** Their target lies in the source token's group only 2.5–3% and 0.8–1% of the time, and roughly half of that is subword continuation. A loss keyed on "target in the same group as the source" barely touches the far heads.
3. The teacher in S3 is 2.5× more accurate than the student on in-group pairs (58% vs 23%). There is real headroom to distil.
4. The eval split for heads 2 and 3 will have a tiny in-group n: about 1–3% of roughly 26k positions, so about 250–800 tokens. In-group numbers for those heads need counts or confidence intervals.

## 2. Candidates

**S1 · Boundary-aware re-weighting.** L = Σ_d Σ_t (1 + α·[g(t) = g(u)]) · CE_d(t).
Rejected. It changes the weight of ≤10% of head-1 positions and ≤3% of head-2/3 positions, about half of which come from subword continuation. It adds no new signal. Not piloted.

**S2 · Boundary probe per head.** For each head d: b_d(t) = σ(w_dᵀ h_d(t) + c_d), with
L_S2,d = BCE(b_d(t), s(u)) averaged over valid (t, u).
- The label is dense: every valid position, at every head. The positive rate is about 63–67%, so no reweighting is needed.
- It is the only candidate that gives heads 2 and 3 a signal at every position.
- Cost: k × (H+1) parameters, negligible.
- At inference, b_d is exactly what `GroupAware` needs (JN-9).
- Risk: the probe may be solved by the residual-block weights W_d without changing the token logits, which would leave token accuracy flat. The pilot measures this.
- Proposal: probe all k heads, including head 0. Head 0's probe says whether the next token starts a new group, which JN-9 also uses. Its gradient reaches the backbone only through LoRA, at weight λ, so it is safe at λ = 0.1.

**S3 · In-group self-consistency (distillation).** The far head should match a better-informed head on the same target u, but only where the extra context that head saw lies inside the target's group:
L_S3,d = mean over valid (t, u) with g(t+1) = g(u) of KL( sg[p_T(·)] ‖ p_d(·|t) ).
Two choices of teacher. **Both are piloted** (decision 1 in section 4):
- **S3-chain** (as in the task file): teacher = head d−1 at t+1. For d ≥ 2 the teacher is itself an extra head and weak: head 1 is 23% top-1 in-group, the others are worse.
- **S3-h0:** teacher = head 0 at u−1, the pretrained LM. It is the strongest teacher available and is good from step 0. Its in-group mask is g(t+1) = … = g(u), which is equivalent to g(t+1) = g(u) because group IDs never decrease. Mask rates are about 33–38% for head 1, 8–10% for head 2 and 2.5–3% for head 3, so S3 mostly trains head 1 and some of head 2.
- **Cost:** full-vocab KL (V = 30k) on the log-softmaxes the CE already computes. That is one extra [B, T, V] fp32 temporary per head, about 125 MB at B = 8, T = 128. Top-k sparse KL is not needed at this size.
- **Control (S3-all):** the same KL with no mask. That is plain self-distillation and carries no linguistic content. If S3 does not beat S3-all, the mask, and therefore the grouping, contributes nothing. This is the same logic as R6, applied one level down.

## 3. Pilot plan (Phase C, needs H3; final pick needs H4)

- **Setup:** R2 setup (k = 4, zero-init resblock heads, fixed weighting 0.8^d on CE), 2,000 steps, seed 42, the `hi_rules_v1` cache.
- **Weights:** λ_S2 = 0.1 (BCE ≈ 0.6) and λ_S3 = 0.5 (KL ≈ 2.8), both fixed for the pilots. Adaptive weighting comes later, in JN-7.

| config | structural terms |
|---|---|
| R2 @ step 2000 | none (reference, taken from the R2 run, same seed and data) |
| `pilot_S2` | S2 on heads 0–3 |
| `pilot_S3` | S3-h0, in-group mask |
| `pilot_S3chain` | S3-chain, in-group mask |
| `pilot_S23` | S2 + the better of S3-h0 / S3-chain |
| `pilot_S3all` | S3-h0, no mask (control) |

**Metric logging names:** `struct/boundary_bce/h{d}` and `struct/consistency/h{d}`.

**Selection rule:**
- **Primary metrics**, on `indiccorp_eval` plus `flores_hi`:
  - head-1 in-group top-1;
  - mean overall top-1 over heads 1–3.
- **Guard:** head-0 CE no worse than R2@2000 + 0.02.
- **Heads 2 and 3 in-group top-1:** reported with n, not used for the decision.
- **Probe quality:** report b_d AUROC per head, since JN-9 depends on it.
- **Teacher choice:** S3-h0 vs S3-chain is decided by the same rule; `pilot_S23` and `pilot_S3all` run with the winning teacher.
- **The winner becomes R3.** If S3 does not beat S3-all, report that, and keep S2 for its use in JN-9.

## 4. Decisions (2026-10-09)

1. **S3 teacher:** pilot both S3-h0 (head 0 at u−1) and S3-chain (head d−1 at t+1). S3-h0 keeps the proposal's intent of penalising inconsistency inside a group, with a stronger teacher; the pilot decides.
2. **k = 4.** Kept for comparability with the Medusa literature. With 1.5-token groups, head 3 almost never sees in-group structure; report it as the "structure-free" head.
3. **Pilot weights:** λ_S2 = 0.1, λ_S3 = 0.5, fixed. JN-7 revisits them.
4. **S1** is not piloted.

**Still open (other owners):**
- **JI-8 / Marathi (Jai):** recompute section 1A with the full `hi_rules_v1` lists and with the Misal tokenizer. Longer groups in tokens make the method's case stronger.
- **Blank rows in IndicCorp (Om, OM-1):** every other row is a blank separator. "First 1,000 rows" means 500 sentences, and the laptop's 100k-row runs saw about 50k sentences. `corpus.py` should state whether splits count raw rows or non-blank sentences. Gate B compares against laptop numbers produced with raw rows.

## 5. Pilot results

_To be filled in after the Phase C pilots._

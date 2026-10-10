# Run log

One row per run. Owner: Jainam (training); Om adds the evaluated numbers from `results/`.
Eval sets: `eval_small` = IndicCorp raw rows 0-99 (~50 sentences), `eval` = rows 0-999 (~500). "h{d}" = head d (predicts token t+d+1); loss = mean CE; % = top-1.

## Laptop (RTX 4060, bf16), before the package

| run | what | result |
|---|---|---|
| NTP baseline | LoRA r=8, 100k raw rows (~50k sentences), legacy script | eval_small h0 **3.05**, flat (lr/rank sweeps did not help) |
| MTP k=2 linear | legacy `MedusaWrapper`, 12,500 batches, heads summed | eval_small h0 **3.09**, h1 8.10 → **5.84** |

## `scripts/train.py` checks (laptop, bf16)

| run | steps | eval set | h0 | h1 | h2 | h3 | notes |
|---|---|---|---|---|---|---|---|
| R0 | 100 | eval_small | 3.054 | | | | matches laptop NTP (3.069 @100) |
| R1 | 100 | eval_small | 3.096 | 7.90 | | | matches laptop MTP (3.102 / 8.10 @100); same trainable-param count |
| R0 | 2000 | eval | 2.836 (42.5%) | | | | reference for the α pilot |
| R2 α=0.0 | 2000 | eval | 2.836 (42.5%) | 5.688 (12.5%) | 6.598 (6.4%) | 6.996 (4.1%) | heads detached: LM identical to R0 |
| R2 α=0.1 | 2000 | eval | 2.847 (42.3%) | 5.595 (12.7%) | 6.487 (6.7%) | 6.878 (4.4%) | **chosen** (`head_backbone_grad: 0.1`) |
| R2 α=1.0 | 2000 | eval | 3.034 (40.1%) | 5.422 (13.2%) | 6.263 (7.1%) | 6.634 (4.9%) | heads best, verifier 0.20 worse |

α = `head_backbone_grad`, the share of the extra heads' gradient that reaches LoRA. Zero-init resblock heads start as copies of head 0, so at init h1/h2/h3 = 9.37 / 10.59 / 10.95 (random linear heads: ~10.5).

## Kaggle (T4 x2, fp32 weights + fp16 autocast, ~0.34 s/step)

Step 12500, eval_small (50 sentences, `data.eval_n: 100` raw rows). Cell: loss (top-1). In-group / at-boundary top-1 from `train.py`'s reference eval with `hi_rules_v0`.

| ID | commit | config | Kaggle dataset | h0 | h1 | h2 | h3 | notes |
|---|---|---|---|---|---|---|---|---|
| R0 | fe87853 | R0.yaml | `jainamrampuria/mtp-run-R0` | 3.049 (39.8%) | | | | 2026-10-09; 0.29 s/step; h0 in-group / boundary 57.7 / 31.4% |
| R1 | fe87853 | R1.yaml | `jainamrampuria/mtp-run-R1` | 3.068 (39.3%) | 5.570 (13.9%) | | | 2026-10-09; 0.36 s/step; in-group / boundary h0 56.0 / 31.5%, h1 25.2 / 13.0% |
| R2 | fe87853 | R2.yaml (α=0.1) | `jainamrampuria/mtp-run-R2` | 3.052 (39.6%) | 5.533 (13.8%) | 6.402 (7.8%) | 6.801 (4.4%) | 2026-10-09; 0.38 s/step; in-group / boundary h0 56.5 / 31.7%, h1 24.5 / 13.0%, h2 19.3 / 7.5%, h3 13.3 / 4.4% |

Datasets hold `config.yaml`, `metrics.jsonl`, `train.log`, `step_12000/`, `step_12500/` (private; shared with Om).

**Gate B: passed.** R0 h0 3.049 vs laptop 3.05; R1 h0 3.068 vs 3.09 (both within 0.05). R1 h1 5.57 vs 5.84 is 0.27 *better*, which fits the known differences: the legacy script summed head losses, the laptop trained on raw rows 100+ with half of them blank, and bf16 vs fp16 autocast. Head 0 of the MTP runs stays within 0.02 of R0, so `head_backbone_grad: 0.1` holds the verifier at NTP quality over the full 12.5k steps.

On eval_small the h0 loss is flat from step 250 (3.052) to 12500 (3.049), so the gains are in heads 1-3. The full 500-sentence eval set and FLORES come from Om's `evaluate.py` (OM-4/OM-7).

### Method runs (Kaggle T4 x2, account `jainam2142`)

Pilots that led here: `docs/design_structural_loss.md` §5. Step 12500. Logged eval: R3 on eval_small (like R0-R2); Rsd on `eval` (~500 sentences).

| ID | commit | config | Kaggle dataset | h0 | h1 | h2 | h3 | notes |
|---|---|---|---|---|---|---|---|---|
| R3 | 75b1db5 | R3.yaml (S23_mix, λ 0.25 / 0.25, chain) | `jainam2142/mtp-run-R3` | 3.056 (39.5%) | 5.497 (15.0%) | 6.382 (8.1%) | 6.773 (4.9%) | 2026-10-10; eval_small; h1 in-group / boundary 28.8 / 14.0% |
| Rsd | a4f1ff4 | Rsd.yaml (frozen backbone, self-distillation, 100k texts) | `jainam2142/mtp-run-Rsd` | 2.865 (42.2%) | 5.967 (13.9%) | 6.869 (7.1%) | 7.250 (4.5%) | 2026-10-10; `eval`; head 0 = base model exactly |
| Rsd_s43 | a4f1ff4 | Rsd_s43.yaml (= Rsd, seed 43 + `data.shuffle`) | not published | 2.865 (42.2%) | 5.969 (14.0%) | 6.878 (7.2%) | 7.257 (4.3%) | replicate of Rsd |
| R6a | 1124204 | R6a.yaml (= R3, `grouper: random`) | — | | | | | running 2026-10-10; compare with R3 using `--grouper hi_rules_v0` |

**Acceptance (Jainam's check, not the official eval):** FixedK on 100 IndicCorp-eval prompts (`make_prompts(..., seed=1)`), RTX 4060, paired bootstrap vs R2.

| run | tokens/step | h1 / h2 / h3 acceptance | h1 in-group top-1 (`eval`, real text) | speed-up (fp32, 20 flores_hi prompts) |
|---|---|---|---|---|
| R2 | 1.370 | 30.1 / 6.1 / 1.4% | 23.4% | 1.24× |
| R3 | 1.380 (n.s.) | 30.9 / 6.3 / 1.5% | **+2.3 ✓** | — |
| Rsd | **1.649 (+0.28 ✓)** | **41.0 / 17.1 / 8.2%** | −3.4 | **1.53×** |
| Rsd_s43 | 1.664 (+0.29 ✓) | 41.6 / 18.0 / 8.3% | −3.3 | — |

## Official evaluation (OM-7 `scripts/evaluate.py`, OM-8)

Step 12500, grouper `hi_rules_v0`, Kaggle T4. JSONs in `results/{run}/eval_{dataset}.json` (INTERFACES §10).
Cell: loss (top-1). In-group / boundary = top-1 when the target is / is not in the source token's word group.

**IndicCorp eval (raw rows 0-999, ~500 sentences)**

| run | h0 | h1 | h2 | h3 | in-group / boundary top-1 (h0 · h1 · h2 · h3) |
|---|---|---|---|---|---|
| R0 | 2.835 (42.5%) |  |  |  | 58.4% / 34.7% |
| R1 | 2.866 (42.0%) | 5.468 (15.4%) |  |  | 58.2% / 34.2% · 28.7% / 14.3% |
| R2 | 2.841 (42.4%) | 5.367 (15.0%) | 6.290 (8.1%) | 6.707 (5.3%) | 58.3% / 34.6% · 23.4% / 14.3% · 19.0% / 7.8% · 17.5% / 5.2% |

Targets per head (R2): h0 26,563 (in-group 8,729), h1 26,063 (in-group 1,886), h2 25,563 (in-group 557), h3 25,063 (in-group 171).

**FLORES-200 devtest hin_Deva (1,012 sentences)**

| run | h0 | h1 | h2 | h3 | in-group / boundary top-1 (h0 · h1 · h2 · h3) |
|---|---|---|---|---|---|
| R0 | 3.956 (30.2%) |  |  |  | 44.1% / 21.7% |
| R1 | 3.957 (30.1%) | 6.377 (10.4%) |  |  | 43.9% / 21.7% · 15.0% / 9.8% |
| R2 | 3.946 (30.3%) | 6.358 (10.3%) | 7.070 (5.9%) | 7.341 (4.4%) | 44.1% / 21.9% · 12.5% / 10.0% · 12.1% / 5.7% · 12.1% / 4.3% |

Targets per head (R2): h0 29,135 (in-group 11,025), h1 28,123 (in-group 3,116), h2 27,111 (in-group 1,009), h3 26,099 (in-group 298).

Checks and observations:

- R0 h0 on `eval` = 2.835, matching the laptop R0 at 2k steps (2.836). Head 0 of R2 is within +0.006 of R0 (R1, α = 1: +0.031).
- In-group targets are easier than boundary targets for every head. Ratio of top-1 (R2) on `eval`: 1.7× / 1.6× / 2.4× / 3.4× for h0-h3; on FLORES 2.0× / 1.25× / 2.1× / 2.8×. FLORES h1 has the smallest gap (12.5% vs 10.0%).
- R1's linear head 1 beats R2's resblock head 1 on in-group top-1 (28.7% vs 23.4% on `eval`) at similar overall top-1 (15.4% vs 15.0%). Worth a look before R3 is compared against R2.
- In-group counts for h3 are small (171 on `eval`), so h3 in-group numbers are noisy.
- FLORES is harder for every head (h0 ~3.95 vs ~2.84): a different domain from the IndicCorp training text.

## Self-speculative decoding (OM-6, OM-8)

Step 12500, Kaggle T4 (fp32 weights + fp16 autocast), batch size 1. Prompts: first 8-16 words of 200 sentences per dataset, 64 new tokens.
Accepted length = tokens per step (greedy = 1.0). Speed-up = tokens/s vs greedy decoding with the base LM on the same prompts.

| run | dataset | policy | accepted len | accept rate h1 / h2 / h3 | speed-up | identical to greedy (fp16) | fp32 re-check (20) | Group Integrity |
|---|---|---|---|---|---|---|---|---|
| R1 | indiccorp_eval | fixed_k | 1.31 | 31.0% | ×1.25 | 87.5% of prompts | 20/20 | 68.0% |
| R1 | indiccorp_eval | confidence_cut | 1.08 | 80.8% | ×1.06 | 88.5% of prompts | 20/20 | 55.0% |
| R1 | flores_hi | fixed_k | 1.28 | 28.7% | ×1.23 | 81.5% of prompts | 20/20 | 77.3% |
| R1 | flores_hi | confidence_cut | 1.05 | 79.9% | ×1.04 | 81.0% of prompts | 20/20 | 70.7% |
| R2 | indiccorp_eval | fixed_k | 1.39 | 30.1% / 7.2% / 2.4% | ×1.14 | 87.5% of prompts | 19/20 | 75.7% |
| R2 | indiccorp_eval | confidence_cut | 1.10 | 77.0% / 73.2% / 84.8% | ×0.94 | 87.5% of prompts | 20/20 | 67.1% |
| R2 | flores_hi | fixed_k | 1.36 | 29.5% / 6.2% / 1.3% | ×1.12 | 89.0% of prompts | 20/20 | 79.0% |
| R2 | flores_hi | confidence_cut | 1.07 | 75.9% / 63.6% / 65.0% | ×0.91 | 89.0% of prompts | 20/20 | 74.2% |

- **Speed-up:** FixedK gives ×1.12-1.25. ConfidenceCut (τ = 0.5) proposes few drafts (~1.05-1.10 tokens/step) and is slower than greedy for R2. The far heads of R2 are rarely accepted (h2 6-7%, h3 1-2%), so R2's extra heads add cost without adding accepted tokens; R1's single linear head gives the best speed-up.
- **Correctness:** in fp16, 81-89% of prompts are identical to greedy. Every divergence happens at a near-tie: greedy's top-2 logit margin there is at most 0.0156 = 2⁻⁶, one fp16 step at these logit sizes (some exactly 0, true ties). The fp32 re-check of the same engine (autocast off) is 20/20 identical in 11 of 12 cases.
- **Resolved:** R2 / indiccorp_eval / fixed_k is 19/20 in the fp32 re-check. The rerun (2026-10-10, same numbers: ×1.147 vs ×1.143) records the fp32 divergence margin: **0.0**, an exact tie between greedy's top-2 logits, so argmax tie-breaking differs and the engine is correct.

## R3 vs R2 (OM-8, official eval)

R3 = `R3_hi_k4_struct` (S23_mix, λ_S3 0.25, λ_S3_all 0.25, chain teacher), `jainam2142/mtp-run-R3`, step 12500. Same eval as R0-R2: Kaggle T4, grouper `hi_rules_v0`, 200 spec-decode prompts.

| dataset | head | loss R2 → R3 | top-1 R2 → R3 | in-group top-1 R2 → R3 (n targets) | boundary top-1 R2 → R3 |
|---|---|---|---|---|---|
| indiccorp_eval | h0 | 2.841 → 2.849 (+0.008) | 42.4% → 42.3% | 58.3% → 58.3% (8,729) | 34.6% → 34.6% |
| indiccorp_eval | h1 | 5.367 → 5.320 (-0.047) | 15.0% → 15.5% | 23.4% → 25.8% (1,886) | 14.3% → 14.7% |
| indiccorp_eval | h2 | 6.290 → 6.256 (-0.033) | 8.1% → 8.4% | 19.0% → 24.1% (557) | 7.8% → 8.0% |
| indiccorp_eval | h3 | 6.707 → 6.678 (-0.030) | 5.3% → 5.5% | 17.5% → 22.2% (171) | 5.2% → 5.3% |
| flores_hi | h0 | 3.946 → 3.944 (-0.002) | 30.3% → 30.2% | 44.1% → 44.1% (11,025) | 21.9% → 21.7% |
| flores_hi | h1 | 6.358 → 6.299 (-0.059) | 10.3% → 10.6% | 12.5% → 14.3% (3,116) | 10.0% → 10.1% |
| flores_hi | h2 | 7.070 → 7.016 (-0.053) | 5.9% → 6.2% | 12.1% → 14.0% (1,009) | 5.7% → 5.9% |
| flores_hi | h3 | 7.341 → 7.296 (-0.045) | 4.4% → 4.5% | 12.1% → 13.1% (298) | 4.3% → 4.4% |

| dataset | policy | accepted len R2 → R3 | speed-up R2 → R3 | R3 identical to greedy (fp16) | R3 fp32 re-check |
|---|---|---|---|---|---|
| indiccorp_eval | fixed_k | 1.39 → 1.39 | ×1.14 → ×1.15 | 84.5% of prompts | 20/20 |
| indiccorp_eval | confidence_cut | 1.10 → 1.11 | ×0.94 → ×0.94 | 84.5% of prompts | 20/20 |
| flores_hi | fixed_k | 1.36 → 1.35 | ×1.12 → ×1.12 | 87.5% of prompts | 20/20 |
| flores_hi | confidence_cut | 1.07 → 1.07 | ×0.91 → ×0.91 | 88.0% of prompts | 20/20 |

- **In-group accuracy of the far heads goes up, on both datasets.** IndicCorp h1 / h2 / h3: +2.4 / +5.1 / +4.7 points; FLORES (out of domain): +1.8 / +1.9 / +1.0. Boundary top-1 barely moves (≤ +0.4), so the gain is where the loss targets it. Loss improves for every extra head (−0.03 to −0.06).
- **Head 0 is held:** +0.008 on IndicCorp eval, −0.002 on FLORES.
- **Acceptance and speed are unchanged** (1.39 vs 1.39 tokens/step, ×1.15 vs ×1.14 FixedK on IndicCorp), matching Jainam's 100-prompt check (design note §5). The structural loss improves *what* the heads predict inside groups, not how often head 0 agrees with them.
- **Correctness:** fp32 re-check 20/20 for every policy and dataset; fp16 divergences again only at near-ties (margin ≤ 0.0156).
- h3 in-group counts are small (171 IndicCorp, 298 FLORES): read h3's in-group numbers as noisy.

## Rsd: frozen backbone + self-distillation (OM-8, official eval)

`Rsd_hi_k4_frozen_sd` (`jainam2142/mtp-run-rsd`) and its replicate `Rsd_hi_k4_frozen_sd_s43` (`jainam2142/mtp-run-rsd-s43`, seed 43 + shuffled data), step 12500. Same eval as R0-R3: Kaggle T4, grouper `hi_rules_v0`, 200 spec-decode prompts. Head 0 is ganga-1b with no adapter, so it trails R0 (LoRA NTP) by 0.03 loss on IndicCorp eval and is ahead by 0.03 on FLORES.

| run | dataset | h0 | h1 | h2 | h3 | in-group / boundary top-1 (h1 · h2 · h3) |
|---|---|---|---|---|---|---|
| R2 | indiccorp_eval | 2.841 (42.4%) | 5.367 (15.0%) | 6.290 (8.1%) | 6.707 (5.3%) | 23.4 / 14.3 · 19.0 / 7.8 · 17.5 / 5.2 |
| Rsd | indiccorp_eval | 2.865 (42.2%) | 5.967 (13.9%) | 6.869 (7.1%) | 7.250 (4.5%) | 19.9 / 13.4 · 13.1 / 7.0 · 13.5 / 4.5 |
| Rsd_s43 | indiccorp_eval | 2.865 (42.2%) | 5.969 (14.0%) | 6.878 (7.2%) | 7.257 (4.3%) | 20.0 / 13.5 · 12.6 / 7.1 · 12.9 / 4.3 |
| R2 | flores_hi | 3.946 (30.3%) | 6.358 (10.3%) | 7.070 (5.9%) | 7.341 (4.4%) | 12.5 / 10.0 · 12.1 / 5.7 · 12.1 / 4.3 |
| Rsd | flores_hi | 3.984 (29.9%) | 6.909 (8.8%) | 7.577 (4.6%) | 7.822 (3.3%) | 9.5 / 8.7 · 6.9 / 4.5 · 8.7 / 3.3 |
| Rsd_s43 | flores_hi | 3.984 (29.9%) | 6.908 (8.5%) | 7.580 (4.4%) | 7.822 (2.9%) | 9.1 / 8.5 · 5.9 / 4.4 · 5.0 / 2.9 |

Per-head numbers are against the *real* next tokens; Rsd's heads were trained on the base model's greedy text, so they are worse here and better where it matters, on head 0's own continuations:

| run | dataset | policy | accepted len | accept rate h1 / h2 / h3 | speed-up (tok/s) | identical to greedy (fp16) | fp32 re-check (20) | Group Integrity |
|---|---|---|---|---|---|---|---|---|
| R2 | indiccorp_eval | fixed_k | 1.39 | 30.1 / 7.2 / 2.4% | ×1.15 (31.8) | 87.5% | 19/20 (exact tie) | 75.7% |
| Rsd | indiccorp_eval | fixed_k | **1.67** | 41.7 / 17.4 / 9.1% | **×1.36** (37.5) | 87.5% | 20/20 | 74.2% |
| Rsd_s43 | indiccorp_eval | fixed_k | 1.66 | 40.7 / 17.6 / 8.9% | ×1.35 (37.4) | 87.5% | 20/20 | 74.4% |
| Rsd | indiccorp_eval | confidence_cut | 1.36 | 70.5 / 64.3 / 67.8% | ×1.14 | 87.0% | 20/20 | 72.2% |
| Rsd_s43 | indiccorp_eval | confidence_cut | 1.37 | 70.5 / 65.9 / 67.8% | ×1.14 | 88.0% | 20/20 | 71.5% |
| R2 | flores_hi | fixed_k | 1.36 | 29.5 / 6.2 / 1.3% | ×1.12 (31.1) | 89.0% | 20/20 | 79.0% |
| Rsd | flores_hi | fixed_k | **1.60** | 39.0 / 15.5 / 7.1% | **×1.32** (36.4) | 82.5% | 20/20 | 80.2% |
| Rsd_s43 | flores_hi | fixed_k | 1.62 | 39.6 / 15.7 / 7.6% | ×1.33 (36.8) | 82.5% | 20/20 | 78.5% |
| Rsd | flores_hi | confidence_cut | 1.30 | 72.6 / 64.9 / 65.8% | ×1.09 | 83.0% | 20/20 | 76.4% |
| Rsd_s43 | flores_hi | confidence_cut | 1.31 | 71.3 / 67.6 / 72.6% | ×1.10 | 82.5% | 20/20 | 76.6% |

Greedy baseline: 27.6-27.8 tok/s on every run.

- **Rsd is the fastest run: ×1.32-1.36 on T4** (vs ×1.12-1.15 for R2/R3), 1.60-1.67 tokens/step. It matches Jainam's 100-prompt check (1.649 tokens/step); his 1.53× was fp32 on an RTX 4060, where the verify pass is relatively cheaper.
- **Replicate agrees:** Rsd_s43 is within 0.01-0.02 tokens/step and ±0.01 speed-up of Rsd on both sets.
- **Far heads are where it gains:** h2 / h3 acceptance 17 / 9% vs R2's 7 / 2%.
- **Correctness:** fp32 re-check 20/20 for every Rsd policy and dataset; fp16 divergences only at near-ties (margin ≤ 0.0156).
- ConfidenceCut is now faster than greedy (×1.09-1.14) but still behind FixedK, consistent with dropping stop-early policies (#33).

## Verify-pass cost on T4 (`bench_verify`, issue #33)

One forward pass of the MTP model (all heads) scoring N positions after a 64-token prompt, KV cache rolled back after each pass. Median of 20 repeats; fp32 weights + fp16 autocast. Greedy = base LM scoring 1 position.

| positions | 1 | 4 | 8 | 16 | 25 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| R2 (ms) | 41.9 | 42.9 | 43.1 | 44.1 | 45.6 | 45.8 | 47.1 |
| Rsd (ms) | 41.6 | 42.6 | 43.0 | 43.8 | 45.3 | 45.5 | 46.5 |
| Rsd_s43 (ms) | 41.4 | 42.5 | 42.8 | 43.7 | 45.1 | 45.2 | 46.2 |
| × greedy pass (Rsd) | 1.17 | 1.20 | 1.21 | 1.23 | 1.27 | 1.27 | 1.30 |

Greedy pass: 35.4-35.7 ms.

- **At batch 1 the T4 is memory-bound:** scoring 25 positions costs 9% more than scoring 1, and 64 positions costs 12% more. Tree size is not the constraint up to 64 nodes. The fixed cost is the extra heads: 1 position is already ×1.17 greedy.
- **The model predicts the measured chain speed-up:** Rsd chain = 1.67 tokens/step ÷ 1.20 (4 positions) = ×1.39, against ×1.36 measured (the rest is drafting overhead). On the same model, Jainam's offline 25-node tree (2.51 tokens/step, #33) ÷ 1.27 gives a **×1.98 ceiling** for `generate_tree()` on T4.


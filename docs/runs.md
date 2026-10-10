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

Pilots that led here: `docs/design_structural_loss.md` §5. Step 12500. Logged eval (`train.py`): eval_small for the R3 line, R8-R10a and A1; `eval` (~500 sentences) for the Rsd line. Logged in-group / boundary uses each run's **own** grouper, so for runs trained with random groups (R6a, R6a_s43, R10a, Rsd_mr) it is not comparable; the comparisons below fix the labels.

| ID | commit | config | Kaggle dataset | h0 | h1 | h2 | h3 | notes |
|---|---|---|---|---|---|---|---|---|
| R3 | 75b1db5 | R3.yaml (S23_mix, λ 0.25 / 0.25, chain) | `jainam2142/mtp-run-R3` | 3.056 (39.5%) | 5.497 (15.0%) | 6.382 (8.1%) | 6.773 (4.9%) | 2026-10-10; eval_small; h1 in-group / boundary 28.8 / 14.0% |
| Rsd | a4f1ff4 | Rsd.yaml (frozen backbone, self-distillation, 100k texts) | `jainam2142/mtp-run-Rsd` | 2.865 (42.2%) | 5.967 (13.9%) | 6.869 (7.1%) | 7.250 (4.5%) | 2026-10-10; `eval`; head 0 = base model exactly |
| Rsd_s43 | a4f1ff4 | Rsd_s43.yaml (= Rsd, seed 43 + `data.shuffle`) | `jainam2142/mtp-run-Rsd-s43` | 2.865 (42.2%) | 5.969 (14.0%) | 6.878 (7.2%) | 7.257 (4.3%) | replicate of Rsd |
| R6a | 1124204 | R6a.yaml (= R3, `grouper: random`) | `jainam2142/mtp-run-R6a` | 3.057 (39.5%) | 5.474 (14.8%) | 6.342 (8.0%) | 6.743 (5.4%) | 2026-10-10; eval_small; compare with R3 under `hi_rules_v0` (below) |
| R3_s43 | #44 | R3_s43.yaml (= R3, seed 43 + `data.shuffle`) | `jainam2142/mtp-run-R3-s43` | 3.057 (39.5%) | 5.494 (14.4%) | 6.387 (7.6%) | 6.786 (4.6%) | 2026-10-10; laptop RTX 4060 (bf16); in-group h1 / h2 / h3 25.2 / 24.6 / 20.0% |
| R6a_s43 | #44 | R6a_s43.yaml (= R6a, seed 43 + `data.shuffle`) | `jainam2142/mtp-run-R6a-s43` | 3.055 (39.5%) | 5.461 (14.4%) | 6.354 (7.2%) | 6.751 (4.9%) | 2026-10-10; account `jainamrampuria` |
| A1_alpha0 | #35 | A1_alpha0.yaml (= R2, α = 0) | `jainamrampuria/mtp-run-A1-alpha0` | 3.051 (39.7%) | 5.596 (13.5%) | 6.481 (7.8%) | 6.880 (4.3%) | official eval below |
| A1_alpha1 | #35 | A1_alpha1.yaml (= R2, α = 1) | `jainamrampuria/mtp-run-A1-alpha1` | 3.175 (38.1%) | 5.392 (14.5%) | 6.230 (8.4%) | 6.611 (4.9%) | official eval below |
| R8 | — | R8.yaml (Misal-1B, NTP) | `jainam2142/mtp-run-R8` | 4.127 (31.9%) | | | | eval_small (mr) |
| R9 | — | R9.yaml (Misal-1B, = R2) | `jainam2142/mtp-run-R9` | 4.132 (31.6%) | 6.576 (13.5%) | 7.373 (8.7%) | 7.716 (7.2%) | eval_small (mr) |
| R10 | #43 | R10.yaml (= R3 recipe, `mr_rules_v1`) | `jainam2142/mtp-run-R10` | 4.159 (31.2%) | 6.494 (13.7%) | 7.296 (9.3%) | 7.635 (7.1%) | eval_small (mr) |
| R10a | #43 | R10a.yaml (= R10, `random_mr_v1`) | `jainam2142/mtp-run-R10a` | 4.157 (31.3%) | 6.490 (13.8%) | 7.282 (9.0%) | 7.628 (7.2%) | eval_small (mr) |
| Rsd_soft | #42 | Rsd_soft.yaml (= Rsd + soft labels, S3_all λ 1, teacher h0) | `jainam2142/mtp-run-Rsd-soft` | 2.865 (42.2%) | 5.693 (14.3%) | 6.638 (7.4%) | 7.025 (4.9%) | `eval` |
| Rsd_k6 | #42 | Rsd_k6.yaml (= Rsd, 6 heads) | `jainam2142/mtp-run-Rsd-k6` | 2.865 (42.2%) | 5.967 (13.8%) | 6.869 (7.1%) | 7.249 (4.5%) | `eval`; h4 7.447 (3.5%), h5 7.573 (2.9%) |
| Rsd_mr | #42 | Rsd_mr.yaml (frozen Misal, self-distillation, `min_new_tokens 64`) | `jainam2142/mtp-run-Rsd-mr` | 4.890 (27.8%) | 7.833 (9.3%) | 8.552 (5.7%) | 8.840 (4.3%) | `eval` (mr); head 0 = Misal-instruct as is |
| Rsd_mr_soft | #42 | Rsd_mr_soft.yaml (= Rsd_mr + soft labels) | `jainam2142/mtp-run-Rsd-mr-soft` | 4.890 (27.8%) | 7.666 (9.3%) | 8.394 (5.6%) | 8.683 (4.3%) | `eval` (mr) |
| Rsd_long, Rsd_data2x | #42 | 2× data and 2× steps / 2× data, same steps | — | | | | | running 2026-10-10 |

`mtp-run-R3-s43`, `-R6a-s43`, `-Rsd-mr`, `-Rsd-mr-soft` are public; the others are private and shared with Om.

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

## R6a: random-group control for R3 (OM-8, official eval)

`R6a_hi_k4_struct_random` (`jainam2142/mtp-run-R6a`) = R3 with `data.grouper: random` (JI-4), step 12500. Evaluated with `--grouper hi_rules_v0`, so in-group / boundary use the **linguistic** groups for every run. The target counts match R2/R3 exactly (h1 in-group 1,886 on IndicCorp, 3,116 on FLORES).

| dataset | head | top-1 R2 / R3 / R6a | in-group top-1 R2 / R3 / R6a (n) | boundary top-1 R2 / R3 / R6a |
|---|---|---|---|---|
| indiccorp_eval | h0 | 42.4 / 42.3 / 42.3 | 58.3 / 58.3 / 58.3 (8,729) | 34.6 / 34.6 / 34.5 |
| indiccorp_eval | h1 | 15.0 / 15.5 / 15.4 | 23.4 / **25.8** / **25.2** (1,886) | 14.3 / 14.7 / 14.7 |
| indiccorp_eval | h2 | 8.1 / 8.4 / 8.4 | 19.0 / **24.1** / 21.5 (557) | 7.8 / 8.0 / 8.1 |
| indiccorp_eval | h3 | 5.3 / 5.5 / 5.4 | 17.5 / **22.2** / 19.9 (171) | 5.2 / 5.3 / 5.3 |
| flores_hi | h0 | 30.3 / 30.2 / 30.2 | 44.1 / 44.1 / 44.2 (11,025) | 21.9 / 21.7 / 21.8 |
| flores_hi | h1 | 10.3 / 10.6 / 10.6 | 12.5 / **14.3** / 13.7 (3,116) | 10.0 / 10.1 / 10.2 |
| flores_hi | h2 | 5.9 / 6.2 / 6.2 | 12.1 / **14.0** / 12.9 (1,009) | 5.7 / 5.9 / 6.0 |
| flores_hi | h3 | 4.4 / 4.5 / 4.5 | 12.1 / **13.1** / 11.1 (298) | 4.3 / 4.4 / 4.5 |

| dataset | FixedK tokens/step R2 / R3 / R6a | speed-up R2 / R3 / R6a | R6a fp32 re-check |
|---|---|---|---|
| indiccorp_eval | 1.39 / 1.39 / 1.40 | ×1.15 / ×1.15 / ×1.14 | 20/20 (both policies) |
| flores_hi | 1.36 / 1.35 / 1.37 | ×1.12 / ×1.12 / ×1.12 | 20/20 (both policies) |

- **Head 1: random groups give most of R3's in-group gain.** R6a is at +1.8 vs R2 on IndicCorp (R3 +2.4) and +1.2 on FLORES (R3 +1.8). It does **not** reproduce the −11.5 point drop of the laptop check in #39.
- **Heads 2-3: R3 stays ahead of R6a** by 2.3 to 2.6 points on IndicCorp and 1.1 to 2.0 on FLORES, about half of R3's gain over R2. These counts are small (557 / 171 targets on IndicCorp), so the binomial SE is about 1.7 / 3.1 points. A paired bootstrap needs token dumps (`--dump_tokens`).
- Head 0, acceptance and speed: same as R2/R3.

## α ablation A1 (JN-10, OM-8, official eval)

`A1_alpha0` / `A1_alpha1` (`jainamrampuria/mtp-run-A1-alpha{0,1}`) = R2 with `head_backbone_grad` 0 / 1; R2 is α = 0.1. Step 12500, `hi_rules_v0`, 200 prompts.

| α | dataset | h0 loss (top-1) | h1 / h2 / h3 top-1 | FixedK tokens/step | h1 / h2 / h3 accept | speed-up | fp32 re-check |
|---|---|---|---|---|---|---|---|
| 0 | indiccorp_eval | 2.834 (42.5%) | 14.7 / 7.9 / 5.2 | 1.38 | 29.7 / 6.8 / 2.3% | ×1.14 | 20/20 |
| 0.1 (R2) | indiccorp_eval | 2.841 (42.4%) | 15.0 / 8.1 / 5.3 | 1.39 | 30.1 / 7.2 / 2.4% | ×1.15 | 19/20 (exact tie) |
| 1 | indiccorp_eval | **2.981 (40.6%)** | 15.6 / 8.7 / 5.7 | **1.44** | 34.1 / 8.3 / 2.0% | **×1.18** | 20/20 |
| 0 | flores_hi | 3.955 (30.2%) | 10.0 / 5.8 / 4.3 | 1.32 | 27.2 / 5.0 / 0.8% | ×1.09 | 20/20 |
| 0.1 (R2) | flores_hi | 3.946 (30.3%) | 10.3 / 5.9 / 4.4 | 1.36 | 29.5 / 6.2 / 1.3% | ×1.12 | 20/20 |
| 1 | flores_hi | **4.019 (29.5%)** | 10.7 / 6.3 / 4.7 | **1.40** | 32.5 / 7.0 / 1.1% | **×1.16** | 20/20 |

- **The trade-off holds on T4:** α = 1 gives +0.04-0.05 tokens/step and +0.03-0.04 speed-up over R2, but head 0 (the model being sped up) is worse by +0.14 nats / −1.8 points top-1 on IndicCorp and +0.07 / −0.8 on FLORES.
- **α = 0.1 is the right default:** its verifier is within 0.01 of α = 0, and its heads are slightly better.
- This matches Jainam's laptop check: +0.049 tokens/step for α = 1.

## Marathi: R8 / R9 (Misal-1B, OM-8, official eval)

`R8_mr_ntp` / `R9_mr_mtp_k4_resblock` (`jainam2142/mtp-run-R8`, `-R9`), base `smallstepai/Misal-1B-instruct-v0.1` (adds BOS; 1.61 tokens/word). Step 12500, grouper `mr_rules_v1`, IndicCorp mr eval (raw rows 0-999) + FLORES devtest mar_Deva, 200 prompts.

| run | dataset | h0 | h1 | h2 | h3 | in-group / boundary top-1 (h0 · h1 · h2 · h3) |
|---|---|---|---|---|---|---|
| R8 | indiccorp_eval (mr) | 4.252 (30.2%) | | | | 46.2 / 20.1 |
| R9 | indiccorp_eval (mr) | 4.258 (30.1%) | 6.824 (11.5%) | 7.621 (7.6%) | 7.969 (6.0%) | 46.4 / 19.7 · 28.3 / 8.6 · 17.4 / 7.2 · 11.7 / 6.0 |
| R8 | flores_mr | 4.903 (23.7%) | | | | 39.8 / 12.6 |
| R9 | flores_mr | 4.905 (23.6%) | 7.520 (7.1%) | 8.122 (5.0%) | 8.318 (4.5%) | 39.5 / 12.6 · 16.2 / 5.4 · 8.2 / 4.9 · 8.1 / 4.5 |

In-group targets per head (R9, IndicCorp): h0 10,214, h1 3,931, h2 1,256, h3 444. That's about twice Hindi's share at h1 (3,931 vs 1,886 for a similar number of targets).

| run | dataset | policy | tokens/step | accept h1 / h2 / h3 | speed-up (tok/s) | identical (fp16) | fp32 re-check | Group Integrity |
|---|---|---|---|---|---|---|---|---|
| R9 | indiccorp_eval | fixed_k | 1.31 | 24.4 / 6.0 / 1.2% | ×1.07 (24.4 vs 22.9) | 85.0% | 20/20 | 77.3% |
| R9 | indiccorp_eval | confidence_cut | 1.07 | 58.4 / 36.5 / 14.1% | ×0.89 | 85.5% | 20/20 | 84.9% |
| R9 | flores_mr | fixed_k | 1.30 | 23.8 / 5.5 / 1.0% | ×1.05 (24.1 vs 23.0) | 90.0% | 20/20 | 79.4% |
| R9 | flores_mr | confidence_cut | 1.06 | 51.3 / 22.5 / 8.2% | ×0.88 | 88.5% | 20/20 | 78.1% |

- **Head 0 is held:** R9 − R8 = +0.006 (IndicCorp) / +0.002 (FLORES).
- **Marathi speed-up is smaller than Hindi:** ×1.05-1.07 vs ×1.12-1.15 for R2, because head 1 is accepted less often (24% vs 30%). Greedy is also slower (23 vs 27.7 tok/s; Misal is a Llama with a 32k vocabulary).
- **In-group vs boundary gap is much larger than in Hindi:** h1 28.3% in-group vs 8.6% at boundaries (3.3×; Hindi R2 1.6×). That's the word-internal structure Jainam's tree oracle picks up (#33).
- **EOS is not an issue for these prompts:** greedy took ~2.8 s per prompt at 23 tok/s ≈ 64 tokens, i.e. almost no prompt stopped early. Prompts are cut mid-sentence, unlike the train prompts in #39.

## JN-7b: loss-weighting pilots (Table 5, #36)

4 pilots × 2k steps on the R3 recipe (`configs/pilot_w_*.yaml`), each differing from `R3.yaml` only in `max_steps: 2000` and the `weighting` line. Paired bootstrap vs `pilot_w_fixed` (2000 resamples; 500 IndicCorp eval sentences for top-1, 100 prompts for tokens/step; labels `hi_rules_v0`; 95% CI). Points.

| pilot | weighting | h0 eval loss | h1 in-group top-1 | mean top-1 h1-3 | FixedK tokens/step |
|---|---|---|---|---|---|
| `pilot_w_fixed` | fixed 0.8^d (= R3) | 3.057 | ref | ref | ref |
| `pilot_w_unc` | uncertainty, head 0 fixed | 3.054 | +0.6 [−0.2, +1.4] | −0.12 [−0.26, +0.01] | −0.012 [−0.035, +0.014] |
| `pilot_w_unc_free` | uncertainty, head 0 free | **3.198 (+0.14)** | **+1.8 [+0.8, +2.8]** | +0.24 [+0.09, +0.40] | +0.014 [−0.023, +0.047] |
| `pilot_w_dwa` | DWA | 3.057 | +0.05 [−0.3, +0.4] | −0.03 [−0.07, +0.01] | −0.011 [−0.030, +0.002] |

- **Negative row:** with head 0 protected (`unc`, DWA), adaptive weighting is flat on every metric. `unc_free` gains in-group top-1 only by down-weighting head 0, so the verifier loses 0.14 nats: the same trade-off as α = 1 (A1). So R5 / R6 are not run (#36).
- **On a frozen backbone (Rsd), weighting is a no-op:** each head has its own parameters, and AdamW divides out a per-parameter loss scale. Rsd config, 200 steps, only `head_decay` changed: h1 eval loss 6.4244 (0.8^d) / 6.4248 (equal) / 6.4263 (0.3^d).

## R3 vs R6a: paired bootstrap (Jainam's check, #39)

Same token-level evaluation as above (500 IndicCorp eval sentences, labels `hi_rules_v0` for **every** run, so the in-group targets are the same: 1,886 / 557 / 171 for h1 / h2 / h3; matches OM-8). Difference vs R2 in points, 95% CI.

| vs R2 | h1 in-group | h2 in-group | h3 in-group | accepted len (100 prompts) |
|---|---|---|---|---|
| R3 | +2.4 [+1.5, +3.4] | +5.0 [+2.6, +7.7] | +4.7 [+0.5, +9.5] | −0.003 [−0.024, +0.020] |
| R6a (random groups) | +1.8 [+1.0, +2.6] | +2.5 [+0.4, +4.9] | +2.3 [−0.7, +5.6] | −0.002 [−0.024, +0.021] |

- At h1 the extra loss itself gives most of the gain (1.8 of 2.4 points). At h2 / h3 about half of R3's gain needs the linguistic groups.
- The direct R3 − R6a CI and the seed-43 pair (R3_s43 / R6a_s43) follow in #39.

## Marathi: R10 / R10a (structural loss, Jainam's check)

R10 = R3 recipe on Misal-1B with `mr_rules_v1`; R10a = the same with `random_mr_v1` (length-matched random groups). Labels `mr_rules_v1` for all three runs (in-group targets h1 / h2 / h3: 3,931 / 1,256 / 444). Difference vs R9 in points, 95% CI.

| vs R9 | h1 in-group | h2 in-group | h3 in-group | accepted len (100 prompts) |
|---|---|---|---|---|
| R10 (`mr_rules_v1`) | +2.2 [+1.5, +3.0] | +1.2 [+0.1, +2.3] | +1.6 [−0.5, +3.7] | +0.028 [−0.009, +0.068] |
| R10a (`random_mr_v1`) | +2.6 [+1.9, +3.4] | +1.4 [+0.3, +2.6] | +0.9 [−1.4, +3.3] | +0.027 [−0.007, +0.066] |

- **In Marathi the structural loss gains the same with random groups** (R10 ≈ R10a at every head). The linguistic groups still matter at decoding time: on R9's draft trees, groups beat random and word starts (#33).

## Rsd line: variants (Jainam's check)

FixedK chain on 100 IndicCorp eval prompts (RTX 4060 or T4), paired bootstrap vs Rsd. Offline tree oracle (`scripts/fit_tree.py`, accepted drafts per step on 200 eval prompts, teacher-forced; +1 = tokens/step): the per-position oracle overstates real decoding (real / offline ≈ 0.86 for k = 4).

| run | chain tokens/step vs Rsd | tree n25 | tree n40 | tree n64 (entropy-shaped) |
|---|---|---|---|---|
| Rsd | ref (offline chain 0.908) | 1.554 | 1.656 | 1.752 |
| Rsd_soft (soft labels) | −0.001 | 1.543 | 1.641 | 1.741 |
| Rsd_k6 (6 heads) | +0.037 [+0.023, +0.055] | 1.695 | 1.818 | 1.933 |

- Soft labels: same as hard labels at full length (the 2k-step pilot's +0.019 did not hold). Negative row.
- k = 6: small chain gain; the offline tree gain (+0.14-0.18) is to be checked on T4 with `generate_tree()` (OM).

**Marathi Rsd (frozen Misal-1B-instruct).** Misal-instruct stops after one sentence (median 6 generated tokens), so the self-distilled text is generated with `--min_new_tokens 64`. At decode time head 0 also stops: 66% of the 100 eval prompts end within 10 tokens. With head 0's `</s>` logit masked (still lossless with respect to the masked model):

| run | tokens/step, stop at `</s>` (mean generated) | tokens/step, `</s>` masked | vs R9 (masked) |
|---|---|---|---|
| R9 | 1.306 (64.0) | 1.306 | ref |
| Rsd_mr | 1.356 (14.0) | **1.539** | **+0.233 [+0.174, +0.299]** |
| Rsd_mr_soft | 1.415 (14.0) | 1.526 | +0.220 [+0.160, +0.287] |

- Self-distillation transfers to Marathi (Hindi: Rsd − R2 = +0.28 on the same check). Soft = hard again.
- **Caveat:** Rsd_mr speeds up Misal-instruct as is, whose head 0 (4.89 on `eval`) is much weaker on IndicCorp than the LoRA-tuned R8/R9 (4.25 official). Evaluate it with `ignore_eos` (OM).

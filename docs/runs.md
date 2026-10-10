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
- **Open:** R2 / indiccorp_eval / fixed_k is 19/20 in the fp32 re-check. Its divergence margin was not recorded, so a sub-1e-5 fp32 tie is likely but not shown; ConfidenceCut on the same 20 prompts is 20/20. Next run records fp32 margins.

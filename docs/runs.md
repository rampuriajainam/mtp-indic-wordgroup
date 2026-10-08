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

| ID | commit | config | Kaggle dataset | h0 | h1 | h2 | h3 | notes |
|---|---|---|---|---|---|---|---|---|
| R0 | 09e3176 | R0.yaml | | | | | | started 2026-10-09 night |
| R1 | 09e3176 | R1.yaml | | | | | | started 2026-10-09 night |
| R2 | 09e3176 | R2.yaml (α=0.1) | | | | | | started 2026-10-09 night |

Gate B: Kaggle R0 / R1 within ~0.05 of the laptop numbers (3.05 / 3.09 / 5.84) on eval_small.
Note: tonight's runs predate the in-group eval; R2's step-12500 checkpoint gets the in-group split from Om's evaluate.py (or `pilot_R2ref` for the 2k-step comparison).

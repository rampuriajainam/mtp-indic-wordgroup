# Evaluation status (OM-8)

Which runs are evaluated on which datasets. ⏳ = running, – = dropped (group-aware decoding stopped, #33).

| run | step | indiccorp_eval | flores_hi | spec: fixed_k | spec: confidence_cut | spec: group_aware | notes |
|---|---|---|---|---|---|---|---|
| R0 | 12500 | ✅ | ✅ | – | – | – | NTP, 1 head (no drafts) |
| R1 | 12500 | ✅ | ✅ | ✅ | ✅ | – | spec: fp16 81-89% identical (near-ties), fp32 re-check 20/20 |
| R2 | 12500 | ✅ | ✅ | ✅ | ✅ | – | spec: fp32 re-check 20/20 except indiccorp fixed_k 19/20 = exact tie (margin 0.0); `bench_verify` ✅ |
| R3 | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-R3`; fp32 re-check 20/20 everywhere; in-group h1-h3 up, acceptance unchanged vs R2 |
| Rsd | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-rsd`; frozen backbone (head 0 = base model), self-distillation; FixedK ×1.32-1.36, fp32 20/20; `bench_verify` ✅ |
| Rsd_s43 | 12500 | ✅ | ✅ | ✅ | ✅ | – | `jainam2142/mtp-run-rsd-s43`; replicate of Rsd (within 0.02 tokens/step); fp32 20/20; `bench_verify` ✅ |
| R6a | 12500 | ⬜ | ⬜ | ⬜ | ⬜ | – | `jainam2142/mtp-run-R6a`; eval with `--grouper hi_rules_v0` (#39) |
| A1_alpha0 | 12500 | ⬜ | ⬜ | ⬜ | ⬜ | – | `jainamrampuria/mtp-run-A1-alpha0`; R2 with α = 0 (JN-10) |
| A1_alpha1 | 12500 | ⬜ | ⬜ | ⬜ | ⬜ | – | `jainamrampuria/mtp-run-A1-alpha1`; R2 with α = 1 (JN-10) |
| R8 (mr) | 12500 | ⬜ | flores_mr ⬜ | – | – | – | `jainam2142/mtp-run-R8`; Misal-1B NTP, `--grouper mr_rules_v1` |
| R9 (mr) | 12500 | ⬜ | flores_mr ⬜ | ⬜ | ⬜ | – | `jainam2142/mtp-run-R9`; Misal-1B k=4 resblock, `--grouper mr_rules_v1` |

Sanity flags: ConfidenceCut speed-up < 1 for R2 (few drafts proposed); R2 indiccorp fixed_k fp32 re-check 19/20 resolved (exact tie). Head 0 of R1/R2 within 0.031 / 0.006 of R0 on `indiccorp_eval`.

# Evaluation status (OM-8)

Which runs are evaluated on which datasets. Spec-decode columns fill in once OM-6 lands.

| run | step | indiccorp_eval | flores_hi | spec: fixed_k | spec: confidence_cut | spec: group_aware | notes |
|---|---|---|---|---|---|---|---|
| R0 | 12500 | ✅ | ✅ | – | – | – | NTP, 1 head (no drafts) |
| R1 | 12500 | ✅ | ✅ | ✅ | ✅ | ⬜ | spec: fp16 81-89% identical (near-ties), fp32 re-check 20/20 |
| R2 | 12500 | ✅ | ✅ | ✅ | ✅ | ⬜ | spec: fp32 re-check 20/20 except indiccorp fixed_k 19/20 (open) |
| R3 | | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | after JN-5 pilots |
| R6a | | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | needs JI-4 |

Sanity flags: ConfidenceCut speed-up < 1 for R2 (few drafts proposed); R2 indiccorp fixed_k fp32 re-check 19/20 (open). Head 0 of R1/R2 within 0.031 / 0.006 of R0 on `indiccorp_eval`.

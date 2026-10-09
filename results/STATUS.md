# Evaluation status (OM-8)

Which runs are evaluated on which datasets. Spec-decode columns fill in once OM-6 lands.

| run | step | indiccorp_eval | flores_hi | spec: fixed_k | spec: confidence_cut | spec: group_aware | notes |
|---|---|---|---|---|---|---|---|
| R0 | 12500 | ✅ | ✅ | – | – | – | NTP, 1 head (no drafts) |
| R1 | 12500 | ✅ | ✅ | ⬜ | ⬜ | ⬜ | |
| R2 | 12500 | ✅ | ✅ | ⬜ | ⬜ | ⬜ | |
| R3 | | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | after JN-5 pilots |
| R6a | | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | needs JI-4 |

Sanity flags: none. Head 0 of R1/R2 within 0.031 / 0.006 of R0 on `indiccorp_eval`.

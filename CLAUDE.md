# Project context for Claude

**Project:** Word-Group Guided Multi-Token Prediction with Adaptive Loss Weighting for Hindi and Marathi.
**Team:** Jainam (Modeling & Training), Jai (Data & Linguistics), Om (Evaluation & Infrastructure).
**I am Jainam.** My tasks: `docs/plan/tasks/JAINAM.md`.

## Read first
- `docs/plan/README.md`: status, order of work (phases A–C now; experiments + write-up later), dependencies/handoffs H1–H10, run matrix R0–R10
- `docs/plan/INTERFACES.md`: target `mtp/` package layout and every function signature / file format. Code must match it.

## Current state
- Base model `LingoIITGN/ganga-1b` (vocab 30k), LoRA r=8 on q_proj/v_proj, lr 2e-4, batch 8, max_len 128, grad clip 1.0.
- Data: `ai4bharat/IndicCorpV2`, config `indiccorp_v2`, split `hin_Deva`, streamed.
- NTP baseline (`train_ntp_baseline_final.py`): held-out loss 3.05, flat. lr/rank sweeps confirmed it's not a bug.
- MTP k=2 (`train_mtp_baseline.py` + `medusa_heads.py`, random-init linear extra head): 12,500 batches, head0 3.09, head1 8.10 → 5.84.
- Word groups: `word_group_boundaries.py` (rule-based, SHORT v0 lists; the expanded lists are in `validate_on_real_data.py`) + `boundary_alignment.py` (offset_mapping → token labels).
- Old scripts still flat in repo root; plan moves them to `legacy/` (OM-1).
- Next for me: JN-1 (heads module with zero-init resblock heads), JN-2 (structural-loss design note), JN-3 (`scripts/train.py`).

## Environment (Windows, PowerShell)
- venv: `.venv` in this folder → `.\.venv\Scripts\Activate.ps1`
- Always `$env:PYTHONUTF8 = "1"` before scripts that read Devanagari.
- Local GPU: 8 GB laptop RTX. Kaggle (T4/P100, no native bf16) for real runs, so never hard-code bfloat16.
- Checkpoints/logs are git-ignored (`checkpoints/`, `training_log*.txt`).

## How I like to work
- Terse answers. Verify against the actual code before suggesting changes.

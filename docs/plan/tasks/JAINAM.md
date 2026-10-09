# Jainam: Modeling & Training

> **How to use this file with Claude:** open Claude in the repo (it reads `CLAUDE.md`), or paste the *Context* block + `INTERFACES.md` into a chat, then name the task. Tasks marked **argue first** need a design discussion before code. Branch `jainam/<task>`, PR against `main`.

## Context (paste this to Claude)

We are building *Word-Group Guided Multi-Token Prediction for Hindi and Marathi*. Base: `LingoIITGN/ganga-1b` (Mistral, 1B, hidden 2048, vocab 30k, untied `lm_head`; tokenizer adds no BOS, ~1.12 tokens/word), LoRA r=8 on `q_proj`/`v_proj`, lr 2e-4, batch 8, max_len 128, grad clip 1.0, IndicCorpV2 `hin_Deva`.

The code is the `mtp/` package (see `docs/plan/INTERFACES.md`). I own the model (`mtp/model/heads.py`), every loss (`mtp/losses/`), `scripts/train.py`, the run configs (`configs/`), all training runs (on my Kaggle account, T4 x2, ~0.34 s/step), the `GroupAware` draft policy, and the method/results write-up. Jai provides groupers (`mtp/data/grouping/`); Om provides evaluation (`mtp/eval/`, `scripts/evaluate.py`) and the speculative-decoding engine.

What exists: `MTPModel` (k heads, `linear` or zero-init `resblock` sharing the frozen LM head, `head_backbone_grad`, optional boundary probes); `per_head_ce`; `StructuralLoss` (S2, S3_h0, S3_chain, S3_all, S23); `LossWeighting` (fixed, uncertainty, dwa); `train.py` (YAML, bit-exact resume, on-the-fly group labels with `hi_rules_v0`, in-group / boundary eval); configs R0-R10 + pilots.

Findings: NTP flat at 3.05 on eval_small. Groups span ~1.5 tokens: token t+d is in t's group 33/8/2.5/0.8% of the time for d=1..4. On the laptop MTP model, in-group targets are ~2x easier (head 1: 28% vs 14% top-1). Zero-init resblock heads with full gradient into the backbone make head 0 0.20 nats worse at 2k steps; `head_backbone_grad` 0.1 keeps it within 0.01 of NTP (`docs/runs.md`).

## Done
JN-1 heads · JN-4 baselines R0-R2 (Gate B passed, `mtp-run-R0/R1/R2`) · JN-2 design note · JN-3 train.py + configs · JN-5 structural-loss code · JN-7 weighting code · OM-1/2/3 infra + Kaggle notebook (now Om's to own) · grouping contract code + `hi_rules_v0` · cleanup (`legacy/`).

## Tasks

### JN-4 · Finish the baselines · P0 · S · done 2026-10-09
- Read the overnight Kaggle output (cell 4 / `train_*.log`). Fill `docs/runs.md` (commit, final eval per head).
- **Gate B:** R0 and R1 within ~0.05 of the laptop numbers on eval_small (3.05 / 3.09 / 5.84). Known differences to keep in mind if it misses: Kaggle trains on non-blank rows from raw row 1,000 (laptop: raw rows 100+, half of them blank), fp16 autocast vs bf16.
- Publish each run folder as a Kaggle Dataset `mtp-run-R0` / `-R1` / `-R2` and post the names (handoff D1 → Om).
- Done when: runs.md has the three rows, gate verdict written, datasets posted.

### JN-5 · Structural-loss pilots → R3 · P0 · S · argue first on the result
- One Kaggle session (notebook cell 3): `GPU_QUEUES = {0: ["pilot_R2ref", "pilot_S2", "pilot_S3"], 1: ["pilot_S3chain", "pilot_S3all", "pilot_S23"]}`. 2,000 steps each, eval on the full IndicCorp eval set (~500 sentences) every 250 steps; ~45 min total.
- Watch: at init the S3 KL is ~7-9 (λ_s3 = 0.5 makes it ~4 nats per head). If head 0 or head 1 is worse than `pilot_R2ref` at step 500, add a run with `--set losses.structural.lambda_s3=0.1`.
- Decide with the rule in `docs/design_structural_loss.md` §3: head-1 in-group top-1 and mean top-1 over heads 1-3 vs `pilot_R2ref`, guard head-0 loss ≤ ref + 0.02; S3 vs S3_all says whether the mask (the linguistics) matters. If S3_chain beats S3_h0, rerun `pilot_S23` with `--set losses.structural.s3_teacher=chain`.
- Write the numbers into the design note §5 and `docs/runs.md`; set `configs/R3.yaml` `variant:` to the winner.
- Done when: the pilot table is in the design note and R3.yaml has no `TBD`.

### JN-8a · R3 and its random control R6a · P0 · S
- R3: 12.5k steps (one GPU, ~75 min). Publish `mtp-run-R3`.
- R6a: copy R3.yaml to `configs/R6a.yaml` with `data.grouper: random` once Jai's `random` grouper is on `main` (D2). Same seed, same everything else. Publish.
- Done when: both are published and Om has evaluated them (review package item 2).

### JN-7b · Weighting pilots → R5, R6 · P1 · S · argue first
- On top of R3's losses, 2,000 steps each: `fixed`, `uncertainty` (fix_head0 true), `uncertainty` (fix_head0 false), `dwa`. Uncertainty's log-variances move slowly at lr 2e-4; try `weighting.lr: 1e-2`.
- Pick by the same rule as JN-5. Log the effective weights (`weight/*`); they are Om's figure (c).
- Fill R5.yaml (scheme, variant, `contrastive.enabled` false unless R4 helped) and R6.yaml (= R5, grouper random). Run both.

### JN-9 · GroupAware draft policy · P1 · M · argue first
- File `mtp/eval/group_aware.py`, class `GroupAware(tau=0.3, use_probes=True, grouper=None)`, contract in INTERFACES §11.
- Rule: walk heads d = 1..k-1; stop at the first head whose max prob < τ; also stop before a draft whose boundary probe says it starts a new word group (sigmoid(boundary_logits[d]) > 0.5); always allow at least 1 draft if head 1 is confident. Fallback without probes: run the grouper on the decoded context and draft only while the current word group is open.
- Unit tests with fake logits (no engine needed). Real numbers need Om's engine (D3): tune τ on 50 held-out FLORES prompts, report on the 200 test prompts against FixedK and ConfidenceCut, on R2 and R3/R5.

### JN-8b · Marathi R8, R9, R10 · P1 · M
- Base `smallstepai/Misal-1B-instruct-v0.1` (Llama). Before launching: tokenizer has offsets and pad, `get_output_embeddings()` works through PEFT, LoRA targets are `q_proj`/`v_proj`, one 100-step local run.
- R8 and R9 need nothing from Jai (eval in-group split is skipped until `mr_rules_v1` exists); R10 needs `mr_rules_v1` (JI-7).
- Fallback if Misal misbehaves: ganga-1b, reported as cross-lingual transfer.

### JN-10 · α ablation (C4) · P2 · S
- R2 at full length with `--set head_backbone_grad=0.0` and `=1.0` (run names `A1_alpha0`, `A1_alpha1`). Table: head-0 loss, per-head top-1, speed-up (from Om).

### JN-6 · Contrastive loss → R4 · P2 · M · argue first
- `mtp/losses/contrastive.py`: SupCon over tokens; project `hidden` with a 2-layer MLP to 128-d, L2-normalise; positives = same (sentence, group_id); ≤ 512 tokens per batch; temperature 0.1. Returns `{"contrastive/supcon": ...}`.
- Low priority because groups are ~1.5 tokens (few positives per anchor) and pulling token representations together can hurt head 0. Only if Phase 1-2 are done. R4 = R3 + contrastive.

### JN-11 · Extras · P2
k ∈ {2, 3, 6} on R3; second seed for R2 and R3; R7 (other grouper, once JI-5 picks).

### Write-up
Method (heads, α, structural losses + pilot comparison, weighting, GroupAware); results (main table, R3/R5 vs R6a/R6, S3 vs S3_all, in-group vs boundary, decoding, Marathi vs Hindi, α); limitations (flat NTP baseline, single seed, rule-based groups, short groups in tokens).

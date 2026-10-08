# Om: Evaluation & Infrastructure

> **How to use this file with Claude:** paste the "Context" block below, then `INTERFACES.md`, then the one task you're on (e.g. "Do OM-4"). Ask for the module **and** its pytest. Run the test, then open a PR on an `om/<task>` branch.

## Context (paste this to Claude)

We are building *Word-Group Guided Multi-Token Prediction for Hindi and Marathi*. A 1B Hindi LM (`LingoIITGN/ganga-1b`, vocab 30k) is fine-tuned with PEFT LoRA (r=8, `q_proj`/`v_proj`) plus extra "Medusa-style" heads. Head i predicts the token i+1 positions ahead, and all heads read the same last hidden state. Training runs on Kaggle (T4/P100 GPUs, 16 GB, sessions that end after a time limit). My job is to make the codebase a clean, reproducible package, make it run on Kaggle with resume, and build the full evaluation suite: per-head perplexity/accuracy, self-speculative decoding speedup, and the tables and figures for the paper.

Existing code (flat in the repo root, to be moved to `legacy/`): `train_ntp_baseline_final.py`, `train_mtp_baseline.py`, `medusa_heads.py`, `word_group_boundaries.py`, `boundary_alignment.py`, `validate_on_real_data.py`, `check_datasets.py`, `test_load_model.py`, `test_lora_training.py`. The training scripts share a `log()` helper that appends to a txt file, a `batchify()` that pads with `tokenizer.pad_token = eos_token`, and a per-head loss that masks padding with -100. Models load with `dtype=torch.bfloat16`, which must become configurable because Kaggle T4/P100 have no native bf16.

---

> **Priority:** OM-1 → OM-2 → OM-3 come first. Together they are handoffs H1 and H2, which training on Kaggle waits on. Post in the team chat as each one merges.

## OM-1 · Package skeleton + shared utilities
**Files:** `mtp/__init__.py` and all sub-package `__init__.py`, `mtp/config.py`, `mtp/device.py`, `mtp/utils/logging.py`, `mtp/data/corpus.py`, `mtp/data/collate.py`, `requirements.txt`, `pyproject.toml` (so `pip install -e .` works), `README.md`, `legacy/`
- `git mv` every existing root script into `legacy/` unchanged. Keep `.gitignore`, and add `runs/`, `boundary_cache/`, `wandb/`.
- `config.py`: `RunConfig` dataclass with nested dataclasses matching INTERFACES §6. Provide `load_config(path)`, `save_config(cfg, path)`, and CLI overrides like `--set optim.lr=1e-4`.
- `device.py`: `pick_device()`, `pick_dtype(cfg.dtype)`. For `auto`: bf16 if `torch.cuda.is_bf16_supported()`, else load weights in fp32 and train under `torch.autocast(fp16)` with `GradScaler`. Expose a context manager `autocast_ctx(cfg)` and `make_scaler(cfg)` so `train.py` and `evaluate.py` never branch on dtype themselves.
- `logging.py`: `MetricLogger(run_dir)` with `.log(step, split, name, value)` → JSONL (INTERFACES §9), plus a pretty console line. Optional W&B mirror if `WANDB_API_KEY` is set.
- `corpus.py`: `load_split(lang, split, n)` implementing the fixed splits in INTERFACES §4 (IndicCorp rows 0–999 = eval, ≥1000 = train, FLORES devtest), with `eval_small` = first 100.
- `collate.py`: `Collator(pad_id)` per INTERFACES §3. It must also work when `group_*` columns are absent (plain NTP runs).
- `requirements.txt`: pin `torch`, `transformers`, `peft`, `datasets`, `pyyaml`, `pytest`. Match the laptop venv versions (`pip freeze` there).
- Tests: config round-trip, collate padding values, `pick_dtype` under mocked `is_bf16_supported`.

## OM-2 · Checkpointing + resume
**File:** `mtp/model/checkpoint.py` (INTERFACES §8)
- `save_run`: PEFT `save_pretrained` for LoRA, `state_dict` for extra heads and the loss-weighting module, optimizer + scaler + RNG states for resume, and `config.yaml` once.
- `load_run`: rebuild base model → apply LoRA → wrap `MTPModel` with `num_heads`/`head_type` from config → load heads. Use a tiny stand-in for `MTPModel` until Jainam's lands, since only the signature matters.
- `latest_step(run_dir)`, and keep only the last 2 step folders + any folder listed in `keep_steps`.
- Test with a tiny random `LlamaForCausalLM` config (2 layers, hidden 64, vocab 100) so it runs on CPU in seconds.

## OM-3 · Kaggle training notebook
**File:** `notebooks/kaggle_train.ipynb`
- Cell 1: `git clone` the repo at a given branch/commit and `pip install -e .`. Read HF token and (optional) W&B key from **Kaggle Secrets**, never hard-coded.
- Cell 2: attach Kaggle Dataset `mtp-boundary-cache` (from Jai) and set `data.cache_dir`.
- Cell 3: `!python scripts/train.py --config configs/<RUN>.yaml --resume auto`, writing to `/kaggle/working/runs/<run_name>`.
- Resume across sessions: at the end of a session, save `runs/<run_name>` as a Kaggle Dataset version (`kaggle datasets version`) and restore it at the start of the next. Document the 3-click process in `docs/kaggle.md`.
- Print GPU type, dtype chosen, and estimated steps/hour at start.

## OM-4 · Evaluation: perplexity + per-head accuracy
**Files:** `mtp/eval/perplexity.py`, `mtp/eval/head_accuracy.py`
- For each head i: mean shifted CE (same masking as `compute_head_loss` in `train_mtp_baseline.py`), perplexity, top-1 and top-5 accuracy.
- **Split by structure** (needs the boundary cache columns): `top1_in_group` = targets whose `group_id` equals the source token's `group_id`, `top1_at_boundary` = the rest. Do the same for loss.
- `--dump_tokens` writes `results/{run}/tokens_{dataset}.jsonl` with `sent_id, text, token_idx, token, head, correct, group_start, group_id` (Jai uses this for error analysis).
- Datasets: `indiccorp_eval` (1,000), `flores_hi`, later `flores_mr`.
- Test: tiny model, hand-built batch where you know the right answer.

## OM-5 · Blind double annotation
- Jai gives you 50 sentence IDs from the Hindi gold set (text only). Annotate them yourself with `scripts/annotate.py` following `docs/annotation_guide.md`, **without** looking at Jai's groups. Save as `data/gold/hi_gold_om50.jsonl`. This gives the inter-annotator agreement number in the paper.

## OM-6 · Self-speculative decoding engine
**File:** `mtp/eval/spec_decode.py` (+ `FixedK` and `ConfidenceCut` in `mtp/eval/draft_policy.py`, contract in INTERFACES §11)
- Loop: run the model on the current sequence → heads 1..k-1 at the last position propose up to k-1 future tokens (greedy argmax) → `policy.num_draft_tokens(...)` decides how many to keep → one forward pass over `context + drafts` → head 0 verifies left to right, accepting while head 0's argmax equals the draft. Always emit at least 1 token (head 0's own prediction at the first mismatch).
- Use the KV cache (`past_key_values`) and roll it back on rejection. Get the no-cache version correct first, then optimise.
- Stats: tokens generated, forward passes, mean accepted length, per-head acceptance rate, tokens/sec, and the same for plain greedy on the same prompts → speedup.
- **Correctness test (must pass):** for 20 prompts, spec-decode output ids == plain greedy output ids, exactly.
- Group Integrity: given a grouper, the fraction of accepted multi-token spans that end on a group boundary.
- Prompts: first 8–16 words of 200 FLORES devtest sentences, `max_new_tokens=64`, batch size 1, timed with `torch.cuda.synchronize()`.

## OM-7 · Eval entry point + Kaggle eval notebook
**Files:** `scripts/evaluate.py`, `notebooks/kaggle_eval.ipynb`
- `python scripts/evaluate.py --run_dir <...> --datasets indiccorp_eval flores_hi --spec_decode --policies fixed_k confidence_cut group_aware`
- Writes `results/{run_name}/eval_{dataset}.json` exactly per INTERFACES §10. Commit these JSONs.
- The notebook attaches a run's Kaggle Dataset (checkpoints) + `mtp-boundary-cache` and runs the script. It runs on your own Kaggle GPU quota.

## OM-8 · Evaluate every run
- As each run R0–R10 finishes, Jainam posts its run Dataset name. Run OM-7 on it and push the JSON. Keep `results/STATUS.md` with a checklist of which runs are evaluated on which datasets.
- Sanity-flag anything odd, e.g. head 0 of an MTP run much worse than R0, or speedup < 1.

## OM-9 · Tables + figures
**File:** `scripts/make_tables.py` → `results/tables/*.md` + `*.tex`, `results/figures/*.pdf`
- **Table 1:** main results. Rows R0–R7 (hi), columns: head0 ppl, per-head top-1 (h1–h3), in-group top-1 (h1–h3), mean accepted length, speedup.
- **Table 2:** Marathi (R8–R10). **Table 3:** grouper ablation (R5 vs R6 vs R7). **Table 4:** draft policies (FixedK / ConfidenceCut / GroupAware) on R2 and R5.
- **Figures:** (a) eval loss per head vs step for R2 and R5 (from `metrics.jsonl`); (b) per-head accuracy, in-group vs at-boundary, grouped bars; (c) learned loss weights over training for R5; (d) **lookahead heatmap**: one sentence, each token coloured by the smallest head index that predicted it correctly, with word-group brackets drawn above the tokens. (d) is the paper's headline figure, so make it look good: a Devanagari font (Noto Sans Devanagari), colour-blind-safe palette, and vector PDF.

## Write-up (Phase E)
- **Experimental setup:** hardware, dtype handling, hyper-parameters (pull from configs automatically), eval sets, decoding setup.
- **Evaluation section:** metric definitions (per-head accuracy, in-group split, acceptance length, Group Integrity), all tables and figures, appendix with full per-run numbers.

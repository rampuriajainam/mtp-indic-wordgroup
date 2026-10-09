# Om: Evaluation & Infrastructure

> **How to use this file with Claude:** open Claude in the repo (it reads `CLAUDE.md`), or paste the *Context* block + `INTERFACES.md` into a chat, then name the task (e.g. "Do OM-4"). Ask for the module **and** its CPU-only pytest. Branch `om/<task>`, PR against `main`.

## Context (paste this to Claude)

**Scheduling authority:** use the uploaded PDF's Phases A–C now and D+E later, and the H1–H10 table in `docs/plan/README.md`. Existing implementation details below are preserved. No P0/P1/P2 label overrides the PDF phase order.

**Phase A:** OM-1 package → OM-2 checkpoints → OM-3 Kaggle notebook (H1/H2) already have implementations; OM-0 reviews them. OM-4/5 are Phase B, OM-6/7 Phase C; OM-8/9 and write-up are later. H3 supplies shared Hindi caches even though current labelling works on the fly.

We are building *Word-Group Guided Multi-Token Prediction for Hindi and Marathi*. A 1B Hindi LM (`LingoIITGN/ganga-1b`, vocab 30k, tokenizer adds no BOS) is fine-tuned with LoRA plus extra prediction heads: head d predicts token t+d+1 from the same last hidden state (`MTPModel`, INTERFACES §5). Word groups (multi-word units like जा रहा था) come from Jai's groupers; Jainam trains the runs on Kaggle and publishes each as a Kaggle Dataset `mtp-run-<ID>`. My job: the evaluation suite (per-head loss/accuracy split by word-group structure, self-speculative decoding with speed-up), evaluating every run, the tables and figures, and owning the infrastructure modules.

Already on `main` (infra I now own, written by Jainam to unblock training):
- `mtp/config.py` (YAML namespace, `cfg_get`, `apply_overrides`), `mtp/device.py` (bf16 only on native-bf16 GPUs; T4 → fp32 weights + fp16 autocast), `mtp/utils/logging.py`.
- `mtp/data/corpus.py` (`load_split`), `mtp/data/collate.py` (`Collator`).
- `mtp/model/build.py` (`build_model`), `mtp/model/checkpoint.py` (`save_run`, `latest_step`, `load_weights`, **`load_run(run_dir, device, step)`** → `(model, tokenizer, cfg)` for evaluation).
- `mtp/data/grouping/` (`get_grouper("hi_rules_v0")`, `label_batch(texts, tokenizer, grouper)` → token labels with `group_start` / `group_id`), so the in-group split works today with no cache.
- `scripts/train.py` has a reference `evaluate()` (token-weighted per-head CE + top-1, in-group / at-boundary split). The official evaluation must agree with it.
- `notebooks/kaggle_train.ipynb`; `tests/conftest.py` with a tiny tokenizer and tiny model; `tests/test_infra.py` shows build → train → `load_run` on CPU.

Kaggle notes: T4 x2; `pip uninstall -y torchao` before installing (`requirements.txt`); the repo is public (plain `git clone`). Windows: `$env:PYTHONUTF8 = "1"`.

## Tasks

### OM-0 · Take over the infra · Phase A (review existing OM-1–OM-3) · S
- Read `mtp/config.py`, `device.py`, `utils/logging.py`, `data/corpus.py`, `data/collate.py`, `model/build.py`, `model/checkpoint.py`, `notebooks/kaggle_train.ipynb`; run `python -m pytest -q`.
- Fix or open an issue for anything wrong. From now on, changes to these go through you.

### OM-4 · Per-head evaluation · Phase B · M
- `mtp/eval/head_accuracy.py` (+ `perplexity.py` if you prefer to split):
  ```python
  evaluate_heads(model, examples, collator, cfg, device, top_k=(1, 5), dump_path=None, texts=None) -> list[dict]
  ```
  `examples` come from `label_batch(texts, tokenizer, get_grouper(name))`. Returns one dict per head with every field of INTERFACES §10 `per_head`: loss, ppl, top1, top5, n; and top1 / loss split in-group vs at-boundary, with n per split.
- `dump_path` writes the per-token dump (§10) that Jai uses for error analysis.
- Datasets: `indiccorp_eval`, `indiccorp_eval_small`, `flores_hi`, later `flores_mr` (`load_split`).
- Tests:
  - a hand-built batch on the tiny model where you know the answer (e.g. force logits);
  - **agreement:** on a tiny trained run, `evaluate_heads` equals `scripts/train.py`'s `evaluate` to 1e-3.

### OM-5 · Blind double annotation · Phase B · S
Jai gives you 50 gold sentence IDs (text only). Annotate them with `scripts/annotate.py` following `docs/annotation_guide.md` without looking at Jai's groups → `data/gold/hi_gold_om50.jsonl`. This is the agreement κ in the paper.

### OM-6 · Self-speculative decoding engine · Phase C · L
- `mtp/eval/spec_decode.py` with `generate(model, tokenizer, prompt_ids, max_new_tokens, policy, grouper=None) -> (output_ids, stats)`. `mtp/eval/draft_policy.py` holds `DraftPolicy`, `FixedK`, `ConfidenceCut(tau)` and `POLICIES`; import `GroupAware` from `mtp/eval/group_aware.py` inside a `try`, since Jainam writes it (INTERFACES §11).
- Loop:
  1. Heads 1..k-1 at the last position propose k-1 tokens (greedy argmax).
  2. `policy.num_draft_tokens(...)` keeps n of them.
  3. One forward pass over context + n drafts.
  4. Head 0 verifies left to right, accepting while its argmax equals the draft.
  5. Always emit at least one token (head 0's prediction at the first mismatch).
- Build `step_state` with `boundary_logits` from `aux` when the model has probes.
- Get the no-cache version right first. Then add the KV cache: `model(..., use_cache=True)` puts `past_key_values` in `aux`; roll back rejected positions with `DynamicCache.crop`.
- Stats: tokens generated, forward passes, mean accepted length, per-head acceptance rate, tokens/s, the same for plain greedy → speed-up. Plus **Group Integrity**: the share of accepted multi-token spans that end on a word-group boundary (run the grouper on the decoded text).
- **Correctness test (must pass):** on the tiny model, for 20 random prompts and every policy (including a policy that proposes garbage), output ids == plain greedy ids exactly.
- Prompts for real runs: the first 8-16 words of 200 FLORES devtest sentences, `max_new_tokens=64`, batch size 1, timed with `torch.cuda.synchronize()`.

### OM-7 · `scripts/evaluate.py` + Kaggle eval notebook · Phase C · M
- `python scripts/evaluate.py --run_dir <...> [--step N] --datasets indiccorp_eval flores_hi --grouper hi_rules_v0 --spec_decode --policies fixed_k confidence_cut group_aware [--dump_tokens]`
- Writes `results/{run_name}/eval_{dataset}.json` exactly per INTERFACES §10 (commit these).
- `notebooks/kaggle_eval.ipynb`: attach the run's Kaggle Dataset, `git clone`, uninstall torchao, install, run the script, show the JSON. Runs on your Kaggle quota.
- Until H10 lands, test end to end on a tiny local run (see `tests/test_infra.py`). The archive run notes report R0/R1/R2 published as `mtp-run-R0/R1/R2`.

### OM-8 · Evaluate every run · Phase D+E · ongoing
- As Jainam posts each `mtp-run-<ID>`, run OM-7 on it and push the JSON. Keep `results/STATUS.md`: a checklist of runs × datasets × policies.
- Sanity-flag anything odd: head 0 of an MTP run much worse than R0, speed-up < 1, `outputs_match_greedy` false.

### OM-9 · Tables + figures · Phase D+E · M
- `scripts/make_tables.py` → `results/tables/*.md` + `*.tex`, `results/figures/*.pdf`.
- **Table 1:** main results R0-R7 (hi). Columns: head-0 ppl, per-head top-1 (h1-h3), in-group top-1 (h1-h3), mean accepted length, speed-up.
- **Table 2:** Marathi R8-R10.
- **Table 3:** controls: R3 vs R6a, R5 vs R6, pilot S3 vs S3_all, grouper ablation R7.
- **Table 4:** draft policies (FixedK / ConfidenceCut / GroupAware) on R2 and R3/R5.
- **Table 5:** α ablation (C4).
- **Figures:**
  - (a) eval loss per head vs step, R2 vs R3/R5 (from `metrics.jsonl`);
  - (b) per-head top-1, in-group vs at-boundary, grouped bars;
  - (c) learned loss weights over training (R5);
  - (d) **lookahead heatmap**, the headline figure: one sentence, each token coloured by the smallest head that predicted it correctly, word-group brackets above. Use a Devanagari font (Noto Sans Devanagari), a colour-blind-safe palette, and vector PDF.
- Hyper-parameters in the setup table are read from the run configs automatically.

### OM-10 · Decoding by group type · Phase later optional · S
Accepted length and Group Integrity broken down by the group type the draft starts in (`grouper.group_types`). Shows where structure-aware drafting helps.

### Write-up · Phase E (later)
Experimental setup (hardware, dtype policy, hyper-parameters from configs, eval sets, decoding setup); evaluation section (metric definitions: per-head accuracy, in-group split, acceptance length, Group Integrity; all tables and figures; appendix with per-run numbers).

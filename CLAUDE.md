# Project context for Claude

**Project:** Word-Group Guided Multi-Token Prediction for Hindi and Marathi (see `README.md`).
**Team:** Jainam (modeling & training), Jai (data & linguistics), Om (evaluation & infrastructure). Several people use Claude on this repo: if you don't know who you are working with, check `git config user.name` or ask, then read that person's task file.

## Read first
- `docs/plan/README.md`: status, PDF phases A–C (priority) / D+E (later), H1–H10 handoffs, critical path, run matrix.
- `docs/plan/INTERFACES.md`: every function signature and file format. Code must match it; if a contract has to change, change INTERFACES.md in the same PR and say so in the PR title.
- `docs/plan/tasks/{JAINAM,JAI,OM}.md`: one file per person, each with a Context block and task specs.

## Facts that are easy to get wrong
- `mtp/` must not download anything at import time; load inside functions.
- ganga-1b tokenizer: SentencePiece-like, **no BOS/EOS added**, pad id 0, ~1.12 tokens/word. Offsets of `▁word` tokens include the leading space (`align.py` handles it).
- IndicCorpV2 `hin_Deva`: every other row is blank. Splits are on raw row numbers, blanks dropped (`mtp/data/corpus.py`): eval = rows 0-999 (~500 sentences), eval_small = rows 0-99 (~50), train = rows >= 1000.
- Never hard-code `torch.bfloat16`. `mtp.device`: bf16 only on GPUs with *native* bf16 (sm >= 8). `torch.cuda.is_bf16_supported()` is True on Kaggle T4 via emulation and trains 6x slower.
- On Kaggle: `pip uninstall -y torchao` first (peft 0.21 refuses LoRA with Kaggle's torchao 0.10).
- MTP heads: zero-init resblocks share the frozen `lm_head`; `head_backbone_grad: 0.1` keeps head 0 (the verifier) at NTP quality.

## Environment
- Windows + PowerShell on the laptops: `.venv`, `$env:PYTHONUTF8 = "1"`. Kaggle T4 x2 for real runs (`notebooks/kaggle_train.ipynb`).
- `python -m pytest -q` (CPU-only, tiny models) must pass before every PR.
- Branches `jainam/*`, `jai/*`, `om/*`, each PR against `main` (don't stack PRs). Checkpoints, caches and logs never go in git; `results/*.json` and figures do.

## How the team likes to work
- Terse answers. Verify against the actual code before suggesting changes.
- For design tasks marked "argue first" in a task file, discuss the design before writing code.

## Plan reconciliation for this handoff
- Use the supplied PDF for phase order; preserve implemented signatures in INTERFACES. First Jai: JI-1 → JI-6a (H3). First Om: OM-1 → OM-2 → OM-3 (H1/H2), already implemented.
- `hi_rules_v1` is registered and implemented. Switch a run explicitly with `--set data.grouper=hi_rules_v1`; existing experiment configs remain unchanged for comparability.
- `legacy/validate_on_real_data.py` saves expanded lists but is only a demo, not a real-data validator. Its compound set is unused. Do not infer coverage numbers from it.
- See `docs/JI-1_HANDOFF.md` for rule choices, tests, and the completed corpus measurement. Do not claim JI-6a or H3 is complete from the grouper alone.

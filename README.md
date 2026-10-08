# Word-Group Guided Multi-Token Prediction for Hindi and Marathi

Multi-token prediction (MTP) heads let a language model draft several future tokens at once and verify them in one pass (self-speculative decoding), so generation gets faster with **exactly the same output**. The far-ahead heads are the weak link: they guess wrong, drafts get rejected, and the speed-up disappears.

Hindi and Marathi pack one unit of meaning into several words (जा रहा था "was going", घर से "from home", के बारे में "about"). **Our hypothesis:** if the prediction heads know where these *word groups* begin and end, they predict better inside a group, drafts get accepted for longer, and decoding gets faster.

Team: **Jainam** (modeling & training) · **Jai** (data & linguistics) · **Om** (evaluation & infrastructure).
Base model: [`LingoIITGN/ganga-1b`](https://huggingface.co/LingoIITGN/ganga-1b) (Hindi, 1B) with LoRA; Marathi: `smallstepai/Misal-1B-instruct-v0.1`. Data: IndicCorpV2, FLORES-200.

## What is new here

1. **Structure-aware MTP losses.** Per-head word-group boundary probes (S2) and in-group self-distillation from the LM head (S3), with two controls that isolate the linguistic part: random boundaries with the same length distribution (R6), and the same distillation without the group mask (S3-all).
2. **Group-aware speculative decoding.** Draft until the predicted end of the current word group, plus a *Group Integrity* metric (do accepted drafts end on group boundaries?).
3. **How far does structure reach?** With ganga's tokenizer (1.12 tokens/word), a word group spans ~1.5 tokens, so the next token is in the same group 33-38% of the time but the 4th-next only ~1%. We measure this per lookahead, per grouper and per tokenizer, and compare analytic Hindi with agglutinative Marathi.
4. **Shared-LM-head MTP heads hurt the verifier.** Zero-init residual heads that reuse the frozen LM head pull the backbone away from next-token prediction (+0.20 nats at 2k steps). Scaling their gradient into the backbone to 0.1 keeps the LM within 0.01 of the NTP baseline.
5. **Where does the LM encode word groups?** Layer-wise probing of ganga-1b for group boundaries.
6. **Resources:** Hindi (200) and Marathi (150) gold word-group annotations with inter-annotator agreement; rule and parser-based groupers.

Plan, status and who does what: [`docs/plan/README.md`](docs/plan/README.md). Interfaces: [`docs/plan/INTERFACES.md`](docs/plan/INTERFACES.md). Results so far: [`docs/runs.md`](docs/runs.md).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .\.venv\Scripts\Activate.ps1
pip install torch                                       # the CUDA build for your machine
pip install -r requirements.txt && pip install -e .
python -m pytest -q                                     # CPU-only, ~30 s
# Windows: $env:PYTHONUTF8 = "1" before anything that reads Devanagari
python scripts/train.py --config configs/R2.yaml --set optim.max_steps=100     # short local run
```

Training on Kaggle: [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) (T4 x2, ~0.34 s/step, a 12.5k-step run takes ~75 min).

## Repository

```
mtp/                 the package (import mtp)
  config.py device.py           YAML configs; bf16 vs fp16/fp32 policy
  data/                         corpus splits, collator, grouping/ (groupers + token alignment)
  model/                        MTPModel heads, build, checkpoints (save_run / load_run)
  losses/                       per-head CE, structural (S2/S3), weighting (fixed/uncertainty/dwa)
  eval/                         evaluation + speculative decoding (Om)
scripts/train.py     the one training entry point
configs/             one YAML per run (R0-R10, pilots)
docs/                plan, interface contracts, design notes, run log
tests/               pytest, CPU-only
legacy/              the original laptop scripts, kept for reference
```

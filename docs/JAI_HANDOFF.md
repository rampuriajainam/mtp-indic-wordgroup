# Jai implementation handoff against new.zip

This package builds on the **newly uploaded code**, not the previous project snapshot. The latest teammate sync controls priority: JI-1 → start JI-2 → JI-8 → JI-3/JI-5 → JI-10 → JI-7. Marathi was also implemented here because it is the only grouper dependency blocking R10. Supporting JI-6 and JI-9 tools are included.

## Completion status

| Task | What is delivered | What still needs doing |
|---|---|---|
| JI-1 | Registered `hi_rules_v1`, expanded explicit lists, longest compound matching, documented light-verb/negation decisions, tests; measured on 5,000 real Hindi train sentences | Human boundary-quality assessment in JI-5 |
| JI-2 | Guide, deterministic source selection, 200 Hindi IDs, unreviewed rule suggestions, annotation/resume CLI, **50 text-only Hindi IDs for Om** | Jai's 200 human annotations and Om's independent 50; no gold or κ is fabricated |
| JI-8 | Full-text measurements on 5,000 Hindi sentences × 4 groupers × all 5 requested tokenizers; Marathi on 5,000 × 3 groupers × 2 tokenizers; figures and top-30 groups | Extend Marathi to other tokenizers if wanted |
| JI-3 | Lazy Trankit and Stanza adapters, sentence-local dependency mapping, MWT offset handling, contiguous-only grouping, CPU tests | Trankit installation/model validation; Stanza is the separately named working alternative |
| JI-5 | Attachment P/R/F1, exact groups, per-type scores, κ, scoring CLI, draft/partial-gold safeguards | Review gold, obtain Om's annotations, score all chosen groupers and choose the final winner |
| JI-10 | Frozen causal-model layer probes, shifted targets, streaming CPU logistic regression, disjoint train/eval guards; **one shared model forward per sentence across all label sets**, tiny-model verification | Run the full 2,000-sentence frozen ganga study; no production-model probe F1 is claimed |
| JI-7 | Registered `mr_rules_v1`, suffix-preserving Marathi rules, tests, 150 source-selected Marathi sentences/drafts, actual reach statistics and v1 histogram | Human Marathi annotations and parser-vs-gold comparison |
| JI-6 | Boundary cache builder/loader, metadata validation, CLI; tested on tiny fixtures and a 32-row real Hindi cache | Build production-size caches if needed; publish to Kaggle yourself |
| JI-9 | Token-dump analysis by target group type and group position, improved/worse examples, comparison guards | Run on actual matched R2/R3/R5/R6 token dumps |

`data/gold/*_gold.jsonl` is intentionally not populated with automatic rule labels. Those files are created by `annotate.py` as the humans annotate. Draft files are under `data/annotation/`, explicitly labelled `unreviewed_rule_draft`. The scorer refuses them by default; `--allow-unreviewed` is a smoke-test mode and never recommends a final grouper. It also refuses to select a final winner from a partially completed human set.

## Collision avoidance and random refit

The uploaded teammate's `random_grouper.py`, `test_random_grouper.py`, and `configs/R6a.yaml` are **byte-for-byte unchanged**. The existing `random` registry line is retained. New registry entries are additive. No training, model, speculative-decoding, run config or existing evaluation implementation was overwritten.

The v1 length histograms were remeasured from 5,000 real training sentences per language and saved with sample hashes in `mtp/data/grouping/histograms_v1.json`. New opt-in names `random_hi_v1` / `random_mr_v1` read them. The original `random` name retains the teammate's v0 defaults, so R6a stays reproducible.

Jainam can opt into a matched final-method comparison without changing old configs:

```bash
python scripts/train.py --config configs/R5.yaml --set data.grouper=hi_rules_v1
python scripts/train.py --config configs/R6.yaml --set data.grouper=random_hi_v1
```

These are **future run examples**, not runs launched by this handoff. The existing R5/R6 configs still need Jainam's final loss/weight choices. `configs/R10.yaml` already requests `mr_rules_v1`; that registry dependency now resolves. Marathi rule quality remains to be assessed.

## Immediate annotation handoff

Send Om **only** `data/annotation/hi_om50_text_only.jsonl`. Each row has just `id` and `text`; it contains 25 FLORES and 25 IndicCorp sentences. It was prepared but **not sent**. Both annotators can start independently now:

```bash
python scripts/annotate.py --input data/annotation/hi_sentences.jsonl --out data/gold/hi_gold.jsonl --annotator jai
python scripts/annotate.py --input data/annotation/hi_om50_text_only.jsonl --out data/gold/hi_gold_om50.jsonl --annotator om --blind
python scripts/annotate.py --input data/annotation/mr_sentences.jsonl --out data/gold/mr_gold.jsonl --annotator jai
```

Gold samples are 6–25 whitespace words, unique by text and ID. Hindi: 100 FLORES devtest + 100 IndicCorp training raw rows 1,000–5,000. Marathi: 75+75. Seed 42; source row IDs retained in the main packets. The blind packet contains no groups, types or draft hints. All CLI displays are text-only, even without `--blind`.

Read `docs/annotation_guide.md` before annotating. Human negation and वाला subtypes use `other` plus a note; rule types use their attaching category. This type difference is documented; attachment metrics are independent of it.

## Measurements completed here

Hindi v1 vs v0 on the identical 5,000 IndicCorp training sentences (280,928 whitespace words):

| Metric | v0 | v1 |
|---|---:|---:|
| Attached words (%) | 24.722 | 29.080 |
| Words/group | 1.3284 | 1.4100 |
| Tokens/observed group, max length 128 | 1.4841 | 1.5741 |

The fuller C3 study does **not** truncate, so its Hindi v1 ganga tokens/group is **1.5817** rather than 1.5741. Full-text figures, raw pair denominators, sample hashes and top groups are saved in `stats_hi.json` and `stats_mr.json`. Pair counts never cross sentence boundaries; special tokens are excluded.

Hindi v1 same-group rate by lookahead 1–4:

| Tokenizer | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| ganga-1b | 37.37% | 10.47% | 3.05% | 0.96% |
| Misal-1B | 61.85% | 33.32% | 16.27% | 6.95% |
| mT5-small | 65.77% | 34.71% | 15.44% | 6.08% |
| BLOOM-560m | 47.16% | 19.07% | 6.77% | 1.98% |
| XLM-R | 51.42% | 23.83% | 9.49% | 3.10% |

Marathi v1 averages **1.0569 words/group**, **1.6399 tokens/group with Misal**, and **2.5947 with ganga**, on its own 5,000-sentence corpus sample. These are heuristic grouper statistics, not evidence of boundary accuracy or improved training. Coverage alone is not a grouper-quality criterion. Similar reach in the random control is also not an R5/R6 training result.

## Reproduce or extend

Install the team's requirements and the optional script dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-jai.txt
python -m pytest -q
```

Windows: set `$env:PYTHONUTF8 = "1"`. A platform torch build is still chosen separately, as in the original project.

```bash
python scripts/ji1_coverage.py --corpus --n 5000 --tokenizer LingoIITGN/ganga-1b --out results/grouping/ji1_coverage.json
python scripts/refit_random.py --lang hi --n 5000
python scripts/refit_random.py --lang mr --n 5000
python scripts/group_stats.py --lang hi --n 5000
python scripts/group_stats.py --lang mr --n 5000 --tokenizers smallstepai/Misal-1B-instruct-v0.1 LingoIITGN/ganga-1b
python scripts/compare_group_lengths.py
```

After **human annotation** (use `stanza` as the explicit backend here; use `trankit` in its compatible environment):

```bash
python scripts/score_groupers.py --lang hi --gold data/gold/hi_gold.jsonl --agreement data/gold/hi_gold_om50.jsonl --groupers hi_rules_v0 hi_rules_v1 stanza random_hi_v1
python scripts/score_groupers.py --lang mr --gold data/gold/mr_gold.jsonl --groupers mr_rules_v1 stanza random_mr_v1
```

Full C5 study (frozen-model forwards on GPU if available, CPU probe updates; no LoRA, LM backward pass or GPU model training):

```bash
python scripts/probe_layers.py --lang hi --n 2000 --device auto
```

Each token's feature predicts the **next token's** boundary. BOS/EOS and padding are ignored. Exact duplicate eval texts are excluded from train. Reports include actual model layer count, counts, hashes, majority baseline and the one-epoch SGD methodology. Compare class prevalences/baselines as well as raw F1 across label definitions; do not infer a preferred layer before running the full study.

Optional caches (stored outside git, for example on Kaggle):

```bash
python scripts/build_boundary_cache.py --lang hi --grouper hi_rules_v1 --split train --n 200000 --out /kaggle/working/boundary_cache
python scripts/build_boundary_cache.py --lang hi --grouper hi_rules_v1 --split eval --out /kaggle/working/boundary_cache
python scripts/build_boundary_cache.py --lang hi --grouper hi_rules_v1 --split flores --out /kaggle/working/boundary_cache
```

Run `python scripts/group_error_analysis.py --help` for JI-9's matched token-dump inputs.

## Data and parser compatibility

The installed modern HF `facebook/flores` loader required authentication. Annotation selection uses the **original official public FLORES-200 archive**, linked from Meta's original repository, not a newer benchmark substituted silently. `mtp/data/flores.py` reads the requested archive member without extracting arbitrary paths and supports a local archive via `MTP_FLORES_ARCHIVE` or its function argument. No download occurs at module import. The cache CLI also uses this original-archive route for FLORES. The teammate's corpus loader is unchanged.

Stanza's Hindi package has no MWT model, while Marathi adds the MWT processor automatically. The adapter requests tokenize/pos/lemma/depparse and lets Stanza resolve language-specific dependencies. Trankit and Stanza follow the specified UD attachment relation list; they do not expand that list just to make examples look better. Some parser outputs miss desired vector-verb groups; gold scoring is needed. Non-contiguous dependency components are split; intervening unrelated words are never absorbed.

Primary API sources used for adapters:
- https://trankit.readthedocs.io/en/latest/posdep.html
- https://trankit.readthedocs.io/en/latest/overview.html
- https://stanfordnlp.github.io/stanza/data_objects.html
- https://stanfordnlp.github.io/stanza/pipeline.html

## Review and merge

Use `CHANGED_FILES.json` for task-sized PRs against current main, rather than replacing the whole checkout. The archive also contains a complete integrated tree for convenience. Pull main first, retain the teammate's random registry line, and add only the new entries. Existing experiment/checkpoint/evaluation results in the upload are preserved.

One existing evaluation test used `mr_rules_v1` as its deliberately unknown name; that test now uses `unregistered_test_grouper` because Marathi is registered. Its test purpose is unchanged. Extracted archives lack Git metadata, so verification used a local temporary commit only to exercise the existing git-commit provenance tests; it is not an original project revision and is not exported.

No messages were sent, no remote PR was opened, no training run was started, and no Kaggle dataset was published. Review/annotation and the full production probe/cache runs remain the team's next actions.

## Verification result

**287 CPU tests passed** on the final implementation. Both real Stanza language pipelines also processed three sentences successfully (`stanza_hi_smoke.json`, `stanza_mr_smoke.json`). A 32-row real Hindi cache passed schema/count/alignment round-trip checks. All five Hindi tokenizers and both Marathi tokenizers completed their 5,000-sentence studies; sidecar status files mark the runs complete. Figures were rendered and inspected. Tiny-model probing verifies shifted targets, no train/eval overlap, unchanged model weights, and shared forwards across labels.

Verification used Python 3.12 with torch 2.14.1+cpu, transformers 5.19.0, datasets 5.1.0, tokenizers 0.23.3, peft 0.21.2, pytest 9.1.1, sklearn 1.8.0, matplotlib 3.10.8 and Stanza 1.15.0. The team's original pinned requirements were not rewritten. `requirements-jai.txt` lists the additional tool dependencies.

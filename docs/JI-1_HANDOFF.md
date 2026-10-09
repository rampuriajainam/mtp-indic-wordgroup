# JI-1 handoff: Hindi rule grouper v1

Implemented and verified against the supplied code and team-plan PDF. No team messages were sent and no remote PR or experiment was launched. This package is ready for Jai/Jainam to review and merge.

## What the source review found

The attached archive contains a v2 plan and more implementation progress than the PDF snapshot: package infrastructure, resume, training, token alignment, head evaluation, structural losses and weighting already exist. Preserve that code. Per the requested reconciliation, phase order now follows the PDF: A–C first, D+E later. H1–H10 and the critical path are restored; JI-6 is split into JI-6a (Phase A, rules cache) and JI-6b (Phase B, Trankit/random).

The expanded lists **were saved** in `legacy/validate_on_real_data.py`: light verbs, जाता/करता, वाला, particles, and compound postpositions. However, its compound set is unused. It has no dataset loading, no assertions and no coverage calculation; it prints eight hand-picked examples. Historical 25% / 1.33 figures cannot be reproduced by running that file alone.

Legacy `check_datasets.py` uses `hin` / `train`; package `corpus.py` correctly uses `indiccorp_v2` / `hin_Deva`. Legacy alignment downloads a tokenizer at import; package `align.py` already avoids this by accepting a tokenizer. Legacy files remain untouched.

## Implementation

- `WORD_LISTS["v1"]` contains the PDF lists, contextual habitual forms recovered from the legacy expansion, and explicit verb bases for light-verb context.
- `HindiRuleGrouperV1`, alias `HindiRuleGrouper`, is registered as `hi_rules_v1`.
- Compound postpositions match longest first, before single-word attachment. The legacy `की वजह` construction is included with `की वजह से` to exercise overlapping windows.
- Lookup strips punctuation but output preserves every original word. Clause-ending punctuation blocks cross-clause attachment.
- `group_types` returns one contract label per group, with compound > light verb > auxiliary > postposition precedence.
- v0 behaviour, alignment interfaces, training code and experiment configs are preserved. No model/data downloads happen at grouping import.

## Linguistic decisions for Jai to review

| Input | v1 groups | Choice |
|---|---|---|
| वह घर के बारे में बात कर रहा है | वह / घर के बारे में / बात / कर रहा है | Keep बात separate from the main verb, as specified in the task example |
| उसने काम कर दिया है | उसने / काम / कर दिया है | Recognised verb + vector verb, then auxiliaries |
| उसने पैसे दिए | उसने / पैसे / दिए | Do not attach a main verb to a noun merely because its spelling matches a light verb |
| घर के लिए | घर के लिए | Compound matching wins |
| उसके लिए | उसके लिए | Standalone लिए follows the PDF's postposition list |
| वह नहीं जा रहा है | वह / नहीं जा रहा है | Preverbal negation attaches forward when the next word is a recognised verb |
| वह जाता नहीं है | वह / जाता नहीं है | Postverbal negation stays with the verbal group |
| वह काम करता है | वह / काम / करता है | Habitual forms do not automatically attach to arbitrary nouns |
| किया जाता है | किया जाता है | Habitual form after a recognised verb stays in the chain |
| जाने वाला भी | जाने वाला भी | वाला-forms and focus particles attach left |

These are lexical heuristics, not POS tagging or a validated annotation guideline. Unknown verb forms can miss attachments; ambiguous recognised forms can over-attach. Auxiliary/copular attachment inherits the broad v0 convention. Increased coverage is **not** evidence of improved boundary accuracy: JI-2 gold annotations and JI-5 scoring are still needed.

## Validation completed

`python -m pytest -q`: **182 passed**. This includes 63 new checks (59 grouper checks and 4 coverage-tool checks) plus the existing 119 checks. All ran on CPU; the tiny-tokenizer/model fixtures require no downloads.

The suite covers every listed compound and light verb, negation, habitual/वाला forms, clause punctuation, empty input, language rejection, offline imports, v0 compatibility, token-label integration, coverage arithmetic, local-input CLI, and 1,000 deterministic synthetic partition cases. The corpus run separately checked partition and type counts on **all 5,000 actual sentences**, exceeding the PDF's 1,000-sentence real-corpus partition requirement.

Environment: Python 3.12; pytest 9.1.1; torch 2.14.1+cpu; transformers 5.19.0; peft 0.21.2; datasets 5.1.0; tokenizers 0.23.3; PyYAML 6.0.3. These are the verification environment, not edits to the team's pinned requirements.

## Actual corpus measurement

Source: first 5,000 non-blank Hindi training sentences from `ai4bharat/IndicCorpV2`, config `indiccorp_v2`, split `hin_Deva`, raw rows >= 1,000. Identical sample for both groupers: 280,928 whitespace words. Sentence hash, counts, types and tokenizer settings are saved in `results/grouping/ji1_coverage.json`.

| Metric | hi_rules_v0 | hi_rules_v1 |
|---|---:|---:|
| Sentences | 5,000 | 5,000 |
| Words attached | 69,451 | 81,694 |
| Words attached (%) | 24.722 | 29.080 |
| Words per group | 1.3284 | 1.4100 |
| Tokens per observed group | 1.4841 | 1.5741 |
| Full-text groups | 211,477 | 199,234 |
| Partition checks | passed | passed |

v1 produced 4,106 compound-postposition groups and 1,048 light-verb groups. Attachment coverage rose by **4.358 percentage points**. Word statistics cover full sentences. Token statistics use ganga's actual tokenizer with training-style truncation at 128 tokens, excluding special tokens; the last retained group can be partial. They are not full-sentence token averages.

Reproduce:

```bash
python scripts/ji1_coverage.py --corpus --n 5000 --tokenizer LingoIITGN/ganga-1b --out results/grouping/ji1_coverage.json
```

For a local UTF-8, one-sentence-per-line sample:

```bash
python scripts/ji1_coverage.py --input hindi_sentences.txt --n 5000
```

Without `--tokenizer`, tokens/group is explicitly null. The JSON records whether the requested sample size was reached; no synthetic or short sample is silently labelled as 5,000 corpus sentences.

## Review and use

The ZIP contains the updated repo without `.git`, environment files, caches or model weights. Copy the changed files into the current checkout, preserving newer unrelated work, on a `jai/JI-1-rules` branch. Run the CPU test suite, then review a PR against main. There is no need to replace unchanged model/training modules.

To opt into v1 without changing historical experiment configs:

```bash
python scripts/train.py --config configs/R3.yaml --set data.grouper=hi_rules_v1
```

This is a usage example, not a request to launch R3 now. **Next: JI-6a**, Hindi `hi_rules_v1` train/eval/FLORES boundary caches. H3 is not complete until those caches are built and published. The existing training path can still label on the fly.

## Changed files

- `mtp/data/grouping/hindi_rules.py`, `mtp/data/grouping/base.py`
- `scripts/ji1_coverage.py`
- `tests/test_hindi_rules_v1.py`, `tests/test_ji1_coverage.py`
- `results/grouping/ji1_coverage.json`
- `docs/plan/README.md`, `docs/plan/INTERFACES.md`, `docs/plan/tasks/JAI.md`, `docs/plan/tasks/JAINAM.md`, `docs/plan/tasks/OM.md`
- `README.md`, `CLAUDE.md`, this handoff note
- `docs/plan/MTP_Indic_Team_Plan.pdf` (unchanged copy of the supplied scheduling authority)

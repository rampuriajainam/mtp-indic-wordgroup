# Jai: Data & Linguistics

> **How to use this file with Claude:** open Claude in the repo (it reads `CLAUDE.md`), or paste the *Context* block + `INTERFACES.md` into a chat, then name the task (e.g. "Do JI-4"). Ask for the module **and** its CPU-only pytest. Branch `jai/<task>`, PR against `main`.

## Context (paste this to Claude)

**Scheduling authority:** use the uploaded PDF's Phases A–C now and D+E later, and the H1–H10 table in `docs/plan/README.md`. Existing implementation details below are preserved. No P0/P1/P2 label overrides the PDF phase order.

**First:** JI-1 → JI-6a (H3), then JI-2. The expanded lists are already saved in `legacy/validate_on_real_data.py`, including light verbs, जाता/करता and वाला. Its compound set is unused; it does not load or validate real data. JI-1 is implemented and measured on 5,000 real IndicCorp training sentences (see `docs/JI-1_HANDOFF.md`).

We are building *Word-Group Guided Multi-Token Prediction for Hindi and Marathi*. A 1B Hindi LM (`LingoIITGN/ganga-1b`, SentencePiece-style tokenizer, vocab 30k, adds no BOS, ~1.12 tokens per word) is fine-tuned with LoRA plus extra prediction heads that guess tokens t+2, t+3, t+4. Hypothesis: those far heads do better if they know where **word groups** begin and end. A word group is a multi-word unit of meaning: verb + auxiliaries (जा रहा था), noun + postposition (घर से), compound postposition (के बारे में), light-verb construction (कर दिया). My job: produce word groups reliably, measure how good they are and how much signal they carry, and supply the linguistic analysis for the report.

The code is the `mtp/` package (see `docs/plan/INTERFACES.md` §2, §7, §12). Already on `main`:
- `mtp/data/grouping/base.py`: the `Grouper` protocol (`name`, `lang`, `group_words`, `group_types`), `REGISTRY` (name → "module:Class"), `get_grouper`, `check_partition`.
- `mtp/data/grouping/hindi_rules.py`: `hi_rules_v0` = the original rules (21 auxiliaries + 8 postpositions; a word joins the previous group if, punctuation stripped, it is in a list). On IndicCorp: 24.6% of words attach, 1.33 words/group.
- `mtp/data/grouping/align.py`: `label_tokens` / `label_batch` turn word groups into token labels (`group_start`, `group_id`) via `offset_mapping`. Training labels data on the fly with these, so a grouper registered in `REGISTRY` is immediately usable for training (`data.grouper: <name>`) and evaluation.
- `mtp/data/corpus.py`: `load_split(lang, "eval" | "eval_small" | "train" | "flores", n)`.
- `legacy/validate_on_real_data.py` has saved expanded attaching lists used by its demo, plus an unused compound set (light verbs, वाला-forms, तक/भी/ही/साथ/पास/नहीं, compound postpositions) to start v1 from.
- `scripts/jn2_structural_stats.py` (part A) computes same-group rates per lookahead; start JI-8 from it.

Numbers so far (v0, ganga tokenizer, 2,500 IndicCorp sentences): group = 1.49 tokens; token t+d is in t's group 33.4 / 7.8 / 2.5 / 0.8% for d = 1..4 (each-word-its-own-group baseline: 11.0 / 3.9 / 1.4 / 0.5%).

Data: `load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="hin_Deva", streaming=True)` (Marathi `mar_Deva`); FLORES-200 devtest. Every other IndicCorp row is blank (`load_split` drops them). Windows: `$env:PYTHONUTF8 = "1"`.

## Tasks

### JI-1 · Hindi rule grouper v1 (implemented and measured) · Phase A · S-M
- In `mtp/data/grouping/hindi_rules.py`: add `WORD_LISTS["v1"]` and `class HindiRuleGrouperV1` (`name = "hi_rules_v1"`), register it in `REGISTRY`.
- Lists:
  - **Auxiliaries / aspect / modal:** है हैं था थी थे हो होगा होगी होंगे हूँ रहा रही रहे गया गई गए गये सकता सकती सकते चुका चुकी चुके पाया पाई पाए
  - **Light / vector verbs after a verb:** लगा लगी लगे पड़ा पड़ी पड़े दिया दी दिए लिया ली लिए डाला डाली डाले
  - **Postpositions / particles:** का की के को से में पर ने तक भी ही लिए साथ बारे पास नहीं वाला वाली वाले
  - **Compound postpositions** (multi-word, longest match first): के लिए · के साथ · के बारे में · के बाद · के पास · के अंदर · के बाहर · के ऊपर · के नीचे · के दौरान · के अनुसार · की ओर · की तरह · के कारण
  - **Punctuation:** strip `।॥,.!?;:"'()` for lookup only, and keep the original word.
- `group_types` must return the right type per group, including `compound_postposition` and `light_verb`.
- Decide and document the ambiguous cases in the module docstring: दिया / लिया as main verbs ("gave"), लिए alone vs के लिए, नहीं before vs after the verb.
- Tests in `tests/test_hindi_rules_v1.py` (existing v0 tests remain in `tests/test_grouping.py`):
  - `"मैं कल बाजार जा रहा था।"` → `[["मैं"],["कल"],["बाजार"],["जा","रहा","था।"]]`;
  - `"वह घर के बारे में बात कर रहा है"` gives your documented grouping;
  - `check_partition` holds on the test sentences.
- `python scripts/ji1_coverage.py --corpus --n 5000 --tokenizer LingoIITGN/ganga-1b --out results/grouping/ji1_coverage.json`; run it on 5,000 train sentences and put the coverage numbers (% words attached, words/group, tokens/group) in the PR description.
- Done when: `get_grouper("hi_rules_v1")` works, tests pass, numbers posted. Then hand off the grouper and proceed to JI-6a (H3): his configs switch with one line.

### JI-6a · Hindi rules boundary cache · Phase A · H3
- Do immediately after JI-1, before the gold set. Build `hi_rules_v1` × {train, eval, flores}; default 200k train rows, held-out split per INTERFACES §4.
- On-the-fly labelling already works, but this shared cache is the PDF's H3 deliverable.
- `mtp/data/boundary_cache.py` + `scripts/build_boundary_cache.py --lang hi --grouper hi_rules_v1 --split train --n 200000 --out boundary_cache/`, using `label_batch`, the format in INTERFACES §3, and `meta.json` inside each folder.
- Upload as Kaggle Dataset `mtp-boundary-cache`.

### JI-2 · Annotation guide + Hindi gold set · Phase A · L
- `docs/annotation_guide.md` (1-2 pages):
  - what counts as a group, with 3+ examples per type (aux chain, postposition, compound postposition, light verb, negation, वाला-forms);
  - what does not (adjective + noun, coordination, numbers).
- `scripts/annotate.py`: shows a sentence with word indices, you type spans (`3-5 7-8`) and types, and it appends to the JSONL (format: INTERFACES §7).
- `data/gold/hi_gold.jsonl`: 200 sentences of 6-25 words, 100 from FLORES devtest `hin_Deva` and 100 from IndicCorp train rows 1,000-5,000.
- Give Om the IDs of 50 of them, text only (H9).

### JI-3 · Trankit grouper · Phase B · M
- `mtp/data/grouping/trankit_grouper.py`, `name = "trankit"`, lang hi and mr.
- A word attaches to its head's group if its deprel ∈ {aux, aux:pass, case, mark, compound:lv, cop}, or it is a PART with a negation feature. Non-contiguous attachments stay separate.
- Map Trankit tokens back to whitespace words; `check_partition` must hold.
- Lazy-load the pipeline. Add `batch_group_words(sentences)`.
- If Trankit won't install on Python 3.12/3.13, use Stanza with the same logic (`name = "stanza"`).
- Parsing is slow, so runs with this grouper need the cache (JI-6).

### JI-4 · Random grouper (the control) · Phase B · S
- `mtp/data/grouping/random_grouper.py`, `class RandomGrouper`, `name = "random"`, registered.
- `__init__(lang="hi", histogram=None, seed=0)`: group-length distribution `{1: 0.7, 2: 0.2, ...}` (in words). Default: measured from `hi_rules_v1` (or v0 until v1 exists) on 5,000 train sentences and stored as a constant in the module, so `get_grouper("random")` needs no arguments.
- `group_words`: sample group lengths until the sentence is covered (the last group is cut to fit), deterministic per `(seed, sentence)`; use a hash of the sentence, not Python's `hash()`, which changes between runs. `group_types`: all `"other"`.
- `RandomGrouper.fit_from(grouper, sentences, seed=0)` builds one with another grouper's histogram.
- Tests: deterministic across instances, partition holds, the length histogram over 2,000 synthetic sentences matches the input within a few percent.
- Why it matters: R6a/R6 (same losses, random boundaries) is the experiment that shows the gain comes from linguistics. **This unblocks Jainam's R6a.**

### JI-5 · Score groupers against gold · Phase B · S
- `mtp/eval/group_metrics.py` + `scripts/score_groupers.py` → `results/grouping/hi_scores.json` + a markdown table.
- Boundary-level P/R/F1, where a boundary = a position between words i and i+1 that is inside a group (an attachment). Also exact-match group accuracy, per group type, and Cohen's κ on Om's 50 (H8).
- Compare `hi_rules_v0`, `hi_rules_v1`, `trankit`, `random`.
- **Decision:** the best F1 becomes the default grouper for the final runs; the runner-up is R7. Post it.

### JI-6b · Trankit + random caches · Phase B · H5
- After JI-3/JI-4/JI-5, add each grouper × {train, eval, flores} with the same tokenizer, splits and metadata. Publish alongside the grouper decision (H5).

### JI-7 · Marathi · Phase C · L
- `mtp/data/grouping/marathi_rules.py`, `name = "mr_rules_v1"`, registered.
- Marathi postpositions are mostly suffixes (घरात, घराला, घरासाठी), so fewer separate words attach.
- Separate attaching words:
  - auxiliaries: आहे आहेत होता होती होते होतो असेल असतील नाही;
  - aspect/modal: लागला लागली लागले शकतो शकते शकतात;
  - compound postpositions written separately: च्या साठी, च्या बद्दल, च्या नंतर, च्या आधी, च्या मध्ये. Check real text for which are split vs joined.
- 150 gold sentences (`data/gold/mr_gold.jsonl`, FLORES + IndicCorp `mar_Deva`); score `mr_rules_v1` vs `trankit` (mr).
- JI-8 stats with the Misal tokenizer.

### JI-8 · How far does structure reach? (C3) · Phase C · M
- `scripts/group_stats.py` → `results/grouping/stats_{lang}.json` (INTERFACES §12) + 2 figures in `results/figures/`.
- For groupers {`words` (each word its own group), `hi_rules_v0`, `hi_rules_v1`, `random`} × tokenizers {ganga-1b, Misal-1B (Marathi model), `google/mt5-small`, `bigscience/bloom-560m`, `xlm-roberta-base`} on 5,000 Hindi train sentences:
  - words/group, tokens/group, tokens/word;
  - **same-group rate per lookahead d = 1..4**: the fraction of token positions t where token t+d is in t's group.
- Repeat for Marathi (`mar_Deva`) with the Marathi groupers once they exist.
- Figure 1: same-group rate vs lookahead, one line per tokenizer (does a finer tokenizer give the far heads more structure?). Figure 2: group length in words and in tokens, hi vs mr.
- Also: the 30 most frequent multi-word groups.
- This is the "how far does linguistic structure reach" result that frames the whole method section.

### JI-9 · Error analysis by group type · Phase D+E · M
From Om's per-token dumps (INTERFACES §10), re-run the grouper on the text to attach group types. Per-head top-1 by group type and by position in group (first / middle / last) for R2 vs R3/R5 vs R6. One table, plus 10 examples where R5 gets a far-head prediction right that R2 gets wrong, and 5 where it is worse.

### JI-10 · Where does ganga-1b encode word groups? (C5) · Phase later optional · M
- `scripts/probe_layers.py` → `results/probing/hi_{grouper}.json` + a figure.
- Frozen ganga-1b with `output_hidden_states=True` (forward only; fits on an 8 GB laptop GPU in bf16, or Kaggle). For every layer ℓ = 0..16, train a logistic-regression probe on the token representations of 2,000 train sentences to predict `group_start` of the **next** token. Evaluate F1 on the eval split.
- Label sets:
  - `hi_rules_v1` (and v0);
  - `random` (control: should be near chance);
  - **word start** (does the next token start a new word?), the baseline that separates "knows word boundaries" from "knows word-group boundaries".
- The interesting result: the layer where group-boundary F1 rises above word-boundary F1, if any. Compare with the S2 probe accuracy on the trained heads (Jainam's runs log `struct/boundary_bce`).

### Write-up · Phase E (later)
Data & annotation (corpora, rules, guideline summary, gold sizes, κ, grouper scores); C3 reach analysis; C5 probing; related work (MTP: Gloeckle et al. 2024, Medusa, Aynetdinov & Akbik; morphology-aware tokenization; Indic LMs; multi-task weighting: Kendall et al. 2018, DWA) with `docs/references.bib`.

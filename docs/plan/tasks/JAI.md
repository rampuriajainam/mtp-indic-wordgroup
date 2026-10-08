# Jai: Data & Linguistics

> **How to use this file with Claude:** paste the "Context" block below, then `INTERFACES.md`, then the one task you're on (e.g. "Do JI-3"). Ask for the module **and** its pytest. Run the test, then open a PR on a `jai/<task>` branch.

## Context (paste this to Claude)

We are building *Word-Group Guided Multi-Token Prediction for Hindi and Marathi*. A 1B Hindi LM (`LingoIITGN/ganga-1b`, SentencePiece tokenizer, vocab 30k) is fine-tuned with LoRA and extra "Medusa-style" heads that predict tokens t+2, t+3, t+4. Our hypothesis is that these far-ahead heads do better if they know where **word groups** begin and end. A word group is a multi-word meaning unit: a verb + auxiliary chain (जा रहा था), a noun + postposition (घर से), a compound postposition (के बारे में), a light-verb construction (कर दिया). My job is to produce those word groups reliably, measure how good they are, and ship them as token-level labels the training code can read.

Existing code in the repo (being moved to `legacy/`):
- `word_group_boundaries.py`: `find_word_groups(sentence) -> list[list[str]]`. A word joins the previous group if (after stripping `।,.!?`) it is in `AUXILIARIES` or `POSTPOSITIONS`. Lists are short (21 auxiliaries, 8 postpositions).
- `boundary_alignment.py`: tokenizes the full sentence with `return_offsets_mapping=True` and marks a token as a group start if a group's start char falls in `[tok_start, tok_end)`, which handles SentencePiece's leading space. Special tokens get 0.
- `validate_on_real_data.py`: despite the name, the on-disk file is a copy of `find_word_groups` with **expanded** lists (light verbs, जाता/करता, वाला-forms, तक/भी/ही/साथ/पास/नहीं, and a `COMPOUND_POSTPOSITIONS` set that is defined but never used). Start JI-1 from these lists. The coverage code that gave 25.0% of words attaching and 1.33 words/group is not in the file; rewrite it for JI-1's stats.

Dataset: `load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="hin_Deva", streaming=True)` (Marathi: `mar_Deva`). FLORES-200 devtest for clean evaluation text.

---

## JI-1 · Hindi rule grouper v1
**File:** `mtp/data/grouping/hindi_rules.py` (class `HindiRuleGrouper`, `name="hi_rules_v1"`), `mtp/data/grouping/base.py` (Protocol + `get_grouper` registry)

Start from `find_word_groups` and extend it:
- **Auxiliaries / aspect / modal:** है हैं था थी थे हो होगा होगी होंगे हूँ रहा रही रहे गया गई गए गये सकता सकती सकते चुका चुकी चुके पाया पाई पाए
- **Light / vector verbs that attach to a preceding verb:** लगा लगी लगे पड़ा पड़ी पड़े दिया दी दिए लिया ली लिए डाला डाली डाले
- **Single postpositions / particles:** का की के को से में पर ने तक भी ही लिए साथ बारे पास नहीं वाला वाली वाले
- **Compound postpositions (multi-word windows, match longest first):** के लिए · के साथ · के बारे में · के बाद · के पास · के अंदर · के बाहर · के ऊपर · के नीचे · के दौरान · के अनुसार · की ओर · की तरह · के कारण
- Punctuation: strip `।॥,.!?;:"'()` for lookup only, and keep the original word.
- Put the word lists in a module-level dict so they're easy to edit and can be cited in the report.

**Checks (pytest):**
- `"मैं कल बाजार जा रहा था।"` → `[["मैं"],["कल"],["बाजार"],["जा","रहा","था।"]]`
- `"वह घर के बारे में बात कर रहा है"` groups `घर के बारे में` and `बात` / `कर रहा है` correctly (write down what you decide and test it)
- Concatenated groups == `sentence.split()` for 1,000 random corpus sentences (property test)
- Re-run coverage stats and record the new % and words/group in the PR description

## JI-2 · Annotation guideline + Hindi gold set
**Files:** `docs/annotation_guide.md`, `data/gold/hi_gold.jsonl` (format in INTERFACES §7)

- Write a 1–2 page guide: what counts as a group, with 3+ examples per type (aux chain, postposition, compound postposition, light verb, negation, वाला-forms) and what does **not** group (adjective + noun, coordinated words, numbers).
- Annotate **200 Hindi sentences**: 100 from FLORES-200 devtest `hin_Deva`, 100 from IndicCorp rows 1,000–5,000. Pick sentences of 6–25 words.
- Write a tiny helper `scripts/annotate.py` that shows a sentence with word indices and lets you type group spans (e.g. `3-5 7-8`) and appends to the JSONL. This makes the annotation much faster.
- Hand 50 of them (IDs only, no groups) to Om for blind double-annotation (used in JI-5).

## JI-3 · Trankit grouper
**File:** `mtp/data/grouping/trankit_grouper.py` (`name="trankit"`, supports `lang="hi"` and `"mr"`)

- Use Trankit's dependency parse. A word attaches to its head's group if its deprel is one of `aux`, `aux:pass`, `case`, `mark`, `compound:lv` (light verb), `cop`, or the word is `PART` with a negation feature. Then convert to contiguous groups (if an attachment is non-contiguous, keep the words separate).
- Map Trankit tokens back to whitespace words (Trankit may split punctuation). Must satisfy the "concatenation == split()" rule.
- Lazy-load the pipeline (no download at import). Add `batch_group_words(sentences)` for speed.
- If Trankit won't install on the Python version we use, try Stanza with the same deprel logic and name it `stanza`. Note which one worked in the PR.

## JI-4 · Random grouper (control)
**File:** `mtp/data/grouping/random_grouper.py` (`name="random"`)
- Constructor takes a group-length distribution (a histogram dict `{1: 0.7, 2: 0.2, 3: 0.08, 4: 0.02}`) and a seed.
- `group_words` samples group lengths from that distribution until the sentence is covered, deterministic per `(seed, sentence)`.
- Add `RandomGrouper.fit_from(grouper, sentences)` that measures the histogram from another grouper, so R6 matches the real grouper's length distribution exactly.

## JI-5 · Score groupers against gold
**Files:** `mtp/eval/group_metrics.py`, `scripts/score_groupers.py`

- Boundary-level precision / recall / F1, where a "boundary" is a position between word i and i+1 that is **inside** a group (i.e. an attachment). Also report exact-match group accuracy.
- Report per group type, using a type tag you add during annotation, or infer it from the attached word's list membership.
- Inter-annotator agreement on Om's 50: Cohen's κ over word-pair attach / don't-attach decisions.
- Output `results/grouping/hi_scores.json` and a markdown table. Compare `hi_rules_v1`, `trankit`, `random`.
- **Decision:** the grouper with the best F1 becomes the default for R3–R5 and the other one is R7. Post the decision in the team chat.

## JI-6 · Boundary cache builder (Hindi)
> **Priority:** this is what unblocks training (handoff H3). Do it **right after JI-1**, before the gold set: ship **JI-6a** = the `hi_rules_v1` cache only (train / eval / flores), then later **JI-6b** = add `trankit` + `random` once JI-3 / JI-4 exist.

**Files:** `mtp/data/boundary_cache.py`, `scripts/build_boundary_cache.py`
- CLI: `python scripts/build_boundary_cache.py --lang hi --grouper hi_rules_v1 --split train --n 200000 --out boundary_cache/`
- Uses `label_tokens` from `align.py` (port of `boundary_alignment.py`, generalized to take any grouper and tokenizer). Train split = IndicCorp rows ≥ 1,000; eval = rows 0–999; `flores` = FLORES devtest.
- Writes the HF dataset + `meta.json` (INTERFACES §3). Uses `datasets.map(batched=True, num_proc=...)` so 200k rows finish in reasonable time on CPU.
- Build JI-6a: `hi_rules_v1 × {train, eval, flores}`. JI-6b: add `trankit` and `random` (random histogram fitted from the default grouper). Upload the whole `boundary_cache/` folder as **one Kaggle Dataset** named `mtp-boundary-cache` (shared with Jainam and Om).
- Test: 20 sentences, `group_id` monotonic, lengths equal, special tokens = -1.

## JI-7 · Marathi
**Files:** `mtp/data/grouping/marathi_rules.py` (`name="mr_rules_v1"`), `data/gold/mr_gold.jsonl`
- Marathi postpositions are mostly **suffixes inside the word** (घरात, घराला, घरासाठी), so fewer separate words attach. Separate attaching words to cover: auxiliaries आहे आहेत होता होती होते होतो असेल असतील नाही; aspect/modal: लागला लागली लागले शकतो शकते शकतात; compound postpositions written separately: च्या साठी, च्या बद्दल, च्या नंतर, च्या आधी, च्या मध्ये (check real text for which are written split vs joined).
- 150 Marathi gold sentences (FLORES `mar_Deva` + IndicCorp `mar_Deva`). Score `mr_rules_v1` vs `trankit` (mr).
- Build the Marathi caches with the **Marathi model's tokenizer** (`smallstepai/Misal-1B-instruct-v0.1`) and add them to the same Kaggle Dataset.

## JI-8 · Group statistics (motivation numbers for the paper)
**File:** `scripts/group_stats.py` → `results/grouping/stats_{lang}.json` + 2 plots
- Distribution of group length in words and in tokens (hi vs mr).
- **Key number:** for each lookahead d ∈ {1,2,3,4}, the fraction of positions t where token t+d is in the **same group** as token t. This directly tells us how much "within-group" signal each head can exploit, and it's the number the method section opens with.
- Most frequent 30 multi-word groups.

## JI-9 · Error analysis by group type
After Om's eval writes per-token dumps (`results/{run}/tokens_{dataset}.jsonl`: `sent_id, text, token_idx, token, head, correct, group_start, group_id`). Group **type** is not in the dump. Re-run your grouper on `text` and attach the type yourself, which is why the grouper should expose `group_types(sentence) -> list[str]` alongside `group_words`.
- For R2 vs R5 vs R6: per-head top-1 broken down by group type (aux chain / postposition / compound postposition / light verb / single word) and by position-in-group (first, middle, last).
- One table + 10 qualitative examples where R5 gets a far-head prediction right that R2 gets wrong, and 5 where R5 is worse.

## Write-up (Phase E)
- **Data & annotation section:** corpora, grouping rules, guideline summary, gold set size, agreement κ, grouper scores, group statistics.
- **Related work:** MTP (Gloeckle et al. 2024; Medusa; Aynetdinov & Akbik), morphology-aware tokenization (MorphTok), Indic LMs (AI4Bharat, ganga), multi-task loss weighting (Kendall et al. 2018; DWA). Keep a shared `docs/references.bib`.

# Word-group annotation guide (JI-2 / JI-7)

A group is a contiguous unit of whitespace words. Preserve spelling and punctuation exactly. Every word index must appear once, in order. Do not split suffixes or punctuation off a whitespace word. Annotate meaning and syntax, independently of rule predictions. This guideline is an operational research definition, not a universal definition of Hindi/Marathi constituents.

## Labels and examples

Slashes below separate groups. Mark the smallest verbal or postpositional unit consistently; avoid broad noun phrases. One contract label per group: `single`, `aux_chain`, `postposition`, `compound_postposition`, `light_verb`, `other`.

| Type | Hindi examples | Decision |
|---|---|---|
| aux_chain | जा रहा था / खा चुका है / पढ़ सकता हूँ | Main verb plus aspect, tense and modality |
| postposition | घर से / पार्क में / बच्चों को | Noun/pronoun plus case marker; do not absorb an adjective |
| compound_postposition | घर के बारे में / काम के लिए / नियम के अनुसार | Host plus the entire compound, longest complete expression |
| light_verb | कर दिया / पढ़ लिया / लिख डाला | Main verb plus vector verb; include following auxiliaries |
| other (negation) | नहीं जा रहा है / जाता नहीं है / नहीं पढ़ सका | Keep negation with its verbal predicate, not the preceding subject |
| other (वाला forms) | जाने वाला / आने वाली / खेलने वाले | Infinitive/participle plus वाला-family |
| single | मैं / सुंदर / किताब | Unattached words; one word per group |

Rule predictions classify negation as `aux_chain` and वाला/focus particles as `postposition` for compatibility. Human gold uses `other` for those subtypes. Record `notes: "negation"` or `"wala"` so analysis can distinguish them. This is an intentional type-label difference; attachment scoring is unaffected. If using a different convention, change this guide before starting the set, not halfway through annotation.

After a full compound or light-verb construction, choose the strongest group label: compound > light verb > auxiliary > simple postposition > other. Use notes to record secondary phenomena.

## Ambiguous cases

- `उसने / पैसे / दिए` uses दिए as a main verb; `उसने / काम / कर दिए` uses a vector verb. Do not attach every दिया/लिया-family word to the noun before it.
- `वह / घर के बारे में / बात / कर रहा है` keeps बात separate; the project deliberately does not automatically group noun + main verb. Keep lexical noun+verb ambiguities in notes for adjudication.
- `उसके लिए` and `घर के लिए` attach लिए as a postposition/compound. A verb before लिए can instead make a light-verb construction.
- `देश है` may be a copular auxiliary group. `सुंदर / देश है` keeps the adjective out.
- Punctuation ending a clause closes a group: `घर। / में`; punctuation attached to the final word stays there. Standalone punctuation is a singleton. A comma normally closes the span; record unusual exceptions in notes.
- Never merge an adjective + noun alone (`सुंदर / घर`), coordinated words (`राम / और / श्याम`), or a number + noun alone (`तीन / किताबें`). Case markers still attach to their actual host.

## Marathi

Fused suffixes remain one whitespace word (`घरात`, `घराला`, `घरासाठी`). Separate auxiliaries and explicit case phrases can form groups:

- Auxiliary: `जात आहे`, `खेळत आहेत`, `वाचत होता`.
- Modal/aspect: `खेळू शकतो`, `करायला लागला`, `वाचू शकते` (context decides auxiliary vs vector-verb use).
- Explicit multiword postposition: `घर च्या साठी`, `घर च्या मध्ये`, `घर च्या जवळ` when the source text actually separates च्या; never invent that split.
- Negation: `जात नाही`, `करत नाही`, `नाही येणार`. Keep it with the predicate. Marathi rule heuristics are provisional until reviewed gold and parser comparison.

## Annotation workflow and blind comparison

1. Jai uses `data/annotation/hi_sentences.jsonl` (200 IDs; 100 FLORES, 100 IndicCorp) or `mr_sentences.jsonl` (150; 75+75). Open the text-only file before viewing any rule suggestions.
2. Run `python scripts/annotate.py --input data/annotation/hi_sentences.jsonl --out data/gold/hi_gold.jsonl --annotator jai`. The CLI shows zero-based indices. Enter merged spans such as `3-5 7-8`; all unselected words become singletons. Enter a type for each merged span.
3. Use `q` to stop. Output is flushed per sentence; rerun to resume by ID. All indices/types are validated.
4. Om independently runs the CLI on `hi_om50_text_only.jsonl` with `--blind --annotator om` and writes `hi_gold_om50.jsonl`. Do not show him Jai's labels, draft labels or scores before submission.
5. Run `score_groupers.py` only on reviewed annotations. Keep original disagreements; calculate κ before adjudication. Boundary κ includes attachment and nonattachment positions and is undefined when both annotators have no label variation.
6. Jai resolves uncertain cases using notes; document any guideline change, reannotating earlier examples consistently.

No human gold labels or blind agreement are supplied by the rule draft. The prepared IDs can be used by Om immediately, but his packet has not been sent automatically.

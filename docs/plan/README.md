# Team Plan

**Word-Group Guided Multi-Token Prediction with Adaptive Loss Weighting for Hindi and Marathi**
Team: Jainam · Jai · Om

Read this file first. It covers where we are, the order of work, and who owns what.
- [`INTERFACES.md`](INTERFACES.md) has the exact function signatures, file formats and repo layout. Everything plugs together through it.
- [`tasks/`](tasks/) has one file per person. Each one is written so you can paste it straight into Claude (together with `INTERFACES.md`) and have it generate the code.

---

## 1. Where we are

The pipeline already works end to end on one laptop GPU (RTX, 8 GB).

| Piece | Result |
|---|---|
| Base model | `LingoIITGN/ganga-1b` (1B, Hindi) loads and generates. Tokenizer gives clean Devanagari subwords |
| LoRA | r=8 on `q_proj, v_proj`, lr 2e-4, grad clip 1.0, batch 8, max_len 128 |
| Data | `ai4bharat/IndicCorpV2` → `indiccorp_v2` / `hin_Deva`, streamed |
| NTP baseline | 100k sentences. Held-out loss **3.05**. Flat across lr/rank sweeps: the base model is already well fit to this distribution, and that is a finding we report |
| Token-level MTP baseline (k=2, linear head) | 12,500 batches. Head 0 **3.09**, head 1 **8.10 → 5.84** |
| Word groups v0 | Rule-based (auxiliary + postposition lists). On real corpus text: 25% of words attach, **1.33 words/group** |
| Token alignment | Word groups → token labels via `offset_mapping`, verified |
| Medusa wrapper | `MedusaWrapper` with linear extra heads, bf16-safe |

What's missing: the actual contribution (structural + contrastive losses, adaptive weighting), Marathi, a proper evaluation suite (FLORES, per-head accuracy, speculative-decoding speedup), and a reproducible setup on Kaggle.

---

## 2. The idea in one picture

```
 sentence ─► grouper ─► word groups ─► token labels (group_start, group_id)
                                              │
 tokens ─► ganga-1b + LoRA ─► hidden ─┬─► head 0 (t+1) ─┐
                                      ├─► head 1 (t+2) ─┤
                                      ├─► head 2 (t+3) ─┼─► per-head CE ─┐
                                      └─► head 3 (t+4) ─┘                ├─► adaptive weighting ─► total loss
                                           structural loss (uses labels) ┤
                                           contrastive loss (uses labels)┘

 inference: heads 1..k-1 draft tokens ─► head 0 verifies ─► speedup
            (group-aware policy: stop drafting at the predicted group boundary)
```

Hypothesis: Hindi and Marathi pack one meaning unit into several tokens (जा रहा था, घर से, के बारे में). If the far-ahead heads are told where those units start and end, they should learn faster, be more accurate inside a unit, and give longer accepted drafts at inference time.

---

## 3. Roles

| Person | Owns |
|---|---|
| **Jainam**: Modeling & Training | Model heads, all loss functions, adaptive weighting, `train.py`, every training run, group-aware decoding policy, method + results analysis |
| **Jai**: Data & Linguistics | Word grouping (Hindi + Marathi, rules + Trankit + random control), gold annotation set, grouper scoring, boundary-label cache on Kaggle, group statistics + error analysis |
| **Om**: Evaluation & Infrastructure | Package structure, config/logging/checkpoints, Kaggle notebooks, evaluation suite (perplexity, per-head accuracy, speculative decoding), all eval runs, tables + figures |

All training runs happen in one place (Jainam's Kaggle account), so checkpoints, seeds and logs stay consistent. Evaluation runs on Om's and Jai's Kaggle accounts, which spreads the GPU quota across three accounts.

---

## 4. Order of work

No dates, just order. **Phases A–C are the current priority. Experiments and write-up come later.** Inside a phase the three tracks run in parallel; inside a cell, go top to bottom. H1–H10 mark handoffs (§5).

### Phase A: Foundation (start immediately · priority)
| Jainam | Jai | Om |
|---|---|---|
| **JN-1** Heads module: k heads, `linear` + zero-init `resblock`, per-position CE | **JI-1** Hindi rule grouper v1: full lists, compound postpositions, light verbs | **OM-1** Package skeleton, `legacy/`, config, dtype, logging, collate → **H1** |
| **JN-2** Structural-loss design note (S1 / S2 / S3) | **JI-6a** Hindi boundary cache, rules grouper → **H3** | **OM-2** Checkpoint save / load / resume |
| **JN-3** `scripts/train.py` driven by YAML (stub Om's parts until H1) | **JI-2** Annotation guide + 200-sentence Hindi gold set | **OM-3** Kaggle training notebook with resume → **H2** |

### Phase B: Baselines on Kaggle + grouping quality (needs H2 · priority)
| Jainam | Jai | Om |
|---|---|---|
| **JN-4** Reproduce R0, R1 on Kaggle; train R2 (k=4 resblock) | **JI-3** Trankit grouper | **OM-4** Eval: per-head ppl + top-k, in-group vs boundary → **H4** |
| | **JI-4** Random grouper (control) | **OM-5** Blind double-annotation (50) |
| | **JI-5** Score groupers vs gold, pick default → **H5** | |
| | **JI-6b** Add trankit + random caches | |

**Gate:** Kaggle R0 / R1 within ~0.05 of the laptop numbers before moving on.

### Phase C: The method (needs H3, H4 · priority)
| Jainam | Jai | Om |
|---|---|---|
| **JN-5** Structural loss (pilots → pick) | **JI-7** Marathi rules + 150 gold + Marathi cache → **H7** | **OM-6** Self-speculative decoding engine + baseline policies → **H6** |
| **JN-6** Contrastive loss | **JI-8** Group statistics: same-group rate per lookahead | **OM-7** `evaluate.py` + Kaggle eval notebook |
| **JN-7** Adaptive weighting: fixed / uncertainty / DWA | | |

### Later, not a priority yet: Phase D + E (experiments and write-up, start after C)
| Jainam | Jai | Om |
|---|---|---|
| **JN-8** Run matrix R3 → R10 | **JI-9** Error analysis by group type | **OM-8** Evaluate every run |
| **JN-9** Group-aware draft policy (needs H6) | § Data + annotation, related work | **OM-9** Tables + figures, incl. lookahead heatmap |
| § Method, results, conclusion | | § Setup, evaluation, figures |

---

## 5. Dependencies and handoffs

Most of the work is independent. These are the only points where one person has to finish something before someone else can move. Each handoff = a merged PR or published Kaggle Dataset + a message in the team chat.

**Critical path:** H2 and H3 are what can block training. So Om's first three tasks (OM-1 → OM-3) and Jai's first two (JI-1 → JI-6a) come before everything else on their lists. The gold set, Trankit and the random control matter for the final results but not for getting the method running, so they come after.

**Nobody waits on Jainam until H10.** Jainam's early tasks (JN-1 → JN-3) don't need anyone else's work, because they only use the contracts.

| # | Handoff | From → To | What it unblocks | Until it lands |
|---|---|---|---|---|
| H1 | Package skeleton + `config` / `device` / `collate` / `logging` merged (OM-1) | Om → everyone | Everyone writes code inside `mtp/` | Write your module in a standalone file using the signatures in `INTERFACES.md`, then move it in |
| H2 | Checkpointing + Kaggle training notebook (OM-2, OM-3) | Om → Jainam | Running anything on Kaggle (JN-4 onwards) | JN-1 → JN-3, testing on the laptop |
| H3 | Hindi boundary cache, rules grouper (JI-1 → JI-6a) | Jai → Jainam, Om | Training with structural / contrastive losses (JN-5, JN-6); in-group eval split (OM-4) | Unit-test the losses on hand-built batches; run baselines R0–R2 (they need no labels) |
| H4 | Eval with the in-group / boundary split (OM-4) | Om → Jainam | Choosing the structural-loss variant from pilot runs (JN-5) | Pilots log their own eval losses; pick once H4 lands |
| H5 | Grouper decision + `trankit` / `random` caches (JI-5, JI-6b) | Jai → Jainam | Final runs R5, R6, R7 | Develop the method on the rules cache from H3 |
| H6 | Speculative-decoding engine (OM-6) | Om → Jainam | `GroupAware` draft policy (JN-9) | — (JN-9 is in the later block anyway) |
| H7 | Marathi caches (JI-7) | Jai → Jainam | Marathi runs R8–R10 | Hindi runs |
| H8 | 50 blind annotations (OM-5) | Om → Jai | Agreement number in JI-5 | Score groupers without κ, add it when it arrives |
| H9 | Gold sentence IDs (JI-2) | Jai → Om | OM-5 | — |
| H10 | Trained runs (JN-4, JN-8) published as Kaggle Datasets | Jainam → Om | Evaluating runs (OM-8) | Test eval on R1/R2 checkpoints |

---

## 6. Run matrix

Same seed, same eval sets, same step budget unless noted. Every run gets a YAML in `configs/`.

| ID | Lang | Heads | Head type | Extra losses | Weighting | Grouper | Why |
|---|---|---|---|---|---|---|---|
| R0 | hi | 1 | – | – | – | – | NTP baseline |
| R1 | hi | 2 | linear | – | fixed | – | Reproduce laptop MTP |
| R2 | hi | 4 | resblock | – | fixed | – | Token-level MTP baseline (the real comparison) |
| R3 | hi | 4 | resblock | structural | fixed | best | Method, part 1 |
| R4 | hi | 4 | resblock | structural + contrastive | fixed | best | Method, part 2 |
| R5 | hi | 4 | resblock | structural + contrastive | adaptive | best | **Full method** |
| R6 | hi | 4 | resblock | structural + contrastive | adaptive | random | Control: does linguistic structure matter? |
| R7 | hi | 4 | resblock | structural + contrastive | adaptive | other grouper | Grouper ablation |
| R8 | mr | 1 | – | – | – | – | Marathi NTP |
| R9 | mr | 4 | resblock | – | fixed | – | Marathi MTP |
| R10 | mr | 4 | resblock | structural + contrastive | adaptive | mr best | Marathi full method |

Optional if compute allows: k ∈ {2, 3, 4, 6} sweep on R5; a second seed for R2 and R5.

Marathi base model: `smallstepai/Misal-1B-instruct-v0.1` (Llama architecture). Fallback if it fails to load: keep `ganga-1b` and report Marathi as cross-lingual transfer.

---

## 7. Ideas that make this stand out

1. **Random-boundary control (R6).** Same losses, but boundaries drawn at random with the same group-length distribution. If R5 beats R6, the gain comes from linguistic structure, not just from having an extra loss. This is the strongest single experiment in the paper.
2. **In-group vs at-boundary accuracy.** Split every head's accuracy by whether the target token continues a group or starts a new one. We expect the structural loss to raise in-group accuracy of the far heads, and this shows where it helps.
3. **Group-aware speculative decoding.** Instead of always drafting k-1 tokens, draft up to the predicted end of the current word group. Report mean accepted length and wall-clock speedup against `FixedK` and `ConfidenceCut`.
4. **Group Integrity score.** Fraction of accepted draft spans that end on a group boundary rather than mid-group. A new, simple metric tied to our hypothesis.
5. **Lookahead heatmap figure.** One sentence, tokens coloured by which head first predicts them correctly, with word-group brackets drawn above. This is the figure that explains the paper at a glance.
6. **Hindi vs Marathi contrast.** Hindi is analytic (separate postpositions), Marathi is agglutinative (suffixes inside the word). Groups should be fewer words but more tokens in Marathi. Report how the method behaves across that difference.

---

## 8. Working rules

- **Branches:** `jainam/*`, `jai/*`, `om/*`. Merge to `main` through a PR that the other two can see. Keep PRs small.
- **Contracts first:** never change a signature in `INTERFACES.md` silently. Edit the file in the same PR and say so in the PR title.
- **Tests:** every module gets a CPU-only pytest with a tiny input (2–3 sentences, a mocked tiny model where needed). `pytest -q` must pass before merge.
- **No big files in git:** checkpoints, caches and logs live on Kaggle (`/kaggle/working`, Kaggle Datasets). `results/*.json` and figures are committed.
- **Windows:** set `$env:PYTHONUTF8 = "1"` before running anything that reads Devanagari text.
- **Kaggle GPUs (T4 / P100)** don't have native bf16. `mtp/device.py` picks fp32 weights + fp16 autocast + GradScaler there, and bf16 where supported. Never hard-code `torch.bfloat16`.
- **Using Claude to write code:** paste `INTERFACES.md` + your task file + the specific task ID. Ask for the module plus its test. Run the test before pushing.

---

## 9. Using Claude to generate the code

1. Pull `main`, create your branch (`jai/JI-3-trankit`).
2. Open a new Claude chat. Paste the **Context** block of your task file, then all of `INTERFACES.md`, then: *"Do JI-3. Write the module and a CPU-only pytest. Match the contracts exactly."*
3. Run the test locally. Fix until green. Run the module once on real data.
4. PR to `main` with what it does, test output, and any number it produced.

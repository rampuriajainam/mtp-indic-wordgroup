> **Latest sync override (2026-10-10):** Jai's current priority is JI-1 → start JI-2 → JI-8 → JI-3/JI-5 → JI-10 → JI-7. JI-4 is already teammate-owned in this upload. See `docs/JAI_HANDOFF.md` for actual implementation/measurement status; human annotation, full C5 runs and final grouper scoring remain separate. The historical plan below is retained as research context, not a claim that these results are complete.

# Team Plan (v2)

**Word-Group Guided Multi-Token Prediction for Hindi and Marathi**
Team: Jainam (modeling & training) · Jai (data & linguistics) · Om (evaluation & infrastructure)

This replaces the first plan (`docs/archive/MTP_Indic_Team_Plan_v1.pdf`). The ideas and task IDs are the same. What changed: most of the infrastructure is built, the measurements we made reshaped the method, and the work is re-cut so that **nobody is blocked by anybody today**.

- [`INTERFACES.md`](INTERFACES.md): every signature and file format. Code must match it.
- [`tasks/JAINAM.md`](tasks/JAINAM.md), [`tasks/JAI.md`](tasks/JAI.md), [`tasks/OM.md`](tasks/OM.md): one file per person. Paste its *Context* block + `INTERFACES.md` into Claude, then name the task.
- [`../runs.md`](../runs.md): every number so far. [`../design_structural_loss.md`](../design_structural_loss.md): the structural-loss design (JN-2).

---

## 1. Where we are

| piece | status |
|---|---|
| Package `mtp/` | config, device/dtype policy, corpus splits, collator, metrics log, checkpoints (`save_run` / `load_run`), model build. Tests: 108 passing, CPU-only |
| Model | `MTPModel`: k heads, `linear` or zero-init `resblock` sharing the frozen LM head, optional boundary probes, `head_backbone_grad` |
| Losses | per-head CE; structural S2 / S3 (+ S3-chain, S3-all, S23); weighting `fixed` / `uncertainty` / `dwa` |
| Training | `scripts/train.py` (YAML configs, bit-exact resume, Kaggle time limits, in-group / boundary eval); `notebooks/kaggle_train.ipynb` |
| Word groups | contract code: grouper registry, `hi_rules_v0` (legacy rules), `label_tokens` (= legacy alignment, verified on 500 sentences). Training labels data on the fly, so **no cache is needed to train** |
| Runs | R0, R1, R2 done on Kaggle (T4 x2, 0.29-0.38 s/step), Gate B passed, published as `mtp-run-R0/R1/R2` (`docs/runs.md`) |
| Not started | grouper v1, random / Trankit groupers, gold set, evaluation suite, speculative decoding, Marathi, contrastive loss, group-aware decoding |

**Findings so far** (details in `docs/runs.md`, `docs/design_structural_loss.md`):
1. **The NTP baseline is flat.** Held-out loss is 3.05 whatever the lr or rank. The base model already fits this data, which we report rather than fight.
2. **Word groups are short in tokens.** ganga's tokenizer averages 1.12 tokens per word, so a group spans about 1.5 tokens. Token t+d is in token t's group 33% / 8% / 2.5% / 0.8% of the time for d = 1..4. Structure reaches heads 1 and 2; head 3 almost never sees it.
3. **Structure is real.** On the trained laptop model, in-group targets are about twice as easy as boundary targets for every head (head 1: 28% vs 14% top-1).
4. **Shared-LM-head heads hurt the verifier.** Zero-init resblock heads trained with full gradient make head 0 worse by 0.20 nats at 2k steps. `head_backbone_grad` = 0.1 keeps it within 0.01 of R0 while the heads still learn faster than when detached.

## 2. The story for the professors

**Claim:** telling MTP heads where Hindi/Marathi word groups start and end makes the far heads more accurate *inside* groups and gives longer accepted drafts, i.e. faster decoding with identical output. **The evidence that it is the linguistics and not just an extra loss** is that the same losses with random boundaries (R6) or without the group mask (S3-all) help less.

What is new (each item names its owner):

| # | contribution | evidence we will show | owner |
|---|---|---|---|
| C1 | Structure-aware MTP losses: boundary probes (S2), in-group self-distillation from the LM head (S3) | per-head top-1, in-group vs boundary; R3/R5 vs R2; controls R6 and S3-all | Jainam |
| C2 | Group-aware speculative decoding + Group Integrity metric | accepted length, speed-up vs FixedK / ConfidenceCut; outputs identical to greedy | Om (engine), Jainam (policy) |
| C3 | How far structure reaches: same-group rate vs lookahead, grouper and tokenizer; Hindi vs Marathi | figure + table | Jai |
| C4 | Gradient isolation for shared-LM-head MTP heads | α ∈ {0, 0.1, 1} table | Jainam |
| C5 | Where ganga-1b encodes word groups: layer-wise probes | probe F1 by layer, rules vs random labels | Jai |
| C6 | Gold word-group sets (hi 200, mr 150) with agreement κ; groupers scored against them | κ, P/R/F1 table | Jai, Om |
| — | Headline figure: lookahead heatmap (tokens coloured by the first head that predicts them, group brackets above) | figure | Om |

**Review package** (what to have ready for the next meeting with the professors; all P0):
1. Baselines R0-R2 + pilot table (S2 / S3 / S3-chain / S23 / S3-all vs R2 at 2k steps), with in-group vs boundary top-1.
2. R3 (best structural loss) vs R2 vs R6a (R3 with random boundaries), evaluated on IndicCorp eval + FLORES.
3. Speculative-decoding speed-up for R2 and R3 (FixedK, ConfidenceCut), outputs verified identical to greedy.
4. The C3 figure (same-group rate vs lookahead, two or three tokenizers) and the C4 α table.

## 3. Priorities

- **P0:** needed for the review package and the core claim. Do these first.
- **P1:** needed for the full report/paper.
- **P2:** extras, if time and GPU quota allow.

Effort: S = up to half a day, M = 1-2 days, L = 3+ days.

| person | P0 | P1 | P2 |
|---|---|---|---|
| Jainam | JN-4 baselines, JN-5 pilots → R3, JN-8a R3 + R6a | JN-7 weighting → R5 + R6, JN-9 GroupAware, JN-8b Marathi | JN-6 contrastive (R4), JN-10 α ablation, R7, k sweep / seeds |
| Jai | JI-1 rules v1, JI-4 random grouper, JI-8 reach + tokenizer study | JI-2 gold set, JI-3 Trankit, JI-5 scoring, JI-7 Marathi, JI-10 layer probing | JI-6 cache, JI-9 error analysis |
| Om | OM-4 eval, OM-6 spec decoding, OM-7 evaluate.py, OM-8 evaluate runs | OM-9 tables + heatmap, OM-5 blind annotation | OM-10 Group Integrity by group type |

Already done (by Jainam): JN-1 heads, JN-2 design, JN-3 train.py, JN-5 code, JN-7 code; OM-1 package, OM-2 checkpoints, OM-3 Kaggle notebook. Om owns those infra modules from now on (review them first, OM-0).

## 4. Order of work

No dates, just order. Inside a phase the three columns run in parallel.

### Phase 1 · Baselines, pilots, eval (now)
| Jainam | Jai | Om |
|---|---|---|
| JN-4 finish R0-R2, gate, publish for Om | JI-1 `hi_rules_v1` | OM-0 review infra |
| JN-5 run 6 pilots (one Kaggle session), pick → R3 | JI-4 `random` grouper | OM-4 eval: per-head loss/ppl/top-k, in-group split, token dumps |
| JN-8a R3, then R6a (R3 + random) | JI-8 reach + tokenizer study (C3) | OM-6 spec-decode engine + FixedK + ConfidenceCut |
| | JI-2 start the gold set (long task) | OM-7 `scripts/evaluate.py` + Kaggle eval notebook |

**Gate 1 (review package ready):** R2, R3, R6a evaluated by OM-7 on IndicCorp eval + FLORES, spec decoding correct (outputs = greedy) with a speed-up number.

### Phase 2 · Full method, controls, decoding
| Jainam | Jai | Om |
|---|---|---|
| JN-7 weighting pilots → R5, then R6 | JI-2 finish gold set (+ 50 IDs to Om) | OM-8 evaluate every run |
| JN-9 GroupAware policy | JI-3 Trankit, JI-5 scoring + κ | OM-5 blind annotation (50) |
| JN-10 α ablation (P2), JN-6 contrastive (P2) | JI-10 layer-wise probing (C5) | OM-9 tables + figures, incl. heatmap |

### Phase 3 · Marathi, ablations, write-up
| Jainam | Jai | Om |
|---|---|---|
| JN-8b R8-R10 (Marathi), R7 | JI-7 Marathi rules + gold, JI-9 error analysis | OM-8/9 Marathi tables, Group Integrity |
| § Method, results, conclusion | § Data + annotation, related work, C3/C5 | § Setup, evaluation, figures |

## 5. Dependencies (the only ones)

Everything else works off `main` today: groupers exist (`hi_rules_v0`), training labels data itself, `load_run` exists for evaluation.

| # | from → to | what | until it lands |
|---|---|---|---|
| D1 | Jainam → Om | finished runs as Kaggle Datasets `mtp-run-<ID>` | Om tests on a tiny local run (`tests/test_infra.py` shows how) |
| D2 | Jai → Jainam | `random` grouper (R6a/R6), `hi_rules_v1` (final runs), `trankit` (R7), `mr_rules_v1` (R10) | runs use `hi_rules_v0`; switching is one config line |
| D3 | Om → Jainam | spec-decode engine (`generate`) for GroupAware numbers | JN-9 is developed and unit-tested against the DraftPolicy contract with fake logits |
| D4 | Om → Jai | per-token dumps (JI-9), 50 blind annotations (κ) | Jai scores groupers without κ |
| D5 | Jai → Om | 50 gold sentence IDs (OM-5) | — |

Each handoff = a merged PR (or published Kaggle Dataset) + a message in the team chat.

## 6. Run matrix

Same seed (42), same data order, eval during training on `eval_small` (pilots on `eval`). Official numbers come from OM-7 on IndicCorp `eval` + FLORES. Every run has a YAML in `configs/`. α = `head_backbone_grad` (0.1 for all k=4 runs).

| ID | P | lang | heads | extra losses | weighting | grouper | why | status |
|---|---|---|---|---|---|---|---|---|
| R0 | P0 | hi | 1 | – | – | – | NTP baseline | done |
| R1 | P0 | hi | 2 linear (α=1) | – | sum | – | reproduce laptop MTP (gate) | done |
| R2 | P0 | hi | 4 resblock | – | fixed 0.8^d | – | token-level MTP baseline | done |
| pilots | P0 | hi | 4 resblock | S2 / S3 / S3-chain / S23 / S3-all | fixed | v0 | pick the structural loss (+ `pilot_R2ref`) | ready |
| R3 | P0 | hi | 4 resblock | structural (S23_mix 0.25/0.25, chain) | fixed | v0 | method, part 1 | done (`mtp-run-R3`) |
| R6a | P0 | hi | 4 resblock | = R3 | fixed | **random** | control: is it the linguistics? | done (`mtp-run-R6a`); seed-43 pair R3_s43 / R6a_s43 done |
| R5 | P1 | hi | 4 resblock | = R3 | adaptive (JN-7b pilot winner) | v0 | full method (unfrozen line) | **dropped**: no `pilot_w_*` that keeps head 0 beats `pilot_w_fixed` (#36; numbers in `docs/runs.md`) |
| R6 | P1 | hi | = R5 | | | **random** | control for the full method | dropped (R5 dropped) |
| Rsd | P0 | hi | 4 resblock, **frozen backbone** | – (CE on self-distilled text) | fixed (weighting is a no-op when frozen, #36) | – | fast line: lossless, + trees (JN-9) | done (`mtp-run-Rsd`, `mtp-run-Rsd-s43`); variants soft / k6 / mr / mr_soft done, long / data2x running |
| R4 | P2 | hi | 4 resblock | structural + contrastive | fixed | best | does contrastive add anything? | needs JN-6 |
| R7 | P2 | hi | = R5 | | | other grouper | grouper ablation | needs JI-3/JI-5 |
| R8 | P1 | mr | 1 | – | – | – | Marathi NTP | done |
| R9 | P1 | mr | 4 resblock | – | fixed | – | Marathi MTP | done |
| R10 | P1 | mr | = R3 (R5 dropped) | | | `mr_rules_v1`; R10a `random_mr_v1` | Marathi structural run + control | done (`mtp-run-R10`, `-R10a`) |
| A1 | P2 | hi | R2 with α = 0 and 1 | | | | C4 ablation, full length | done |

Optional (P2): k ∈ {2, 3, 6} on R3; a second seed for R2 and R3.
Compute: one 12.5k-step run ≈ 75 min on one T4; one Kaggle session with T4 x2 does 2 runs in parallel; the 6 pilots fit in about one hour.
Marathi base model: `smallstepai/Misal-1B-instruct-v0.1`. Fallback if it misbehaves: ganga-1b and report Marathi as cross-lingual transfer.

## 7. Risks and what we do about them

| risk | sign | response |
|---|---|---|
| Structural losses don't beat R2 | pilots within noise of `pilot_R2ref` | Report it honestly with the C3 reach analysis (structure reaches only heads 1-2); focus on head 1 and on decoding (C2); try λ ramps |
| S3 is too strong early | consistency KL is ~7-9 at init (vs ~2.8 when trained); head 0 or head losses get worse | lower `lambda_s3` or ramp it (JN-5) |
| Random boundaries do as well as linguistic ones | R6a ≈ R3 | a real finding about what MTP heads use; report with C5 probing |
| Speed-up < 1 on T4 | OM-6 timings | report accepted length (hardware-independent) next to wall-clock |
| Kaggle quota | 3 accounts, T4 x2 | training on Jainam's account, evaluation on Om's and Jai's |

## 8. Working rules

- **Branches:** `jainam/*`, `jai/*`, `om/*`; one PR per task, **always against `main`** (don't stack PRs). Keep PRs small; the others should be able to see them.
- **Contracts first:** never change a signature in `INTERFACES.md` silently. Edit it in the same PR and put "[contract change]" in the PR title.
- **Tests:** every module gets a CPU-only pytest with tiny inputs (`tests/conftest.py` has a tiny tokenizer and model). `python -m pytest -q` passes before merge.
- **No big files in git:** checkpoints, caches and logs live on Kaggle. `results/**/*.json` and figures are committed.
- **Windows:** `$env:PYTHONUTF8 = "1"` before anything that reads Devanagari.
- **dtype:** never hard-code `torch.bfloat16`; use `mtp.device`.

## 9. Using Claude

1. Pull `main`, create your branch (`jai/JI-4-random`).
2. Open Claude in the repo (it reads `CLAUDE.md`), or paste into a chat: the **Context** block of your task file, then `INTERFACES.md`, then "Do JI-4. Write the module and a CPU-only pytest. Match the contracts exactly."
3. Run the tests locally; run the module once on real data.
4. PR to `main` with what it does, the test output, and any number it produced. Post in the team chat.

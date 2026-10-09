# Team Plan · PDF-aligned phases and handoffs

**Word-Group Guided Multi-Token Prediction for Hindi and Marathi**
Team: Jainam (modeling & training) · Jai (data & linguistics) · Om (evaluation & infrastructure)

The uploaded [MTP_Indic_Team_Plan.pdf](MTP_Indic_Team_Plan.pdf) is authoritative for phase order and H1–H10 handoffs. The archive already contains implementation progress beyond its snapshot; keep that progress and the tested package interfaces. This update reconciles scheduling without rolling back code. Historical measurements below come from the repository's run notes, not a new run here.

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
| Word groups | contract code: grouper registry, `hi_rules_v0` (legacy rules), `hi_rules_v1` (JI-1), `label_tokens` (= legacy alignment, verified on 500 sentences). Training labels data on the fly, so **no cache is needed to train** |
| Runs | R0, R1, R2 done on Kaggle (T4 x2, 0.29-0.38 s/step), Gate B passed, published as `mtp-run-R0/R1/R2` (`docs/runs.md`) |
| Not started | random / Trankit groupers, gold set, evaluation suite, speculative decoding, Marathi, contrastive loss, group-aware decoding |

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

Phases A–C are the current priority. Phases D+E (experiments and write-up) are later, after C. Existing P0/P1/P2 research notes do not override this order. Completed modules remain completed; review them rather than rebuilding them.

## 4. Order of work

| Phase | Jainam | Jai | Om |
|---|---|---|---|
| A · Foundation | JN-1 heads → JN-2 design → JN-3 train.py | **JI-1 rules → JI-6a Hindi rules cache → JI-2 guide + 200 gold** | **OM-1 package → OM-2 checkpoints → OM-3 Kaggle notebook** |
| B · Baselines + grouping quality; needs H2 | JN-4 reproduce R0/R1, train R2 | JI-3 Trankit → JI-4 random → JI-5 score/pick → JI-6b Trankit/random caches | OM-4 in-group eval → OM-5 blind annotations |
| C · Method; needs H3/H4 | JN-5 structural pilots → JN-6 contrastive → JN-7 adaptive weighting | JI-7 Marathi rules/gold/cache → JI-8 group statistics | OM-6 speculative engine → OM-7 eval entry point/notebook |
| D+E · Later, after C | JN-8 R3–R10 experiments → JN-9 GroupAware; method/results | JI-9 error analysis; data/annotation/related work | OM-8 evaluate runs → OM-9 tables/figures; setup/evaluation |

Gate B: R0/R1 within about 0.05 of the laptop losses before proceeding. The supplied run notes report this passed. Additional archive tasks (JI-10 probes, JN-10/11 ablations, OM-10) are optional later extensions, not PDF critical-path work.

## 5. Dependencies and H1–H10 handoffs

Each handoff is a merged PR or published Kaggle Dataset plus a team-chat message. A local implementation alone does not prove that a publication handoff landed.

| ID | Deliverable | From → To | Unblocks | Until it lands |
|---|---|---|---|---|
| H1 | OM-1 package/config/device/collate/logging | Om → everyone | modules inside `mtp/` | standalone modules matching INTERFACES |
| H2 | OM-2 checkpoints + OM-3 Kaggle training notebook | Om → Jainam | JN-4 onward on Kaggle | JN-1–JN-3 and laptop tests |
| H3 | JI-1 + JI-6a Hindi rules boundary caches | Jai → Jainam, Om | JN-5/JN-6 labels, OM-4 split | hand-built loss tests; R0–R2 need no labels |
| H4 | OM-4 in-group/boundary evaluation | Om → Jainam | structural-loss selection | pilot-local logging |
| H5 | JI-5 grouper decision + JI-6b Trankit/random caches | Jai → Jainam | later R5–R7 | rules cache from H3 |
| H6 | OM-6 speculative engine | Om → Jainam | later JN-9 GroupAware | fake-logit policy tests |
| H7 | JI-7 Marathi caches | Jai → Jainam | later R8–R10 | Hindi runs |
| H8 | OM-5 50 blind annotations | Om → Jai | agreement in JI-5 | score without κ |
| H9 | JI-2 gold sentence IDs, text only | Jai → Om | OM-5 | prepare annotation tooling |
| H10 | JN-4/JN-8 trained runs published on Kaggle | Jainam → Om | OM-8 run evaluation | tiny runs or R1/R2 checkpoints |

### Critical path

**H2 and H3 are the PDF's training blockers.** Om's OM-1 → OM-2 → OM-3 and Jai's JI-1 → JI-6a come first; the gold set, Trankit and random control follow. Nobody needs to wait on Jainam's early tasks; his outgoing run handoff is H10.

Implementation note: the current `train.py` already labels text on the fly, so caches are not a technical requirement for that path. This does **not** remove the planned H3 cache deliverable. Package/checkpoint/notebook code and baselines already exist in the archive; don't regress them. H3 remains unfinished until the Hindi train/eval/FLORES rules caches are built and published.

## 6. Run matrix (archive research detail; experiments scheduled in D+E)

Same seed (42), same data order, eval during training on `eval_small` (pilots on `eval`). Official numbers come from OM-7 on IndicCorp `eval` + FLORES. Every run has a YAML in `configs/`. α = `head_backbone_grad` (0.1 for all k=4 runs).

| ID | P | lang | heads | extra losses | weighting | grouper | why | status |
|---|---|---|---|---|---|---|---|---|
| R0 | P0 | hi | 1 | – | – | – | NTP baseline | done per run notes |
| R1 | P0 | hi | 2 linear (α=1) | – | sum | – | reproduce laptop MTP (gate) | done per run notes |
| R2 | P0 | hi | 4 resblock | – | fixed 0.8^d | – | token-level MTP baseline | done per run notes |
| pilots | P0 | hi | 4 resblock | S2 / S3 / S3-chain / S23 / S3-all | fixed | v0 | pick the structural loss (+ `pilot_R2ref`) | ready |
| R3 | P0 | hi | 4 resblock | structural (pilot winner) | fixed | best | method, part 1 | after pilots |
| R6a | P0 | hi | 4 resblock | = R3 | fixed | **random** | control: is it the linguistics? | needs JI-4 |
| R5 | P1 | hi | 4 resblock | structural (+ contrastive if R4 helps) | adaptive (pilot winner) | best | full method | after JN-7 pilots |
| R6 | P1 | hi | = R5 | | | **random** | control for the full method | |
| R4 | P2 | hi | 4 resblock | structural + contrastive | fixed | best | does contrastive add anything? | needs JN-6 |
| R7 | P2 | hi | = R5 | | | other grouper | grouper ablation | needs JI-3/JI-5 |
| R8 | P1 | mr | 1 | – | – | – | Marathi NTP | |
| R9 | P1 | mr | 4 resblock | – | fixed | – | Marathi MTP | |
| R10 | P1 | mr | = R5 | | | mr best | Marathi full method | needs JI-7 |
| A1 | P2 | hi | R2 with α = 0 and 1 | | | | C4 ablation, full length | |

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

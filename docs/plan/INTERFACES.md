# Interface Contracts (v2)

**Project:** Word-Group Guided Multi-Token Prediction for Hindi and Marathi

The single source of truth for how the pieces connect. Code that matches these signatures can be written by any of us independently and will plug together. To change a contract: edit this file in the same PR and put "[contract change]" in the PR title.

Status tags: **[exists]** = on `main` with tests; **[todo: name]** = to be written by that person.

---

## 0. What exists

| module | what | status |
|---|---|---|
| `mtp/config.py` | YAML ↔ attribute namespace, `apply_overrides`, `cfg_get` | [exists] |
| `mtp/device.py` | `pick_device`, `pick_dtype`, `native_bf16`, `autocast_ctx`, `make_scaler` | [exists] |
| `mtp/utils/logging.py` | `MetricLogger`, `read_metrics` | [exists] |
| `mtp/data/corpus.py` | `load_split(lang, split, n)` | [exists] |
| `mtp/data/collate.py` | `Collator(pad_id)` | [exists] |
| `mtp/data/grouping/base.py` | `Grouper` protocol, `REGISTRY`, `get_grouper`, `check_partition` | [exists] |
| `mtp/data/grouping/hindi_rules.py` | `hi_rules_v0`, `hi_rules_v1` | [exists] |
| `mtp/data/grouping/align.py` | `label_tokens`, `label_batch` | [exists] |
| `mtp/model/heads.py` | `MTPModel`, `MTPOutput` | [exists] |
| `mtp/model/build.py` | `build_model(cfg, device)`, `load_tokenizer` | [exists] |
| `mtp/model/checkpoint.py` | `save_run`, `latest_step`, `load_weights`, `load_run` | [exists] |
| `mtp/losses/mtp_ce.py` | `per_head_ce`, `shift_targets` | [exists] |
| `mtp/losses/structural.py` | `StructuralLoss` (S2, S3_h0, S3_chain, S3_all, S23, S3_mix, S23_mix) | [exists] |
| `mtp/losses/weighting.py` | `LossWeighting` (fixed, uncertainty, dwa) | [exists] |
| `scripts/train.py` | training entry point | [exists] |
| `notebooks/kaggle_train.ipynb` | Kaggle training | [exists] |
| `mtp/losses/contrastive.py` | SupCon over word groups | [todo: Jainam, P2] |
| `mtp/eval/draft_policy.py` | `DraftPolicy`, `FixedK`, `ConfidenceCut`, `POLICIES` registry, `get_policy` | [exists] |
| `mtp/eval/group_aware.py` | `GroupAware` policy | [todo: Jainam] |
| `mtp/eval/head_accuracy.py` | `evaluate_heads` (per-head loss/ppl/top-k, in-group split, token dump; ppl lives here, no separate `perplexity.py`) | [exists] |
| `mtp/eval/spec_decode.py` | self-speculative decoding engine (`generate`, `greedy_generate`, `evaluate_spec_decode`, `make_prompts`) | [exists] |
| `scripts/evaluate.py`, `notebooks/kaggle_eval.ipynb` | evaluation entry point: per-head eval + `--spec_decode` → §10 JSON | [exists] |
| `mtp/data/grouping/{random_grouper,trankit_grouper,marathi_rules,random_refit,words}.py` | groupers; original random stays teammate-owned | [exists; Trankit live setup optional; Stanza smoke tested] |
| `mtp/eval/group_metrics.py`, `scripts/{score_groupers,group_stats,probe_layers,annotate}.py` | grouper quality, statistics, probing | [exists; human gold/full C5 run pending] |
| `mtp/data/boundary_cache.py`, `scripts/build_boundary_cache.py` | optional pre-labelled cache | [exists; production cache publication pending] |
| `scripts/make_tables.py` | tables + figures | [todo: Om] |

`legacy/` holds the original laptop scripts (see `legacy/README.md`).

Rule: **nothing under `mtp/` may download a model or dataset at import time.** Loading happens inside functions.

## 1. Repo layout

```
mtp/            config.py device.py
  data/         corpus.py collate.py boundary_cache.py*  grouping/{base,align,hindi_rules,marathi_rules*,random_grouper*,trankit_grouper*}.py
  model/        heads.py build.py checkpoint.py
  losses/       mtp_ce.py structural.py weighting.py contrastive.py*
  eval/         perplexity.py* head_accuracy.py* spec_decode.py* draft_policy.py* group_aware.py* group_metrics.py*
  utils/        logging.py
scripts/        train.py jn2_structural_stats.py evaluate.py* make_tables.py* build_boundary_cache.py*
                score_groupers.py* group_stats.py* probe_layers.py* annotate.py*
configs/        R0..R10, pilot_*.yaml (Jainam owns run configs)
notebooks/      kaggle_train.ipynb kaggle_eval.ipynb*
data/gold/      hi_gold.jsonl* mr_gold.jsonl* hi_gold_om50.jsonl*
results/        eval JSON, tables, figures (committed)
docs/           plan/, design notes, runs.md, annotation_guide.md*
tests/          pytest, CPU-only (conftest.py: tiny tokenizer + tiny model fixtures)
legacy/         original scripts, unchanged
```
(* = todo)

## 2. Grouping contract (Jai owns; everyone consumes)

```python
# mtp/data/grouping/base.py  [exists]
GROUP_TYPES = ("single", "aux_chain", "postposition", "compound_postposition", "light_verb", "other")
REGISTRY = {"hi_rules_v0": "mtp.data.grouping.hindi_rules:HindiRuleGrouperV0", ...}   # name -> "module:Class"

class Grouper(Protocol):
    name: str            # "hi_rules_v1", "trankit", "random", ...
    lang: str            # "hi" | "mr"
    def group_words(self, sentence: str) -> list[list[str]]: ...   # concatenation == sentence.split()
    def group_types(self, sentence: str) -> list[str]: ...         # one GROUP_TYPES entry per group

def get_grouper(name: str, lang: str = "hi", **kwargs) -> Grouper   # constructor gets (lang=..., **kwargs)
def check_partition(sentence: str, groups) -> None                  # raises ValueError
```
- A new grouper = a class with `__init__(self, lang="hi", **kw)` + one line in `REGISTRY`.
- Optional, for slow groupers (Trankit): `batch_group_words(sentences) -> list[list[list[str]]]`.
- `random` grouper: `RandomGrouper(lang, histogram: dict[int, float], seed: int = 0)`, deterministic per (seed, sentence), and `RandomGrouper.fit_from(grouper, sentences, seed=0)` to copy another grouper's group-length histogram. Registered name `random`; `get_grouper("random", lang, histogram=...)` or a default histogram measured from `hi_rules_v1` on 5,000 train sentences and stored in the module.

```python
# mtp/data/grouping/align.py  [exists]
def label_tokens(sentence, tokenizer, grouper, max_length=128) -> dict
def label_batch(sentences, tokenizer, grouper, max_length=128) -> list[dict]   # same, one tokenizer call
# dict: input_ids, attention_mask, group_start (1 = first token of a group), group_id (0,0,1,2,2,...; -1 = special/pad)
```
Rules: uses `offset_mapping` on the full sentence (a token starts group g if g's first character is in `[tok_start, tok_end)`, which covers SentencePiece's leading space); `group_id` is non-decreasing over real tokens; truncation may cut the last group. Identical to `legacy/boundary_alignment.py` on all 500 eval sentences.

## 3. Boundary cache (optional; Jai)

Training does not need it: `scripts/train.py` labels raw text on the fly with `label_batch`. The cache is for slow groupers (Trankit) and for a fixed, shareable artifact.

A HuggingFace `datasets.Dataset` (`save_to_disk`) per `boundary_cache/{lang}_{grouper}_{split}/` with columns `text, input_ids, attention_mask, group_start, group_id` (unpadded; §2), and `meta.json` **inside** the folder: `{"grouper", "tokenizer", "max_length", "n_rows", "words_per_group", "tokens_per_group", "pct_tokens_group_start"}`. `train.py` checks `meta.json["tokenizer"] == cfg.model_name`. Splits as in §4. One Kaggle Dataset `mtp-boundary-cache`.

Collate (`Collator(pad_id)`) right-pads `input_ids` with pad id, `attention_mask` and `group_start` with 0, `group_id` with -1; works without the group columns.

## 4. Fixed evaluation sets

`load_split(lang, split, n=None) -> list[str]` [exists]. IndicCorpV2 has a blank row between documents, so splits are on **raw row numbers with blank rows dropped**:

| split | content | size (hi) | eval dataset name |
|---|---|---|---|
| `eval` | IndicCorp raw rows 0-999 | ~500 sentences | `indiccorp_eval` |
| `eval_small` | raw rows 0-99 (the laptop runs' held-out set) | ~50 | `indiccorp_eval_small` |
| `train` | the first n non-blank rows from raw row 1,000 | n | – |
| `flores` | FLORES-200 devtest (`hin_Deva` / `mar_Deva`) | 1,012 | `flores_hi`, `flores_mr` |

Gold grouping sets: `data/gold/hi_gold.jsonl`, `data/gold/mr_gold.jsonl` (§7).

## 5. Model contract (Jainam)

```python
# mtp/model/heads.py  [exists]
@dataclass
class MTPOutput:
    logits: list[torch.Tensor]   # length k; logits[d] is [B, T, V]; head d predicts token t+d+1
    hidden: torch.Tensor         # [B, T, H] last hidden state (after the final norm)
    aux: dict                    # see below

class MTPModel(nn.Module):
    def __init__(self, base_model, num_heads: int, head_type: str = "linear", n_layers: int = 1,
                 backbone_grad: float = 1.0, boundary_probes: bool = False, **kw): ...
    def forward(self, input_ids, attention_mask=None, use_cache: bool = False, **kw) -> MTPOutput: ...
    def head_state_dict(self) -> dict          # extra heads + probes = heads.pt
    def load_head_state_dict(self, state)      # also accepts the legacy extra_heads-only format
    def draft_chain(self, h, t0: int, n: int) -> list[int]   # head_type "seq" only: n greedy chain drafts after t0
    num_heads: int; head_type: str; extra_heads: nn.ModuleList; boundary_probes: nn.ModuleList
```
- `num_heads` counts all heads including head 0 (the base LM head).
- `head_type`: `"linear"` (fresh `nn.Linear(hidden, vocab)`, R1), `"resblock"` (h_d = h + SiLU(W h + b), W, b zero-init, logits = frozen `lm_head(h_d)`), `"seq"` (sequential heads: s_0 = h_t, s_d = s_{d-1} + SiLU(W_d [s_{d-1}; e(x_{t+d})] + b_d), e = detached input embeddings, W_d zero-init, logits = `lm_head(s_d)`; `forward` teacher-forces x_{t+d} from `input_ids` (zeros past the end), so decoding gets drafts from `draft_chain`; `spec_decode.generate` does this automatically).
- `backbone_grad`: share of heads 1..k-1's gradient that reaches the backbone (0 = detached). Forward is identical for every value.
- `aux["head_hidden"]`: list of k `[B, T, H]` (what each head's output layer reads). `aux["boundary_logits"]`: list of k `[B, T]`, logit that token t+d+1 starts a word group (only with probes). `aux["past_key_values"]`: only with `use_cache=True`.

```python
# mtp/model/build.py  [exists]
def build_model(cfg, device) -> tuple[MTPModel, tokenizer]   # tokenizer -> base (pick_dtype) -> LoRA -> MTPModel
```

## 6. Run config contract

YAML loaded by `load_config` into an attribute namespace (`cfg.optim.lr`). Optional fields are read with `cfg_get(cfg, "a.b", default)`, so adding a field never breaks old configs. CLI: `--set optim.lr=1e-4` (`apply_overrides`).

```yaml
run_name: R2_hi_mtp_k4_resblock
lang: hi
model_name: LingoIITGN/ganga-1b
num_heads: 4
head_type: resblock               # linear | resblock | seq
head_layers: 1                    # resblocks per head
head_backbone_grad: 0.1           # α; 1.0 = full joint training
lora: {r: 8, alpha: 16, dropout: 0.05, targets: [q_proj, v_proj]}
optim: {lr: 2.0e-4, batch_size: 8, grad_clip: 1.0, max_steps: 12500, grad_accum: 1, weight_decay: 0.01}
data: {cache_dir: /kaggle/input/mtp-boundary-cache, grouper: hi_rules_v0, max_length: 128, eval_split: eval_small, eval_n: 100}
losses:
  structural: {enabled: false, variant: null, weight: 1.0, lambda_s2: 0.1, lambda_s3: 0.5, lambda_s3_all: 0.25, s3_teacher: null}
  contrastive: {enabled: false, weight: 0.05, temperature: 0.1}
weighting: {scheme: fixed, head_decay: 0.8}   # fixed | uncertainty | dwa; + fix_head0, dwa_window, dwa_temperature, lr
log_every: 20
eval_every: 250
save_every: 500
keep_steps: []                    # step folders never deleted
seed: 42
dtype: auto                       # auto | bf16 | fp16 | fp32
# git_commit: written by train.py into the run's config.yaml
```
Structural variants: `S2`, `S3_h0`, `S3_chain`, `S3_all`, `S23`, `S3_mix`, `S23_mix` (see `docs/design_structural_loss.md`). `s3_teacher` (`h0` | `chain`) overrides the default teacher of `S23` (h0), `S3_mix` and `S23_mix` (chain); `lambda_s3_all` weights the `struct/consistency_all/h{d}` terms of the `_mix` variants. `TBD` in a config = not decided yet; train.py fails loudly on it.

## 7. Gold annotation format (Jai)

`data/gold/hi_gold.jsonl`, one sentence per line:
```json
{"id": "hi_0001", "source": "flores_devtest", "text": "मैं कल बाजार जा रहा था।", "groups": [[0],[1],[2],[3,4,5]], "types": ["single","single","single","aux_chain"], "annotator": "jai", "notes": ""}
```
`groups` are lists of word indices into `text.split()` (contiguous, covering every word); `types` from `GROUP_TYPES`, one per group.

## 8. Checkpoint contract

```python
# mtp/model/checkpoint.py  [exists]
def save_run(run_dir, model, cfg, step, optimizer=None, weighting=None, keep_last=2) -> None
def latest_step(run_dir) -> int | None
def load_weights(model, step_dir) -> None                                   # LoRA + heads.pt into a built model
def load_run(run_dir, device="auto", step=None) -> tuple[MTPModel, tokenizer, cfg]   # eval mode; model.loaded_step
```
```
runs/{run_name}/
  config.yaml            once, incl. git_commit
  metrics.jsonl
  step_{N}/lora/         PEFT save_pretrained
  step_{N}/heads.pt      model.head_state_dict()
  step_{N}/weighting.pt  LossWeighting state
  step_{N}/optim.pt      written by train.py: {"step", "optimizer", "scaler", "rng"}
```
Only the last 2 step folders are kept, plus `keep_steps`. A published run = this folder as Kaggle Dataset `mtp-run-<ID>`.

## 9. Metrics log contract

`metrics.jsonl`, one object per line: `{"step": 500, "split": "train" | "eval", "name": "...", "value": 6.12}`.

| name | split | meaning |
|---|---|---|
| `loss/head{d}`, `loss/total` | train, eval | per-head CE (mean over valid positions) |
| `struct/boundary_bce/h{d}`, `struct/consistency/h{d}` | train | structural terms, unweighted |
| `weight/head{d}`, `weight/struct/...` | train | effective loss weights |
| `acc/top1/head{d}` | eval | top-1 |
| `acc/top1_in_group/head{d}`, `acc/top1_at_boundary/head{d}` | eval | top-1 split by whether target t+d+1 is in the source token t's group |
| `lr`, `grad_norm`, `time/sec_per_step` | train | |

## 10. Evaluation outputs (Om)

`results/{run_name}/eval_{dataset}.json`:
```json
{
  "run_name": "R2_hi_mtp_k4_resblock", "dataset": "flores_hi", "step": 12500, "grouper": "hi_rules_v0", "git_commit": "...",
  "per_head": [{"head": 0, "loss": 3.05, "ppl": 21.1, "top1": 0.41, "top5": 0.63, "n": 25000,
                "top1_in_group": 0.55, "top1_at_boundary": 0.33, "loss_in_group": 1.6, "loss_at_boundary": 3.5,
                "n_in_group": 8000, "n_at_boundary": 17000}],
  "spec_decode": [{"policy": "fixed_k", "mean_accepted_len": 1.42, "accept_rate_per_head": [0.42, 0.20, 0.09],
                   "tokens_per_sec": 31.5, "greedy_tokens_per_sec": 24.0, "speedup": 1.31,
                   "outputs_match_greedy": true, "match_rate": 1.0, "max_mismatch_margin": null,
                   "fp32_check": {"n_prompts": 20, "match_rate": 1.0, "outputs_match_greedy": true, "max_mismatch_margin": null},
                   "group_integrity": 0.71, "n_prompts": 200}]
}
```
`outputs_match_greedy` = every prompt identical to greedy in the timed run; `match_rate` = share of prompts identical; `max_mismatch_margin` = largest greedy top-1 minus top-2 logit at a point where an output diverged (tiny = a near-tie flipped by fp16 rounding). `fp32_check` (only under fp16 autocast, e.g. T4) re-runs the first 20 prompts with autocast off: it must be all identical, else the engine has a bug (its own `max_mismatch_margin` tells a sub-1e-5 fp32 tie from a real bug); `null` when the run has no autocast.

In-group / at-boundary are defined exactly as `scripts/train.py`'s `evaluate` (target t+d+1 in source t's group). For the same model, data and step, OM-4's numbers must match train.py's logged eval to 1e-3. A split with no positions (`n_in_group` or `n_at_boundary` = 0, e.g. head 3 on short groups) has `null` top1/loss; train.py logs 0.0 there.

```python
# mtp/eval/head_accuracy.py  [exists]
def evaluate_heads(model, examples, collator, cfg, device, top_k=(1, 5), dump_path=None, texts=None, tokenizer=None) -> list[dict]
# one dict per head (the per_head fields above, top{k} for each k); dump_path needs tokenizer (token strings);
# texts (aligned with examples) fill the dump's "text" field, null if not given
```

Verify-pass cost (`--bench_verify`), `results/{run_name}/bench_verify.json`: the wall-clock of ONE verification pass (full MTPModel forward on a cached 64-token prefix, then rolled back) vs the number of positions it scores (chain drafts or tree nodes), for sizing draft trees (#33):
```json
{"run_name": "...", "step": 12500, "device": "cuda:0", "gpu": "Tesla T4", "fp16_autocast": true, "prefix_len": 64,
 "greedy_ms": 36.0, "passes": [{"positions": 1, "ms": 37.0, "ms_p90": 38.1, "x_greedy": 1.03}, {"positions": 25, "ms": 40.2, "ms_p90": 41.0, "x_greedy": 1.12}]}
```
`greedy_ms` = greedy's single-token base-LM pass; `ms` = median over repeats. (Example numbers are illustrative.)

Per-token dump (`--dump_tokens`), `results/{run_name}/tokens_{dataset}.jsonl`, one line per (sentence, position, head):
```json
{"sent_id": 17, "text": "मैं कल बाजार जा रहा था।", "token_idx": 5, "token": "▁रहा", "head": 2, "target_idx": 8, "correct": true, "rank": 1, "group_start": 0, "group_id": 3, "target_group_id": 3}
```

## 11. Draft policy contract (Om: engine + FixedK + ConfidenceCut; Jainam: GroupAware)

```python
# mtp/eval/draft_policy.py
class DraftPolicy(Protocol):
    name: str
    def num_draft_tokens(self, head_logits: list[torch.Tensor], context_ids: torch.Tensor, step_state: dict) -> int:
        """head_logits: k tensors [V], every head's logits at the last position (index 0 = head 0).
        context_ids: [T] tokens so far. Return how many of the k-1 drafts (heads 1..k-1) to propose, 0..k-1."""

class FixedK:         # always k-1
class ConfidenceCut:  # stop at the first head d >= 1 whose max softmax prob < tau
class GroupAware:     # stop at the predicted end of the current word group (Jainam, in mtp/eval/group_aware.py)

POLICIES = {"fixed_k": FixedK, "confidence_cut": ConfidenceCut, "group_aware": GroupAware}   # name -> class
```
`GroupAware` lives in its own file so Om and Jainam never edit the same file; `draft_policy.py` imports it inside a `try` until it exists.
`step_state` (filled by the engine every step): `{"boundary_logits": list[float] | None` (k values at the last position, from `aux`), `"tokenizer": tokenizer, "grouper": Grouper | None, "step": int}`.

```python
# mtp/eval/spec_decode.py
def generate(model, tokenizer, prompt_ids, max_new_tokens, policy, grouper=None, *, use_cache=True, amp=None) -> tuple[list[int], dict]
def greedy_generate(model, tokenizer, prompt_ids, max_new_tokens, *, use_cache=True, amp=None) -> tuple[list[int], dict]  # base LM = head 0
def evaluate_spec_decode(model, tokenizer, prompts, policies, max_new_tokens=64, grouper=None, use_cache=True, amp=None) -> list[dict]  # §10 spec_decode entries
def make_prompts(texts, tokenizer, n=200, min_words=8, max_words=16, seed=0) -> list[list[int]]
# amp: a context-manager factory, e.g. lambda: autocast_ctx(cfg). The engine clamps a policy's answer to 0..k-1.
```
Tree drafts (issue #33; Jainam: policies in `mtp/eval/tree_policy.py`; Om: `generate_tree()` verifies them):
```python
# mtp/eval/tree_policy.py
class TreePolicy(Protocol):
    name: str
    max_nodes: int   # the largest tree it ever returns (size the verify pass / benchmark with it)
    def tree(self, head_logits: list[torch.Tensor], context_ids: torch.Tensor, step_state: dict) -> list[tuple[int, int, int]]:
        """Nodes (parent, d, rank), parents before children. parent = node index, -1 = the root t0
        (head 0's argmax, always kept); d = the drafting head, = parent's d + 1 (root's d = 0);
        rank = 0-based rank in head d's logits at the last position (0 = argmax)."""

class StaticTree:   # same tree every step: fitted Medusa sparse tree, or StaticTree.from_widths((3, 2, 1))
class EntropyTree:  # one fitted tree per entropy cell (head d's entropy above its calibration median)
TREE_POLICIES = {"static_tree": StaticTree, "entropy_tree": EntropyTree}
def draft_tokens(nodes, head_logits) -> list[int]        # token id per node
def accepted_length(nodes, ranks) -> int                 # oracle count; generate_tree must match it
def save_policy(policy, path); def load_policy(path) -> TreePolicy   # JSON
```
`scripts/fit_tree.py --run_dir <run>` fits both on IndicCorp train prompts and writes `results/{run_name}/tree_policy_{static,entropy}_n{N}.json` plus `tree_fit.json` (oracle accepted drafts/step on IndicCorp eval).

Correctness: the output must be **identical** to plain greedy decoding with head 0 (the test). Stats: tokens generated, forward passes, mean accepted length, per-head acceptance, tokens/s, Group Integrity (share of accepted multi-token spans that end on a group boundary).

## 12. Grouping statistics and probing outputs (Jai)

- `results/grouping/stats_{lang}.json`: `{"grouper": ..., "tokenizer": ..., "n_sentences": ..., "words_per_group": ..., "tokens_per_group": ..., "tokens_per_word": ..., "same_group_rate": {"1": 0.33, "2": 0.08, "3": 0.025, "4": 0.008}}`, one object per (grouper, tokenizer) in a list.
- `results/grouping/{lang}_scores.json`: per grouper, boundary P/R/F1, exact-group accuracy, per type; κ.
- `results/probing/{lang}_{grouper}.json`: `{"layer": [0..L], "f1": [...], "acc": [...], "majority_baseline": ...}`.

## 13. Jai implementation extensions (additive; 2026-10-10 sync)

Existing model, training, checkpoint, eval and grouping signatures are unchanged. The teammate-owned `random` registry entry and its v0 histogram remain unchanged. New names:

- `hi_rules_v1`, `mr_rules_v1`: language-specific rule groupers (Hindi has alias `HindiRuleGrouper` for `HindiRuleGrouperV1`).
- `words`: whitespace singleton baseline, hi or mr.
- `trankit`, `stanza`: separate lazy parser backends; optional injected pipeline for CPU tests. No implicit fallback or silently changed grouper name. `batch_group_words` is available. A bounded last-sentence cache avoids parsing again for `group_types`.
- `random_hi_v1`, `random_mr_v1`: measured v1 controls from `histograms_v1.json`. The existing `random` is reproducible for R6a.

```python
# mtp/data/annotation.py
validate_record(row, require_groups=True) -> dict
read_jsonl(path, require_groups=False) -> list[dict]
parse_spans(value, n_words) -> list[list[int]]
# mtp/data/boundary_cache.py
build_cache(texts, tokenizer, grouper, out_dir, max_length=128, batch_size=64, overwrite=False) -> dict
load_cache(path, tokenizer_name=None, max_length=None) -> (Dataset, meta)
# mtp/data/flores.py: original FLORES-200 public archive, no import-time downloads
load_flores_text(lang, split='devtest', archive_path=None) -> list[str]
# mtp/eval/group_statistics.py
measure_grouping(texts, tokenizer, grouper, max_length=None) -> dict
# mtp/eval/group_metrics.py
score_records(gold, predictions) -> dict
agreement(a, b) -> dict
# mtp/eval/layer_probes.py
run_layer_probes(model, tokenizer, grouper, train_texts, eval_texts,
                 device='cpu', max_length=128, max_positions=64, seed=42) -> dict
run_multi_layer_probes(model, tokenizer, groupers, train_texts, eval_texts,
                       device='cpu', max_length=128, max_positions=64, seed=42) -> dict[str, dict]
```

Gold format §7 is unchanged; optional `annotation_status` records review provenance. `unreviewed_rule_draft` is not human gold. Scoring rejects drafts by default and never picks a final winner from drafts or partial gold. Blind agreement requires the 50 reviewed IDs. Exact-group accuracy is matched gold spans / all gold spans; per-type scores require both exact span and type. Cohen κ uses every between-word position, including nonattachments; degenerate/no-variation κ is null.

Statistics §12 remain a list of objects, with additive sample hashes, raw pair counts, top groups and truncation metadata. Full-text statistics default to no truncation. Special tokens are excluded. Probe outputs retain layer/F1/accuracy/majority fields plus methodology/provenance. All label sets share one frozen-model forward per sentence; logistic updates run on CPU.

The cache metadata remains inside each cache directory to match current `train.py`. Cache and gold publication, human annotations, full C5 model runs and final grouper choice are distinct from implementation completion. See `docs/JAI_HANDOFF.md`.

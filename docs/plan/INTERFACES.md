# Interface Contracts

**Project:** Word-Group Guided Multi-Token Prediction with Adaptive Loss Weighting for Hindi and Marathi

This file is the single source of truth for how the pieces connect. Every task file points here.
If you need to change a contract, change it here first and tell the other two people.
Code that matches these signatures can be written by any of us independently and will plug together.

---

## 0. Ground truth: what already exists in the repo

| File | What it does | Status |
|---|---|---|
| `test_load_model.py` | Loads `LingoIITGN/ganga-1b`, generates Hindi | Done |
| `test_lora_training.py` | PEFT/LoRA wrap + 3 dummy steps | Done |
| `check_datasets.py` | Loads `ai4bharat/IndicCorpV2`, config `indiccorp_v2`, split `hin_Deva` | Done |
| `train_ntp_baseline_final.py` | NTP baseline: LoRA r=8 on `q_proj,v_proj`, lr 2e-4, batch 8, max_len 128, 100 held-out sentences, grad clip 1.0 | Done. Held-out loss **3.05** |
| `medusa_heads.py` | `MedusaWrapper(base_model, num_extra_heads)`: extra heads are plain `nn.Linear(hidden, vocab)` on the last hidden state | Done (k=2 tested) |
| `train_mtp_baseline.py` | Token-level MTP, k=2, same LoRA setup, summed per-head CE | Done. 12,500 batches: head0 **3.09**, head1 **8.10 → 5.84** |
| `word_group_boundaries.py` | `find_word_groups(sentence) -> list[list[str]]`, rule-based (aux + postposition lists) | v0, **short** lists (21 aux, 8 postpositions). The expanded lists (light verbs, वाला-forms, more particles, a `COMPOUND_POSTPOSITIONS` set that is defined but unused) live in `validate_on_real_data.py` |
| `boundary_alignment.py` | `get_aligned_boundary_labels(sentence)` returns token-level `1 = group start` labels via `offset_mapping` | Done, verified by eye |
| `validate_on_real_data.py` | Despite the name, the on-disk version is a copy of `word_group_boundaries.py` with the expanded lists; the corpus-coverage code that produced 25% words matched / 1.33 words/group is not in it | Lists only |

These all live flat in the repo root. Step 1 of the plan moves them into the package below (old copies go to `legacy/`).

---

## 1. Target repo layout

```
mtp-indic-research/
├── mtp/                          # importable package:  import mtp
│   ├── __init__.py
│   ├── config.py                 # RunConfig dataclass + YAML load/save               [Om]
│   ├── device.py                 # dtype/autocast selection (bf16 vs fp16 vs fp32)    [Om]
│   ├── data/
│   │   ├── corpus.py             # streaming load, fixed held-out split, batching     [Om]
│   │   ├── collate.py            # pads input_ids / attention_mask / group_id         [Om]
│   │   ├── boundary_cache.py     # build + load pre-labelled datasets                 [Jai]
│   │   └── grouping/
│   │       ├── base.py           # Grouper protocol + registry                        [Jai]
│   │       ├── hindi_rules.py    # rule grouper, Hindi                                [Jai]
│   │       ├── marathi_rules.py  # rule grouper, Marathi                              [Jai]
│   │       ├── trankit_grouper.py# dependency-parse grouper (hi + mr)                 [Jai]
│   │       ├── random_grouper.py # control: random boundaries, same length dist.     [Jai]
│   │       └── align.py          # word groups -> token labels (from boundary_alignment.py) [Jai]
│   ├── model/
│   │   ├── heads.py              # MTPModel, head variants                            [Jainam]
│   │   └── checkpoint.py         # save_run / load_run                                [Om]
│   ├── losses/
│   │   ├── mtp_ce.py             # per-head shifted CE (from train_mtp_baseline.py)   [Jainam]
│   │   ├── structural.py         # word-group structural loss                         [Jainam]
│   │   ├── contrastive.py        # group contrastive loss                             [Jainam]
│   │   └── weighting.py          # fixed / uncertainty / DWA weighting                [Jainam]
│   ├── eval/
│   │   ├── perplexity.py         # per-head loss + ppl                                [Om]
│   │   ├── head_accuracy.py      # top-1/top-5 per head, split in-group vs boundary   [Om]
│   │   ├── spec_decode.py        # self-speculative decoding engine + speed metrics   [Om]
│   │   ├── draft_policy.py       # pluggable draft-length policies                    [Jainam]
│   │   └── group_metrics.py      # grouper P/R/F1 vs gold, group stats                [Jai]
│   └── utils/
│       └── logging.py            # JSONL metric logger                                [Om]
├── scripts/
│   ├── train.py                  # single training entry point                        [Jainam]
│   ├── evaluate.py               # single eval entry point                            [Om]
│   ├── build_boundary_cache.py   # CLI around boundary_cache                          [Jai]
│   ├── score_groupers.py         # groupers vs gold set                               [Jai]
│   └── make_tables.py            # results/*.json -> tables + plots                   [Om]
├── configs/                      # one YAML per run (see §6)                          [Jainam owns run configs]
├── notebooks/
│   ├── kaggle_train.ipynb                                                             [Om]
│   └── kaggle_eval.ipynb                                                              [Om]
├── data/gold/                    # hand-annotated gold groups                         [Jai]
├── docs/                         # annotation guide, plan, report drafts
├── results/                      # eval JSON + tables + figures (committed)
├── tests/                        # pytest, CPU-only, tiny inputs
├── legacy/                       # the original root-level scripts, untouched
├── requirements.txt
└── README.md
```

Rule: **nothing under `mtp/` may download a model or dataset at import time.** Loading happens inside functions.

---

## 2. Grouping contract (Jai builds, everyone consumes)

```python
# mtp/data/grouping/base.py
from typing import Protocol

class Grouper(Protocol):
    name: str            # e.g. "hi_rules_v1", "trankit", "random"
    lang: str            # "hi" or "mr"
    def group_words(self, sentence: str) -> list[list[str]]:
        """Whitespace words of `sentence`, partitioned into contiguous groups.
        Concatenating all groups in order MUST give exactly sentence.split()."""
    def group_types(self, sentence: str) -> list[str]:
        """One type per group, same order as group_words():
        'single' | 'aux_chain' | 'postposition' | 'compound_postposition' | 'light_verb' | 'other'"""

def get_grouper(name: str, lang: str, **kwargs) -> Grouper: ...
```

```python
# mtp/data/grouping/align.py
def label_tokens(sentence: str, tokenizer, grouper: Grouper, max_length: int = 128) -> dict:
    """Tokenize the FULL sentence once (same call as training) and return:
        input_ids:      list[int]
        attention_mask: list[int]
        group_start:    list[int]  # 1 if token is first token of a word-group, else 0
        group_id:       list[int]  # 0,0,0,1,1,2,...  index of the group each token belongs to
                                   # -1 for special tokens (BOS/EOS) and padding
    All four lists have identical length."""
```

Rules:
- Uses `offset_mapping`, same as `boundary_alignment.py` (SentencePiece leading-space handling included).
- `group_id` is monotonic non-decreasing over real tokens.
- Truncation at `max_length` may cut a group. That is fine; the cut group just ends early.

## 3. Boundary cache contract (Jai builds, Jainam trains on it, Om evaluates on it)

A HuggingFace `datasets.Dataset` saved with `save_to_disk`, uploaded as a Kaggle Dataset.

| column | type | notes |
|---|---|---|
| `text` | str | raw sentence |
| `input_ids` | list[int] | unpadded |
| `attention_mask` | list[int] | unpadded |
| `group_start` | list[int] | §2 |
| `group_id` | list[int] | §2 |

Folder naming: `boundary_cache/{lang}_{grouper}_{split}/` e.g. `hi_hi_rules_v1_train/`, `hi_trankit_eval/`, `mr_mr_rules_v1_train/`.
Splits: `train` (N configurable, default 200k), `eval` (the **fixed** held-out set, see §4), `flores` (FLORES-200 devtest).
A `meta.json` next to each folder: grouper name, tokenizer name, max_length, n rows, avg group length (words + tokens), % tokens that are group starts.

Collate (`mtp/data/collate.py`, Om): right-pad `input_ids` with pad id, `attention_mask` with 0, `group_start` with 0, `group_id` with -1. Returns tensors.

## 4. Fixed evaluation sets

So numbers from different people are comparable:
- **IndicCorp held-out:** first 1,000 rows of the `hin_Deva` stream (and `mar_Deva` for Marathi). Training draws from row 1,000 onward. (The current scripts use the first 100. Keep that subset as `eval_small` for quick checks.)
- **FLORES-200 devtest:** `hin_Deva`, `mar_Deva`.
- **Gold grouping set:** `data/gold/hi_gold.jsonl`, `data/gold/mr_gold.jsonl` (§7).

## 5. Model contract (Jainam builds)

```python
# mtp/model/heads.py
@dataclass
class MTPOutput:
    logits: list[torch.Tensor]   # length k; logits[i] is [B, T, V]; head i predicts token t+i+1
    hidden: torch.Tensor         # [B, T, H] last hidden state (for contrastive loss)
    aux: dict                    # optional extras, e.g. boundary-prediction logits

class MTPModel(nn.Module):
    def __init__(self, base_model, num_heads: int, head_type: str = "linear", **kw): ...
    def forward(self, input_ids, attention_mask=None) -> MTPOutput: ...
```
- `num_heads` counts **all** heads including head 0 (the original LM head). Current `medusa_heads.py` uses `num_extra_heads`; `num_heads = num_extra_heads + 1`.
- `head_type`: `"linear"` (current), `"resblock"` (Medusa-1 style residual block, init so it starts as identity + the LM head).
- Optional `__init__` kwargs (JN-1/JN-3/JN-5): `n_layers=1` (resblocks per head), `backbone_grad=1.0` (scales the gradient heads 1..k-1 send into the backbone; 0 = detached), `boundary_probes=False` (one linear probe per head for S2 / `GroupAware`).
- `forward(..., use_cache=False)`. `aux` keys: `head_hidden` (list of k `[B, T, H]`, the state each head's output layer reads), `boundary_logits` (list of k `[B, T]`, only with probes), `past_key_values` (only with `use_cache=True`).
- `head_state_dict()` / `load_head_state_dict(state)`: every non-base parameter (extra heads + probes). This is what `heads.pt` holds. `load_head_state_dict` also accepts the legacy `extra_heads.state_dict()` format.

## 6. Run config contract (Om builds loader, Jainam writes configs)

```yaml
run_name: hi_mtp_k4_struct_adaptive
lang: hi
model_name: LingoIITGN/ganga-1b
num_heads: 4
head_type: resblock
lora: {r: 8, alpha: 16, dropout: 0.05, targets: [q_proj, v_proj]}
optim: {lr: 2.0e-4, batch_size: 8, grad_clip: 1.0, max_steps: 12500}
data: {cache_dir: /kaggle/input/mtp-boundary-cache, grouper: hi_rules_v1, max_length: 128}
losses:
  structural: {enabled: true, variant: S2, weight: 0.1}
  contrastive: {enabled: false, weight: 0.05, temperature: 0.1}
weighting: {scheme: uncertainty}     # fixed | uncertainty | dwa
eval_every: 250
save_every: 500
seed: 42
dtype: auto                          # auto | bf16 | fp16 | fp32
```
Optional fields `scripts/train.py` reads with defaults (JN-3; `RunConfig` must accept them): `head_layers: 1`, `head_backbone_grad: 1.0`, `optim.grad_accum: 1`, `optim.weight_decay: 0.01`, `data.eval_n: 100`, `data.eval_split: eval_small`, `losses.structural.{lambda_s2: 0.1, lambda_s3: 0.5, s3_teacher: h0}`, `weighting.head_decay: 0.8`, `log_every: 20`, `keep_steps: []`, and `git_commit` (written by train.py). CLI overrides: `apply_overrides(cfg, ["optim.lr=1e-4", ...])`.

## 7. Gold annotation format (Jai)

`data/gold/hi_gold.jsonl`, one sentence per line:
```json
{"id": "hi_0001", "source": "flores_devtest", "text": "मैं कल बाजार जा रहा था।", "groups": [[0],[1],[2],[3,4,5]], "annotator": "jai", "notes": ""}
```
`groups` are lists of word indices into `text.split()`.

## 8. Checkpoint contract (Om builds, Jainam calls)

```python
# mtp/model/checkpoint.py
def save_run(run_dir: str, model: MTPModel, cfg: RunConfig, step: int, optimizer=None, weighting=None) -> None
def load_run(run_dir: str, device: str = "auto") -> tuple[MTPModel, "Tokenizer", RunConfig]
def latest_step(run_dir: str) -> int | None   # for resuming after a Kaggle session ends
```
Layout:
```
runs/{run_name}/
  config.yaml
  step_{N}/lora/            # PEFT save_pretrained
  step_{N}/heads.pt         # model.head_state_dict(): extra heads + boundary probes
  step_{N}/weighting.pt     # learned loss weights, if any
  step_{N}/optim.pt         # optional, for resume
  metrics.jsonl
```
`scripts/train.py` calls `save_run(..., optimizer=None, weighting=...)` and then writes `optim.pt` itself: `{"step", "optimizer", "scaler", "rng"}`. Resume restores LoRA via `set_peft_model_state_dict`, `heads.pt` via `load_head_state_dict`, then `optim.pt`.

## 9. Metrics log contract (Om builds, everyone writes)

`metrics.jsonl`, one JSON object per line:
```json
{"step": 500, "split": "train", "name": "loss/head1", "value": 6.12}
{"step": 500, "split": "eval",  "name": "loss/head0", "value": 3.07}
{"step": 500, "split": "train", "name": "weight/structural", "value": 0.083}
```
Name prefixes: `loss/`, `acc/`, `weight/`, `lr`, `time/`.

## 10. Eval output contract (Om builds)

`results/{run_name}/eval_{dataset}.json`:
```json
{
  "run_name": "...", "dataset": "flores_hi", "step": 12500,
  "per_head": [{"head": 0, "loss": 3.05, "ppl": 21.1, "top1": 0.41, "top5": 0.63,
                "top1_in_group": 0.55, "top1_at_boundary": 0.33}],
  "spec_decode": {"policy": "fixed_k", "mean_accepted_len": 1.42, "accept_rate_per_head": [0.42],
                  "tokens_per_sec": 31.5, "greedy_tokens_per_sec": 24.0, "speedup": 1.31,
                  "outputs_match_greedy": true},
  "group_integrity": 0.71
}
```

## 11. Draft policy contract (Om builds engine, Jainam writes policies)

```python
# mtp/eval/draft_policy.py
class DraftPolicy(Protocol):
    name: str
    def num_draft_tokens(self, head_logits: list[torch.Tensor], context_ids: torch.Tensor, step_state: dict) -> int:
        """Return how many of the k-1 drafted tokens to propose this step (0..k-1)."""

class FixedK:         # always propose all k-1       (baseline, Om ships this one)
class ConfidenceCut:  # stop at first head whose max prob < tau
class GroupAware:     # stop at predicted group boundary (Jainam)
```
`spec_decode.generate(model, tokenizer, prompt_ids, max_new_tokens, policy) -> (output_ids, stats)`.
Greedy verification: final output **must** be identical to plain greedy decoding with head 0. That is the correctness test.

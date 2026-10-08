# legacy/

The original laptop scripts (before the `mtp/` package), kept unchanged for reference and reproducibility of the first numbers (NTP 3.05, MTP k=2 head1 5.84, grouper v0 25% / 1.33 words per group). Nothing in `mtp/` or `scripts/` imports these, except `scripts/jn2_structural_stats.py`, which reads the v0 word lists from here.

| file | replaced by |
|---|---|
| `medusa_heads.py` | `mtp/model/heads.py` (`MTPModel`) |
| `train_ntp_baseline_final.py`, `train_mtp_baseline.py` | `scripts/train.py` + `configs/R0.yaml`, `configs/R1.yaml` |
| `word_group_boundaries.py` (short lists) | `mtp/data/grouping/hindi_rules.py` (`hi_rules_v0`) |
| `validate_on_real_data.py` (expanded lists, no coverage code) | starting point for `hi_rules_v1` (JI-1) |
| `boundary_alignment.py` | `mtp/data/grouping/align.py` (`label_tokens`) |
| `check_datasets.py`, `test_load_model.py`, `test_lora_training.py` | `tests/`, `mtp/data/corpus.py` |

Run them from the repo root as `python legacy/<file>.py` if ever needed (they download the model at import).

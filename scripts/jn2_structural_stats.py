"""
JN-2 evidence for docs/design_structural_loss.md.

Part A (CPU): how much same-group signal each head can see, per grouper.
  For every pair of real tokens (t, t+L): is group_id[t] == group_id[t+L]?
  Head d (predicting t+d+1) sees lookahead L = d+1 for S1, and its S3 mask
  (t+1 and u = t+d+1 in one group) is lookahead L = d.
  "words" = every whitespace word is its own group, i.e. the same-group rate
  that comes from subword continuation alone. Grouping only adds signal
  above that line.

Part B (GPU): the laptop R1 checkpoint (k=2 linear, batch 12500) on the fixed
  IndicCorp eval rows 0-999, split in-group vs at-boundary, plus the S3 pair
  (head 1 at t vs head 0 at t+1, both predicting t+2).

Usage:
  $env:PYTHONUTF8 = "1"
  python scripts/jn2_structural_stats.py --part a --n 5000
  python scripts/jn2_structural_stats.py --part b
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL_NAME = "LingoIITGN/ganga-1b"
R1_CKPT = ROOT / "checkpoints" / "mtp_baseline" / "batch_12500"


def load_legacy_grouper(filename):
    """find_word_groups from a root-level (or legacy/) script, without running its __main__."""
    for base in (ROOT, ROOT / "legacy"):
        path = base / filename
        if path.exists():
            spec = importlib.util.spec_from_file_location(path.stem, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.find_word_groups
    raise FileNotFoundError(filename)


GROUPERS = {
    "words": lambda s: [[w] for w in s.split()],
    "v0_short": load_legacy_grouper("word_group_boundaries.py"),
    "v0_expanded": load_legacy_grouper("validate_on_real_data.py"),
}


def token_group_ids(text, tokenizer, grouper, max_length=128):
    """Same alignment rule as boundary_alignment.py; group_id -1 for special tokens.
    max_length=None: no truncation (corpus statistics)."""
    enc = tokenizer(text, return_offsets_mapping=True, truncation=max_length is not None, max_length=max_length)
    starts, pos = [], 0
    groups = grouper(text)
    words = text.split()
    assert sum(len(g) for g in groups) == len(words)
    w = 0
    for g in groups:
        start = text.index(words[w], pos)
        starts.append(start)
        for _ in g:
            pos = text.index(words[w], pos) + len(words[w])
            w += 1
    gid, cur = [], -1
    for a, b in enc["offset_mapping"]:
        if a == b:
            gid.append(-1)
            continue
        if any(a <= s < b for s in starts) or cur == -1:
            cur += 1
        gid.append(cur)
    return enc["input_ids"], gid, len(groups), len(words)


def load_texts(start, n):
    from datasets import load_dataset
    raw = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="hin_Deva", streaming=True)
    texts = [ex["text"].strip() for ex in raw.skip(start).take(n)]  # includes blank separator lines
    return [t for t in texts if t]


def part_a(n):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    texts = load_texts(1000, n)  # train region, INTERFACES §4
    res = {"n_sentences": len(texts)}
    for name, grouper in GROUPERS.items():
        same = {L: [0, 0] for L in range(1, 5)}
        n_tok = n_starts = n_groups = n_words = n_multi = 0
        for text in texts:
            _, gid, ng, nw = token_group_ids(text, tok, grouper, max_length=None)
            n_groups += ng
            n_words += nw
            n_multi += sum(len(g) > 1 for g in grouper(text))
            real = [g for g in gid if g >= 0]
            n_tok += len(real)
            n_starts += sum(1 for i, g in enumerate(real) if i == 0 or g != real[i - 1])
            for L in same:
                for i in range(len(real) - L):
                    same[L][0] += real[i] == real[i + L]
                    same[L][1] += 1
        res[name] = {
            "words_per_group": n_words / n_groups,
            "tokens_per_group": n_tok / n_groups,
            "tokens_per_word": n_tok / n_words,
            "pct_words_attached": 100 * (n_words - n_groups) / n_words,
            "pct_groups_multiword": 100 * n_multi / n_groups,
            "pct_tokens_group_start": 100 * n_starts / n_tok,
            "same_group_rate": {L: same[L][0] / same[L][1] for L in same},
        }
    return res


@torch.no_grad()
def part_b():
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mtp.model.heads import MTPModel

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).cuda()
    base = PeftModel.from_pretrained(base, f"{R1_CKPT}_lora")
    model = MTPModel(base, num_heads=2, head_type="linear").eval()
    model.extra_heads.load_state_dict(torch.load(f"{R1_CKPT}_extra_heads.pt", weights_only=True))

    texts = load_texts(0, 1000)  # fixed eval rows 0-999
    res = {"n_sentences": len(texts)}
    for gname in ("v0_short", "v0_expanded"):
        grouper = GROUPERS[gname]
        acc = {f"h{h}_{s}": [0, 0, 0.0] for h in (0, 1) for s in ("in", "bd")}  # correct, count, ce
        s3 = {s: {"n": 0, "agree": 0, "kl": 0.0, "teacher_ok": 0, "student_ok": 0} for s in ("in", "bd")}
        for i in range(0, len(texts), 8):
            chunk = texts[i:i + 8]
            rows = [token_group_ids(t, tok, grouper) for t in chunk]
            T = max(len(r[0]) for r in rows)
            ids = torch.full((len(rows), T), tok.pad_token_id)
            gid = torch.full((len(rows), T), -1)
            for b, (x, g, _, _) in enumerate(rows):
                ids[b, :len(x)] = torch.tensor(x)
                gid[b, :len(g)] = torch.tensor(g)
            mask = (torch.arange(T)[None] < torch.tensor([len(r[0]) for r in rows])[:, None]).long()
            ids, gid, mask = ids.cuda(), gid.cuda(), mask.cuda()
            out = model(ids, mask)
            logp = [torch.log_softmax(lg.float(), -1) for lg in out.logits]

            for h in (0, 1):
                s = h + 1
                src, tgt = gid[:, :-s], gid[:, s:]
                ok = (src >= 0) & (tgt >= 0)
                lp = logp[h][:, :-s]
                correct = lp.argmax(-1) == ids[:, s:]
                ce = -lp.gather(-1, ids[:, s:, None]).squeeze(-1)
                for name, m in (("in", ok & (src == tgt)), ("bd", ok & (src != tgt))):
                    a = acc[f"h{h}_{name}"]
                    a[0] += correct[m].sum().item()
                    a[1] += m.sum().item()
                    a[2] += ce[m].sum().item()

            # S3 pair: student = head 1 at t, teacher = head 0 at t+1, target u = t+2
            stu, tea = logp[1][:, :-2], logp[0][:, 1:-1]
            u = ids[:, 2:]
            g_t, g_t1, g_u = gid[:, :-2], gid[:, 1:-1], gid[:, 2:]
            ok = (g_t >= 0) & (g_t1 >= 0) & (g_u >= 0)
            kl = (tea.exp() * (tea - stu)).sum(-1)
            agree = stu.argmax(-1) == tea.argmax(-1)
            t_ok = tea.argmax(-1) == u
            s_ok = stu.argmax(-1) == u
            for name, m in (("in", ok & (g_t1 == g_u)), ("bd", ok & (g_t1 != g_u))):
                d = s3[name]
                d["n"] += m.sum().item()
                d["agree"] += agree[m].sum().item()
                d["kl"] += kl[m].sum().item()
                d["teacher_ok"] += t_ok[m].sum().item()
                d["student_ok"] += s_ok[m].sum().item()

        res[gname] = {
            "head_split": {k: {"top1": v[0] / v[1], "ce": v[2] / v[1], "n": v[1]} for k, v in acc.items()},
            "s3_pair_h1_vs_h0": {k: {"n": v["n"], "top1_agree": v["agree"] / v["n"], "kl_teacher_student": v["kl"] / v["n"],
                                     "teacher_top1": v["teacher_ok"] / v["n"], "student_top1": v["student_ok"] / v["n"]}
                                 for k, v in s3.items()},
        }
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b"], required=True)
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()
    result = part_a(args.n) if args.part == "a" else part_b()
    print(json.dumps(result, indent=2, ensure_ascii=False))

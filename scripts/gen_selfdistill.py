"""
Self-distillation training text: the base model's own greedy continuations of training prompts.

A draft is accepted only if it equals head 0's greedy token. On text that head 0 generated
greedily, the CE target of head d at t (token t+d+1) IS that greedy token, so training the heads
on this text trains them for acceptance directly (the Medusa-2 self-distillation recipe).

    python scripts/gen_selfdistill.py --lang hi --n 16000 --out /kaggle/working/sd/hi_sd.jsonl \
        [--shard 0 --num_shards 2] [--max_new_tokens 64] [--batch_size 64]

Prompts: the first 8-16 words (seeded) of IndicCorp train sentences (same split train.py uses).
Output: one {"text": prompt + continuation} per line; train.py reads it via data.train_file.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--model_name", default="LingoIITGN/ganga-1b")
    ap.add_argument("--run_dir", default=None, help="generate with a trained run's head 0 (base + its LoRA) instead")
    ap.add_argument("--n", type=int, default=16000, help="sentences in total (over all shards)")
    ap.add_argument("--skip", type=int, default=0, help="skip the first SKIP train sentences (new prompts for more data)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--min_new_tokens", type=int, default=0,
                    help="suppress EOS until this many tokens: instruct models (Misal) stop after one sentence")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from mtp.data.corpus import load_split
    from mtp.device import pick_device

    device = pick_device()
    # fp16 inference is fine on T4 (no training here); bf16 where native, fp32 on CPU
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(args.dtype)
    if dtype is None:
        dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16) \
            if device == "cuda" else torch.float32
    if args.run_dir:  # head 0 of the run = base model + LoRA (the PEFT model generates with the adapter)
        from mtp.model.checkpoint import load_run
        mtp_model, tok, _ = load_run(args.run_dir, device=device)
        model = mtp_model.base_model.merge_and_unload().to(dtype).eval()
    else:
        tok = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype).to(device).eval()
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.unk_token

    texts = load_split(args.lang, "train", args.skip + args.n)[args.skip:]
    rng = random.Random(args.seed)
    prompts = []
    for t in texts:  # every sentence gets a prompt length, so shards are deterministic
        words = t.split()
        w = rng.randint(8, 16)
        prompts.append(" ".join(words[: min(w, max(len(words) - 1, 1))]))
    prompts = prompts[args.shard::args.num_shards]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f, torch.no_grad():
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i:i + args.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True).to(device)
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, min_new_tokens=args.min_new_tokens or None)
            for ids in gen:  # decode prompt + continuation together: a lone "▁word" would lose its space
                text = tok.decode(ids, skip_special_tokens=True).strip()
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            if (i // args.batch_size) % 20 == 0:
                print(f"{i + len(batch)}/{len(prompts)}", flush=True)
    print(f"wrote {len(prompts)} texts to {out}")


if __name__ == "__main__":
    main()

"""
Token-level MTP baseline (no word-group loss yet -- that's next).

Head 0: predicts next token (standard, same as NTP baseline)
Head 1: predicts 2-tokens-ahead (new -- this is the actual MTP objective)

Uses the same LoRA + batching + clipping + held-out-eval setup that
worked for the NTP baseline, so results are comparable apples-to-apples.
"""

import os
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from medusa_heads import MedusaWrapper

MODEL_NAME = "LingoIITGN/ganga-1b"
NUM_SENTENCES = 100000
NUM_EVAL_SENTENCES = 100
BATCH_SIZE = 8
MAX_LENGTH = 128
SAVE_EVERY = 100
EVAL_EVERY = 100
NUM_EXTRA_HEADS = 1  # k=2 total heads
CHECKPOINT_DIR = "./checkpoints/mtp_baseline"
LOG_FILE = "training_log_mtp.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def batchify(texts, tokenizer, batch_size):
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        yield tokenizer(
            chunk, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_LENGTH,
        )


def compute_head_loss(logits, input_ids, attention_mask, shift):
    """
    Loss for a head predicting `shift` tokens ahead.
    shift=1 -> head 0 (standard next-token)
    shift=2 -> head 1 (2-tokens-ahead)
    """
    if logits.shape[1] <= shift:
        return None  # sequence too short for this shift, skip

    pred_logits = logits[:, :-shift, :]
    targets = input_ids[:, shift:].clone()
    target_mask = attention_mask[:, shift:]
    targets[target_mask == 0] = -100  # ignore padding positions

    vocab_size = pred_logits.shape[-1]
    loss = F.cross_entropy(
        pred_logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=-100,
    )
    return loss


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log(f"=== Run started {time.ctime()} ===")
    log(f"Device: {device} | Batch size: {BATCH_SIZE} | Extra heads: {NUM_EXTRA_HEADS}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, lora_config)

    model = MedusaWrapper(base_model, num_extra_heads=NUM_EXTRA_HEADS).to(device)
    model.train()

    # Only train what's actually trainable: LoRA deltas + new extra heads.
    # The frozen base weights are excluded automatically via requires_grad.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    log(f"Trainable params: {num_trainable:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)

    total_to_pull = NUM_SENTENCES + NUM_EVAL_SENTENCES
    log(f"Loading {total_to_pull} Hindi sentences...")
    raw = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="hin_Deva", streaming=True)
    all_texts = [ex["text"] for ex in raw.take(total_to_pull)]

    eval_texts = all_texts[:NUM_EVAL_SENTENCES]
    train_texts = all_texts[NUM_EVAL_SENTENCES:]
    log(f"Loaded {len(train_texts)} training + {len(eval_texts)} held-out eval sentences.")

    def run_eval():
        model.eval()
        total_losses = [0.0] * (NUM_EXTRA_HEADS + 1)
        count = 0
        with torch.no_grad():
            for enc in batchify(eval_texts, tokenizer, BATCH_SIZE):
                enc = {k: v.to(device) for k, v in enc.items()}
                all_logits = model(**enc)
                for h, logits in enumerate(all_logits):
                    l = compute_head_loss(logits, enc["input_ids"], enc["attention_mask"], shift=h + 1)
                    if l is not None:
                        total_losses[h] += l.item()
                count += 1
        model.train()
        return [tl / max(count, 1) for tl in total_losses]

    log("Starting training loop...")
    batch_num = 0
    running_losses = [0.0] * (NUM_EXTRA_HEADS + 1)

    for enc in batchify(train_texts, tokenizer, BATCH_SIZE):
        enc = {k: v.to(device) for k, v in enc.items()}
        all_logits = model(**enc)

        head_losses = []
        for h, logits in enumerate(all_logits):
            l = compute_head_loss(logits, enc["input_ids"], enc["attention_mask"], shift=h + 1)
            if l is not None:
                head_losses.append(l)
                running_losses[h] += l.item()

        total_loss = sum(head_losses)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        batch_num += 1

        if batch_num % 20 == 0:
            avg_str = " | ".join(f"head{h}: {running_losses[h]/20:.4f}" for h in range(len(running_losses)))
            log(f"Batch {batch_num} | {avg_str}")
            running_losses = [0.0] * (NUM_EXTRA_HEADS + 1)

        if batch_num % SAVE_EVERY == 0:
            ckpt_path = f"{CHECKPOINT_DIR}/batch_{batch_num}"
            base_model.save_pretrained(f"{ckpt_path}_lora")
            torch.save(model.extra_heads.state_dict(), f"{ckpt_path}_extra_heads.pt")
            log(f"  -> checkpoint saved: {ckpt_path}")

        if batch_num % EVAL_EVERY == 0:
            eval_losses = run_eval()
            eval_str = " | ".join(f"head{h}: {eval_losses[h]:.4f}" for h in range(len(eval_losses)))
            log(f"  >> HELD-OUT EVAL at batch {batch_num}: {eval_str}")

    log(f"=== Run finished {time.ctime()} ===")


if __name__ == "__main__":
    main()

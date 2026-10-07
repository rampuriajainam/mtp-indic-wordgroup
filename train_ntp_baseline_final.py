"""
NTP baseline fine-tune, v2: real batching + bigger eval set.

Fixes from v1:
- Batched training (BATCH_SIZE sentences at once, padded) instead of
  one sentence per step -- faster, and gradients are less noisy.
- 100 held-out eval sentences instead of 20 -- trustworthy trend,
  not mostly noise.

pip install datasets peft --break-system-packages
"""

import os
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "LingoIITGN/ganga-1b"
NUM_SENTENCES = 100000       # real run
NUM_EVAL_SENTENCES = 100
BATCH_SIZE = 8
MAX_LENGTH = 128
SAVE_EVERY = 100
EVAL_EVERY = 100
CHECKPOINT_DIR = "./checkpoints/ntp_baseline_final"
LOG_FILE = "training_log_final.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def batchify(texts, tokenizer, batch_size):
    """Yield padded batches of tokenized text."""
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(
            chunk, return_tensors="pt", padding=True,
            truncation=True, max_length=MAX_LENGTH,
        )
        yield enc


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log(f"=== Run started {time.ctime()} ===")
    log(f"Device: {device} | Batch size: {BATCH_SIZE}")

    log("Loading tokenizer + base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # needed for padding to work

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).to(device)

    lora_config = LoraConfig(
        r=8, lora_alpha=16,  # back to the known-stable config
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    total_to_pull = NUM_SENTENCES + NUM_EVAL_SENTENCES
    log(f"Loading {total_to_pull} Hindi sentences...")
    raw = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="hin_Deva", streaming=True)
    all_texts = [ex["text"] for ex in raw.take(total_to_pull)]

    eval_texts = all_texts[:NUM_EVAL_SENTENCES]
    train_texts = all_texts[NUM_EVAL_SENTENCES:]
    log(f"Loaded {len(train_texts)} training + {len(eval_texts)} held-out eval sentences.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)  # known-stable, settling here for the official baseline

    def run_eval():
        model.eval()
        total_loss, count = 0.0, 0
        with torch.no_grad():
            for enc in batchify(eval_texts, tokenizer, BATCH_SIZE):
                enc = {k: v.to(device) for k, v in enc.items()}
                labels = enc["input_ids"].clone()
                labels[enc["attention_mask"] == 0] = -100  # ignore padding in loss
                out = model(**enc, labels=labels)
                total_loss += out.loss.item()
                count += 1
        model.train()
        return total_loss / max(count, 1)

    log("Starting training loop...")
    batch_num = 0
    running_loss = 0.0
    for enc in batchify(train_texts, tokenizer, BATCH_SIZE):
        enc = {k: v.to(device) for k, v in enc.items()}
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        outputs = model(**enc, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevents the blow-up we just saw
        optimizer.step()

        running_loss += loss.item()
        batch_num += 1

        if batch_num % 20 == 0:
            avg_loss = running_loss / 20
            log(f"Batch {batch_num} (~{batch_num * BATCH_SIZE} sentences) | avg loss: {avg_loss:.4f}")
            running_loss = 0.0

        if batch_num % SAVE_EVERY == 0:
            ckpt_path = f"{CHECKPOINT_DIR}/batch_{batch_num}"
            model.save_pretrained(ckpt_path)
            log(f"  -> checkpoint saved: {ckpt_path}")

        if batch_num % EVAL_EVERY == 0:
            eval_loss = run_eval()
            log(f"  >> HELD-OUT EVAL LOSS at batch {batch_num}: {eval_loss:.4f}")

    final_path = f"{CHECKPOINT_DIR}/final"
    model.save_pretrained(final_path)
    log(f"=== Run finished {time.ctime()}. Final checkpoint: {final_path} ===")


if __name__ == "__main__":
    main()
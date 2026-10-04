"""
NTP baseline fine-tune -- safe to leave running overnight.

- Logs everything to training_log.txt (check it in the morning)
- Saves a checkpoint every SAVE_EVERY steps, so a crash/interruption
  doesn't lose everything
- Uses a real (if modest) slice of Hindi data, not just a few samples

pip install datasets peft --break-system-packages
"""

import os
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "LingoIITGN/ganga-1b"
NUM_SENTENCES = 100000        # adjust based on how much ran overnight you want
NUM_EVAL_SENTENCES = 20      # fixed held-out set, scored repeatedly to see real trend
MAX_LENGTH = 128
SAVE_EVERY = 200             # steps between checkpoint saves
EVAL_EVERY = 200             # steps between held-out eval runs
CHECKPOINT_DIR = "./checkpoints/ntp_baseline"
LOG_FILE = "training_log.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log(f"=== Run started {time.ctime()} ===")
    log(f"Device: {device}")

    log("Loading tokenizer + base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    ).to(device)

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    total_to_pull = NUM_SENTENCES + NUM_EVAL_SENTENCES
    log(f"Loading {total_to_pull} Hindi sentences from AI4Bharat IndicCorpV2...")
    raw = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="hin_Deva", streaming=True)
    all_texts = [ex["text"] for ex in raw.take(total_to_pull)]

    # First NUM_EVAL_SENTENCES are held out and NEVER trained on -- fixed
    # eval set so we can measure real learning, not per-sentence noise.
    eval_texts = all_texts[:NUM_EVAL_SENTENCES]
    texts = all_texts[NUM_EVAL_SENTENCES:]
    log(f"Loaded {len(texts)} training sentences + {len(eval_texts)} held-out eval sentences.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    def run_eval():
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for text in eval_texts:
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
                ).to(device)
                if inputs["input_ids"].shape[1] < 2:
                    continue
                labels = inputs["input_ids"].clone()
                out = model(**inputs, labels=labels)
                total_loss += out.loss.item()
                count += 1
        model.train()
        return total_loss / max(count, 1)

    step = 0
    running_loss = 0.0
    log("Starting training loop...")
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        ).to(device)
        if inputs["input_ids"].shape[1] < 2:
            continue  # skip empty/too-short lines

        labels = inputs["input_ids"].clone()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        step += 1

        if step % 50 == 0:
            avg_loss = running_loss / 50
            log(f"Step {step}/{len(texts)} | avg loss (last 50): {avg_loss:.4f}")
            running_loss = 0.0

        if step % SAVE_EVERY == 0:
            ckpt_path = f"{CHECKPOINT_DIR}/step_{step}"
            model.save_pretrained(ckpt_path)
            log(f"  -> checkpoint saved: {ckpt_path}")

        if step % EVAL_EVERY == 0:
            eval_loss = run_eval()
            log(f"  >> HELD-OUT EVAL LOSS at step {step}: {eval_loss:.4f}  "
                f"(compare this number across checkpoints, not the training loss above)")

    final_path = f"{CHECKPOINT_DIR}/final"
    model.save_pretrained(final_path)
    log(f"=== Run finished {time.ctime()}. Final checkpoint: {final_path} ===")


if __name__ == "__main__":
    main()
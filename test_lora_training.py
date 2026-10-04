"""
Step 2: confirm LoRA fine-tuning loop runs end-to-end.

This does NOT train on real data yet -- it just proves the training
loop (forward, backward, optimizer step, checkpoint save) works before
you build Medusa heads or real data loading on top of it.

pip install peft --break-system-packages
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_NAME = "LingoIITGN/ganga-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading base model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
)
model.to(device)

# --- Wrap with LoRA ---
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # adjust if model uses different names
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # should show a tiny fraction of total params

# --- Fake a few training steps on dummy sentences ---
dummy_sentences = [
    "भारत एक विशाल और विविधतापूर्ण देश है।",
    "मुंबई भारत की वित्तीय राजधानी है।",
    "हिंदी भारत की सबसे व्यापक रूप से बोली जाने वाली भाषा है।",
]

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()

print("\nRunning 3 dummy training steps...")
for step, text in enumerate(dummy_sentences):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    labels = inputs["input_ids"].clone()

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"  Step {step + 1}: loss = {loss.item():.4f}")

# --- Save LoRA adapter only (small file, not the full model) ---
save_path = "./lora_test_checkpoint"
model.save_pretrained(save_path)
print(f"\n✅ LoRA adapter saved to {save_path}")
print("If loss values printed above with no errors, your training loop works.")

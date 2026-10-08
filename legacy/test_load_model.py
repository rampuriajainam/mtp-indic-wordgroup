"""
Week 1, Step 1: confirm base model access + one forward pass.

Run this first. If it prints logits shape + a generated sentence
without crashing, your biggest Week 1 risk is cleared.

pip install torch transformers accelerate --break-system-packages
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Pick ONE to start with. Swap MODEL_NAME to test the other language. ---
MODEL_NAME = "LingoIITGN/ganga-1b"              # Hindi, 1B params
# MODEL_NAME = "smallstepai/Misal-1B-instruct-v0.1"  # Marathi, 1B params

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print(f"Loading tokenizer + model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
)
model.eval()
if device == "cpu":
    model.to(device)

print(f"Loaded. Param count: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# --- 1. Confirm a basic forward pass works (this is the real Week 1 test) ---
sample_text = "भारत एक विशाल और विविधतापूर्ण देश है।"  # Hindi sample sentence
inputs = tokenizer(sample_text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

print(f"\nForward pass OK. Logits shape: {outputs.logits.shape}")
# Shape should be [batch_size, seq_len, vocab_size]

# --- 2. Confirm generation works end-to-end ---
with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
    )

decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"\nGenerated continuation:\n{decoded}")

print("\n✅ If you see text above with no errors, the base model is ready to build on.")

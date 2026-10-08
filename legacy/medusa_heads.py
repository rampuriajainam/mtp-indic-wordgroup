"""
Step: add Medusa-style extra prediction heads to the base model.

Standard model: one output head predicts the NEXT token.
Medusa-style MTP: k heads, each predicting one token further ahead
(head 0 = next token, head 1 = token after that, etc.), all sharing
the same backbone hidden state.

This is the "Linear Layers (LL)" variant from the Aynetdinov & Akbik
paper you cited -- simplest version, extra linear layers on the
shared hidden state. Transformer-layer heads (TL) are more complex
and can come later if this works.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class MedusaWrapper(nn.Module):
    """
    Wraps a causal LM with k-1 extra prediction heads.
    Head 0 is the model's original LM head (next token).
    Heads 1..k-1 are new linear layers predicting further-ahead tokens.
    """

    def __init__(self, base_model, num_extra_heads=1):
        super().__init__()
        self.base_model = base_model
        self.num_extra_heads = num_extra_heads

        hidden_size = base_model.config.hidden_size
        vocab_size = base_model.config.vocab_size

        # One new linear layer per extra head, each mapping
        # hidden_state -> vocab_size logits, just like the original LM head.
        # match dtype to the base model (bf16) or the two won't multiply together
        model_dtype = next(base_model.parameters()).dtype
        self.extra_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False, dtype=model_dtype)
            for _ in range(num_extra_heads)
        ])

    def forward(self, input_ids, attention_mask=None):
        # Get the shared hidden states from the base model's backbone
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # outputs.logits is head 0's prediction (next token) already
        head_0_logits = outputs.logits

        # Shared hidden state used by all extra heads
        last_hidden = outputs.hidden_states[-1]

        extra_logits = [head(last_hidden) for head in self.extra_heads]

        # Returns a list: [head_0_logits, head_1_logits, ..., head_k-1_logits]
        # head_i predicts the token (i+1) positions ahead
        return [head_0_logits] + extra_logits


if __name__ == "__main__":
    MODEL_NAME = "LingoIITGN/ganga-1b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)

    NUM_EXTRA_HEADS = 1  # k=2 total (head 0 + 1 extra) to start simple
    model = MedusaWrapper(base_model, num_extra_heads=NUM_EXTRA_HEADS).to(device)

    print(f"Wrapped model with {NUM_EXTRA_HEADS} extra head(s). Testing forward pass...")

    sample_text = "भारत एक विशाल और विविधतापूर्ण देश है।"
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)

    with torch.no_grad():
        all_logits = model(**inputs)

    print(f"\nNumber of heads returned: {len(all_logits)}")
    for i, logits in enumerate(all_logits):
        print(f"  Head {i} logits shape: {logits.shape}  (predicts {i+1} token(s) ahead)")

    print("\n✅ If you see k shapes above, all matching [1, seq_len, vocab_size],")
    print("   the Medusa head architecture is wired up correctly.")
    print("\nNOTE: extra_heads are randomly initialized and untrained right now --")
    print("   their predictions will be garbage until you actually train them.")
    print("   This script only confirms the ARCHITECTURE works, not that it's useful yet.")
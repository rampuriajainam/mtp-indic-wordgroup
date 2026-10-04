"""
Step 3: download AI4Bharat Hindi data + FLORES-200 devtest, and sanity-check
that tokenization isn't silently mangling Devanagari script.

pip install datasets --break-system-packages
"""

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "LingoIITGN/ganga-1b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- 1. AI4Bharat IndicCorp v2 (Hindi) ---
# Note: this dataset is large. We stream it and only pull a small sample
# to sanity-check before committing to a full download.
print("Loading a sample of AI4Bharat IndicCorp v2 (Hindi)...")
try:
    indic_corp = load_dataset(
        "ai4bharat/IndicCorpV2",
        "hin",  # ISO code for Hindi
        split="train",
        streaming=True,
    )
    sample = list(indic_corp.take(5))
    print(f"Pulled {len(sample)} samples. Example:")
    print(sample[0])
except Exception as e:
    print(f"IndicCorpV2 load failed: {e}")
    print("If this fails, check the exact dataset config name on the HF page:")
    print("https://huggingface.co/datasets/ai4bharat/IndicCorpV2")

# --- 2. FLORES-200 devtest (Hindi) ---
print("\nLoading FLORES-200 devtest (Hindi)...")
try:
    flores = load_dataset("facebook/flores", "hin_Deva", split="devtest")
    print(f"Loaded {len(flores)} sentences. Example:")
    print(flores[0])
except Exception as e:
    print(f"FLORES load failed: {e}")

# --- 3. Tokenization sanity check ---
print("\n--- Tokenization check ---")
test_sentences = [
    "भारत एक विशाल और विविधतापूर्ण देश है।",
    "मैं कल बाजार जा रहा था।",  # contains a verb+auxiliary (word-group test case)
    "उसने किताब पढ़ी।",
]

for sent in test_sentences:
    tokens = tokenizer.tokenize(sent)
    print(f"\nText: {sent}")
    print(f"Tokens ({len(tokens)}): {tokens}")
    # Red flag: if token count is way higher than word count, check for
    # byte-level fallback (tokens look like weird unicode escapes).

print("\n✅ If sentences above loaded and tokens look like real Devanagari")
print("   subword pieces (not garbled bytes), data pipeline basics are sound.")

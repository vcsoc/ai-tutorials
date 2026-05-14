"""
Step 3 — Fine-Tune the LLM
────────────────────────────
Loads the model saved by step 1, continues training on a second, more targeted
dataset to specialise the model's knowledge, then saves the updated weights back
to the same directory.

Prerequisite:
    python 1-train/script.py   (creates ./llm_small/)

Run:
    python 3-finetune/script.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

MODEL_DIR  = "./llm_small"      # written by 1-train/script.py
OUTPUT_DIR = "./llm_small_ft"   # fine-tuned weights saved here
# Note: we save to a *different* directory because Windows holds an open
# memory-map on the safetensors file loaded from MODEL_DIR, and won't allow
# that file to be overwritten while the process is running (os error 1224).

print("=== Step 3: Fine-tuning LLM ===")

# ── Load the trained model from step 1 ───────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

# ── Fine-tune dataset ─────────────────────────────────────────────────────────
# These two sentences are more precise than the original training data.
# Running another training pass nudges the model's weights toward this phrasing.

fine_tune_data = [
    {"text": "RAG combines a retriever with a generator to answer questions accurately."},
    {"text": "Python is easy to learn and widely used for AI applications."},
]

fine_tune_dataset = Dataset.from_list(fine_tune_data)


def tokenize(batch):
    enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=32)
    enc["labels"] = enc["input_ids"].copy()
    return enc


tokenized_fine_tune = fine_tune_dataset.map(tokenize, batched=True)

# ── Training ──────────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=2,
    learning_rate=2e-5,              # lower than step 1 — adjusting, not relearning
    logging_steps=1,
    save_steps=500,
    save_total_limit=1,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_fine_tune,
)

trainer.train()

# ── Test ──────────────────────────────────────────────────────────────────────

inputs  = tokenizer("Explain RAG simply.", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print("\nFine-tuned model output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# ── Save updated weights ──────────────────────────────────────────────────────

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Fine-tuned model saved to {OUTPUT_DIR}/")
print("   Next: run  python 4-export/script.py  (export to GGUF for Ollama/LM Studio)")

"""
Step 1 — Create & Train a Small LLM
────────────────────────────────────
Loads a tiny pre-trained GPT-2 model and runs a short training loop on three
example sentences. The trained model and tokenizer are saved to ../llm_small/
so the next scripts (2-rag and 3-finetune) can load them.

Run:
    python 1-train/script.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

MODEL_NAME  = "sshleifer/tiny-gpt2"   # 2.5 MB, 2 transformer layers
OUTPUT_DIR  = "./llm_small"            # model saved here after training

print("=== Step 1: Creating and training small LLM ===")

# ── Load model & tokenizer ────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# GPT-2 ships without a padding token; use EOS as pad so batched padding works
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ── Dataset ───────────────────────────────────────────────────────────────────

train_data = [
    {"text": "Hello, how are you?"},
    {"text": "I am learning AI."},
    {"text": "RAG stands for Retrieval-Augmented Generation."},
]

dataset = Dataset.from_list(train_data)


def tokenize(batch):
    enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=32)
    # labels = input_ids → model learns to predict the next token at every position
    enc["labels"] = enc["input_ids"].copy()
    return enc


tokenized_dataset = dataset.map(tokenize, batched=True)

# ── Training ──────────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=2,
    learning_rate=5e-5,
    logging_steps=1,
    save_steps=500,
    save_total_limit=1,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
# Save both the model weights and the tokenizer so downstream scripts can load
# everything from a single directory without re-downloading from the Hub.

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Model saved to {OUTPUT_DIR}/")
print("   Next: run  python 2-rag/script.py")

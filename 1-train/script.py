from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Sample data
data = [
    {"text": "Hello, how are you?"},
    {"text": "I am learning AI."},
    {"text": "RAG stands for Retrieval-Augmented Generation."},
]

dataset = Dataset.from_list(data)

# Tokenizer & model (small GPT2)
model_name = "sshleifer/tiny-gpt2"  # very small GPT2
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize
def tokenize(batch):
    enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=32)
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./llm_small",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=1,
    save_steps=5,
    save_total_limit=1,
    logging_dir="./logs",
    remove_unused_columns=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train
trainer.train()
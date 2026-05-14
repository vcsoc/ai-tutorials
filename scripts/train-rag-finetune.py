# llm_rag_finetune.py
"""
Full Walkthrough: Small LLM → RAG → Fine-Tune
Requires: pip install torch transformers datasets faiss-cpu sentence-transformers
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

# -----------------------------
# 1️⃣ Create & Train Small LLM
# -----------------------------
print("=== Step 1: Creating and training small LLM ===")

# Sample training data
train_data = [
    {"text": "Hello, how are you?"},
    {"text": "I am learning AI."},
    {"text": "RAG stands for Retrieval-Augmented Generation."},
]

dataset = Dataset.from_list(train_data)

# Use a tiny GPT2 model
model_name = "sshleifer/tiny-gpt2"
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

# Train the model
trainer.train()

# -----------------------------
# 2️⃣ RAG: Retrieval-Augmented Generation
# -----------------------------
print("\n=== Step 2: RAG setup ===")

# Knowledge base
documents = [
    "Python is a programming language.",
    "Hugging Face provides the transformers library.",
    "RAG combines retrieval and generation to answer questions better.",
]

# Create embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents)

# Build FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Example query
query = "What is RAG in AI?"
query_embedding = embed_model.encode([query])
D, I = index.search(np.array(query_embedding), k=1)
retrieved_doc = documents[I[0][0]]

print("Query:", query)
print("Retrieved doc:", retrieved_doc)

# Generate LLM response using retrieved doc
input_text = f"Use the following document to answer the question.\nDocument: {retrieved_doc}\nQuestion: {query}"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50)
print("LLM output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# -----------------------------
# 3️⃣ Fine-Tune LLM on Improved Data
# -----------------------------
print("\n=== Step 3: Fine-tuning LLM ===")

# Fine-tune dataset
fine_tune_data = [
    {"text": "RAG combines a retriever with a generator to answer questions accurately."},
    {"text": "Python is easy to learn and widely used for AI applications."},
]

fine_tune_dataset = Dataset.from_list(fine_tune_data)
tokenized_fine_tune = fine_tune_dataset.map(tokenize, batched=True)

# Update trainer and train
trainer.train_dataset = tokenized_fine_tune
trainer.train()

# Test fine-tuned LLM
test_query = "Explain RAG simply."
test_input = f"{test_query}"
inputs = tokenizer(test_input, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print("Fine-tuned LLM output:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# -----------------------------
print("\n✅ Done! Small LLM created, used RAG, and fine-tuned.")
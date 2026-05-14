Here's a full, simple Python walkthrough covering the three steps: creating a small LLM, using RAG, and fine-tuning. We'll keep it beginner-friendly, using Hugging Face Transformers and FAISS for retrieval.

The full runnable script is in [`scripts/train-rag-finetune.py`](scripts/train-rag-finetune.py).

# Python Walkthrough: LLM, RAG, Fine-Tuning

## 1️⃣ Setup

Install the necessary packages:

```
pip install torch transformers datasets faiss-cpu sentence-transformers accelerate
```

| Package | Purpose |
|---|---|
| `torch` | Deep learning backend used by all models |
| `transformers` | LLM models, tokenizers, and the Trainer API |
| `datasets` | Lightweight dataset loading and preprocessing |
| `faiss-cpu` | Vector similarity search engine for RAG retrieval |
| `sentence-transformers` | Embedding model that converts text to vectors |
| `accelerate` | Required backend for the Hugging Face Trainer |

---

## 2️⃣ Creating & Training a Small LLM

We load a pre-trained tiny GPT-2 model (~2.5 MB) and run a short training loop on three example sentences. The goal is to show how the training pipeline works, not to produce a capable model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Sample training data — three short sentences the model will train on
train_data = [
    {"text": "Hello, how are you?"},
    {"text": "I am learning AI."},
    {"text": "RAG stands for Retrieval-Augmented Generation."},
]

dataset = Dataset.from_list(train_data)

# Load a tiny GPT-2 model (2.5 MB, 2 transformer layers — fast but nonsensical output)
model_name = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT-2 has no padding token by default — set it to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize: convert text to token IDs, pad/truncate to 32 tokens
# labels = input_ids tells the model to predict the next token at each position (causal LM objective)
def tokenize(batch):
    enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=32)
    enc["labels"] = enc["input_ids"].copy()
    return enc

tokenized_dataset = dataset.map(tokenize, batched=True)

# TrainingArguments controls the training loop behaviour
training_args = TrainingArguments(
    output_dir="./llm_small",          # where to save model checkpoints
    per_device_train_batch_size=2,      # 2 samples per gradient update
    num_train_epochs=2,                 # pass over the full dataset twice
    logging_steps=1,                    # print loss after every step
    save_steps=5,                       # save a checkpoint every 5 steps
    save_total_limit=1,                 # keep only the latest checkpoint
    remove_unused_columns=False,        # keep all dataset columns (including labels)
)

# Trainer wraps the model, data, and training arguments into a single train() call
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
```

**Expected output:**

```
{'loss': '10.48', 'grad_norm': '2.18', 'learning_rate': '5e-05', 'epoch': '0.5'}
{'loss': '10.48', 'grad_norm': '1.87', 'learning_rate': '3.75e-05', 'epoch': '1.0'}
...
{'train_runtime': '0.21', 'train_loss': '10.48', 'epoch': '2'}
```

A high loss (~10.48) is expected — `tiny-gpt2` has random weights and only 3 training samples. The loss would drop significantly with a larger model and more data. The trained model is saved to `./llm_small/`.

---

## 3️⃣ Using RAG (Retrieval-Augmented Generation)

RAG splits the answering process into two stages:
1. **Retrieve** — find the most relevant document from a knowledge base using vector similarity
2. **Generate** — pass that document as context to the LLM to guide its output

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Knowledge base — in production this could be thousands of documents
documents = [
    "Python is a programming language.",
    "Hugging Face provides the transformers library.",
    "RAG combines retrieval and generation to answer questions better.",
]

# SentenceTransformer converts each document into a 384-dimensional embedding vector
# Semantically similar sentences will have vectors that are close together
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents)

# FAISS builds an index over those vectors for fast nearest-neighbour search
dimension = doc_embeddings.shape[1]   # 384
index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance
index.add(np.array(doc_embeddings))

# At query time: embed the question, search for the closest document vector
query = "What is RAG in AI?"
query_embedding = embed_model.encode([query])
D, I = index.search(np.array(query_embedding), k=1)  # return top-1 result
retrieved_doc = documents[I[0][0]]

print("Query:", query)
print("Retrieved doc:", retrieved_doc)

# Combine the retrieved document with the question and feed it to the LLM
input_text = (
    f"Use the following document to answer the question.\n"
    f"Document: {retrieved_doc}\n"
    f"Question: {query}"
)
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print("LLM output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Expected output:**

```
Query: What is RAG in AI?
Retrieved doc: RAG combines retrieval and generation to answer questions better.
LLM output: Use the following document to answer the question.
Document: RAG combines retrieval and generation to answer questions better.
Question: What is RAG in AI? <repetitive noise from tiny-gpt2>
```

The retrieval step works correctly — FAISS returns the most semantically relevant document. The generation is nonsensical because `tiny-gpt2` is too small to follow instructions. With a real model (e.g. `gpt2`, `mistral`, or any instruction-tuned LLM) the answer would be coherent.

---

## 4️⃣ Fine-Tuning the LLM

Fine-tuning continues training the model on a new, more targeted dataset. We reuse the same `Trainer` and `tokenize` function from Step 1, just swapping in a different dataset.

```python
# Improved dataset with more precise, domain-specific phrasing
fine_tune_data = [
    {"text": "RAG combines a retriever with a generator to answer questions accurately."},
    {"text": "Python is easy to learn and widely used for AI applications."},
]

fine_tune_dataset = Dataset.from_list(fine_tune_data)
tokenized_fine_tune = fine_tune_dataset.map(tokenize, batched=True)

# Swap the training dataset and re-run — the model's weights continue updating
trainer.train_dataset = tokenized_fine_tune
trainer.train()

# Test the fine-tuned model
inputs = tokenizer("Explain RAG simply.", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print("Fine-tuned LLM output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Expected output:**

```
{'loss': '10.48', 'grad_norm': '1.29', 'epoch': '1'}
{'loss': '10.48', 'grad_norm': '1.90', 'epoch': '2'}
Fine-tuned LLM output: Explain RAG simply. <repetitive noise from tiny-gpt2>
```

The fine-tuning step runs successfully. Again, coherent output requires a larger model. The weights are updated and the model is saved to `./llm_small/`.

---

## 5️⃣ Summary

| Step | What it does | Key components |
|---|---|---|
| **1 — Train** | Load a pre-trained model and continue training on custom data | `AutoModelForCausalLM`, `Trainer`, `TrainingArguments` |
| **2 — RAG** | Retrieve the most relevant document at query time, then generate a response grounded in it | `SentenceTransformer`, `faiss`, `model.generate()` |
| **3 — Fine-tune** | Update the model weights further on a smaller, targeted dataset | `Trainer` with a new `train_dataset` |

This walkthrough keeps everything small and fast, so it runs on a laptop in under a minute.
For production, you can scale:

- Bigger LLMs: `gpt2`, `flan-t5-small`, `bloom-560m`, `mistral-7b`
- Larger document corpora for RAG (thousands of PDFs, wikis, databases)
- Mixed precision training (`fp16=True` in `TrainingArguments`) for GPU speedup
- LoRA / QLoRA for memory-efficient fine-tuning of large models

---

Choosing between RAG (Retrieval-Augmented Generation) and fine-tuning a LLM depends on the use case, the type of data you have, and your goals. Here's a detailed breakdown:

### 1. Retrieval-Augmented Generation (RAG)

RAG is when you combine a general-purpose LLM with an external knowledge source (like a document database, PDFs, wiki, or private data). The LLM fetches relevant info from your documents at runtime to answer questions.

Use RAG when:

- Dynamic knowledge: Your data changes frequently (e.g., manuals, internal docs, legal contracts, company policies). No need to retrain every time data updates.
- Large knowledge base: You have more information than can practically fit in a model’s context window.
- Cost and time efficiency: Fine-tuning can be expensive and time-consuming. RAG lets you leverage a base LLM without retraining.
- Accuracy-critical references: You want answers grounded in specific documents or citations.
- One-off or low-frequency queries: Useful when you don’t need a specialized model but want contextually relevant answers.

Example: Answering questions from your company’s 10,000-page technical documentation or internal wiki using an LLM at query time.

### 2. Fine-Tuning

Fine-tuning involves training the LLM on your specific dataset so the model itself internalizes your domain knowledge.

Use fine-tuning when:

- Stable domain knowledge: Your information doesn’t change often, so retraining isn’t frequently needed.
- Task-specific performance: You want the LLM to perform a specialized task exceptionally well (e.g., legal contract summarization, medical advice, coding style enforcement).
- Better context handling: The model can respond without looking up external documents, useful if latency or retrieval complexity is an issue.
- Brand or style alignment: You want the model to adopt your organization’s tone, style, or policy in every response.
- Offline or restricted environment: When external retrieval isn’t feasible due to security or connectivity.

Example: A fine-tuned LLM trained specifically on your company’s proprietary code and documentation to help developers write compliant code.

### 3. Practical Rule of Thumb
Scenario | Prefer
--|--|
Knowledge changes frequently | RAG
Data is static & domain-specific | Fine-tuning
Must reference external docs verbatim | RAG
Need highly optimized task-specific responses | Fine-tuning
Limited budget / fast deployment | RAG
Security-sensitive offline deployment | Fine-tuning

💡 Tip: In many real-world setups, a hybrid approach is used: fine-tune the model on general company knowledge but use RAG to pull in dynamic or very large datasets. This way, you get both accuracy and flexibility.
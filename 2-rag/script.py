"""
Step 2 — RAG: Retrieval-Augmented Generation
─────────────────────────────────────────────
Loads the model trained in step 1, builds a FAISS vector index over a small
knowledge base, retrieves the most relevant document for a query, then uses
the model to generate a response grounded in that document.

Prerequisite:
    python 1-train/script.py   (creates ./llm_small/)

Run:
    python 2-rag/script.py
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "./llm_small"   # written by 1-train/script.py

print("=== Step 2: RAG — Retrieval-Augmented Generation ===")

# ── Load the trained model from step 1 ───────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

# ── Knowledge base ────────────────────────────────────────────────────────────
# In production this could be thousands of documents loaded from PDFs, a wiki,
# or a database. Here we use three short sentences for demonstration.

documents = [
    "Python is a programming language.",
    "Hugging Face provides the transformers library.",
    "RAG combines retrieval and generation to answer questions better.",
]

# ── Build FAISS index ─────────────────────────────────────────────────────────
# SentenceTransformer encodes each document into a 384-dimensional dense vector.
# Semantically similar texts produce vectors that are close together in that space.

embed_model   = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(documents)

dimension = doc_embeddings.shape[1]            # 384
index     = faiss.IndexFlatL2(dimension)       # exact L2 (Euclidean) nearest-neighbour
index.add(np.array(doc_embeddings))

# ── Retrieve ──────────────────────────────────────────────────────────────────
# Encode the query the same way, then ask FAISS for the single closest document.

query           = "What is RAG in AI?"
query_embedding = embed_model.encode([query])
_distances, indices = index.search(np.array(query_embedding), k=1)
retrieved_doc   = documents[indices[0][0]]

print(f"\nQuery:         {query}")
print(f"Retrieved doc: {retrieved_doc}")

# ── Generate ──────────────────────────────────────────────────────────────────
# Prepend the retrieved document to the question so the LLM has relevant context.
# With tiny-gpt2 the output will be nonsensical — a real instruction model
# (e.g. SmolLM2-135M-Instruct in 4-export/) would give a coherent answer here.

input_text = (
    f"Use the following document to answer the question.\n"
    f"Document: {retrieved_doc}\n"
    f"Question: {query}"
)
inputs  = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

print("\nLLM output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\n✅ RAG complete.")
print("   Next: run  python 3-finetune/script.py")

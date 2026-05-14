"""
4-export/script.py

Fine-tune SmolLM2-135M-Instruct on a small Q&A dataset, then export to GGUF
so the model can be loaded in Ollama or LM Studio and tested with natural language.

Model: HuggingFaceTB/SmolLM2-135M-Instruct
  - 135 M parameters, ~270 MB download
  - Already instruction-tuned — understands the user / assistant chat format
  - LLaMA-based architecture — fully supported by the llama.cpp converter

Extra dependency (on top of the existing requirements):
  pip install gguf sentencepiece
"""

import os
import subprocess
import sys
import urllib.request

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME   = "HuggingFaceTB/SmolLM2-135M-Instruct"
OUTPUT_DIR   = "./my-smollm2"          # HuggingFace model saved here after training
GGUF_PATH    = "./my-smollm2.gguf"     # final GGUF file for Ollama / LM Studio
CONVERT_SCRIPT = "convert_hf_to_gguf.py"

# ── Step 1: Load the base model ───────────────────────────────────────────────
# SmolLM2-135M-Instruct is already instruction-tuned, so we only need a small
# amount of fine-tuning data to specialise it for our domain.

print("=== Step 1: Loading SmolLM2-135M-Instruct ===")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ── Step 2: Fine-tune on a small Q&A dataset ─────────────────────────────────
# We format each example using the model's built-in ChatML template:
#   <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>
# labels = input_ids means the model trains to predict every token,
# including the prompt — sufficient for a small demo.

print("\n=== Step 2: Fine-tuning on custom Q&A data ===")

qa_pairs = [
    (
        "What is machine learning?",
        "Machine learning is a type of AI where models learn patterns from data "
        "rather than being explicitly programmed with rules.",
    ),
    (
        "What is RAG?",
        "RAG stands for Retrieval-Augmented Generation. It retrieves relevant "
        "documents at query time and uses them as context so the LLM can give "
        "accurate, grounded answers.",
    ),
    (
        "What is fine-tuning?",
        "Fine-tuning is the process of continuing to train a pre-trained model on "
        "a smaller, task-specific dataset to specialise its knowledge or style.",
    ),
    (
        "What is a transformer?",
        "A transformer is a neural network architecture that uses self-attention "
        "to process sequences in parallel. It is the foundation of modern LLMs "
        "like GPT, LLaMA, and BERT.",
    ),
    (
        "What is FAISS?",
        "FAISS is a library by Meta for fast similarity search over dense vectors. "
        "It is commonly used in RAG pipelines to find the most relevant documents "
        "for a given query.",
    ),
    (
        "What is a tokenizer?",
        "A tokenizer converts raw text into a sequence of integer token IDs that "
        "a model can process. It also handles special tokens like padding and "
        "end-of-sequence markers.",
    ),
    (
        "When should I use RAG instead of fine-tuning?",
        "Use RAG when your knowledge base changes frequently or is very large. "
        "Use fine-tuning when your data is stable and you want the model to "
        "internalise a specific style or domain deeply.",
    ),
    (
        "What is a GGUF file?",
        "GGUF is a binary format used by llama.cpp to store quantized LLM weights. "
        "It is supported by Ollama, LM Studio, and other local inference tools.",
    ),
]


def make_example(question: str, answer: str) -> dict:
    """Apply the ChatML template and return a dict with a 'text' key."""
    messages = [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


raw_data = [make_example(q, a) for q, a in qa_pairs]
dataset  = Dataset.from_list(raw_data)


def tokenize(batch: dict) -> dict:
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    # labels = input_ids tells the model to predict every token in the sequence
    enc["labels"] = enc["input_ids"].copy()
    return enc


tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-5,              # lower than default — we're fine-tuning, not training from scratch
    logging_steps=1,
    save_steps=500,
    save_total_limit=1,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

trainer.train()

# ── Step 3: Save in HuggingFace format ───────────────────────────────────────
# The converter in Step 4 reads from this directory.

print("\n=== Step 3: Saving model in HuggingFace format ===")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}/")

# ── Step 4: Convert to GGUF ───────────────────────────────────────────────────
# We download convert_hf_to_gguf.py from the llama.cpp repo.
# This is a pure-Python script — no C++ compilation required.
# It needs: pip install gguf sentencepiece
#
# --outtype q8_0  →  8-bit quantization: ~135 MB file, near-lossless quality.
# Other options: f16 (full float16, ~270 MB), f32 (largest, ~540 MB).

print("\n=== Step 4: Converting to GGUF ===")

if not os.path.exists(CONVERT_SCRIPT):
    url = (
        "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/"
        "convert_hf_to_gguf.py"
    )
    print(f"Downloading {CONVERT_SCRIPT} from llama.cpp ...")
    urllib.request.urlretrieve(url, CONVERT_SCRIPT)
    print("Download complete.")

result = subprocess.run(
    [
        sys.executable, CONVERT_SCRIPT,
        OUTPUT_DIR,
        "--outfile", GGUF_PATH,
        "--outtype", "q8_0",
    ],
    capture_output=False,
)

if result.returncode != 0:
    print("\n⚠ GGUF conversion failed.")
    print("Make sure the extra dependencies are installed:")
    print("  pip install gguf sentencepiece")
    sys.exit(1)

print(f"\nGGUF saved to: {os.path.abspath(GGUF_PATH)}")

# ── Step 5: Create Ollama Modelfile ──────────────────────────────────────────
# A Modelfile tells Ollama which GGUF to load and sets inference parameters.

print("\n=== Step 5: Creating Ollama Modelfile ===")

gguf_abs = os.path.abspath(GGUF_PATH)

modelfile_content = f"""\
FROM {gguf_abs}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"

SYSTEM "You are a helpful AI assistant specialised in machine learning and Python."
"""

with open("Modelfile", "w") as f:
    f.write(modelfile_content)

print("Modelfile written.")

# ── Usage instructions ────────────────────────────────────────────────────────

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Load in Ollama
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ollama create my-smollm2 -f Modelfile
  ollama run my-smollm2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Load in LM Studio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Open LM Studio
  2. My Models → Load Model from File
  3. Select: """ + gguf_abs + """

✅ Done!
""")

🏛️ History-Expert-LLM
Specialized Fine-Tuning of Llama-3.2-1B using QLoRA for Factual History Assistance

A domain-specific Large Language Model (LLM) fine-tuned to act as a Factual History Assistant, optimized for accurate historical responses, timeline reasoning, and strict domain boundaries.

This project fine-tunes Meta Llama-3.2-1B-Instruct using QLoRA on a curated historical instruction dataset designed to reduce hallucinations and improve factual consistency. 

small_history_facts

🎯 Project Objective

The goal of this project was to transform a general-purpose instruction model into a specialized historian model capable of:

Providing accurate historical explanations (Ancient, Medieval, Modern history)

Correcting false premises in user questions

Handling comparative chronology and timeline reasoning

Refusing out-of-domain requests (domain guarding)

Reducing factual hallucinations in small models (1B scale)

🧠 Model & Training Approach
Base Model

Model: Llama-3.2-1B-Instruct

Framework: Hugging Face Transformers + PEFT + TRL

Fine-Tuning Method: QLoRA (4-bit quantized training)

QLoRA Configuration

Quantization: NF4 4-bit

LoRA Rank (r): 32

LoRA Alpha (α): 64

Target Modules:

q_proj

k_proj

v_proj

o_proj

gate_proj

up_proj

down_proj 

small_history_facts

Training Setup

Hardware: NVIDIA T4 GPU (Google Colab / Kaggle)

Optimizer: Paged AdamW 8-bit

Epochs: 4

Gradient Accumulation: 8

FP16 Training Enabled

📂 Dataset
Dataset Name

history_dataset_3000_relational_patched.json

Size

~3,000+ instruction–response pairs

Dataset Design Philosophy

The dataset was built to avoid common small-model issues:

✅ Standard Historical Q&A

Textbook-style factual answers

Indian & World history coverage

✅ Relational / Chronological Samples

Timeline comparisons

Earlier vs later event reasoning

Cross-era differentiation

✅ Hallucination Protection

Multi-entity confusion fixes

Explicit correction samples

✅ Domain Guardrails

Refusal examples for non-history questions

Example:

User:

Did Chandragupta Maurya lead the French Revolution?

Assistant:

No. Chandragupta Maurya founded the Mauryan Empire in the 4th century BCE, while the French Revolution occurred in 18th-century France. 

history_dataset_3000_relational…

⚙️ Training Pipeline
1️⃣ Data Formatting

Each sample is converted into Llama-3 chat format:

<|begin_of_text|>
system → factual history assistant
user → instruction
assistant → response
2️⃣ Train/Test Split

90% Training

10% Validation

3️⃣ Supervised Fine-Tuning (SFT)

Implemented using trl.SFTTrainer. 

small_history_facts

📊 Performance Summary
Metric	Value
Final Training Loss	~0.08
Validation Loss	~0.08
Token Accuracy	~96%
Training Time	~1.5 hours (T4 GPU)
🧪 Example Outputs
1️⃣ Factual Recall

Q: What was the Gupta Empire?
A: The Gupta Empire (4th–6th century CE) is associated with advancements in science, mathematics, literature, and art.

2️⃣ Fact Correction

Q: Was A.O. Hume a leader of the Mughal Empire?
A: No. A.O. Hume was linked to the founding of the Indian National Congress in 1885, while the Mughal Empire belongs to an earlier historical period.

3️⃣ Domain Guarding

Q: How do I build a website using HTML?
A: I specialize in history and cannot answer questions outside historical topics.

🚀 How to Run
Install Dependencies
pip install transformers peft trl bitsandbytes datasets
Train
python small_history_facts.py

The script will:

Load Llama-3.2-1B

Apply QLoRA adapters

Train on the history dataset

Save the model to:

./history_expert_final
📁 Repository Structure
├── history_dataset_3000_relational_patched.json
├── small_history_facts.py
├── README.md
└── .gitignore
🔬 Key Learnings

Small models (1B) require explicit relational knowledge to avoid fact mixing.

Guardrails improve domain reliability significantly.

Dataset structure matters more than dataset size for specialized fine-tuning.

Chronology samples greatly reduce timeline hallucinations.

📌 Future Improvements

Add retrieval-augmented generation (RAG)

Benchmark against base Llama-3.2-1B

Release merged adapters for inference

# LLAMA Project

This repository provides a simple and minimal implementation for performing inference and Low-Rank Adaptation (LoRA) fine-tuning on Llama2-7B models (need 40GB GPU memory). It is designed with minimal dependencies (only `torch` and `sentencepiece`) to provide a straightforward setup.

---



## 📂 Directory Structure

---

```
Root Directory:
├── .gitignore             # Files to ignore in version control
├── alpaca_data_200.json   # Dataset or configuration for training/evaluation
├── consolidated.00.pth    # Pre-trained model weights
├── finetune.py            # Script for fine-tuning the model
├── inference.py           # Script for model inference
├── README.md              # Project documentation (this file)
├── tokenizer.model        # Pre-trained tokenizer file
├── llama/                 # Core library and model code
│   ├── __init__.py        # Library initialization
│   ├── generation.py      # Functions for text generation
│   ├── lazy_model.py      # token pruning version (used for inference)
│   ├── lora.py            # LoRA fine-tuning implementation
│   ├── model.py           # Model architecture
│   ├── model_train.py     # used for training
│   └── tokenizer.py       # used for inference
└── .vscode/               # VSCode IDE settings (if applicable)



## 📦 Dependencies

Install the required dependencies using:
$ pip install -r requirements.txt

Key dependencies:
- torch
- sentencepiece

```

## 🛠️ Usage

### 1. Fine-tune the Model
Run the following command to fine-tune the model using `alpaca_data_200.json`:
$ python finetune.py

### 2. Perform Inference
Use the script below to generate predictions based on prompts:
$ python inference.py

---

## 📜 File Details

### finetune.py
- Implements supervised fine-tuning using a custom dataset.
- Utilizes **LoRA** (Low-Rank Adaptation) for efficient parameter updates.

### inference.py
- Generates predictions for predefined prompts.
- Demonstrates text generation using the `LLAMA` model's `generate` function.

### llama/tokenizer.py
- Encodes and decodes text using a SentencePiece tokenizer.
- Provides functions to tokenize input sequences for fine-tuning and inference.

### llama/lora.py
- Implements **LoRA**-based updates for efficient fine-tuning.

### llama/model.py
- Defines the core architecture of the LLAMA model.
- for inference

### llama/model_train.py
- Extends `model.py` for full training functionality.
- Supports **LoRA** integration and GPU-based optimization.

### llama/lazy_model.py
- Token pruning to achieve model inference acceleration.

---

## 🧪 Example Usage

### Example Prompts for Inference
Prompt: "I believe the meaning of life is"
Generated: "to seek and create meaningful connections."

Prompt: "Translate English to French:\npeppermint =>"
Generated: "menthe poivrée"

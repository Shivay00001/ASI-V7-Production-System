# AI Neuro-Reflective LLM V7 - Production Multi-Head Architecture

**AI Neuro-Reflective LLM V7** is a production-grade, standalone Large Language Model implementation designed for scalability, robust training, and advanced cognitive modeling. It integrates a Transformer backbone with specialized neuro-reflective heads for meta-cognition and autonomous value alignment.

## 🚀 Features

- **Unified Architecture**: Single-file implementation (`production_asi_v7.py`) containing Model, Trainer, and Data Pipeline.
- **Advanced Cognitive Modeling**:
  - **Backbone**: Causal Language Model (default: GPT-2, extensible to Llama 3/Mistral).
  - **Reflection Head**: Analyze internal states for meta-cognitive confidence monitoring.
  - **Value Head**: RLHF-ready scalar output for alignment and reward modeling.
- **Production-Grade Training**:
  - Automatic Mixed Precision (AMP / FP16).
  - Gradient Accumulation for effective large-batch training.
  - Robust Checkpointing & State Recovery.
  - Dynamic Data Streaming for large datasets.
- **Interactive Interfaces**: Built-in CLI for Training and Chat.

## 🛠️ Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/Shivay00001/ASI-V7-Production-System.git
    cd ASI-V7-Production-System
    ```

2. **Install Dependencies**:
    The system requires PyTorch and Hugging Face Transformers.

    ```bash
    pip install torch transformers pydantic numpy
    ```

    *(Note: For GPU support, ensure you install the CUDA-enabled version of PyTorch).*

## 📖 Usage

 The system is driven by a single CLI entry point: `production_asi_v7.py`.

### 1. Training

Fine-tune the model on any text dataset. The system supports plain text files (one sentence/paragraph per line).

```bash
# Basic Training
python production_asi_v7.py train --data sample_training_data.txt

# Custom Epochs
python production_asi_v7.py train --data your_corpus.txt --epochs 5
```

**Training Artifacts**:

- Checkpoints are saved to `asi_v7_checkpoints/`.
- Logs are written to `asi_v7_system.log` and printed to console.

### 2. Interactive Chat

Talk to the ASI model directly in your terminal.

```bash
python production_asi_v7.py chat
```

*Type `exit` or `quit` to end the session.*

## 🧠 Architecture Overview

### The "Brain" (`ASICognitiveModel`)

The core class wraps a Hugging Face `AutoModelForCausalLM`. It injects two auxiliary heads into the forward pass:

1. **Reflection Head**: Takes the last hidden state -> `Linear` -> `Tanh` -> `Sigmoid`. Represents the model's "confidence" or self-reflection on its own output.
2. **Value Head**: Takes the last hidden state -> `MLP`. Outputs a scalar value used for Reinforcement Learning from Human Feedback (RLHF).

### The "Coach" (`ProductionTrainer`)

A custom training loop built from scratch (not just `Trainer` wrapper) to ensure full control over:

- **Forward Pass**: Handling specialized multiple outputs (Logits + Reflection + Value).
- **Optimization**: AdamW with linear warmup and cosine decay (via scheduler).
- **Precision**: `torch.cuda.amp.GradScaler` for FP16 training stability.

## 📜 License

Copyright (c) 2026 **Shivay Singh Rajput**. All Rights Reserved.

This software is the proprietary intellectual property of Shivay Singh Rajput.
Unauthorized copying, distribution, or modification of this file, via any medium, is strictly prohibited.

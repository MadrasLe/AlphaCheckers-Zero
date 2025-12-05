# ♟️ AlphaCheckers-Zero: Deep RL vs Generative AI Arena

> **A PyTorch implementation of AlphaZero for Checkers, featuring a "Battle Arena" benchmarking Deep Reinforcement Learning against State-of-the-Art LLMs (Llama 3, Kimi) and Classical Search (Minimax).**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?logo=pytorch)
![License](https://img.shields.io/badge/License-Apache_2.0-green)
![Status](https://img.shields.io/badge/Status-Research_Complete-purple)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Madras1/AlphaCherckerZero)

---

##  Project Overview

**AlphaCheckers-Zero** explores the boundaries of **Self-Play Reinforcement Learning** without human priors (Tabula Rasa). Built from scratch, the agent learns optimal strategies solely by playing against itself using a customized **Monte Carlo Tree Search (MCTS)** guided by a Deep Neural Network.

Beyond standard training, this repository hosts a unique **Battle Arena**, designed to answer a modern AI research question: 
*> Can the probabilistic "intuition" of Large Language Models (LLMs) beat the rigorous "calculation" of a specialized RL agent?*

---
### [👉 Click here to play AlphaCheckers-Zero Arena on Hugging Face](https://huggingface.co/spaces/Madras1/AlphaCherckerZero)

##  Training Infrastructure & Hyperparameters

The model achieved convergence efficiently, demonstrating robustness even under constrained compute resources.

* **Compute:** NVIDIA T4 & P100 GPUs (Google Colab / GCP).
* **Training Time:** ~10 Hours.
* **Strategy:** Checkpoint rotation handling for session continuity.

### Configuration
The training pipeline was optimized for stability over speed, using the following hyperparameters:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Num Episodes** | 500 | Total self-play cycles. |
| **Self-Play Games** | 25 | Games played per episode to gather data. |
| **MCTS Simulations** | 80 | Look-ahead steps per move decision. |
| **Batch Size** | 256 | Samples from Replay Buffer for training. |
| **Replay Buffer** | 75,000 | Sliding window of moves to prevent forgetting. |
| **Learning Rate** | 2e-3 | Adam Optimizer initial rate. |
| **PUCT ($c_{puct}$)** | 2.0 | Exploration vs. Exploitation balance factor. |

---

##  The Battle Arena Results

The trained model (`checkers_master_final.pth`) was subjected to rigorous testing against diverse intelligence architectures.

| Opponent Class | Specific Model | Result | Insight / Observation |
| :--- | :--- | :--- | :--- |
| **Biological** | Human Player | ✅ **Win** | Surpassed creator's skill level. |
| **Generative AI** | **Llama 3 70b** (via Groq) | ✅ **Win** | LLM struggled with long-term spatial consistency and hallucinated invalid moves under pressure. |
| **Generative AI** | **Kimi k2** | ✅ **Win** | Failed to maintain strategic defense against forced capture chains (multi-hop jumps). |
| **Classical Algo** | **Minimax (Depth 8)** | 🤝 **Draw** | **Critical Result:** The RL agent converged to a robust defensive policy, matching a brute-force engine capable of millions of future states. |

---

##  Project Structure

The codebase is modularized for clarity and reproducibility:

```bash
├── AlphacheckerTrainer.py   # Core Training Loop (Self-Play + Backprop)               
├── eval.py                  # Play locally (Human vs AI)
├── evaLLM.py                # Arena: AI vs LLMs (Requires Groq API)
├── evalminimax.py           # Arena: AI vs Minimax Algorithm
├── requirements.txt         # Project Dependencies
└── checkers_master_final.pth # Pre-trained Weights
 Getting Started
1. Installation
Clone the repo and install dependencies:
```

```bash
git clone [https://github.com/MadrasLe/AlphaCheckers-Zero.git](https://github.com/MadrasLe/AlphaCheckers-Zero.git)
cd AlphaCheckers-Zero
pip install -r requirements.txt
```

2. Play against the AI
Challenge the trained model yourself via CLI:

```bash
python eval.py
```
3. Run the "Arena" (RL vs LLM)
Test the agent against Llama 3 (requires a Groq API Key):

```bash
export GROQ_API_KEY="your_api_key_here"
python evaLLM.py
```
4. Training from Scratch
To reproduce the results (Warning: Requires GPU):

``bash
python AlphacheckerTrainer.py
```



Technical Details
The Neural Network
Input: (5, 8, 8) Tensor. Encodes [Player Pieces, Opponent Pieces, Kings] and Turn.

Backbone: Convolutional layers with Batch Normalization and ReLU.

Outputs: * Policy Head: Probability distribution over 64 squares (origin).

Value Head: Scalar [-1, 1] predicting win probability.

License & Credits
Developed by Gabriel Yogi. This project serves as a research proof-of-concept for Reinforcement Learning efficiency.

Licensed under Apache 2.0.

# AlphaCheckers-Zero

> A implementation of the **AlphaZero** algorithm for Checkers (Brazilian/International rules), featuring a specialized **Battle Arena** against Large Language Models (LLMs) and classical Minimax algorithms.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Maintained-green)

## About the Project

This repository contains a Deep Reinforcement Learning engine built from scratch using **PyTorch** and **Monte Carlo Tree Search (MCTS)**. The agent learns the game of Checkers solely through self-play, without any prior human knowledge or heuristics.

The project goes beyond standard implementation by including a unique **Arena Mode**, where the AlphaZero agent is benchmarked against:
1.  **Classical Algorithms:** Minimax with Alpha-Beta Pruning.
2.  **Generative AI:** State-of-the-art LLMs (via Groq API) to test logic vs. probabilistic generation.

##  Benchmarks & Performance

The trained model (`checkers_master_final.pth`) was subjected to rigorous testing. 

| Opponent | Type | Result | Notes |
| :--- | :--- | :--- | :--- |
| **Human Player** | Biological | ✅ **Win** | Surpassed the creator. |
| **Llama 3.3 70b** | LLM (Groq) | ✅ **Win** | Exploited the LLM's lack of spatial board consistency. |
| **Llama-4-maverick-17b-128e** | LLM | ✅ **Win** | Consistent tactical superiority. |
| **Kimi k2** | LLM | ✅ **Win** | The LLM failed to maintain long-term strategy. |
| **Minimax (Depth 8)** | Classical Algo | 🤝 **Draw** | **Crucial Result:** Proves the neural network has converged to a robust, defensive optimal policy, matching a brute-force engine calculating millions of moves. |

##  Technical Architecture

*   **Neural Network:** A ResNet-like architecture with a dual head:
    *   **Policy Head:** Outputs move probabilities ($p$).
    *   **Value Head:** Estimates the win probability ($v$) of the current state.
*   **Inference:** Uses MCTS guided by the neural network to simulate future outcomes.
*   **Training:** Continuous self-play loops with Replay Buffer and data augmentation.

## Project Structure

Please rename the source files to match the structure below for better organization:

*   `AlphaCheckerTrainer.py` Main training loop, MCTS logic, and Network Architecture.
*   `eval.py`: Interface to play against the AI locally.
*   `evalLLM.py`: Script to battle against LLMs using Groq API.
*   `evalminimax.py`: Script to battle against the Minimax algorithm.
*   `checkers_master_final.pth`: The AlphaChecker Weight Trained

## Getting Started


Developed by Gabriel Yogi.
This project is for research purposes in the field of Reinforcement Learning.

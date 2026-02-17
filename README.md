# Titanic-Adaptive-Feature-Permutations (Proof of Concept)

This repository contains a **terminal-driven proof of concept** demonstrating how
**learnable permutation matrices**, combined with **temperature annealing** and
**entropy regularization**, allow a neural network to *learn its own feature structure*
rather than relying on fixed input ordering.

The work intentionally avoids notebooks and visual tooling. All experiments are
script-based, reproducible, and architecture-focused.

---

## 🔬 Motivation

Standard tabular models assume:
- Fixed feature ordering
- Uniform feature treatment
- Static architecture topology

This experiment challenges those assumptions by allowing the network to:
- Learn *how* features are grouped and routed
- Specialize parallel branches via different learned permutations
- Gradually collapse soft permutations into near-discrete structures

This is **structure learning**, not just weight learning.

---

## 🧠 Core Idea

At the heart of this project is a **LearnableFeaturePermute layer**:

- Parameterizes a soft permutation matrix
- Uses Sinkhorn normalization for differentiability
- Applies temperature annealing to sharpen permutations over training
- Uses entropy regularization to prevent uniform, degenerate solutions

The result is a network that *reorganizes its own inputs* during training.

---

## 🏗 Architecture Overview

- Input → Learnable Feature Permutation
- Parallel processing branches (distinct permutations)
- Fusion gate (softmax-weighted)
- Output classifier

Each branch learns a different structural interpretation of the same data.

---

## 📊 Experiment

Dataset:
- Titanic (binary classification)

Results:
- Stable validation accuracy ~0.88
- No NaNs or training collapse
- Learned permutations converge and stabilize
- Regularization prevents overfitting despite long training

This confirms feasibility not optimality.

---

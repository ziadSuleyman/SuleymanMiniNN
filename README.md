# SuleymanMiniNN

### A Dynamic Deep Learning Framework Built From Scratch

SuleymanMiniNN is a fully functional deep learning framework implemented from scratch using NumPy.
It replicates the core design principles of modern dynamic frameworks such as PyTorch, with a focus on transparency, modularity, and educational depth.

The framework includes its own:

* Tensor abstraction
* Dynamic computational graph
* Reverse-mode automatic differentiation engine
* Neural network module system
* Optimizers (SGD, Adam)
* Batch Normalization
* Dropout
* Hyperparameter tuning system
* Full training pipeline (MNIST)

---

# Table of Contents

* Overview
* Architecture
* Features
* Installation
* Requirements
* Project Structure
* Usage Example
* Training on MNIST
* Results
* Design Philosophy
* Author

---

# Overview

SuleymanMiniNN is designed to expose the internal mechanics of deep learning systems.
Unlike high-level APIs, this project builds:

* Gradient tracking
* Graph construction
* Topological sorting
* Backpropagation
* Parameter updates

All from scratch.

The goal is to deeply understand how modern frameworks operate internally.

---

# Architecture

The framework follows a dynamic define-by-run paradigm:

1. Operations create graph nodes at runtime
2. Each Tensor tracks:

   * Data
   * Gradient
   * Parents
   * Operation metadata
3. Backward pass performs:

   * Reverse topological traversal
   * Chain rule propagation

Core components:

* `Tensor`
* `AutogradEngine`
* `Module`
* `Linear`
* `Activation Layers`
* `BatchNorm1d`
* `Dropout`
* `Optimizers`
* `Trainer`
* `HyperparameterTuner`

---

# Features

* Dynamic computational graph
* Reverse-mode automatic differentiation
* Modular neural network API
* Sequential model building
* Batch normalization
* Dropout regularization
* SGD & Adam optimizers
* Grid search hyperparameter tuning
* MNIST training pipeline

---

# Requirements

* Python 3.9+
* NumPy
* Matplotlib (for visualization)
* scikit-learn (for dataset utilities)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/ziadSuleyman/SuleymanMiniNN.git
cd SuleymanMiniNN
```

(Optional) Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Project Structure

```
SuleymanMiniNN/
│
├── SuleymanMiniNN/      # Core framework source code
├── examples/            # Example training scripts
├── tests/               # Unit tests
├── README.md
├── requirements.txt
└── LICENSE
```

---

# Usage Example

Simple neural network:

```python
from SuleymanMiniNN.nn import Sequential, Linear, ReLU
from SuleymanMiniNN.optim import Adam

model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)

optimizer = Adam(model.parameters(), lr=0.001)
```

Forward pass:

```python
output = model(x)
loss = loss_fn(output, y)
loss.backward()
optimizer.step()
```

---

# Training on MNIST

Run:

```bash
python examples/train_mnist.py
```

Configuration includes:

* 20 epochs
* Adam optimizer
* BatchNorm
* Dropout
* Grid search tuning

---

# Results

Final Test Accuracy: **96.82%**

* 20 Epochs
* BatchNorm + Dropout
* Adam optimizer
* Hyperparameter tuning

---

# Design Philosophy

This framework prioritizes:

* Mathematical clarity
* Explicit gradient flow
* Structural transparency
* Systems-level understanding

The objective is not performance optimization, but deep architectural comprehension of modern deep learning engines.

---

# Author

Ziad Suleyman
Applied AI Engineer
Deep Learning Systems Enthusiast


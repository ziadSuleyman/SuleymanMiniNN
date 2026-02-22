SuleymanMiniNN

A Dynamic Deep Learning Framework Built From Scratch

SuleymanMiniNN is a fully functional deep learning engine implemented from scratch using NumPy.

It includes:

Dynamic Computational Graph (Define-by-Run)

Reverse-Mode Automatic Differentiation

Modular Neural Network API

Optimizers (SGD, Adam)

BatchNorm & Dropout

Grid Search Hyperparameter Tuning

MNIST Training Pipeline

The project is designed for educational and research purposes to understand how modern frameworks like PyTorch work internally.

Core Philosophy

Unlike wrapper libraries, SuleymanMiniNN implements its own:

Tensor abstraction

Autograd engine

Topological graph traversal

Backpropagation mechanism

Training loop system

This allows full visibility into the computational graph and gradient flow.

Example: MNIST Training

Final Test Accuracy: 96.82%

20 Epoch Training
BatchNorm + Dropout
Adam Optimizer
Grid Search Tuning

Core Components

Tensor (data + gradient + history tracking)

AutogradEngine (topological sorting + chain rule)

Module system (Sequential, Linear, Activation layers)

Optimizers (SGD, Adam)

Trainer abstraction

Hyperparameter tuner

Why This Project Matters

Understanding deep learning at this level builds:

Mathematical intuition

Systems thinking

Research readiness

Applied AI engineering capability

Author

Ziad Suleyman
Applied AI Engineer
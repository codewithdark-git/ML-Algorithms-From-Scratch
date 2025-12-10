# Support Vector Machines (SVM) - From Scratch

A comprehensive Python and NumPy implementation of Support Vector Machines, covering both **Hard Margin** and **Soft Margin** formulations using the **Sequential Minimal Optimization (SMO)** algorithm.

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Mathematical Background](#mathematical-background)
6. [Visualization](#visualization)
7. [File Structure](#file-structure)
8. [Comparison with Sklearn](#comparison-with-sklearn)

---

## Introduction

Support Vector Machine (SVM) is a powerful supervised learning algorithm for classification. The key idea is to find a **hyperplane** that maximizes the **margin** between two classes.

### What Makes This Implementation Special?

- ✅ Pure Python + NumPy (no external ML libraries for training)
- ✅ Educational visualizations with animated GIFs
- ✅ Both Hard and Soft Margin support
- ✅ Multiple kernel functions (Linear, RBF, Polynomial)
- ✅ Comparison benchmarks with sklearn

---

## Features

| Feature | Description |
|---------|-------------|
| **Hard Margin SVM** | For perfectly linearly separable data |
| **Soft Margin SVM** | Allows misclassifications via slack variables |
| **SMO Optimizer** | Sequential Minimal Optimization algorithm |
| **Kernels** | Linear, RBF (Gaussian), Polynomial |
| **Visualizations** | Decision boundary plots, training GIFs |

---

## Installation

```bash
pip install numpy matplotlib scikit-learn
```

---

## Quick Start

### Hard Margin SVM
```python
from svm_hard import HardMarginSVM
import numpy as np

X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])
y = np.array([1, 1, 1, -1, -1])

model = HardMarginSVM(kernel='linear')
model.fit(X, y)
predictions = model.predict(X)
```

### Soft Margin SVM with RBF Kernel
```python
from svm_soft import SoftMarginSVM

model = SoftMarginSVM(C=1.0, kernel='rbf', gamma=0.5)
model.fit(X, y)
predictions = model.predict(X)
```

### Generate Training Visualizations
```python
from svm_hard import HardMarginSVM
from visualizer import SVMVisualizer

model = HardMarginSVM(kernel='linear', record_history=True)
model.fit(X, y)

viz = SVMVisualizer(model, X, y)
viz.plot_decision_boundary(title="Hard Margin SVM", save_path="decision_boundary.png")
viz.create_educational_gif(filename="training_animation.gif")
viz.create_margin_evolution_gif(filename="margin_evolution.gif")
```

---

## Mathematical Background

### The Hyperplane

Given dataset $D = \{(x_1, y_1), ..., (x_n, y_n)\}$ where $x_i \in \mathbb{R}^d$ and $y_i \in \{-1, +1\}$:

$$\mathbf{w} \cdot \mathbf{x} + b = 0$$

Decision rule: $f(x) = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$

### Hard Margin (Linearly Separable)

**Objective:** Maximize margin $\frac{2}{||\mathbf{w}||}$

$$\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2$$
$$\text{s.t.} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \ge 1$$

### Soft Margin (Non-Separable)

Introduces slack variables $\xi_i$ for misclassifications:

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{i=1}^n \xi_i$$
$$\text{s.t.} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

- **C → ∞**: Hard margin (no misclassification allowed)
- **C → 0**: Wide margin (allows more misclassification)

### Dual Problem

Using Lagrange multipliers $\alpha_i$:

$$\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$

**Constraints:** $0 \le \alpha_i \le C$ and $\sum_i \alpha_i y_i = 0$

### SMO Algorithm

Sequential Minimal Optimization solves the QP by:
1. Select two $\alpha_i, \alpha_j$ violating KKT conditions
2. Optimize analytically (closed-form solution)
3. Update bias $b$
4. Repeat until convergence

### Kernel Functions

| Kernel | Formula |
|--------|---------|
| Linear | $K(x, z) = x \cdot z$ |
| RBF | $K(x, z) = \exp(-\gamma ||x - z||^2)$ |
| Polynomial | $K(x, z) = (\gamma \cdot x \cdot z + c)^d$ |

---

## Visualization

The `SVMVisualizer` class provides three visualization methods:

### 1. Static Decision Boundary
```python
viz.plot_decision_boundary(title="SVM", save_path="boundary.png")
```

### 2. Educational Training Animation
Shows step-by-step optimization with annotations:
```python
viz.create_educational_gif(filename="training.gif", fps=8)
```

### 3. Margin Evolution Plot
Dual-panel animation showing boundary and margin width:
```python
viz.create_margin_evolution_gif(filename="margin.gif")
```

---

## File Structure

```
svm/
├── README.md              # This file
├── svm_core.py           # Core SVM with SMO algorithm
├── svm_hard.py           # Hard Margin SVM wrapper
├── svm_soft.py           # Soft Margin SVM wrapper
├── visualizer.py         # Enhanced visualization utilities
├── compare_bmark.py      # Sklearn comparison benchmark
└── outputs/              # Generated plots and GIFs
```

---

## Comparison with Sklearn

Run the benchmark:
```bash
python compare_bmark.py
```

This generates:
- `outputs/hard_margin_final.png` - Final decision boundary
- `outputs/hard_margin_training.gif` - Training animation
- `outputs/hard_margin_evolution.gif` - Margin evolution
- `outputs/soft_margin_final.png` - RBF kernel result
- `outputs/soft_margin_training.gif` - RBF training animation

---

## References

1. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
2. Platt, J. (1998). Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines. *Microsoft Research Technical Report*.

---

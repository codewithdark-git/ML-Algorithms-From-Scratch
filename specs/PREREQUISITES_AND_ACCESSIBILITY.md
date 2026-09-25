# Prerequisites & Multi-Level Accessibility Plan

## Overview

This document defines how to make "Building Machine Learning Algorithms from First Principles" and the accompanying codebase accessible to three distinct audiences:

| Audience | Profile | Goal | Entry Point |
|----------|---------|------|-------------|
| **Beginner** | Knows basic Python, no ML background | Build intuition + working implementations | Part I + Ch4-7 (supervised basics) |
| **Intermediate** | Knows ML concepts, wants from-scratch implementations | Deep understanding + production skills | Part II + Ch8-12 (unsupervised) + Ch16 |
| **Expert** | Experienced ML engineer, wants advanced topics | PINN, production systems, architecture | Part III + Ch13-15 (advanced) + Ch17 |

---

## 1. Prerequisites Structure

### 1.1 Chapter 2: Prerequisites (Expanded)

**Current state**: Chapter 2 exists but may be too brief
**Target**: Comprehensive prerequisite chapter with self-assessment

#### 2.1.1 Mathematical Prerequisites
```
Linear Algebra
├── Vectors, matrices, operations
├── Matrix multiplication, transpose, inverse
├── Eigenvalues/vectors (for PCA, SVD)
└── Norms, orthogonality

Calculus
├── Derivatives, partial derivatives, chain rule
├── Gradient, Hessian
├── Optimization: convexity, stationary points
└── Taylor series (for PINN)

Probability & Statistics
├── Random variables, distributions (Gaussian, Bernoulli, Multinomial)
├── Expectation, variance, covariance
├── Bayes' theorem, conditional probability
├── MLE, MAP estimation
└── Hypothesis testing basics (for Ch17)
```

#### 2.1.2 Programming Prerequisites
```
Python (3.10+)
├── NumPy: arrays, broadcasting, linear algebra (np.linalg)
├── Basic OOP: classes, inheritance, magic methods
├── Type hints (PEP 484), dataclasses
├── File I/O, JSON/YAML config
└── Basic testing: pytest, assertions

Development Environment
├── Virtual environments (venv/conda)
├── Package installation (pip install -e .)
├── Jupyter notebooks for exploration
└── Git basics
```

#### 2.1.3 Self-Assessment Checklist (Append to Chapter 2)
```markdown
## Prerequisite Self-Check

Rate yourself 1-5 on each (1=unfamiliar, 5=comfortable):

### Math
- [ ] Matrix multiplication & broadcasting
- [ ] Gradient descent intuition
- [ ] Gaussian distribution & log-likelihood
- [ ] Eigenvalues/eigenvectors

### Python
- [ ] NumPy array manipulation
- [ ] Writing classes with __init__, fit, predict
- [ ] Using type hints
- [ ] Running pytest

### Scoring
- 0-20: Start with Appendix A (Math Refresher) + Python tutorial
- 21-35: Review Appendix A as needed, begin Chapter 3
- 36-50: Ready for Chapter 3 directly
```

### 1.2 Appendix A: Mathematical Refresher (New)
- 30-page compact reference with worked examples
- One page per key concept: gradient, Hessian, SVD, Bayes, etc.
- "Cheat sheet" style for quick lookup during exercises

---

## 2. Multi-Level Learning Paths

### 2.1 Beginner Path: "Foundations First" (~12 weeks)

```
WEEK 1-2: Part I - Foundations
├── Ch1: Introduction (why from-scratch?)
├── Ch2: Prerequisites (complete self-assessment)
├── Ch3: Gradient Descent (5 exercises)
│   ├── ex01: Unified optimizer (core pattern)
│   ├── ex02: Shuffling effect (data order matters)
│   ├── ex03: LR decay (scheduling)
│   ├── ex04: Scaling experiment (feature scaling)
│   └── ex05: Momentum (acceleration)

WEEK 3-5: Ch4-5 Linear & Multiple Regression (12 exercises)
├── Ch4: Linear Regression (6 exercises)
│   ├── ex01-02: LR sensitivity, noise
│   ├── ex03: No intercept (bias term)
│   ├── ex04: Cost history (visualization)
│   ├── ex05: Normal eq vs GD (analytic vs iterative)
│   └── ex06: Scaling crossover (when scaling matters)
└── Ch5: Multiple/Polynomial (6 exercises)
    ├── ex01-02: Feature engineering basics
    ├── ex03: Coefficient back-transform
    ├── ex04: Residual analysis (diagnostics)
    ├── ex05: Feature comparison
    └── ex06: Interaction terms

WEEK 6-7: Ch6-7 Regularization & Classification (13 exercises)
├── Ch6: Regularized Regression (6 exercises)
│   ├── ex01-02: Ridge/Lasso alpha sweep
│   ├── ex03: Scale sensitivity (key insight!)
│   ├── ex04: Intercept penalty debate
│   ├── ex05: Closed-form vs GD for Ridge
│   └── ex06: Elastic net for correlated features
└── Ch7: Logistic Regression (7 exercises)
    ├── ex01: Threshold exploration (probability → class)
    ├── ex02: Training without scaling (why it fails)
    ├── ex03: 2D decision boundary visualization
    ├── ex04: ROC curve (evaluation)
    ├── ex05: L2 regularization
    ├── ex06: Nonlinear boundary (feature engineering)
    └── ex07: Multinomial extension (softmax)

WEEK 8: Capstone Project - Predict House Prices
├── End-to-end: data → EDA → model → evaluation → report
├── Uses: LinearRegression, RidgeRegression, cross_val_score
└── Deliverable: 3-page report with code
```

**Beginner-Friendly Features in Codebase**:
```python
# 1. Verbose mode in all estimators
model = LinearRegression(verbose=True)  # prints iteration, cost

# 2. Built-in plotting helpers
from ml_from_scratch.utils import plot_learning_curve, plot_decision_boundary

# 3. Detailed docstrings with mathematical formulas
# 4. Starter exercises have heavy comments explaining each step
# 5. Solution files include "Why this works" sections
```

### 2.2 Intermediate Path: "Algorithm Deep Dive" (~10 weeks)

```
WEEK 1-2: Ch8-10 Naive Bayes, SVM, Trees (20 exercises)
├── Ch8: Naive Bayes (7 exercises)
│   ├── ex01: Zero variance handling (numerical stability)
│   ├── ex02: Manual prediction (trace through math)
│   ├── ex03: 2D visualization (Gaussian contours)
│   ├── ex04: Iris multiclass + confusion matrix
│   ├── ex05: Log-space computation (underflow prevention)
│   ├── ex06: Correlated features (violation of assumption)
│   └── ex07: Multinomial NB for text (TF-IDF)
├── Ch9: SVM (7 exercises)
│   ├── ex01: Hard vs soft margin (C parameter)
│   ├── ex02: Gamma sweep (RBF kernel sensitivity)
│   ├── ex03: Outlier robustness (support vectors)
│   ├── ex04: Margin width vs C
│   ├── ex05: Polynomial kernel
│   ├── ex06: RBF kernel comparison
│   └── ex07: Heart disease (real dataset)
└── Ch10: Decision Trees (6 exercises)
    ├── ex01: Entropy vs Gini (impurity measures)
    ├── ex02: Depth vs overfitting (bias-variance)
    ├── ex03: Manual tree trace (white-box understanding)
    ├── ex04: Feature importance (permutation vs impurity)
    ├── ex05: Pruning (cost-complexity)
    └── ex06: Nonlinear boundary

WEEK 3-4: Ch11-12 Clustering (12 exercises)
├── Ch11: K-Means (6 exercises)
│   ├── ex01: KMeans++ vs random init
│   ├── ex02: Elbow method on real data
│   ├── ex03: Wrong K visualization
│   ├── ex04: Silhouette score (cluster quality)
│   ├── ex05: Non-spherical failure (moons dataset)
│   └── ex06: Outlier detection (distance to centroid)
└── Ch12: GMM (6 exercises)
    ├── ex01: BIC vs AIC (model selection)
    ├── ex02: Responsibility vector (E-step)
    ├── ex03: Diagonal covariance (constraints)
    ├── ex04: Regularization sensitivity
    ├── ex05: Iris comparison (GMM vs KMeans)
    └── ex06: KMeans initialization for EM

WEEK 5-6: Ch13 Neural Networks + Ch16 Pipelines (13 exercises)
├── Ch13: Neural Networks (6 exercises)
│   ├── ex01: Architecture sweep (width/depth)
│   ├── ex02: Tanh vs ReLU activation
│   ├── ex03: L2 regularization (weight decay)
│   ├── ex04: Overfitting point (early stopping)
│   ├── ex05: Regression output (MLPRegressor)
│   └── ex06: Minibatch training (SGD vs GD)
└── Ch16: Pipelines (7 exercises)
    ├── ex01: Mode imputation (preprocessing)
    ├── ex02: Data leakage demonstration
    ├── ex03: Regression metrics in pipeline
    ├── ex04: Pipeline-aware CV
    ├── ex05: Stratified KFold
    ├── ex06: Hyperparameter search
    └── ex07: Production pipeline

WEEK 7: Capstone Project - Customer Churn Prediction
├── Compare: LogisticRegression vs SVM vs DecisionTree vs MLP
├── Pipeline with preprocessing + CV + hyperparameter search
├── Model selection with statistical significance (Ch17 ex02)
└── Deliverable: Reproducible pipeline + comparison report
```

**Intermediate-Friendly Features**:
```python
# 1. Access to internal state for inspection
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)
print(model.tree_)  # Full tree structure
model.plot_tree()   # Visualize

# 2. Algorithm comparison utilities
from ml_from_scratch.utils import compare_models
results = compare_models(models, X, y, cv=5)

# 3. Configuration-driven experiments
from ml_from_scratch.production import ConfigManager
config = ConfigManager("config/experiment.yaml")

# 4. Property-based test utilities (Ch17 ex05)
from ml_from_scratch.testing import check_scale_invariance, check_permutation_invariance
```

### 2.3 Expert Path: "Advanced Topics & Production" (~8 weeks)

```
WEEK 1-2: Ch14-15 PINN (7 exercises)
├── Ch14: PINN Theory (already covered in Ch13)
├── Ch15: PINN (7 exercises)
│   ├── ex01: Different forcing functions (Poisson, Helmholtz)
│   ├── ex02: Collocation point strategies (random, grid, Latin hypercube)
│   ├── ex03: First derivative constraints (Neumann BC)
│   ├── ex04: Depth/width sweep (architecture search)
│   ├── ex05: Heat equation (time-dependent PDE)
│   ├── ex06: Autodiff comparison (forward vs reverse)
│   └── ex07: Inverse problem (parameter estimation)

WEEK 3-4: Ch17 Production Software (7 exercises)
├── ex01: Interchangeable interfaces (Strategy pattern)
├── ex02: Comparison test framework (A/B, McNemar)
├── ex03: Monitoring design (drift detection, alerting)
├── ex04: Project reorganization (modular, config-driven)
├── ex05: Property-based tests (invariants)
├── ex06: CI/CD integration (GitHub Actions)
└── ex07: Reflection (algorithm portfolio)

WEEK 5: Capstone - Production ML System
├── Design: data pipeline → training → validation → deployment → monitoring
├── Implement: model registry, A/B testing, drift alerts
├── Automate: CI/CD with quality gates
└── Document: Architecture decision records (ADRs)
```

**Expert-Friendly Features**:
```python
# 1. Extensible base classes
class BaseEstimator:
    def fit(self, X, y): ...
    def predict(self, X): ...
    def get_params(self): ...
    def set_params(self, **params): ...

# 2. Custom optimizer/loss registration
from ml_from_scratch.neural_networks import register_optimizer
@register_optimizer("my_adam")
class MyAdam(Optimizer): ...

# 3. Production monitoring hooks
from ml_from_scratch.production import ModelMonitor
monitor = ModelMonitor(model, reference_data=X_train)
monitor.log_prediction(X_new, y_pred)
alert = monitor.check_drift()

# 4. Distributed training hooks (future)
# 5. ONNX export for deployment
```

---

## 3. Codebase Accessibility Features

### 3.1 Progressive Disclosure in API

```python
# LEVEL 1: High-level (Beginner)
from ml_from_scratch import LinearRegression
model = LinearRegression()
model.fit(X, y)
preds = model.predict(X_test)

# LEVEL 2: With configuration (Intermediate)
from ml_from_scratch import LinearRegression
model = LinearRegression(
    optimizer="sgd",
    learning_rate=0.01,
    max_iter=1000,
    verbose=True
)
model.fit(X, y)
print(model.cost_history_)  # Access training history

# LEVEL 3: Full control (Expert)
from ml_from_scratch.optimization import stochastic_gradient_descent
from ml_from_scratch.linear_models import LinearRegression

coef = stochastic_gradient_descent(
    X, y,
    lr=0.01,
    batch_size=32,
    max_iter=1000,
    callback=lambda w, i: log_metrics(w, i)
)
model = LinearRegression()
model.coef_ = coef
```

### 3.2 Documentation Tiers

| Tier | Audience | Format | Location |
|------|----------|--------|----------|
| **API Reference** | All | Docstrings + Sphinx | `docs/api/` |
| **Tutorials** | Beginner | Step-by-step notebooks | `notebooks/tutorials/` |
| **How-to Guides** | Intermediate | Task-focused | `docs/guides/` |
| **Deep Dives** | Expert | Mathematical + architectural | `docs/deep-dives/` |
| **Exercise Walkthroughs** | All | Per-exercise | `topics/chXX/exercises/exNN/README.md` |

### 3.3 Exercise Difficulty Markers

Each exercise README includes:
```markdown
## Difficulty: ⭐⭐☆☆☆ (Easy/Intermediate/Hard)

### Prerequisites
- [ ] Chapter 3: Gradient Descent
- [ ] NumPy broadcasting

### Estimated Time: 45-60 minutes

### Learning Objectives
1. Implement batch gradient descent from scratch
2. Understand learning rate effect on convergence
3. Visualize cost history

### Extension (for advanced learners)
- Add momentum term
- Implement Adam optimizer
- Compare convergence rates
```

---

## 4. Implementation Plan

### Phase A: Prerequisites Enhancement (Week 1-2)

| Task | Description | Files |
|------|-------------|-------|
| A1 | Expand Chapter 2 with self-assessment | `Book/chapters/chapter02.tex` |
| A2 | Create Appendix A: Math Refresher | `Book/appendices/appendixA.tex` |
| A3 | Add prerequisite badges to each chapter | `Book/chapters/chapter*.tex` |
| A4 | Add `verbose` mode to all estimators | `src/ml_from_scratch/*/*.py` |

### Phase B: Learning Path Documentation (Week 2-3)

| Task | Description | Files |
|------|-------------|-------|
| B1 | Create LEARNING_PATHS.md | `docs/LEARNING_PATHS.md` |
| B2 | Add difficulty markers to all 89 exercise READMEs | `topics/ch*/exercises/ex*/README.md` |
| B3 | Create beginner/intermediate/expert notebook templates | `notebooks/templates/` |
| B4 | Add `utils/plotting.py` with helpers | `src/ml_from_scratch/utils/plotting.py` |

### Phase C: Codebase Features (Week 3-4)

| Task | Description | Files |
|------|-------------|-------|
| C1 | Add `get_params`/`set_params` to all estimators | `src/ml_from_scratch/*/*.py` |
| C2 | Create `ConfigManager` for YAML-driven experiments | `src/ml_from_scratch/production/config.py` |
| C3 | Add `compare_models` utility | `src/ml_from_scratch/utils/comparison.py` |
| C4 | Create property-based test helpers | `src/ml_from_scratch/testing/properties.py` |

### Phase D: Documentation & Examples (Week 4-5)

| Task | Description | Files |
|------|-------------|-------|
| D1 | Build Sphinx docs with 3-tier structure | `docs/source/` |
| D2 | Create 5 beginner tutorials | `notebooks/tutorials/01-05_*.ipynb` |
| D3 | Create 3 intermediate guides | `notebooks/guides/01-03_*.ipynb` |
| D4 | Create 2 expert deep-dives | `notebooks/deep-dives/01-02_*.ipynb` |
| D5 | Add capstone project templates | `projects/capstone_*/` |

---

## 5. Entry Points by Audience

### 5.1 Quick Start Scripts

```bash
# Beginner: Guided tour
python scripts/beginner_tour.py
# → Runs Ch3 ex01, Ch4 ex01, Ch7 ex01 with explanations

# Intermediate: Algorithm comparison
python scripts/algorithm_comparison.py --dataset iris
# → Trains 5 models, compares with CV, prints report

# Expert: Production pipeline demo
python scripts/production_demo.py --config config/production.yaml
# → Full MLOps pipeline with monitoring
```

### 5.2 Web-Based Interactive (Future)

- JupyterLite deployment for browser-based exercises
- Binder links in each exercise README
- Colab notebooks for GPU-required exercises (PINN)

---

## 6. Accessibility Checklist

### For Beginners
- [ ] No unexplained jargon in first 3 chapters
- [ ] Every mathematical symbol defined inline
- [ ] Code has line-by-line comments in starters
- [ ] Visual outputs for every exercise
- [ ] Common error explanations ("Why did this fail?")

### For Intermediates
- [ ] Algorithm variants exposed (SGD, momentum, Adam)
- [ ] Hyperparameter explanations with ranges
- [ ] Comparison utilities built-in
- [ ] Real-world datasets in exercises
- [ ] Pipeline integration examples

### For Experts
- [ ] Extensible base classes documented
- [ ] Custom component registration
- [ ] Production monitoring hooks
- [ ] Performance profiling utilities
- [ ] Architecture decision records template

### Universal
- [ ] All code type-hinted
- [ ] Comprehensive docstrings (NumPy style)
- [ ] Example in every docstring
- [ ] Tests serve as executable documentation
- [ ] CI validates all examples run

---

## 7. Metrics for Success

| Metric | Beginner | Intermediate | Expert |
|--------|----------|--------------|--------|
| Time to first working model | < 30 min | < 15 min | < 10 min |
| Exercise completion rate | > 80% | > 90% | > 95% |
| Capstone project success | > 70% | > 85% | > 90% |
| "Would recommend" score | > 4.5/5 | > 4.5/5 | > 4.5/5 |

---

## 8. Maintenance

- **Quarterly**: Review prerequisite assumptions based on reader feedback
- **Per release**: Update learning paths for new chapters
- **Continuous**: Monitor exercise completion analytics (if telemetry added)
- **Annually**: Major version - restructure paths based on ML landscape changes

---

## Appendix: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│  ML FROM SCRATCH - LEARNING PATH QUICK REFERENCE           │
├──────────────┬──────────────────────────────────────────────┤
│ BEGINNER     │ Ch1-3 (Foundations) → Ch4-7 (Supervised)     │
│ 12 weeks     │ Capstone: House Price Prediction             │
├──────────────┼──────────────────────────────────────────────┤
│ INTERMEDIATE │ Ch8-12 (Unsupervised) → Ch13,16 (NN+Pipeline)│
│ 10 weeks     │ Capstone: Customer Churn                     │
├──────────────┼──────────────────────────────────────────────┤
│ EXPERT       │ Ch14-15 (PINN) → Ch17 (Production)           │
│ 8 weeks      │ Capstone: Production ML System               │
└──────────────┴──────────────────────────────────────────────┘

START HERE: python scripts/self_assess.py
DOCS:       docs/LEARNING_PATHS.md
EXERCISES:  python scripts/list_exercises.py --chapter 3
RUN:        python scripts/run_exercise.py --chapter 3 --exercise 1 --variant starter
```
# Learning Paths

> **Quick Start:** Run `python scripts/self_assess.py` to find your recommended path.

---

## Overview

This book and codebase supports three learning paths, each designed for a different background and goal:

| Path | Audience | Duration | Chapters | Capstone |
|------|----------|----------|----------|----------|
| **Beginner** | Python basics, no ML | 12 weeks | 1-7 | House Price Prediction |
| **Intermediate** | ML concepts, wants implementations | 10 weeks | 8-12, 13, 16 | Customer Churn |
| **Expert** | ML engineer, advanced topics | 8 weeks | 14-15, 17 | Production ML System |

---

## Beginner Path: "Foundations First"

### Prerequisites
- Basic Python (variables, loops, functions)
- High school algebra
- Willingness to learn NumPy

### Week-by-Week Schedule

| Week | Chapters | Exercises | Focus |
|------|----------|-----------|-------|
| 1-2 | 1-3 | Ch3: 5 ex | Gradient descent, optimization |
| 3-4 | 4 | 6 ex | Linear regression fundamentals |
| 5 | 5 | 6 ex | Multiple & polynomial regression |
| 6 | 6 | 6 ex | Regularization (Ridge/Lasso/ElasticNet) |
| 7 | 7 | 7 ex | Logistic regression, classification |
| 8 | - | Capstone | House price prediction |

### Key Exercises to Complete First
1. **Ch3 ex01** - Unified optimizer (core pattern for all GD)
2. **Ch4 ex01** - Large LR (learning rate sensitivity)
3. **Ch4 ex04** - Cost history (visualization)
4. **Ch5 ex01** - Third feature (multiple regression)
5. **Ch6 ex03** - Scale sensitivity (critical insight!)
6. **Ch7 ex01** - Threshold exploration (probability → class)

### Beginner Tips
- **Always run starter first** - see the TODOs, understand the structure
- **Use `--variant solution`** to compare your approach
- **Read the README** - each has "Why this works" section
- **Enable verbose mode**: `model = LinearRegression(verbose=True)`
- **Plot everything**: `from ml_from_scratch.utils import plot_learning_curve`

### Capstone: House Price Prediction
```bash
# End-to-end project
# 1. Load data → 2. EDA → 3. Baseline → 4. Iterate → 5. Report
# Uses: LinearRegression, RidgeRegression, cross_val_score
# Deliverable: 3-page report with code
```

---

## Intermediate Path: "Algorithm Deep Dive"

### Prerequisites
- Completed Beginner path OR
- Solid ML theory (bias-variance, regularization, CV)
- Comfortable with NumPy, OOP, type hints

### Week-by-Week Schedule

| Week | Chapters | Exercises | Focus |
|------|----------|-----------|-------|
| 1-2 | 8-9 | 14 ex | Naive Bayes, SVM |
| 3 | 10 | 6 ex | Decision Trees |
| 4 | 11-12 | 12 ex | K-Means, GMM |
| 5 | 13 | 6 ex | Neural Networks |
| 6 | 16 | 7 ex | Pipelines & Production |
| 7 | - | Capstone | Customer Churn |

### Key Exercises to Complete First
1. **Ch8 ex05** - Log-space (numerical stability)
2. **Ch9 ex02** - Gamma sweep (RBF sensitivity)
3. **Ch10 ex03** - Manual tree trace (white-box)
4. **Ch11 ex04** - Silhouette score (cluster quality)
5. **Ch12 ex01** - BIC vs AIC (model selection)
6. **Ch13 ex04** - Overfitting point (early stopping)
7. **Ch16 ex02** - Leakage demonstration (critical!)

### Intermediate Tips
- **Inspect internals**: `model.tree_`, `model.support_vectors_`, `model.centroids_`
- **Compare algorithms**: `python scripts/algorithm_comparison.py --dataset iris`
- **Use ConfigManager** for reproducible experiments
- **Pipeline-aware CV**: Prevents data leakage automatically
- **Property-based tests**: Verify algorithm invariants

### Capstone: Customer Churn Prediction
```bash
# Compare multiple algorithms with proper CV
# Pipeline: preprocessing → model → evaluation
# Statistical significance testing (McNemar's test)
# Deliverable: Reproducible pipeline + comparison report
```

---

## Expert Path: "Advanced Topics & Production"

### Prerequisites
- Production ML experience
- Deep learning fundamentals
- Software engineering practices (testing, CI/CD, monitoring)

### Week-by-Week Schedule

| Week | Chapters | Exercises | Focus |
|------|----------|-----------|-------|
| 1-2 | 14-15 | 7 ex | PINN (Physics-Informed NN) |
| 3-4 | 17 | 7 ex | Production Software |
| 5 | - | Capstone | Production ML System |

### Key Exercises
1. **Ch14 ex05** - Heat equation (time-dependent PDE)
2. **Ch14 ex07** - Inverse problem (parameter estimation)
3. **Ch17 ex01** - Interchangeable interfaces (Strategy pattern)
4. **Ch17 ex02** - Comparison test framework (A/B, McNemar)
5. **Ch17 ex03** - Monitoring design (drift, alerting)
6. **Ch17 ex05** - Property-based tests (invariants)
7. **Ch17 ex06** - CI/CD integration (GitHub Actions)

### Expert Tips
- **Extend base classes**: `BaseEstimator`, `BaseOptimizer`
- **Register custom components**: `@register_optimizer("my_adam")`
- **Production monitoring**: `ModelMonitor` with drift detection
- **ONNX export** for deployment (future)
- **Distributed training hooks** (architecture ready)

### Capstone: Production ML System
```bash
# Full MLOps pipeline
# Data → Train → Validate → Deploy → Monitor
# Model registry, A/B testing, drift alerts
# CI/CD with quality gates
# Architecture Decision Records (ADRs)
# Deliverable: Deployed system + documentation
```

---

## Exercise Difficulty Guide

Each exercise is marked with difficulty:

| Symbol | Level | Description |
|--------|-------|-------------|
| ⭐☆☆☆☆ | Easy | Core concept, guided implementation |
| ⭐⭐☆☆☆ | Medium | Requires combining concepts |
| ⭐⭐⭐☆☆ | Hard | Open-ended, requires insight |
| 🧠 | Reflection | No code, written analysis |

### By Chapter

| Chapter | Topic | Difficulty Range | Key Hard Exercises |
|---------|-------|------------------|-------------------|
| 3 | Gradient Descent | ⭐-⭐⭐⭐ | ex05: Momentum |
| 4 | Linear Regression | ⭐-⭐⭐ | ex06: Scaling crossover |
| 5 | Multiple/Polynomial | ⭐-⭐⭐ | ex06: Interaction terms |
| 6 | Regularization | ⭐-⭐⭐ | ex06: Elastic net |
| 7 | Logistic Regression | ⭐-⭐⭐⭐ | ex06-07: Nonlinear, Multinomial |
| 8 | Naive Bayes | ⭐-⭐⭐⭐ | ex05: Log-space, ex07: Text |
| 9 | SVM | ⭐-⭐⭐⭐ | ex05-07: Kernels, real data |
| 10 | Decision Trees | ⭐-⭐⭐ | ex05: Pruning, ex06: Nonlinear |
| 11 | K-Means | ⭐-⭐⭐ | ex05: Non-spherical, ex06: Outliers |
| 12 | GMM | ⭐-⭐⭐⭐ | ex04: Regularization, ex06: Init |
| 13 | Neural Networks | ⭐-⭐⭐⭐ | ex04: Overfitting, ex06: Minibatch |
| 14 | PINN | ⭐⭐-⭐⭐⭐⭐ | ex05: Heat eq, ex07: Inverse |
| 15 | Pipelines | ⭐-⭐⭐⭐ | ex02: Leakage, ex06: HPO |
| 16 | Production | ⭐-⭐⭐⭐ | ex05: Properties, ex06: CI/CD |

---

## Quick Commands Reference

```bash
# Self-assessment
python scripts/self_assess.py

# Beginner guided tour
python scripts/beginner_tour.py

# List all exercises
python scripts/list_exercises.py

# Run any exercise (starter or solution)
python scripts/run_exercise.py --chapter 3 --exercise 1 --variant starter
python scripts/run_exercise.py --chapter 3 --exercise 1 --variant solution

# Algorithm comparison (intermediate)
python scripts/algorithm_comparison.py --dataset iris
python scripts/algorithm_comparison.py --dataset breast_cancer --task classification
python scripts/algorithm_comparison.py --dataset synthetic --task regression --n-samples 5000

# Production demo (expert)
python scripts/production_demo.py --create-config  # creates config/production.yaml
python scripts/production_demo.py --config config/production.yaml

# Validate everything
python scripts/check_exercise_coverage.py
python scripts/check_book_routes.py
python -m pytest topics/ -v
```

---

## Getting Help

| Resource | Command/Location |
|----------|------------------|
| Exercise README | `topics/chXX/exercises/exNN_slug/README.md` |
| Solution code | `topics/chXX/exercises/exNN_slug/solution.py` |
| API docs | `docs/api/` (after `make docs`) |
| Tutorials | `notebooks/tutorials/` |
| GitHub Discussions | https://github.com/.../discussions |
| Issues | https://github.com/.../issues |

---

## Progress Tracking

### Beginner Checklist
- [ ] Ch3: All 5 exercises (GD fundamentals)
- [ ] Ch4: All 6 exercises (Linear regression)
- [ ] Ch5: All 6 exercises (Multiple/Polynomial)
- [ ] Ch6: All 6 exercises (Regularization)
- [ ] Ch7: All 7 exercises (Logistic regression)
- [ ] Capstone: House price prediction report

### Intermediate Checklist
- [ ] Ch8: All 7 exercises (Naive Bayes)
- [ ] Ch9: All 7 exercises (SVM)
- [ ] Ch10: All 6 exercises (Decision Trees)
- [ ] Ch11: All 6 exercises (K-Means)
- [ ] Ch12: All 6 exercises (GMM)
- [ ] Ch13: All 6 exercises (Neural Networks)
- [ ] Ch16: All 7 exercises (Pipelines)
- [ ] Capstone: Customer churn prediction

### Expert Checklist
- [ ] Ch14: All 7 exercises (PINN)
- [ ] Ch17: All 7 exercises (Production)
- [ ] Capstone: Production ML system
- [ ] Contribute: PR with new exercise/algorithm

---

## FAQ

**Q: Can I skip chapters?**
A: Beginner path: no, each builds on previous. Intermediate: can skip 1-7 if comfortable. Expert: focus on 14-15, 17.

**Q: What if an exercise is too hard?**
A: Read the solution, understand it, then re-implement. Use the "Extension" section in README for easier variants.

**Q: Do I need GPU?**
A: Only for PINN exercises (Ch14-15). Others run on CPU. Use Colab for GPU exercises.

**Q: How long does each exercise take?**
A: Easy: 30-45 min, Medium: 45-60 min, Hard: 60-90 min. Starters are faster.

**Q: Can I use this in a course?**
A: Yes! Each exercise is self-contained. See `docs/INSTRUCTOR_GUIDE.md` (future).

---

*Last updated: 2024 | Version: 1.0*
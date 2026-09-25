# Project Plan: ML-Algorithms-From-Scratch Restructure & Exercise Implementation

## Overview

This plan covers restructuring the existing flat codebase into a proper Python package (src/ml_from_scratch/) and implementing all 83 exercises from the book chapters with the required infrastructure (scripts, topics directory, tests).

Total Exercises: 83 unique exercises across 14 chapters (Ch3-Ch17; Ch12/13 are duplicates)

---

## Phase 1: Restructure Core Codebase

### 1.1 Create src/ml_from_scratch/ Package Structure

src/ml_from_scratch/
|-- __init__.py
|-- optimization.py
|-- validation.py
|-- random.py
|-- metrics.py
|-- preprocessing.py
|-- datasets/
|   |-- __init__.py
|   |-- synthetic.py
|   |-- loaders.py
|-- linear_models/
|   |-- __init__.py
|   |-- base.py
|   |-- linear_regression.py
|   |-- regularized_regression.py
|   |-- logistic_regression.py
|-- naive_bayes/
|   |-- __init__.py
|   |-- gaussian_nb.py
|-- svm/
|   |-- __init__.py
|   |-- svm.py
|-- trees/
|   |-- __init__.py
|   |-- decision_tree.py
|-- clustering/
|   |-- __init__.py
|   |-- kmeans.py
|   |-- gmm.py
|-- neural_networks/
|   |-- __init__.py
|   |-- layers.py
|   |-- activations.py
|   |-- losses.py
|   |-- optimizers.py
|   |-- mlp.py
|-- pinn/
|   |-- __init__.py
|   |-- pinn.py
|   |-- autodiff.py
|-- pipeline/
|   |-- __init__.py
|   |-- pipeline.py
|   |-- preprocessing_steps.py
|   |-- cross_validation.py
|-- production/
    |-- __init__.py
    |-- monitoring.py
    |-- model_registry.py
    |-- testing.py

### 1.2-1.5 see full plan in specs/PROJECT_PLAN.md (continued below)
---

## Phase 2: Create Exercise Infrastructure

### 2.1 Create scripts/ Directory

scripts/
|-- __init__.py
|-- list_exercises.py
|-- run_exercise.py
|-- check_exercise_coverage.py
|-- check_book_routes.py
|-- regenerate_figures.py

### 2.2 Script Specifications

list_exercises.py: Lists all 83 exercises by chapter
run_exercise.py: Runs starter/solution variants with --chapter --exercise --variant args
check_exercise_coverage.py: Verifies all 83 exercises present with all 4 required files
check_book_routes.py: Validates codepath{} references in .tex files
regenerate_figures.py: Regenerates plots from versioned scripts

### 2.3 Create topics/ Directory Structure (83 exercise dirs)

topics/
|-- ch03_gradient_descent/exercises/ (5: ex01..ex05)
|-- ch04_linear_regression/exercises/ (6: ex01..ex06)
|-- ch05_multiple_polynomial_regression/exercises/ (6: ex01..ex06)
|-- ch06_regularization/exercises/ (6: ex01..ex06)
|-- ch07_logistic_regression/exercises/ (7: ex01..ex07)
|-- ch08_naive_bayes/exercises/ (7: ex01..ex07)
|-- ch09_svm/exercises/ (7: ex01..ex07)
|-- ch10_decision_trees/exercises/ (6: ex01..ex06)
|-- ch11_kmeans/exercises/ (6: ex01..ex06)
|-- ch12_gmm/exercises/ (6: ex01..ex06)
|-- ch14_neural_networks/exercises/ (6: ex01..ex06)
|-- ch15_pinn/exercises/ (7: ex01..ex07)
|-- ch16_pipelines/exercises/ (7: ex01..ex07)
|-- ch17_production/exercises/ (7: ex01..ex07)

### 2.4 Exercise Template Files
Each exNN_slug/ has: starter.py, solution.py, test_solution.py, README.md, expected/, config.yaml

### 2.5 Acceptance Criteria Phase 2
- All 5 scripts executable
- list_exercises.py outputs 83 exercises
- run_exercise.py works for sample
- check_exercise_coverage.py reports 83, 0 missing
- topics/ structure complete with all 83 dirs
- Templates created for each exercise

---

## Phase 3: Implement All 83 Exercises

### 3.1 Exercise Implementation Standards
Each exercise: README.md (chapter, number, name, difficulty, objectives, task, starter guide, commands, acceptance criteria, pitfalls, exploration), starter.py (TODOs, deterministic data, actionable errors), solution.py (complete, reuses src/, explains math, preserves difficulty, actual results), test_solution.py (correctness test, edge case, fixed seeds, plot validation)

### 3.2 Exercise Mapping by Chapter

Ch3 Gradient Descent (5): ex01_unified_optimizer(Easy), ex02_shuffling_effect(Easy), ex03_lr_decay(Medium), ex04_scaling_experiment(Medium), ex05_momentum(Hard)
Ch4 Linear Regression (6): ex01_large_lr(Easy), ex02_more_noise(Easy), ex03_no_intercept(Medium), ex04_cost_history(Medium), ex05_normal_eq_vs_gd(Medium), ex06_scaling_crossover(Hard)
Ch5 Multiple/Polynomial (6): ex01_third_feature(Easy), ex02_sine_wave(Easy), ex03_back_transform_coef(Medium), ex04_residual_analysis(Medium), ex05_feature_comparison(Medium), ex06_interaction_terms(Hard)
Ch6 Regularization (6): ex01_ridge_alpha_sweep(Easy), ex02_lasso_sparsity(Easy), ex03_scale_sensitivity(Medium), ex04_intercept_penalty(Medium), ex05_closed_form_vs_gd(Medium), ex06_correlated_features_enet(Hard)
Ch7 Logistic Regression (7): ex01_threshold_exploration(Easy), ex02_training_without_scaling(Easy), ex03_two_feature_boundary(Medium), ex04_roc_curve(Medium), ex05_l2_regularization(Hard), ex06_nonlinear_boundary(Hard), ex07_multiclass_extension(Hard)
Ch8 Naive Bayes (7): ex01_zero_variance(Easy), ex02_manual_prediction(Easy), ex03_2d_visualization(Medium), ex04_iris_multiclass_cm(Medium), ex05_logspace_predict_proba(Hard), ex06_correlated_features(Hard), ex07_laplace_smoothing(Hard)
Ch9 SVM (7): ex01_hard_vs_soft_margin(Easy), ex02_gamma_sweep(Easy), ex03_outlier_robustness(Medium), ex04_margin_width_vs_c(Medium), ex05_polynomial_kernel(Hard), ex06_rbf_kernel_comparison(Hard), ex07_heart_disease_cm(Hard)
Ch10 Decision Trees (6): ex01_entropy_vs_gini(Easy), ex02_depth_vs_overfitting(Easy), ex03_manual_tree_trace(Medium), ex04_feature_importance(Medium), ex05_pruning(Hard), ex06_nonlinear_boundary(Hard)
Ch11 K-Means (6): ex01_kmeanspp_vs_random(Easy), ex02_elbow_method(Easy), ex03_wrong_k_visualization(Medium), ex04_silhouette_score(Medium), ex05_nonspherical_failure(Hard), ex06_outlier_detection(Hard)
Ch12 GMM (6, Ch13 duplicate skip): ex01_bic_vs_aic(Easy), ex02_responsibility_vector(Easy), ex03_diagonal_covariance(Medium), ex04_regularization_sensitivity(Medium), ex05_iris_comparison(Hard), ex06_kmeans_initialization(Hard)
Ch14 Neural Networks (6): ex01_architecture_sweep(Easy), ex02_tanh_activation(Easy), ex03_l2_regularization(Medium), ex04_overfitting_point(Medium), ex05_regression_output(Hard), ex06_minibatch_training(Hard)
Ch15 PINN (7): ex01_different_forcing(Easy), ex02_collocation_points(Easy), ex03_first_derivative(Medium), ex04_depth_width_sweep(Medium), ex05_heat_equation(Hard), ex06_autodiff(Hard), ex07_inverse_problem(Hard)
Ch16 Pipelines (7): ex01_mode_imputation(Easy), ex02_leakage_demo(Easy), ex03_regression_metrics(Medium), ex04_pipeline_cv(Medium), ex05_stratified_kfold(Hard), ex06_hyperparameter_search(Hard), ex07_model_comparison(Hard)
Ch17 Production (7): ex01_interchangeable_interfaces(Easy), ex02_comparison_test(Easy), ex03_monitoring_design(Medium), ex04_project_reorganization(Medium), ex05_property_based_test(Hard), ex06_ci_cd_pipeline(Hard), ex07_reflection(Reflection)

### 3.3 Implementation Sequence
Batch 1: Ch3-6 (23 exercises) ~5 days
Batch 2: Ch7-9 (21 exercises) ~5 days
Batch 3: Ch10-12 (18 exercises) ~4 days
Batch 4: Ch14-15 (13 exercises) ~3 days
Batch 5: Ch16-17 (14 exercises) ~3 days

### 3.4 Acceptance Criteria Phase 3
- All 83 dirs have 4 required files
- starter.py runs with TODOs
- solution.py runs with output
- test_solution.py passes pytest
- README.md follows CLAUDE.md standards
- No Ch13 duplicates

---

## Phase 4: Validate & Test

### 4.1 Test Suite Organization
tests/
|-- unit/ (existing)
|-- integration/ (existing)
|-- invariants/ (existing)
|-- parity/ (existing)
|-- smoke/ (existing)
|-- exercises/ (NEW)
    |-- test_ch03_exercises.py ... test_ch17_exercises.py

### 4.2 Validation Steps
1. pytest tests/exercises/ -v (100% pass)
2. pytest tests/ -v (all pass)
3. python scripts/check_exercise_coverage.py (83 found, 0 missing)
4. python scripts/check_book_routes.py (all codepath resolve)
5. python scripts/regenerate_figures.py (no errors)
6. mypy src/ && ruff check . (zero errors)
7. Sample run: for ch in 3 4 5 6 7 8 9 10 11 12 14 15 16 17; do python scripts/run_exercise.py --chapter $ch --exercise 1 --variant solution; done
8. cd Book && latexmk -pdf main.tex (clean compile)

### 4.3 Quality Gates (from CLAUDE.md)
1. 83 exercises exactly once (no Ch13 duplicate)
2. Every exercise has starter, solution, passing tests
3. Format, lint, type checks pass
4. One representative solution per topic executes
5. Notebooks valid JSON, representative execute
6. All manuscript code routes resolve
7. Figures regenerate correctly
8. Book compiles cleanly
9. No credentials/private data
10. Dependency groups documented

---

## Dependencies & Critical Path
Phase 1: ~16 days (package structure, shared utils, 14 modules)
Phase 2: ~4 days (5 scripts, topics structure, templates)
Phase 3: ~20 days (83 exercises in 5 batches, parallelizable)
Phase 4: ~3 days (validation)
Total: ~43 days (reducible with parallelization)

---

## Critical Files for Implementation

| File | Purpose |
|------|---------|
| src/ml_from_scratch/__init__.py | Package exports, version |
| src/ml_from_scratch/optimization.py | Ch3 core - all GD variants |
| src/ml_from_scratch/linear_models/linear_regression.py | Ch4, Ch5 core |
| src/ml_from_scratch/linear_models/regularized_regression.py | Ch6 core |
| src/ml_from_scratch/linear_models/logistic_regression.py | Ch7 core |
| src/ml_from_scratch/naive_bayes/gaussian_nb.py | Ch8 core |
| src/ml_from_scratch/svm/svm.py | Ch9 core |
| src/ml_from_scratch/trees/decision_tree.py | Ch10 core |
| src/ml_from_scratch/clustering/kmeans.py | Ch11 core |
| src/ml_from_scratch/clustering/gmm.py | Ch12/13 core |
| src/ml_from_scratch/neural_networks/mlp.py | Ch14 core |
| src/ml_from_scratch/pinn/pinn.py | Ch15 core |
| src/ml_from_scratch/pipeline/pipeline.py | Ch16 core |
| src/ml_from_scratch/production/monitoring.py | Ch17 core |
| scripts/list_exercises.py | Exercise discovery |
| scripts/run_exercise.py | Exercise runner |
| scripts/check_exercise_coverage.py | Coverage validator |
| scripts/check_book_routes.py | Book route validator |
| scripts/regenerate_figures.py | Figure regeneration |
| topics/ch03_gradient_descent/exercises/ex01_unified_optimizer/starter.py | Example exercise starter |
| topics/ch03_gradient_descent/exercises/ex01_unified_optimizer/solution.py | Example exercise solution |
| topics/ch03_gradient_descent/exercises/ex01_unified_optimizer/test_solution.py | Example exercise tests |
| topics/ch03_gradient_descent/exercises/ex01_unified_optimizer/README.md | Example exercise docs |
| tests/exercises/test_ch03_exercises.py | Exercise test aggregation |
| pyproject.toml | Package config (already exists, verify) |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Ch12/13 duplicate confusion | Wasted effort on Ch13 | Explicitly skip Ch13; only implement Ch12 exercises |
| PINN torch dependency | CI failures without GPU | Make torch optional; skip PINN tests in CI without torch |
| Exercise template inconsistency | Poor learner experience | Create a cookiecutter/template script for exercise dirs |
| Book route references break | check_book_routes.py fails | Update codepath{} in .tex files during Phase 1 migration |
| Notebook JSON validity | Notebooks fail validation | Add nbformat validation to CI |
| Type checking failures | mypy errors | Add type stubs; use type: ignore sparingly |

---

## Notes
- Chapter 12 and 13 are duplicates (both GMM). Only implement exercises for Chapter 12.
- topics/ uses chXX_topic naming; exercises use exNN_slug format (zero-padded).
- All exercises import from src/ml_from_scratch/ - no duplicate implementations.
- PINN exercises require pip install -e .[pinn] for torch dependency.

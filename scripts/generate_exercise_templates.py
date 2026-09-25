#!/usr/bin/env python
"""
Generate exercise template files for all exercises.
"""

from pathlib import Path

TOPICS_DIR = Path("topics")

# Exercise configurations with difficulty and imports needed
EXERCISES = {
    # Chapter 3: Gradient Descent
    ("ch03_gradient_descent", "ex01_unified_optimizer"): {
        "difficulty": "Easy",
        "topic": "Unified Optimizer",
        "imports": ["import numpy as np", "from ml_from_scratch.optimization import gradient_descent"],
        "description": "Implement a single gradient_descent(X, y, batch_size, ...) function that unifies all three variants.",
    },
    ("ch03_gradient_descent", "ex02_shuffling_effect"): {
        "difficulty": "Easy",
        "topic": "Effect of Shuffling",
        "imports": ["import numpy as np", "from ml_from_scratch.optimization import stochastic_gradient_descent"],
        "description": "Train SGD with and without shuffling on sorted data and compare.",
    },
    ("ch03_gradient_descent", "ex03_lr_decay"): {
        "difficulty": "Medium",
        "topic": "Learning Rate Decay",
        "imports": ["import numpy as np", "from ml_from_scratch.optimization import stochastic_gradient_descent, learning_rate_schedule"],
        "description": "Implement 1/t decay schedule and compare convergence.",
    },
    ("ch03_gradient_descent", "ex04_scaling_experiment"): {
        "difficulty": "Medium",
        "topic": "Scaling Experiment",
        "imports": ["import numpy as np", "import time", "from ml_from_scratch.optimization import batch_gradient_descent, stochastic_gradient_descent, minibatch_gradient_descent"],
        "description": "Time all three variants on datasets of sizes 100, 1K, 10K, 100K.",
    },
    ("ch03_gradient_descent", "ex05_momentum"): {
        "difficulty": "Hard",
        "topic": "Momentum",
        "imports": ["import numpy as np", "from ml_from_scratch.optimization import MomentumOptimizer"],
        "description": "Implement momentum (beta=0.9) for mini-batch GD on elongated landscape.",
    },
    
    # Chapter 4: Linear Regression
    ("ch04_linear_regression", "ex01_large_lr"): {
        "difficulty": "Easy",
        "topic": "Large Learning Rate",
        "imports": ["import numpy as np", "from ml_from_scratch import LinearRegression, make_regression"],
        "description": "Set LR=0.001 and observe divergence/oscillation.",
    },
    ("ch04_linear_regression", "ex02_more_noise"): {
        "difficulty": "Easy",
        "topic": "More Noise",
        "imports": ["import numpy as np", "from ml_from_scratch import LinearRegression, make_regression, r2_score"],
        "description": "Generate data with noise std $100K vs $30K, compare R2.",
    },
    ("ch04_linear_regression", "ex03_no_intercept"): {
        "difficulty": "Medium",
        "topic": "No Intercept",
        "imports": ["import numpy as np", "from ml_from_scratch import LinearRegression, make_regression, r2_score"],
        "description": "Train with fit_intercept=False, compare R2 to model with intercept.",
    },
    ("ch04_linear_regression", "ex04_cost_history"): {
        "difficulty": "Medium",
        "topic": "Cost History",
        "imports": ["import numpy as np", "from ml_from_scratch import LinearRegression, make_regression"],
        "description": "Modify fit() to save cost_history, plot cost vs iteration.",
    },
    ("ch04_linear_regression", "ex05_normal_eq_vs_gd"): {
        "difficulty": "Medium",
        "topic": "Normal Equation vs Gradient Descent",
        "imports": ["import numpy as np", "import time", "from ml_from_scratch import LinearRegression, make_regression"],
        "description": "Compare Normal Eq vs GD on 100/10K/100K data for accuracy and speed.",
    },
    ("ch04_linear_regression", "ex06_scaling_crossover"): {
        "difficulty": "Hard",
        "topic": "Scaling Crossover",
        "imports": ["import numpy as np", "import time", "from ml_from_scratch import LinearRegression, make_regression"],
        "description": "Log-log plot: GD vs Normal Eq crossover point for 10 features.",
    },
    
    # Chapter 5: Multiple/Polynomial Regression
    ("ch05_multiple_polynomial_regression", "ex01_third_feature"): {
        "difficulty": "Easy",
        "topic": "Third Feature",
        "imports": ["import numpy as np", "from ml_from_scratch import LinearRegression, StandardScaler"],
        "description": "Add house age feature (-$2K/year) and fit model.",
    },
    ("ch05_multiple_polynomial_regression", "ex02_sine_wave"): {
        "difficulty": "Easy",
        "topic": "Sine Wave",
        "imports": ["import numpy as np", "from ml_from_scratch import LinearRegression, PolynomialFeatures, r2_score"],
        "description": "Fit sin(x) + noise with polynomial degrees 1,3,5,7,9.",
    },
    ("ch05_multiple_polynomial_regression", "ex03_backtransform_coef"): {
        "difficulty": "Medium",
        "topic": "Back-Transform Coefficients",
        "imports": ["import numpy as np", "from ml_from_scratch import LinearRegression, StandardScaler"],
        "description": "Standardize -> fit -> back-transform coefficients to original scale.",
    },
    ("ch05_multiple_polynomial_regression", "ex04_residual_analysis"): {
        "difficulty": "Medium",
        "topic": "Residual Analysis",
        "imports": ["import numpy as np", "from ml_from_scratch import LinearRegression, PolynomialFeatures"],
        "description": "Residual plots for linear (U-shape) vs quadratic (random) fits.",
    },
    ("ch05_multiple_polynomial_regression", "ex05_feature_comparison"): {
        "difficulty": "Medium",
        "topic": "Feature Comparison on Heart Data",
        "imports": ["import numpy as np", "from ml_from_scratch import LinearRegression, StandardScaler, load_heart_disease"],
        "description": "Compare age only vs age+trestbps vs all three features on heart data.",
    },
    ("ch05_multiple_polynomial_regression", "ex06_interaction_terms"): {
        "difficulty": "Hard",
        "topic": "Interaction Terms",
        "imports": ["import numpy as np", "from ml_from_scratch import LinearRegression, r2_score"],
        "description": "Generate y = 5x1 + 3x2 + 2x1x2 + noise, compare with/without interaction.",
    },
    
    # Chapter 6: Regularized Regression
    ("ch06_regularized_regression", "ex01_ridge_alpha_sweep"): {
        "difficulty": "Easy",
        "topic": "Ridge Alpha Sweep",
        "imports": ["import numpy as np", "from ml_from_scratch import RidgeRegression, StandardScaler"],
        "description": "Ridge alpha sweep, plot coefficient paths.",
    },
    ("ch06_regularized_regression", "ex02_lasso_sparsity_vs_alpha"): {
        "difficulty": "Easy",
        "topic": "Lasso Sparsity vs Alpha",
        "imports": ["import numpy as np", "from ml_from_scratch import LassoRegression, StandardScaler"],
        "description": "Lasso alpha sweep, count non-zero coefficients.",
    },
    ("ch06_regularized_regression", "ex03_scale_sensitivity"): {
        "difficulty": "Medium",
        "topic": "Scale Sensitivity",
        "imports": ["import numpy as np", "from ml_from_scratch import RidgeRegression, StandardScaler"],
        "description": "Regularization sensitivity to feature scaling.",
    },
    ("ch06_regularized_regression", "ex04_intercept_penalty"): {
        "difficulty": "Medium",
        "topic": "Intercept Penalty Effect",
        "imports": ["import numpy as np", "from ml_from_scratch import RidgeRegression"],
        "description": "Effect of penalizing intercept in Ridge regression.",
    },
    ("ch06_regularized_regression", "ex05_closed_form_vs_gd_ridge"): {
        "difficulty": "Medium",
        "topic": "Closed-Form vs GD for Ridge",
        "imports": ["import numpy as np", "from ml_from_scratch import RidgeRegression"],
        "description": "Compare Ridge closed-form vs gradient descent.",
    },
    ("ch06_regularized_regression", "ex06_correlated_features_elastic_net"): {
        "difficulty": "Hard",
        "topic": "Correlated Features and Elastic Net",
        "imports": ["import numpy as np", "from ml_from_scratch import ElasticNetRegression, StandardScaler"],
        "description": "Elastic Net on correlated features.",
    },
    
    # Chapter 7: Logistic Regression
    ("ch07_logistic_regression", "ex01_threshold_exploration"): {
        "difficulty": "Easy",
        "topic": "Threshold Exploration",
        "imports": ["import numpy as np", "from ml_from_scratch import LogisticRegression, StandardScaler, make_classification"],
        "description": "Threshold sweep, precision/recall tradeoff.",
    },
    ("ch07_logistic_regression", "ex02_training_without_scaling"): {
        "difficulty": "Easy",
        "topic": "Training Without Scaling",
        "imports": ["import numpy as np", "from ml_from_scratch import LogisticRegression, make_classification"],
        "description": "Unscaled features -> convergence issues.",
    },
    ("ch07_logistic_regression", "ex03_two_feature_boundary"): {
        "difficulty": "Medium",
        "topic": "Two-Feature Decision Boundary",
        "imports": ["import numpy as np", "from ml_from_scratch import LogisticRegression, StandardScaler, make_classification"],
        "description": "2D decision boundary visualization.",
    },
    ("ch07_logistic_regression", "ex04_roc_curve"): {
        "difficulty": "Medium",
        "topic": "ROC Curve",
        "imports": ["import numpy as np", "from ml_from_scratch import LogisticRegression, StandardScaler, make_classification, roc_auc_score"],
        "description": "ROC curve, AUC computation.",
    },
    ("ch07_logistic_regression", "ex05_l2_regularization"): {
        "difficulty": "Hard",
        "topic": "L2 Regularization",
        "imports": ["import numpy as np", "from ml_from_scratch import LogisticRegression, StandardScaler"],
        "description": "L2-regularized logistic regression with different C values.",
    },
    ("ch07_logistic_regression", "ex06_nonlinear_boundary"): {
        "difficulty": "Hard",
        "topic": "Nonlinear Decision Boundary",
        "imports": ["import numpy as np", "from ml_from_scratch import LogisticRegression, PolynomialFeatures, StandardScaler"],
        "description": "Polynomial features for nonlinear boundary.",
    },
    ("ch07_logistic_regression", "ex07_multinomial_extension"): {
        "difficulty": "Hard",
        "topic": "Multinomial Extension",
        "imports": ["import numpy as np", "from ml_from_scratch import LogisticRegression, StandardScaler, make_classification"],
        "description": "Multi-class (OvR or softmax) extension.",
    },
    
    # Chapter 8: Naive Bayes
    ("ch08_naive_bayes", "ex01_zero_variance"): {
        "difficulty": "Easy",
        "topic": "Zero Variance",
        "imports": ["import numpy as np", "from ml_from_scratch import GaussianNB"],
        "description": "Handle zero variance features in GaussianNB.",
    },
    ("ch08_naive_bayes", "ex02_manual_prediction"): {
        "difficulty": "Easy",
        "topic": "Manual Prediction",
        "imports": ["import numpy as np", "from ml_from_scratch import GaussianNB"],
        "description": "Manual predict_proba calculation for verification.",
    },
    ("ch08_naive_bayes", "ex03_2d_visualization"): {
        "difficulty": "Medium",
        "topic": "2D Visualization",
        "imports": ["import numpy as np", "from ml_from_scratch import GaussianNB, StandardScaler, make_classification"],
        "description": "2D decision boundary visualization for GaussianNB.",
    },
    ("ch08_naive_bayes", "ex04_iris_multiclass_confusion"): {
        "difficulty": "Medium",
        "topic": "Iris Multiclass Confusion Matrix",
        "imports": ["import numpy as np", "from ml_from_scratch import GaussianNB, confusion_matrix"],
        "description": "Iris dataset with GaussianNB, confusion matrix.",
    },
    ("ch08_naive_bayes", "ex05_logspace_predict_proba"): {
        "difficulty": "Hard",
        "topic": "Log-Space predict_proba",
        "imports": ["import numpy as np", "from ml_from_scratch import GaussianNB"],
        "description": "Log-space predict_proba for numerical stability.",
    },
    ("ch08_naive_bayes", "ex06_correlated_features"): {
        "difficulty": "Hard",
        "topic": "Correlated Features",
        "imports": ["import numpy as np", "from ml_from_scratch import GaussianNB, StandardScaler"],
        "description": "Correlated features impact on Naive Bayes assumptions.",
    },
    ("ch08_naive_bayes", "ex07_multinomial_nb_text"): {
        "difficulty": "Hard",
        "topic": "Multinomial NB Text",
        "imports": ["import numpy as np", "from ml_from_scratch import MultinomialNB"],
        "description": "MultinomialNB on text classification (bag of words).",
    },
    
    # Chapter 9: SVM
    ("ch09_svm", "ex01_hard_vs_soft_margin"): {
        "difficulty": "Easy",
        "topic": "Hard vs Soft Margin",
        "imports": ["import numpy as np", "from ml_from_scratch import KernelSVM, StandardScaler"],
        "description": "C parameter: hard vs soft margin visualization.",
    },
    ("ch09_svm", "ex02_gamma_sweep"): {
        "difficulty": "Easy",
        "topic": "Gamma Sweep",
        "imports": ["import numpy as np", "from ml_from_scratch import KernelSVM, StandardScaler"],
        "description": "RBF gamma sweep, decision boundary changes.",
    },
    ("ch09_svm", "ex03_outlier_robustness"): {
        "difficulty": "Medium",
        "topic": "Outlier Robustness",
        "imports": ["import numpy as np", "from ml_from_scratch import KernelSVM, LogisticRegression, StandardScaler"],
        "description": "Outlier impact on SVM vs logistic regression.",
    },
    ("ch09_svm", "ex04_margin_width_vs_c"): {
        "difficulty": "Medium",
        "topic": "Margin Width vs C",
        "imports": ["import numpy as np", "from ml_from_scratch import LinearSVM, StandardScaler"],
        "description": "Margin width vs C visualization.",
    },
    ("ch09_svm", "ex05_polynomial_kernel"): {
        "difficulty": "Hard",
        "topic": "Polynomial Kernel Comparison",
        "imports": ["import numpy as np", "from ml_from_scratch import KernelSVM, StandardScaler"],
        "description": "Polynomial kernel vs RBF comparison.",
    },
    ("ch09_svm", "ex06_heart_disease_confusion"): {
        "difficulty": "Hard",
        "topic": "Heart Disease Confusion Matrix",
        "imports": ["import numpy as np", "from ml_from_scratch import KernelSVM, StandardScaler, load_heart_disease, confusion_matrix"],
        "description": "Heart disease dataset with SVM, confusion matrix.",
    },
    ("ch09_svm", "ex07_multi_class_svm"): {
        "difficulty": "Hard",
        "topic": "Multi-class SVM",
        "imports": ["import numpy as np", "from ml_from_scratch import KernelSVM, StandardScaler, make_classification"],
        "description": "OvO/OvR multi-class SVM implementation.",
    },
    
    # Chapter 10: Decision Trees
    ("ch10_decision_trees", "ex01_entropy_vs_gini"): {
        "difficulty": "Easy",
        "topic": "Entropy vs Gini",
        "imports": ["import numpy as np", "from ml_from_scratch import DecisionTreeClassifier, make_classification, accuracy_score"],
        "description": "Compare entropy vs Gini splits on same data.",
    },
    ("ch10_decision_trees", "ex02_depth_vs_overfitting"): {
        "difficulty": "Easy",
        "topic": "Depth vs Overfitting",
        "imports": ["import numpy as np", "from ml_from_scratch import DecisionTreeClassifier, make_classification, accuracy_score"],
        "description": "Max depth vs train/test accuracy curve.",
    },
    ("ch10_decision_trees", "ex03_manual_tree_trace"): {
        "difficulty": "Medium",
        "topic": "Manual Tree Trace",
        "imports": ["import numpy as np", "from ml_from_scratch import DecisionTreeClassifier"],
        "description": "Manually trace tree building on small dataset.",
    },
    ("ch10_decision_trees", "ex04_feature_importance"): {
        "difficulty": "Medium",
        "topic": "Feature Importance",
        "imports": ["import numpy as np", "from ml_from_scratch import DecisionTreeClassifier, make_classification"],
        "description": "Compute and visualize feature importance.",
    },
    ("ch10_decision_trees", "ex05_pruning"): {
        "difficulty": "Hard",
        "topic": "Pruning",
        "imports": ["import numpy as np", "from ml_from_scratch import DecisionTreeClassifier, make_classification"],
        "description": "Cost-complexity pruning implementation.",
    },
    ("ch10_decision_trees", "ex06_nonlinear_boundary"): {
        "difficulty": "Hard",
        "topic": "Nonlinear Boundary",
        "imports": ["import numpy as np", "from ml_from_scratch import DecisionTreeClassifier, make_classification"],
        "description": "Tree approximation of nonlinear boundary.",
    },
    
    # Chapter 11: K-Means
    ("ch11_kmeans_clustering", "ex01_kmeanspp_vs_random"): {
        "difficulty": "Easy",
        "topic": "K-Means++ vs Random",
        "imports": ["import numpy as np", "from ml_from_scratch import KMeans, make_blobs"],
        "description": "K-Means++ vs random initialization comparison.",
    },
    ("ch11_kmeans_clustering", "ex02_elbow_method_real_data"): {
        "difficulty": "Easy",
        "topic": "Elbow Method on Real Data",
        "imports": ["import numpy as np", "from ml_from_scratch import KMeans, StandardScaler, load_heart_disease"],
        "description": "Elbow method on heart disease dataset.",
    },
    ("ch11_kmeans_clustering", "ex03_wrong_k_visualization"): {
        "difficulty": "Medium",
        "topic": "Wrong K Visualization",
        "imports": ["import numpy as np", "from ml_from_scratch import KMeans, make_blobs"],
        "description": "Visualize wrong K (too few/too many clusters).",
    },
    ("ch11_kmeans_clustering", "ex04_silhouette_score"): {
        "difficulty": "Medium",
        "topic": "Silhouette Score",
        "imports": ["import numpy as np", "from ml_from_scratch import KMeans, make_blobs"],
        "description": "Silhouette score for K selection.",
    },
    ("ch11_kmeans_clustering", "ex05_non_spherical_failure"): {
        "difficulty": "Hard",
        "topic": "Non-Spherical Failure",
        "imports": ["import numpy as np", "from ml_from_scratch import KMeans, GaussianMixture, make_blobs"],
        "description": "K-means failure on non-spherical clusters vs GMM.",
    },
    ("ch11_kmeans_clustering", "ex06_outlier_detection"): {
        "difficulty": "Hard",
        "topic": "Outlier Detection",
        "imports": ["import numpy as np", "from ml_from_scratch import KMeans, make_blobs"],
        "description": "K-means for outlier detection (distance to centroid).",
    },
    
    # Chapter 12: Gaussian Mixture
    ("ch12_gaussian_mixture", "ex01_bic_vs_aic"): {
        "difficulty": "Easy",
        "topic": "BIC vs AIC",
        "imports": ["import numpy as np", "from ml_from_scratch import GaussianMixture, make_blobs"],
        "description": "BIC vs AIC for model selection on GMM.",
    },
    ("ch12_gaussian_mixture", "ex02_responsibility_vector"): {
        "difficulty": "Easy",
        "topic": "Responsibility Vector",
        "imports": ["import numpy as np", "from ml_from_scratch import GaussianMixture"],
        "description": "Manual responsibility computation for single point.",
    },
    ("ch12_gaussian_mixture", "ex03_diagonal_covariance"): {
        "difficulty": "Medium",
        "topic": "Diagonal Covariance",
        "imports": ["import numpy as np", "from ml_from_scratch import GaussianMixture, make_blobs"],
        "description": "Diagonal vs full covariance comparison.",
    },
    ("ch12_gaussian_mixture", "ex04_regularization_sensitivity"): {
        "difficulty": "Medium",
        "topic": "Regularization Sensitivity",
        "imports": ["import numpy as np", "from ml_from_scratch import GaussianMixture"],
        "description": "Covariance regularization sensitivity (reg_covar parameter).",
    },
    ("ch12_gaussian_mixture", "ex05_iris_comparison"): {
        "difficulty": "Hard",
        "topic": "Iris Comparison",
        "imports": ["import numpy as np", "from ml_from_scratch import GaussianMixture, KMeans, StandardScaler"],
        "description": "GMM vs K-means on Iris dataset.",
    },
    ("ch12_gaussian_mixture", "ex06_kmeans_initialization"): {
        "difficulty": "Hard",
        "topic": "K-Means Initialization",
        "imports": ["import numpy as np", "from ml_from_scratch import GaussianMixture, KMeans"],
        "description": "K-means init for GMM vs random init.",
    },
    
    # Chapter 14: Neural Networks
    ("ch14_neural_networks", "ex01_architecture_sweep"): {
        "difficulty": "Easy",
        "topic": "Architecture Sweep",
        "imports": ["import numpy as np", "from ml_from_scratch import MLPClassifier, make_classification, accuracy_score"],
        "description": "Hidden layer size sweep (e.g., [10], [20], [50], [100], [20,10]).",
    },
    ("ch14_neural_networks", "ex02_tanh_activation"): {
        "difficulty": "Easy",
        "topic": "Tanh Activation",
        "imports": ["import numpy as np", "from ml_from_scratch import MLPClassifier, make_classification, accuracy_score"],
        "description": "Tanh vs ReLU activation comparison.",
    },
    ("ch14_neural_networks", "ex03_l2_regularization"): {
        "difficulty": "Medium",
        "topic": "L2 Regularization",
        "imports": ["import numpy as np", "from ml_from_scratch import MLPClassifier, make_classification, accuracy_score"],
        "description": "L2 regularization (alpha parameter) in MLP.",
    },
    ("ch14_neural_networks", "ex04_overfitting_point"): {
        "difficulty": "Medium",
        "topic": "Overfitting Point",
        "imports": ["import numpy as np", "from ml_from_scratch import MLPClassifier, make_classification, accuracy_score"],
        "description": "Detect overfitting with validation curve (train vs val accuracy).",
    },
    ("ch14_neural_networks", "ex05_regression_output"): {
        "difficulty": "Hard",
        "topic": "Regression Output",
        "imports": ["import numpy as np", "from ml_from_scratch import MLPRegressor, make_regression, r2_score"],
        "description": "MLP for regression (linear output layer).",
    },
    ("ch14_neural_networks", "ex06_minibatch_training"): {
        "difficulty": "Hard",
        "topic": "Mini-batch Training",
        "imports": ["import numpy as np", "from ml_from_scratch import MLPClassifier, make_classification, accuracy_score"],
        "description": "Mini-batch SGD training loop with different batch sizes.",
    },
    
    # Chapter 15: PINN
    ("ch15_pinn", "ex01_different_forcing"): {
        "difficulty": "Easy",
        "topic": "Different Forcing Function",
        "imports": ["import numpy as np", "from ml_from_scratch import PINNBase, create_pinn_data"],
        "description": "PINN with different forcing functions for heat equation.",
    },
    ("ch15_pinn", "ex02_collocation_points"): {
        "difficulty": "Easy",
        "topic": "Collocation Points",
        "imports": ["import numpy as np", "from ml_from_scratch import PINNBase, create_pinn_data"],
        "description": "Number of collocation points effect on accuracy.",
    },
    ("ch15_pinn", "ex03_first_derivative"): {
        "difficulty": "Medium",
        "topic": "First Derivative Extension",
        "imports": ["import numpy as np", "from ml_from_scratch import PINNBase"],
        "description": "First derivative constraint for PINN (e.g., boundary conditions).",
    },
    ("ch15_pinn", "ex04_depth_width_sweep"): {
        "difficulty": "Medium",
        "topic": "Depth and Width Sweep",
        "imports": ["import numpy as np", "from ml_from_scratch import PINNBase"],
        "description": "Network depth/width sweep for PINN accuracy.",
    },
    ("ch15_pinn", "ex05_heat_equation"): {
        "difficulty": "Hard",
        "topic": "Heat Equation",
        "imports": ["import numpy as np", "from ml_from_scratch import PINNBase, HeatEquation1D"],
        "description": "1D heat equation PINN with analytical solution comparison.",
    },
    ("ch15_pinn", "ex06_autodiff"): {
        "difficulty": "Hard",
        "topic": "Automatic Differentiation",
        "imports": ["import numpy as np", "from ml_from_scratch import PINNBase"],
        "description": "Custom autodiff vs PyTorch for PDE residuals.",
    },
    ("ch15_pinn", "ex07_inverse_problem"): {
        "difficulty": "Hard",
        "topic": "Inverse Problem",
        "imports": ["import numpy as np", "from ml_from_scratch import PINNBase"],
        "description": "Parameter estimation (inverse problem) with PINN.",
    },
    
    # Chapter 16: Pipelines & Production
    ("ch16_pipelines_production", "ex01_mode_imputation"): {
        "difficulty": "Easy",
        "topic": "Mode Imputation",
        "imports": ["import numpy as np", "from ml_from_scratch import Pipeline, StandardScaler"],
        "description": "Mode imputation in pipeline for categorical features.",
    },
    ("ch16_pipelines_production", "ex02_demonstrating_leakage"): {
        "difficulty": "Easy",
        "topic": "Demonstrating Leakage",
        "imports": ["import numpy as np", "from ml_from_scratch import Pipeline, StandardScaler, cross_val_score, LogisticRegression"],
        "description": "Data leakage demonstration with/without pipeline.",
    },
    ("ch16_pipelines_production", "ex03_regression_metrics"): {
        "difficulty": "Medium",
        "topic": "Regression Metrics",
        "imports": ["import numpy as np", "from ml_from_scratch import Pipeline, StandardScaler, cross_val_score, LinearRegression, make_regression"],
        "description": "Regression metrics (MSE, R2) in cross-validation.",
    },
    ("ch16_pipelines_production", "ex04_pipeline_aware_cv"): {
        "difficulty": "Medium",
        "topic": "Pipeline-Aware Cross-Validation",
        "imports": ["import numpy as np", "from ml_from_scratch import Pipeline, StandardScaler, cross_val_score, LogisticRegression"],
        "description": "Pipeline-aware CV vs manual scaling CV.",
    },
    ("ch16_pipelines_production", "ex05_stratified_kfold"): {
        "difficulty": "Hard",
        "topic": "Stratified K-Fold",
        "imports": ["import numpy as np", "from ml_from_scratch import StratifiedKFold, cross_val_score, LogisticRegression, make_classification"],
        "description": "Stratified K-fold implementation and verification.",
    },
    ("ch16_pipelines_production", "ex06_hyperparameter_search"): {
        "difficulty": "Hard",
        "topic": "Hyperparameter Search",
        "imports": ["import numpy as np", "from ml_from_scratch import Pipeline, StandardScaler, LogisticRegression, cross_val_score"],
        "description": "Grid/random search with pipeline for hyperparameter tuning.",
    },
    ("ch16_pipelines_production", "ex07_production_pipeline"): {
        "difficulty": "Hard",
        "topic": "Production Pipeline",
        "imports": ["import numpy as np", "from ml_from_scratch import Pipeline, StandardScaler, LogisticRegression, ModelSerializer"],
        "description": "End-to-end production pipeline with serialization.",
    },
    
    # Chapter 17: Production/Software Engineering
    ("ch17_production_software", "ex01_interchangeable_interfaces"): {
        "difficulty": "Easy",
        "topic": "Interchangeable Interfaces",
        "imports": ["import numpy as np", "from ml_from_scratch import LinearRegression, LogisticRegression, DecisionTreeClassifier"],
        "description": "Strategy pattern for interchangeable models (common interface).",
    },
    ("ch17_production_software", "ex02_comparison_test"): {
        "difficulty": "Easy",
        "topic": "Comparison Test",
        "imports": ["import numpy as np", "from ml_from_scratch import LogisticRegression, DecisionTreeClassifier, LinearSVM, StandardScaler, make_classification"],
        "description": "A/B test framework for model comparison.",
    },
    ("ch17_production_software", "ex03_monitoring_design"): {
        "difficulty": "Medium",
        "topic": "Monitoring Design",
        "imports": ["import numpy as np", "from ml_from_scratch import ModelMonitor, LogisticRegression"],
        "description": "Model monitoring/alerting design with drift detection.",
    },
    ("ch17_production_software", "ex04_project_reorganization"): {
        "difficulty": "Medium",
        "topic": "Project Reorganization",
        "imports": ["numpy as np"],
        "description": "Refactor a project for production structure (modular, config-driven).",
    },
    ("ch17_production_software", "ex05_property_based_test"): {
        "difficulty": "Hard",
        "topic": "Property-Based Test",
        "imports": ["import numpy as np", "from ml_from_scratch import LinearRegression, make_regression"],
        "description": "Hypothesis property-based tests for ML invariants.",
    },
    ("ch17_production_software", "ex06_reflection"): {
        "difficulty": "Reflection",
        "topic": "What Mattered Most",
        "imports": [],
        "description": "Reflection exercise on the learning journey.",
    },
    ("ch17_production_software", "ex07_ci_cd_integration"): {
        "difficulty": "Hard",
        "topic": "CI/CD Integration",
        "imports": ["import numpy as np", "from ml_from_scratch import Pipeline, StandardScaler, LogisticRegression, ModelSerializer"],
        "description": "CI/CD pipeline for ML (test, train, serialize, deploy).",
    },
}


def generate_starter(topic_dir, ex_dir, config):
    """Generate starter.py with TODOs."""
    imports = "\n".join(config.get("imports", []))
    topic_name = config.get("topic", "")
    slug = ex_dir.name
    ex_num = int(slug[2:4])
    ch_num = int(topic_dir.name[2:4])
    
    lines = [
        '"""',
        f'Exercise {ex_num}: {topic_name}',
        f'Chapter {ch_num} - {topic_dir.name[5:].replace("_", " ").title()}',
        f'Difficulty: {config.get("difficulty", "Medium")}',
        '',
        config.get('description', ''),
        '"""',
        '',
        '# TODO: Import required modules',
        imports,
        '',
        '# TODO: Set random seed for reproducibility',
        'np.random.seed(42)',
        '',
        '# TODO: Generate or load data',
        '# X, y = ...',
        '',
        '# TODO: Implement the exercise',
        '# ...',
        '',
        '# TODO: Verify your implementation',
        '# ...',
        '',
        'if __name__ == "__main__":',
        f'    print("Exercise {ex_num} starter - implement the TODOs above")',
    ]
    content = "\n".join(lines)
    (ex_dir / "starter.py").write_text(content)


def generate_solution(topic_dir, ex_dir, config):
    """Generate solution.py with reference implementation."""
    topic_name = config.get("topic", "")
    slug = ex_dir.name
    ex_num = int(slug[2:4])
    ch_num = int(topic_dir.name[2:4])
    
    imports = "\n".join(config.get("imports", []))
    
    lines = [
        '"""',
        f'Solution for Exercise {ex_num}: {topic_name}',
        f'Chapter {ch_num} - {topic_dir.name[5:].replace("_", " ").title()}',
        f'Difficulty: {config.get("difficulty", "Medium")}',
        '',
        config.get('description', ''),
        '"""',
        '',
        '# Imports',
        imports,
        '',
        'def main():',
        '    np.random.seed(42)',
        '',
        '    # Solution implementation',
        f'    print(f"Solution for Exercise {ex_num}: {topic_name}")',
        '    print("=" * 60)',
        '',
        '    # TODO: Implement solution here',
        '    pass',
        '',
        'if __name__ == "__main__":',
        '    main()',
    ]
    content = "\n".join(lines)
    (ex_dir / "solution.py").write_text(content)


def generate_test(topic_dir, ex_dir, config):
    """Generate test_solution.py with basic tests."""
    topic_name = config.get("topic", "")
    slug = ex_dir.name
    ex_num = int(slug[2:4])
    ch_num = int(topic_dir.name[2:4])
    topic_title = topic_dir.name[5:].replace("_", " ").title()
    
    lines = [
        '"""',
        f'Tests for Exercise {ex_num}: {topic_name}',
        f'Chapter {ch_num} - {topic_title}',
        '"""',
        '',
        'import numpy as np',
        'import pytest',
        'import sys',
        'from pathlib import Path',
        'sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))',
        '',
        '# Import the solution',
        'from solution import main',
        '',
        '',
        'def test_solution_runs():',
        '    """Test that solution runs without errors."""',
        '    # This should not raise any exceptions',
        '    main()',
        '',
        '',
        'def test_solution_output():',
        '    """Test solution produces expected output."""',
        '    # TODO: Add specific assertions based on exercise requirements',
        '    pass',
        '',
        '',
        'if __name__ == "__main__":',
        '    test_solution_runs()',
        '    test_solution_output()',
        '    print("All tests passed!")',
    ]
    content = "\n".join(lines)
    test_filename = f"test_ch{ch_num:02d}_{slug}.py"
    (ex_dir / test_filename).write_text(content)


def generate_readme(topic_dir, ex_dir, config):
    """Generate README.md with exercise description."""
    topic_name = config.get("topic", "")
    slug = ex_dir.name
    ex_num = int(slug[2:4])
    ch_num = int(topic_dir.name[2:4])
    difficulty = config.get("difficulty", "Medium")
    
    rel_path = ex_dir.relative_to(TOPICS_DIR)
    topic_title = topic_dir.name[5:].replace("_", " ").title()
    
    # Build markdown content using list
    lines = [
        f'# Exercise {ex_num}: {topic_name}',
        '',
        f'**Chapter:** {ch_num} - {topic_title}  ',
        f'**Difficulty:** {difficulty}  ',
        f'**Path:** `{rel_path}`',
        '',
        '---',
        '',
        '## Task Description',
        '',
        config.get('description', 'No description available.'),
        '',
        '---',
        '',
        '## Learning Objectives',
        '',
        '- Understand the core concept',
        '- Implement the algorithm correctly',
        '- Verify the implementation works',
        '',
        '---',
        '',
        '## Starter Guide',
        '',
        'The `starter.py` file contains TODOs that guide you through the implementation:',
        '',
        '1. Import required modules from `ml_from_scratch`',
        '2. Set random seed for reproducibility',
        '3. Generate or load the required data',
        '4. Implement the algorithm',
        '5. Verify your implementation',
        '',
        '### Key Hints',
        '',
        '- Use `np.random.seed(42)` for reproducible results',
        '- Import from `ml_from_scratch` package (e.g., `from ml_from_scratch import LinearRegression`)',
        '- Check the book chapter for mathematical details',
        '- Run the starter to see what fails: `python scripts/run_exercise.py --chapter {ch_num} --exercise {ex_num} --variant starter`',
        '',
        '---',
        '',
        '## Acceptance Criteria',
        '',
        '- [ ] `starter.py` runs without errors after implementation',
        '- [ ] `solution.py` produces correct output',
        f'- [ ] `test_{slug}.py` passes all tests',
        '- [ ] Numerical results match expected tolerances (if applicable)',
        '',
        '### Run Commands',
        '',
        '```bash',
        '# Run starter',
        f'python scripts/run_exercise.py --chapter {ch_num} --exercise {ex_num} --variant starter',
        '',
        '# Run solution',
        f'python scripts/run_exercise.py --chapter {ch_num} --exercise {ex_num} --variant solution',
        '',
        '# Run tests',
        f'python -m pytest topics/{topic_dir.name}/exercises/{slug}/test_ch{ch_num:02d}_{slug}.py -v',
        '```',
        '',
        '---',
        '',
        '## Common Pitfalls',
        '',
        '- Forgetting to set random seed -> non-reproducible results',
        '- Not scaling features before gradient descent -> slow/no convergence',
        '- Using wrong parameter names (check `ml_from_scratch` API)',
        '- Not handling edge cases (empty arrays, single samples)',
        '',
        '---',
        '',
        '## Further Exploration',
        '',
        '- Try different hyperparameters and observe effects',
        '- Compare with scikit-learn implementation',
        '- Extend to handle additional cases',
        '',
        '---',
        '',
        '## Reference',
        '',
        f'See Chapter {ch_num} of the book for mathematical background and detailed explanation.',
    ]
    content = "\n".join(lines)
    (ex_dir / "README.md").write_text(content)


def main():
    """Generate all exercise templates."""
    for (topic_name, ex_name), config in EXERCISES.items():
        topic_dir = TOPICS_DIR / topic_name
        ex_dir = topic_dir / "exercises" / ex_name
        
        if not ex_dir.exists():
            print(f"Warning: {ex_dir} does not exist, skipping")
            continue
        
        generate_starter(topic_dir, ex_dir, config)
        generate_solution(topic_dir, ex_dir, config)
        generate_test(topic_dir, ex_dir, config)
        generate_readme(topic_dir, ex_dir, config)
        
        print(f"Generated templates for {topic_name}/{ex_name}")


if __name__ == "__main__":
    main()
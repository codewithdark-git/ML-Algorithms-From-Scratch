"""ML From Scratch - Building Machine Learning Algorithms from First Principles."""

__version__ = "0.2.0"

# Core optimization (Chapter 3)
from ml_from_scratch.ch03_gradient_descent.optimization import (
    batch_gradient_descent,
    stochastic_gradient_descent,
    minibatch_gradient_descent,
    gradient_descent,
    learning_rate_schedule,
    MomentumOptimizer,
)

# Linear models (Chapters 4, 5, 6)
from ml_from_scratch.ch04_linear_regression.linear_regression import LinearRegression
from ml_from_scratch.ch05_multiple_polynomial_regression.preprocessing import PolynomialFeatures
from ml_from_scratch.ch06_regularized_regression.regularized_regression import (
    RidgeRegression,
    LassoRegression,
    ElasticNetRegression,
)

# Classification (Chapters 7, 8, 9, 10)
from ml_from_scratch.ch07_logistic_regression.logistic_regression import LogisticRegression
from ml_from_scratch.ch08_naive_bayes.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from ml_from_scratch.ch09_svm.svm import LinearSVM, KernelSVM
from ml_from_scratch.ch10_decision_trees.trees import DecisionTreeClassifier, DecisionTreeRegressor

# Clustering (Chapters 11, 12)
from ml_from_scratch.ch11_kmeans_clustering.clustering import KMeans
from ml_from_scratch.ch12_gaussian_mixture.clustering import GaussianMixture

# Neural Networks (Chapter 14)
from ml_from_scratch.ch13_neural_networks.neural_networks import MLPClassifier, MLPRegressor

# PINN (Chapter 15)
from ml_from_scratch.ch14_pinn.pinn import PINNBase, HeatEquation1D, WaveEquation1D, BurgersEquation1D

# Pipeline & Production (Chapters 16, 17)
from ml_from_scratch.ch15_pipelines_production.pipeline import Pipeline, cross_val_score, KFold, StratifiedKFold
from ml_from_scratch.ch16_production_software.production import ModelMonitor, ModelSerializer, ModelServer, ExperimentTracker

# Utilities
from ml_from_scratch.ch03_gradient_descent.validation import validate_data, check_array
from ml_from_scratch.ch03_gradient_descent.random import set_seed, get_rng
from ml_from_scratch.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    silhouette_score,
    silhouette_samples,
    adjusted_rand_score,
)
from ml_from_scratch.ch05_multiple_polynomial_regression.preprocessing import StandardScaler, train_test_split, SimpleImputer
from ml_from_scratch.datasets import (
    load_heart_disease,
    make_regression,
    make_classification,
    make_blobs,
)
from ml_from_scratch.utils import (
    plot_learning_curve,
    plot_decision_boundary,
    plot_regression_line,
    plot_cost_comparison,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_silhouette_analysis,
    plot_elbow_curve,
)

__all__ = [
    # Optimization
    "batch_gradient_descent",
    "stochastic_gradient_descent",
    "minibatch_gradient_descent",
    "gradient_descent",
    "learning_rate_schedule",
    "MomentumOptimizer",
    # Linear models
    "LinearRegression",
    "PolynomialFeatures",
    "RidgeRegression",
    "LassoRegression",
    "ElasticNetRegression",
    # Classification
    "LogisticRegression",
    "GaussianNB",
    "MultinomialNB",
    "BernoulliNB",
    "LinearSVM",
    "KernelSVM",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "KMeans",
    "GaussianMixture",
    "MLPClassifier",
    "MLPRegressor",
    "PINNBase",
    "HeatEquation1D",
    "WaveEquation1D",
    "BurgersEquation1D",
    # Pipeline & Production
    "Pipeline",
    "cross_val_score",
    "KFold",
    "StratifiedKFold",
    "ModelMonitor",
    "ModelSerializer",
    "ModelServer",
    "ExperimentTracker",
    # Utilities
    "validate_data",
    "check_array",
    "set_seed",
    "get_rng",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "roc_auc_score",
    "confusion_matrix",
    "silhouette_score",
    "silhouette_samples",
    "adjusted_rand_score",
    "StandardScaler",
    "SimpleImputer",
    "PolynomialFeatures",
    "train_test_split",
    # Datasets
    "load_heart_disease",
    "make_regression",
    "make_classification",
    "make_blobs",
    # Plotting utilities
    "plot_learning_curve",
    "plot_decision_boundary",
    "plot_regression_line",
    "plot_cost_comparison",
    "plot_feature_importance",
    "plot_confusion_matrix",
    "plot_silhouette_analysis",
    "plot_elbow_curve",
]
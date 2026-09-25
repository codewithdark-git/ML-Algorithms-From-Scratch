#!/usr/bin/env python
"""
Algorithm comparison script for intermediate learners.

Usage:
    python scripts/algorithm_comparison.py --dataset breast_cancer
    python scripts/algorithm_comparison.py --dataset breast_cancer
    python scripts/algorithm_comparison.py --dataset synthetic --n-samples 1000 --n-features 20
"""

import argparse
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, "src")

from ml_from_scratch import (
    LinearRegression, RidgeRegression, LassoRegression, ElasticNetRegression,
    LogisticRegression, GaussianNB, MultinomialNB, BernoulliNB,
    LinearSVM, KernelSVM, DecisionTreeClassifier, DecisionTreeRegressor,
    KMeans, GaussianMixture, MLPClassifier, MLPRegressor,
    StandardScaler, train_test_split, cross_val_score,
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score, adjusted_rand_score,
    make_classification, make_regression, make_blobs,
)


# Dataset loaders
def load_iris() -> Tuple[np.ndarray, np.ndarray]:
    """Load iris dataset (simplified - using synthetic for demo)."""
    from ml_from_scratch.datasets import make_classification
    X, y = make_classification(n_samples=150, n_features=4, n_classes=3, n_informative=3, random_state=42)
    return X, y


def load_breast_cancer() -> Tuple[np.ndarray, np.ndarray]:
    """Load breast cancer dataset (simplified)."""
    from ml_from_scratch.datasets import make_classification
    X, y = make_classification(n_samples=569, n_features=30, n_classes=2, n_informative=15, random_state=42)
    return X, y


def load_synthetic(n_samples: int, n_features: int, task: str = "classification") -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic dataset."""
    if task == "classification":
        from ml_from_scratch.datasets import make_classification
        return make_classification(n_samples=n_samples, n_features=n_features, n_classes=2, random_state=42)
    else:
        from ml_from_scratch.datasets import make_regression
        return make_regression(n_samples=n_samples, n_features=n_features, noise=0.1, random_state=42)


# Model configurations for comparison
CLASSIFICATION_MODELS = {
    "LogisticRegression": LogisticRegression(n_iters=1000, learning_rate=0.1),
    "GaussianNB": GaussianNB(),
    "LinearSVM": LinearSVM(C=1.0, max_iter=1000),
    "KernelSVM (RBF)": KernelSVM(kernel="rbf", C=1.0, gamma=0.1, max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(max_depth=5),
    "MLPClassifier": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, learning_rate_init=0.01),
}

REGRESSION_MODELS = {
    "LinearRegression": LinearRegression(),
    "RidgeRegression": RidgeRegression(alpha=1.0),
    "LassoRegression": LassoRegression(alpha=0.1),
    "ElasticNet": ElasticNetRegression(alpha=0.1, l1_ratio=0.5),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5),
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, learning_rate_init=0.01),
}

CLUSTERING_MODELS = {
    "KMeans": KMeans(n_clusters=3, random_state=42),
    "GaussianMixture": GaussianMixture(n_components=3, random_state=42),
}


def evaluate_classification(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
    """Evaluate classification model with cross-validation."""
    # Simple hold-out for speed (cross_val_score would be better but slower)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale for models that need it
    if hasattr(model, 'fit') and 'SVM' in type(model).__name__ or 'MLP' in type(model).__name__:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = model.predict(X_test)
    
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
        "f1": f1_score(y_test, y_pred, average='macro', zero_division=0),
        "train_time": train_time,
    }


def evaluate_regression(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate regression model."""
    from ml_from_scratch.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale for neural networks
    if 'MLP' in type(model).__name__:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
    else:
        y_test_scaled = y_test
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = model.predict(X_test)
    
    # Inverse transform if needed
    if 'MLP' in type(model).__name__:
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_test_scaled = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    
    return {
        "mse": mean_squared_error(y_test_scaled, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test_scaled, y_pred)),
        "mae": mean_absolute_error(y_test_scaled, y_pred),
        "r2": r2_score(y_test_scaled, y_pred),
        "train_time": train_time,
    }


def evaluate_clustering(model, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Evaluate clustering model."""
    start = time.time()
    model.fit(X)
    train_time = time.time() - start
    
    labels = model.predict(X) if hasattr(model, 'predict') else model.labels_
    
    return {
        "silhouette": silhouette_score(X, labels),
        "adjusted_rand": adjusted_rand_score(y_true, labels),
        "train_time": train_time,
    }


def print_results_table(results: List[Dict], task: str):
    """Print formatted results table."""
    if task == "classification":
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "Time (s)"]
        rows = []
        for r in results:
            rows.append([
                r["name"],
                f"{r['accuracy']:.4f}",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['f1']:.4f}",
                f"{r['train_time']:.3f}",
            ])
    elif task == "regression":
        headers = ["Model", "MSE", "RMSE", "MAE", "R²", "Time (s)"]
        rows = []
        for r in results:
            rows.append([
                r["name"],
                f"{r['mse']:.4f}",
                f"{r['rmse']:.4f}",
                f"{r['mae']:.4f}",
                f"{r['r2']:.4f}",
                f"{r['train_time']:.3f}",
            ])
    else:  # clustering
        headers = ["Model", "Silhouette", "Adj. Rand", "Time (s)"]
        rows = []
        for r in results:
            rows.append([
                r["name"],
                f"{r['silhouette']:.4f}",
                f"{r['adjusted_rand']:.4f}",
                f"{r['train_time']:.3f}",
            ])
    
    # Print table
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in col_widths)
    
    print(f"\n{header_line}")
    print(separator)
    for row in rows:
        print(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(row))))
    
    # Find best
    if task == "classification":
        best = max(results, key=lambda x: x["f1"])
        print(f"\nBest F1: {best['name']} ({best['f1']:.4f})")
    elif task == "regression":
        best = max(results, key=lambda x: x["r2"])
        print(f"\nBest R²: {best['name']} ({best['r2']:.4f})")
    else:
        best = max(results, key=lambda x: x["silhouette"])
        print(f"\nBest Silhouette: {best['name']} ({best['silhouette']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Compare ML algorithms")
    parser.add_argument("--dataset", choices=["iris", "breast_cancer", "synthetic"], default="iris")
    parser.add_argument("--task", choices=["classification", "regression", "clustering"], default="classification")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds (not used in quick mode)")
    parser.add_argument("--quick", action="store_true", help="Single train/test split (faster)")
    args = parser.parse_args()
    
    print(f"ALGORITHM COMPARISON")
    print(f"=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Mode: {'Quick (hold-out)' if args.quick else f'CV={args.cv}'}")
    
    # Load data
    if args.dataset == "iris":
        X, y = load_iris()
    elif args.dataset == "breast_cancer":
        X, y = load_breast_cancer()
    else:
        X, y = load_synthetic(args.n_samples, args.n_features, args.task)
    
    print(f"Shape: X={X.shape}, y={y.shape}")
    
    # Select models
    if args.task == "classification":
        models = CLASSIFICATION_MODELS
        eval_fn = evaluate_classification
    elif args.task == "regression":
        models = REGRESSION_MODELS
        eval_fn = evaluate_regression
    else:
        models = CLUSTERING_MODELS
        eval_fn = evaluate_clustering
    
    # Run evaluation
    results = []
    print(f"\nEvaluating {len(models)} models...")
    
    for name, model in models.items():
        print(f"  {name}...", end=" ", flush=True)
        try:
            metrics = eval_fn(model, X, y)
            metrics["name"] = name
            results.append(metrics)
            print("OK")
        except Exception as e:
            print(f"X ({e})")
    
    # Print results
    print_results_table(results, args.task)
    
    print(f"\nNEXT STEPS:")
    print(f"  • Try hyperparameter tuning for top models")
    print(f"  • Use cross_val_score for more robust estimates")
    print(f"  • Explore: python scripts/run_exercise.py --chapter 9 --exercise 2 --variant solution")
    print(f"  • Full comparison: topics/ch16_pipelines_production/exercises/ex07_production_pipeline/")


if __name__ == "__main__":
    main()
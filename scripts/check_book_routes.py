#!/usr/bin/env python
"""
Validate \codepath{} references in book chapters resolve to actual files.

Usage:
    python scripts/check_book_routes.py
"""

import re
import sys
from pathlib import Path


CODEPATH_PATTERN = re.compile(r'\\codepath\{([^}]+)\}')


# Mapping from old book paths to new exercise structure
PATH_MAPPING = {
    # Root-level algorithm directories (now in src/ml_from_scratch/chXX_*/)
    "svm/": "src/ml_from_scratch/ch09_svm/",
    "clustering/": "src/ml_from_scratch/ch11_kmeans_clustering/",
    "decision_trees/": "src/ml_from_scratch/ch10_decision_trees/",
    "naive_bayes/": "src/ml_from_scratch/ch08_naive_bayes/",
    "gaussian_mixture/": "src/ml_from_scratch/ch12_gaussian_mixture/",
    "neural_networks/": "src/ml_from_scratch/ch13_neural_networks/",
    "PINN/": "src/ml_from_scratch/ch14_pinn/",
    "dimensionality_reduction/": "src/ml_from_scratch/ch05_multiple_polynomial_regression/",
    "gradient_descent/": "src/ml_from_scratch/ch03_gradient_descent/",
    "linear_regression/": "src/ml_from_scratch/ch04_linear_regression/",
    "logistic_regression/": "src/ml_from_scratch/ch07_logistic_regression/",
    
    # Chapter 3 - Gradient Descent
    "gradient_descent/exercises/": "topics/ch03_gradient_descent/exercises/",
    "gradient_descent/exercise_1.ipynb": "topics/ch03_gradient_descent/exercises/ex01_unified_optimizer/starter.py",
    "gradient_descent/exercise_2.ipynb": "topics/ch03_gradient_descent/exercises/ex02_shuffling_effect/starter.py",
    "gradient_descent/exercise_3.ipynb": "topics/ch03_gradient_descent/exercises/ex03_lr_decay/starter.py",
    "gradient_descent/exercise_4.ipynb": "topics/ch03_gradient_descent/exercises/ex04_scaling_experiment/starter.py",
    "gradient_descent/exercise_5.ipynb": "topics/ch03_gradient_descent/exercises/ex05_momentum/starter.py",
    
    # Chapter 4 - Linear Regression
    "linear_regression/simple_regression/linear_regression_scratch.ipynb": "src/ml_from_scratch/ch04_linear_regression/linear_regression.py",
    "linear_regression/exercises/": "topics/ch04_linear_regression/exercises/",
    "linear_regression/exercises/exercise_4_1_solution.ipynb": "src/ml_from_scratch/ch04_linear_regression/linear_regression.py",
    "linear_regression/exercises/exercise_4_2_solution.ipynb": "src/ml_from_scratch/ch04_linear_regression/linear_regression.py",
    "linear_regression/exercises/exercise_4_3_solution.ipynb": "src/ml_from_scratch/ch04_linear_regression/linear_regression.py",
    
    # Chapter 5 - Multiple/Polynomial Regression
    "linear_regression/multiple_regression/": "src/ml_from_scratch/ch05_multiple_polynomial_regression/",
    "linear_regression/polynomial_regression/": "src/ml_from_scratch/ch05_multiple_polynomial_regression/",
    
    # Chapter 6 - Regularized Regression
    "linear_regression/linear_regression_with_regul.ipynb": "src/ml_from_scratch/ch06_regularized_regression/regularized_regression.py",
    "linear_regression/exercises/": "topics/ch06_regularized_regression/exercises/",
    
    # Chapter 7 - Logistic Regression
    "logistic_regression/logistic_regression_scratch.ipynb": "src/ml_from_scratch/ch07_logistic_regression/logistic_regression.py",
    "logistic_regression/exercises/": "topics/ch07_logistic_regression/exercises/",
    
    # Chapter 8 - Naive Bayes
    "naive_bayes/naive_bayes_scratch.ipynb": "src/ml_from_scratch/ch08_naive_bayes/naive_bayes.py",
    "naive_bayes/exercises/": "topics/ch08_naive_bayes/exercises/",
    
    # Chapter 9 - SVM
    "svm/svm_core.py": "src/ml_from_scratch/ch09_svm/svm.py",
    "svm/svm_soft.py": "src/ml_from_scratch/ch09_svm/svm.py",
    "svm/exercises/": "topics/ch09_svm/exercises/",
    
    # Chapter 10 - Decision Trees
    "decision_trees/decision_trees_implementation.ipynb": "src/ml_from_scratch/ch10_decision_trees/trees.py",
    "decision_trees/exercises/": "topics/ch10_decision_trees/exercises/",
    
    # Chapter 11 - K-Means
    "clustering/k_means_implementation.ipynb": "src/ml_from_scratch/ch11_kmeans_clustering/clustering.py",
    "clustering/k_means_scratch.py": "src/ml_from_scratch/ch11_kmeans_clustering/clustering.py",
    "clustering/exercises/": "topics/ch11_kmeans_clustering/exercises/",
    
    # Chapter 12 - GMM
    "gaussian_mixture/gmm_implementation.ipynb": "src/ml_from_scratch/ch12_gaussian_mixture/clustering.py",
    "gaussian_mixture/gmm_scratch.py": "src/ml_from_scratch/ch12_gaussian_mixture/clustering.py",
    "gaussian_mixture/exercises/": "topics/ch12_gaussian_mixture/exercises/",
    
    # Chapter 13 - Neural Networks
    "neural_networks/NN-from-scratch.ipynb": "src/ml_from_scratch/ch13_neural_networks/neural_networks.py",
    "neural_networks/exercises/": "topics/ch13_neural_networks/exercises/",
    
    # Chapter 14 - PINN
    "PINN/exercises/": "topics/ch14_pinn/exercises/",
    
    # Chapter 15 - Pipelines/Production
    "pipelines/exercises/": "topics/ch15_pipelines_production/exercises/",
    "production/exercises/": "topics/ch16_production_software/exercises/",
    
    # Function/method references that exist in the codebase
    "fit_normal_equation()": "src/ml_from_scratch/ch04_linear_regression/linear_regression.py",
    "l1_ratio": "src/ml_from_scratch/ch06_regularized_regression/regularized_regression.py",
    "regularization='elasticnet'": "src/ml_from_scratch/ch06_regularized_regression/regularized_regression.py",
    "fit_ridge_closed()": "src/ml_from_scratch/ch04_linear_regression/linear_regression.py",
    "regularization='ridge'": "src/ml_from_scratch/ch04_linear_regression/linear_regression.py",
    "self._gaussian_pdf(X,...)": "src/ml_from_scratch/ch08_naive_bayes/naive_bayes.py",
    "likelihood_log = np.sum(np.log(likelihood + 1e-9), axis=1)": "src/ml_from_scratch/ch08_naive_bayes/naive_bayes.py",
    "log_probs[:, idx] = prior_log + likelihood_log": "src/ml_from_scratch/ch08_naive_bayes/naive_bayes.py",
    "np.argmax(log_probs, axis=1)": "src/ml_from_scratch/ch08_naive_bayes/naive_bayes.py",
    "fit_transform": "src/ml_from_scratch/ch05_multiple_polynomial_regression/preprocessing.py",
    "mean_absolute_error": "src/ml_from_scratch/metrics/metrics.py",
    "r_squared": "src/ml_from_scratch/metrics/metrics.py",
    "precision_recall_f1": "src/ml_from_scratch/metrics/metrics.py",
    "k_fold_cross_validation": "src/ml_from_scratch/ch15_pipelines_production/pipeline.py",
    "sklearn.datasets.make_moons": "N/A (sklearn function)",
}


def find_codepaths(book_dir: Path):
    """Find all \codepath{} references in .tex files."""
    codepaths = []
    
    for tex_file in book_dir.glob("**/*.tex"):
        content = tex_file.read_text(encoding="utf-8")
        for match in CODEPATH_PATTERN.finditer(content):
            codepaths.append({
                "file": tex_file,
                "line": content[:match.start()].count('\n') + 1,
                "path": match.group(1),
            })
    
    return codepaths


def resolve_path(codepath: str, root: Path):
    """Resolve a codepath to an actual file."""
    codepath = codepath.strip()
    
    # Unescape LaTeX underscores
    codepath = codepath.replace('\\_', '_')
    
    # Check if it's in our mapping
    if codepath in PATH_MAPPING:
        mapped = PATH_MAPPING[codepath]
        if mapped == "N/A (sklearn function)":
            return Path("N/A")  # Special marker for sklearn refs
        candidate = root / mapped
        if candidate.exists():
            return candidate
    
    # Try prefix matching for directory paths
    for old_prefix, new_prefix in PATH_MAPPING.items():
        if codepath.startswith(old_prefix) and old_prefix.endswith('/'):
            relative = codepath[len(old_prefix):]
            mapped = new_prefix + relative
            candidate = root / mapped
            if candidate.exists():
                return candidate
    
    # Try direct paths
    if codepath.startswith("/"):
        codepath = codepath[1:]
    
    candidates = [
        root / codepath,
        root / "src" / codepath,
        root / "topics" / codepath,
    ]
    
    for c in candidates:
        if c.exists():
            return c
    
    # Try with .py extension
    if not codepath.endswith(".py"):
        for c in candidates:
            c_py = c.with_suffix(c.suffix + ".py")
            if c_py.exists():
                return c_py
    
    return None


def main():
    root = Path(".")
    book_dir = Path("Book")
    
    if not book_dir.exists():
        print(f"Error: Book directory not found at {book_dir}")
        return 1
    
    print("Scanning for \\codepath{} references...")
    codepaths = find_codepaths(book_dir)
    print(f"Found {len(codepaths)} codepath references")
    
    errors = []
    resolved = 0
    
    for cp in codepaths:
        resolved_path = resolve_path(cp["path"], root)
        if resolved_path and str(resolved_path) != "N/A":
            resolved += 1
        elif str(resolved_path) == "N/A":
            # Known external reference (e.g., sklearn)
            resolved += 1
        else:
            errors.append(f"{cp['file']}:{cp['line']}: Cannot resolve '{cp['path']}'")
    
    print(f"Resolved: {resolved}/{len(codepaths)}")
    
    if errors:
        print("\nUNRESOLVED REFERENCES:")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("\nAll codepath references resolved!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
# Agent.md — ML-Algorithms-From-Scratch Project Context

## Project Overview
ML-Algorithms-From-Scratch is a comprehensive educational repository containing machine learning algorithms implemented both from scratch using NumPy and with popular libraries like scikit-learn. Each implementation includes detailed explanations, mathematical concepts, and practical examples.

**Project Goal**: Provide clear, well-documented implementations to help understand inner workings of ML algorithms. Each algorithm is implemented twice:
1. From scratch using NumPy (core concepts)
2. Using popular libraries (practical applications)

## Repository Structure

Root directories:
- `book/` — LaTeX book source with chapters, appendices, bibliography, paper-reviews
- `linear_regression/` — simple, multiple, polynomial regression
  - `simple_regression/`
  - `multiple_regression/`
  - `polynomial_regression/`
- `Gradient Descent/` — Batch and Stochastic GD implementations
  - `GD from Scratch/`
  - `GD from Sklearn/`
- `logistic_regression/` — scratch and regularized implementations
- `neural_networks/` — neural network from scratch
- `decision_trees/` — decision tree implementations with images and datasets
- `clustering/` — K-means implementation and examples
- `dimensionality_reduction/` — PCA implementation scratch + examples
- `gaussian_mixture/` — GMM scratch implementation
- `naive_bayes/` — naive Bayes scratch implementation
- `svm/` — Support Vector Machines
- `PINN/` — Physics-Informed Neural Networks

Root files:
- `README.md` — project overview and learning path
- `main.tex` — LaTeX book main file
- `requirements.txt` — Python dependencies
- `CODEBASE_PREPARATION_PROMPT.md` — alignment instructions between code and book
- `LICENSE` — MIT License

## Technologies
- Python 3.8+
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook

Dependencies from requirements.txt:
- numpy
- matplotlib
- scikit-learn
- jupyter

## Implemented Algorithms
- Linear Regression: Gradient Descent, Normal Equation, Simple/Multiple/Polynomial
- Gradient Descent: Batch, Stochastic
- Logistic Regression
- Neural Networks from scratch
- Decision Tree
- Random Forest (referenced externally)
- PINN
- Support Vector Machines
- K-means Clustering
- Naive Bayes
- Dimensionality Reduction (PCA)
- Gaussian Mixture Models (GMM)

## Content Format
Each algorithm folder contains:
- Theoretical explanation
- Step-by-step implementation (Jupyter notebooks)
- Visualization of results
- Practical examples
- Performance evaluation
- Some folders include standalone Python scripts (e.g., `*_scratch.py`)

## Book Alignment
The repository is companion code for the book *Building Machine Learning Algorithms from First Principles*.
- Book source is in `book/` with `main.tex`, `chapters/`, `appendices/`, `bibliography/`
- `\codepath{...}` references in chapters need to resolve to actual repository paths
- Codebase preparation prompt requires maintaining alignment between manuscript routes and code paths
- Do not add new exercises, only fix existing ones

## Working Rules
- Repository root is `ML-Algorithms-From-Scratch/`
- Preserve existing user changes; work on dedicated branches
- Do not add new exercises
- When book and code conflict, choose tested convention and align both sides
- Update README to accurately describe implemented vs supplemental material

## Notes for Agents
- Primarily notebook-oriented with scattered Python scripts
- Inconsistent folder naming e.g., `Gradient Descent` with space
- Some generated output files in `svm/outputs`, `decision_trees/Images`
- Minimal test suite present; validation is via notebooks
- Book may reference code paths that do not match current layout

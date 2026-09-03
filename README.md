# ML Algorithms From Scratch

A learning-oriented Python codebase accompanying *Building Machine Learning
Algorithms from First Principles*. The repository combines mathematical
explanations, NumPy-based implementations, teaching notebooks, and tested
exercise solutions.

## What Is Included

The maintained implementation package is under `src/ml_from_scratch/` and is
organized by chapter:
- Chapter 16: Production software

The public convenience API is exported from
`src/ml_from_scratch/__init__.py`. Implementations use NumPy and follow
scikit-learn-style interfaces where appropriate, including `fit`, `predict`,
`score`, and `get_params`/`set_params` support for selected estimators.

## Repository Layout

```text
src/ml_from_scratch/              Maintained implementations and utilities
src/ml_from_scratch/chXX_*/       Chapter modules and chapter-local notebooks
topics/chXX_*/exercises/          89 starter/solution exercise units
tests/                            Unit, invariant, parity, and smoke tests
figures/generated/chXX_*/         Generated chapter figures
figures/book/                     Figures used by the manuscript
Book/                             LaTeX manuscript and chapter sources
scripts/                          Exercise, route, and figure tooling
data/                             Data documentation and data files only
docs/                             Learning paths and project documentation
```

Each exercise normally contains:

```text
starter.py
solution.py
README.md

```

## Installation

The project requires Python 3.10 or newer. The recommended development setup
installs the package in editable mode with its development tools:

```bash
pip install -e ".[dev]"
```

PINN functionality has an optional PyTorch dependency:

```bash
pip install -e ".[dev,pinn]"
```

The package can then be imported from any Python process:

```python
from ml_from_scratch import LinearRegression, KMeans, LogisticRegression
```

## Learning Workflow

1. Read the relevant chapter in `Book/chapters/`.
2. Open the notebook in the matching `src/ml_from_scratch/chXX_*/notebooks/`
   directory.
3. Work through the corresponding exercise in `topics/`.
4. Compare the starter and solution implementations.
5. Run the focused exercise test and the broader test suite.

The exercise runner supports starter and solution variants:

```bash
python scripts/run_exercise.py --chapter 3 --exercise 1 --variant starter
python scripts/run_exercise.py --chapter 3 --exercise 1 --variant solution
```

## Validation Commands

Run commands from the repository root:

```bash
# Verify all 89 exercises and required resources
python scripts/check_exercise_coverage.py

# Verify manuscript codepath references
python scripts/check_book_routes.py

# Run the complete configured test suite
python -m pytest

# Run quality checks
ruff check .
mypy src/
```

Useful focused test commands include:

```bash
python -m pytest tests/unit/ -v
python -m pytest tests/invariants/ tests/parity/ tests/smoke/ -v
python -m pytest topics/ch03_gradient_descent/ -v
```

## Development Notes
- Put maintained algorithm logic in `src/ml_from_scratch/`; exercises should
- Keep notebooks focused on explanation, visualization, and calls into the
- Generated figures belong under `figures/generated/` and manuscript figures
  under `figures/book/`.

## Contributing

Contributions are welcome. New algorithms, exercises, tests, documentation,
and notebook improvements should follow the existing chapter and exercise
structure. Before submitting changes, run the relevant focused tests plus the
coverage and manuscript route checks.

## License

# ML Algorithms From Scratch

This repository accompanies *Building Machine Learning Algorithms from First
Principles*. It is a learning-oriented Python codebase with mathematical
explanations, NumPy implementations, chapter notebooks, exercise solutions,
and automated tests.

## Project Scope

The maintained implementation package is `src/ml_from_scratch/`. It covers
the following chapters and topics:

| Chapter | Topic |
|---------|-------|
| 2 | Python, NumPy, mathematics, and probability prerequisites |
| 3 | Gradient descent and optimization |
| 4 | Linear regression |
| 5 | Multiple and polynomial regression |
| 6 | Ridge, Lasso, and Elastic Net regression |
| 7 | Logistic regression |
| 8 | Naive Bayes |
| 9 | Support vector machines |
| 10 | Decision trees |
| 11 | K-Means clustering |
| 12 | Gaussian mixture models |
| 13 | Neural networks |
| 14 | Physics-informed neural networks |
| 15 | Pipelines and cross-validation |
| 16 | Production software and model operations |

Implementations are organized in chapter packages such as
`ml_from_scratch.ch03_gradient_descent` and
`ml_from_scratch.ch12_gaussian_mixture`. Common datasets, metrics,
preprocessing, validation, randomness, and plotting utilities are also
provided. The package root exposes the main public APIs for convenient use.

## Repository Layout

```text
src/ml_from_scratch/
    ch02_prerequisites/              Chapter modules and notebooks
    ch03_gradient_descent/
    ...
    ch16_production_software/
    datasets/                         Dataset generators and loaders
    metrics/                          Regression, classification, and clustering metrics
    utils/                            Plotting and shared helpers
topics/                               89 chapter exercises
    chXX_*/exercises/exNN_*/           Starter, solution, README, and focused test
tests/                                Unit, invariant, parity, and smoke tests
figures/generated/chXX_*/             Generated chapter figures
figures/book/                         Manuscript figures
figures/assets/                       Shared figure assets
Book/                                 LaTeX manuscript sources
scripts/                              Exercise, route, figure, and learning tools
data/                                 Data documentation and data files
docs/                                 Learning paths and project documentation
pyproject.toml                        Package metadata and tool configuration
```

Teaching notebooks live beside their chapter implementation, for example:
`src/ml_from_scratch/ch02_prerequisites/notebooks/`. The repository currently
contains 36 notebooks. Exercise notebooks are not required; the canonical
exercise files are Python starter and solution files under `topics/`.

## Installation

Python 3.10 or newer is required. Install the package and development tools
from the repository root:

```bash
pip install -e ".[dev]"
```

The optional `pinn` extra installs PyTorch for PINN workflows:

```bash
pip install -e ".[dev,pinn]"
```

The runtime dependencies include NumPy, SciPy, Matplotlib, scikit-learn, and
pandas. Development dependencies include pytest, ruff, mypy, and Jupyter.

## Quick Start

Import maintained algorithms from the package:

```python
import numpy as np

from ml_from_scratch import LinearRegression

X = np.array([[1.0], [2.0], [3.0]])
y = np.array([3.0, 5.0, 7.0])

model = LinearRegression().fit(X, y)
predictions = model.predict(X)
print(predictions)
```

Run an exercise variant with the exercise runner:

```bash
python scripts/run_exercise.py --chapter 3 --exercise 1 --variant starter
python scripts/run_exercise.py --chapter 3 --exercise 1 --variant solution
```

Start Jupyter from the repository root and open a notebook from the matching
chapter package:

```bash
jupyter lab
```

## Learning Workflow

1. Read the relevant chapter in `Book/chapters/`.
2. Review the chapter README and notebook under `src/ml_from_scratch/`.
3. Work through the matching exercise in `topics/`.
4. Compare the starter and solution implementations.
5. Run the focused test, then the broader checks.

The prerequisite notebooks in `topics/ch02_prerequisites/` introduce Python,
NumPy, linear algebra, and probability before Chapter 3.

## Validation

Run these commands from the repository root:

```bash
# Check the 89-exercise inventory and required files
python scripts/check_exercise_coverage.py

# Check manuscript codepath references
python scripts/check_book_routes.py

# Run all tests configured in pyproject.toml
python -m pytest

# Run static quality checks
ruff check .
mypy src/
```

Useful focused commands:

```bash
python -m pytest tests/unit/ -v
python -m pytest tests/invariants/ tests/parity/ tests/smoke/ -v
python -m pytest topics/ch03_gradient_descent/ -v
```

The coverage checker expects 89 exercises across Chapters 3-16. The book route
checker validates the manuscript's 46 `\\codepath{}` references.

## Development Conventions

- Keep maintained algorithm logic in `src/ml_from_scratch/`.
- Keep exercises educational and import maintained code instead of duplicating
  package implementations.
- Use `numpy.random.default_rng(random_state)` for reproducible stochastic
  behavior.
- Shuffle features and targets together; never shuffle them independently.
- Keep notebooks focused on explanation, visualization, and calls into the
  maintained package.
- Store generated figures under `figures/generated/` and manuscript figures
  under `figures/book/`.
- Add focused tests for mathematical behavior, edge cases, and public API
  contracts when changing an implementation.

## Contributing

Contributions are welcome. New algorithms, exercises, tests, notebooks, and
documentation should follow the existing chapter structure. Before opening a
pull request, run the relevant tests together with the exercise coverage and
manuscript route checks.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
This project is licensed under the MIT License. See [LICENSE](LICENSE).

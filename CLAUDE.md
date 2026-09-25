# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

### Installation
```bash
pip install -e '.[dev]'
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/unit/
pytest tests/integration/
pytest tests/exercises/

# Run exercise-specific tests
python scripts/run_exercise.py --chapter 3 --exercise 1 --variant solution
pytest topics/ch03_gradient_descent/exercises/ex01_unified_optimizer/test_solution.py
```

### Formatting and Linting
```bash
# Check formatting
ruff check .

# Apply formatting
ruff check --fix .

# Type checking
mypy src/
```

## Project Structure

### Core Implementation
- `src/ml_from_scratch/` - Maintained algorithm implementations
  - Each algorithm has its own module (optimization, linear_models, naive_bayes, etc.)
  - Shared utilities in validation.py, random.py, datasets/, metrics/, preprocessing/
  - Production-ready interfaces in pipeline/ and production/

### Exercise Organization
- `topics/chXX_topic/` - Chapter-specific topic directories
  - `topics/chXX_topic/exercises/exNN_slug/` - Individual exercise directories
    - `starter.py` - Starting point with TODOs
    - `solution.py` - Complete, tested solution
    - `test_solution.py` - Focused tests for the exercise
    - `README.md` - Exercise description, learning objectives, setup instructions
    - `expected/` - Small expected outputs (when needed)
    - `config.yaml` - Exercise-specific configuration (when useful)

### Supporting Directories
- `scripts/` - Helper scripts for exercise management
  - `list_exercises.py` - List all exercises
  - `run_exercise.py` - Run starter/solution variants
  - `check_exercise_coverage.py` - Verify all 83 exercises are present
  - `check_book_routes.py` - Validate manuscript code references
  - `regenerate_figures.py` - Generate plots from versioned scripts
- `tests/` - Test suites
  - `unit/` - Mathematical kernel tests
  - `integration/` - Chapter workflow tests
  - `invariants/` - Property-based tests
  - `parity/` - Library comparison tests
  - `smoke/` - Basic execution tests for each exercise
- `docs/` - Documentation
  - `index.md` - Main documentation landing page
  - `topic-index.md` - Browse by topic
  - `exercise-index.md` - Browse by exercise
  - `installation.md` - Setup instructions
  - `data-policy.md` - Data usage guidelines
  - `contributor-guide.md` - Contribution process
- `data/` - Datasets
  - `raw/` - Original, versioned datasets
  - `generated/` - Preprocessed data (not committed)
- `figures/generated/` - Generated plots (not committed)

## Key Conventions

### Algorithm API
All estimators follow a consistent interface:
- `fit(X, y=None) -> self`
- `predict(X) -> predictions or labels`
- `predict_proba(X) -> probabilities (when applicable)`
- `fit_predict(X) -> labels (for clustering)`
- `score(X, y=None) -> metric score`

### Randomness Control
- Use `numpy.random.default_rng(random_state)` for all stochastic operations
- Document seeds in all exercises and examples
- Never shuffle X and y independently

### Notebooks
- Notebooks should be thin: explain, visualize, and call maintained modules
- Not contain core algorithm logic (that belongs in src/)
- Execute from clean checkout when optional dependencies are installed
- Valid JSON format

## Common Commands

### Exercise Management
```bash
# List all exercises
python scripts/list_exercises.py

# Run exercise starter
python scripts/run_exercise.py --chapter 3 --exercise 1 --variant starter

# Run exercise solution
python scripts/run_exercise.py --chapter 3 --exercise 1 --variant solution

# Check exercise coverage
python scripts/check_exercise_coverage.py

# Validate book routes
python scripts/check_book_routes.py

# Regenerate figures
python scripts/regenerate_figures.py
```

### Development Workflow
```bash
# Install development dependencies
pip install -e '.[dev]'

# Run linter
ruff check .

# Run formatter
ruff check --fix .

# Run type checker
mypy src/

# Run full test suite
pytest

# Run exercise tests specifically
pytest tests/exercises/
```

## Algorithm Organization

The repository maintains one implementation per algorithm in `src/ml_from_scratch/`:
- Optimization algorithms: `optimization.py`
- Linear models: `linear_models/` (linear_regression.py, logistic_regression.py, etc.)
- Naive Bayes: `naive_bayes/`
- SVM: `svm/`
- Decision trees: `trees/`
- Clustering: `clustering/` (kmeans.py, gmm.py)
- Neural networks: `neural_networks/` and `pinn/`
- Pipeline utilities: `pipeline/` and `production/`

Exercises import from these maintained modules rather than duplicating implementations.

## Exercise Standards

Each exercise follows this structure:
1. **README.md** - Complete description with:
   - Chapter, exercise number, name, difficulty
   - Learning objectives (observable skills)
   - Complete task description from manuscript
   - Starter guide (explains TODOs without revealing solution)
   - Exact run commands
   - Acceptance criteria (numerical tolerances, required outputs)
   - Common pitfalls to avoid
   - Further exploration (optional, clearly labeled)

2. **Starter** - Runs far enough to expose TODOs, includes:
   - Focused TODO markers (not empty files)
   - Small deterministic data or documented loader
   - Actionable errors for missing prerequisites

3. **Solution** - Complete, readable, tested implementation that:
   - Reuses maintained modules from src/
   - Explains non-obvious mathematical choices
   - Preserves intended difficulty (doesn't trivialize with library calls unless comparison is the task)
   - Reports actual results (not manuscript expectations)

4. **Tests** - Each solution has:
   - At least one focused correctness test
   - One relevant edge case or invariant test
   - For stochastic tests: fixed seeds with tolerant assertions
   - For plotting tests: data/metric validation + file generation check
   - For performance tests: hardware-tolerant checks with environment recording

## Quality Gates

Before considering work complete:
1. All 83 exercises present exactly once (no Chapter 13 duplicate)
2. Every exercise has documented starter, solution, and passing tests
3. Formatting, linting, and type checks pass
4. At least one representative solution from each topic executes successfully
5. All notebooks are valid JSON and representative ones execute
6. All active manuscript code routes resolve
7. Representative figures regenerate correctly from scripts
8. Book compiles cleanly after any route/code-snippet edits
9. No credentials or private data committed
10. Dependency groups documented (core vs optional heavyweight)

This structure allows learners to progress naturally from book equations → tested implementation → runnable example → guided starter → complete solution → clear verification without encountering duplicated logic, missing routes, or insecure access.
# Codebase Upgrade Plan: Consolidation & Restructure

## Current State Analysis

### Duplicate Implementation Locations
| Location | Purpose | Status |
|----------|---------|--------|
| `data/raw/*/` | Original book code (old) | **REMOVE** - Duplicates `src/` |
| `src/ml_from_scratch/` | Main library | **KEEP** - Single source of truth |
| `topics/chXX/exercises/` | Exercise starter/solution | **KEEP** - Educational content |

### Figure Locations
| Location | Status |
|----------|--------|
| `figures/generated/` | **REORGANIZE** - Move to chapter structure |
| `figures/` root | **KEEP** - Add structure |

### Notebook Locations
| Location | Status |
|----------|--------|
| `topics/ch02_prerequisites/notebooks/` | **MOVE** → `src/ml_from_scratch/notebooks/` |
| Exercise directories | **ADD** - Create notebooks alongside starter.py |

---

## Target Structure

```
ML-Algorithms-From-Scratch/
├── src/ml_from_scratch/                 # SINGLE SOURCE OF TRUTH
│   ├── __init__.py
│   ├── optimization.py                  # Core algorithms (.py)
│   ├── linear_models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── linear_regression.py
│   │   └── regularized_regression.py
│   ├── logistic_regression/
│   ├── naive_bayes/
│   ├── svm/
│   ├── trees/
│   ├── clustering/
│   ├── neural_networks/
│   ├── pinn/
│   ├── pipeline/
│   ├── production/
│   ├── preprocessing/
│   ├── metrics/
│   ├── validation.py
│   ├── random.py
│   ├── datasets/
│   ├── utils/
│   └── notebooks/                       # TEACHING NOTEBOOKS (NEW)
│       ├── ch02_prerequisites/
│       │   ├── 01_python_basics.ipynb
│       │   ├── 02_numpy.ipynb
│       │   ├── 03_math_linear_algebra.ipynb
│       │   └── 04_probability_statistics.ipynb
│       ├── ch03_gradient_descent/
│       │   ├── 01_unified_optimizer.ipynb
│       │   ├── 02_shuffling_effect.ipynb
│       │   ├── 03_lr_decay.ipynb
│       │   ├── 04_scaling_experiment.ipynb
│       │   └── 05_momentum.ipynb
│       ├── ch04_linear_regression/
│       │   ├── 01_large_lr.ipynb
│       │   ├── 02_more_noise.ipynb
│       │   ├── 03_no_intercept.ipynb
│       │   ├── 04_cost_history.ipynb
│       │   ├── 05_normal_eq_vs_gd.ipynb
│       │   └── 06_scaling_crossover.ipynb
│       └── ... (one per chapter exercise)
│
├── topics/                              # EXERCISES ONLY
│   ├── ch03_gradient_descent/
│   │   └── exercises/
│   │       ├── ex01_unified_optimizer/
│   │       │   ├── starter.py
│   │       │   ├── solution.py
│   │       │   ├── test_ch03_ex01_unified_optimizer.py
│   │       │   └── README.md
│   │       └── ...
│   ├── ch04_linear_regression/
│   │   └── exercises/
│   └── ... (chapters 3-17)
│
├── figures/                             # ORGANIZED FIGURES
│   ├── generated/                       # Auto-generated from notebooks
│   │   ├── ch03_gradient_descent/
│   │   ├── ch04_linear_regression/
│   │   └── ...
│   ├── book/                            # Book manuscript figures
│   │   ├── ch02_prerequisites/
│   │   ├── ch03_gradient_descent/
│   │   └── ...
│   └── assets/                          # Shared assets (logos, etc.)
│
├── scripts/                             # TOOLING (unchanged)
│   ├── list_exercises.py
│   ├── run_exercise.py
│   ├── check_book_routes.py
│   ├── check_exercise_coverage.py
│   ├── regenerate_figures.py
│   ├── self_assess.py
│   ├── beginner_tour.py
│   ├── algorithm_comparison.py
│   └── production_demo.py
│
├── tests/                               # TESTS (unchanged)
│   ├── unit/
│   ├── integration/
│   ├── invariants/
│   └── smoke/
│
├── Book/                                # MANUSCRIPT (unchanged)
│   ├── chapters/
│   ├── appendices/
│   └── main.tex
│
├── data/                                # DATA ONLY (no code)
│   └── README.md
│
├── specs/                               # PROJECT SPECS (unchanged)
│   └── ...
│
├── docs/                                # DOCUMENTATION
│   ├── BOOK_IMPROVEMENT_GUIDE.md
│   └── LEARNING_PATHS.md
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── CLAUDE.md
```

---

## Implementation Steps

### Step 1: Create Notebook Directory Structure
```bash
mkdir -p src/ml_from_scratch/notebooks/ch02_prerequisites
mkdir -p src/ml_from_scratch/notebooks/ch03_gradient_descent
mkdir -p src/ml_from_scratch/notebooks/ch04_linear_regression
mkdir -p src/ml_from_scratch/notebooks/ch05_multiple_polynomial_regression
mkdir -p src/ml_from_scratch/notebooks/ch06_regularized_regression
mkdir -p src/ml_from_scratch/notebooks/ch07_logistic_regression
mkdir -p src/ml_from_scratch/notebooks/ch08_naive_bayes
mkdir -p src/ml_from_scratch/notebooks/ch09_svm
mkdir -p src/ml_from_scratch/notebooks/ch10_decision_trees
mkdir -p src/ml_from_scratch/notebooks/ch11_kmeans_clustering
mkdir -p src/ml_from_scratch/notebooks/ch12_gaussian_mixture
mkdir -p src/ml_from_scratch/notebooks/ch14_neural_networks
mkdir -p src/ml_from_scratch/notebooks/ch15_pinn
mkdir -p src/ml_from_scratch/notebooks/ch16_pipelines_production
mkdir -p src/ml_from_scratch/notebooks/ch17_production_software
```

### Step 2: Move Prerequisite Notebooks
```bash
mv topics/ch02_prerequisites/notebooks/*.ipynb \
   src/ml_from_scratch/notebooks/ch02_prerequisites/
```

### Step 3: Create Chapter Notebooks from Exercise Solutions
For each chapter exercise, create a teaching notebook in `src/ml_from_scratch/notebooks/chXX/` that:
- Imports from `ml_from_scratch` library
- Demonstrates the algorithm step-by-step
- Generates figures to `figures/generated/chXX/`
- Follows Raschka style (Concept → Visual → Code → Verify)

### Step 4: Reorganize Figures
```bash
mkdir -p figures/generated/ch03_gradient_descent
mkdir -p figures/generated/ch04_linear_regression
mkdir -p figures/generated/ch05_multiple_polynomial_regression
mkdir -p figures/generated/ch06_regularized_regression
mkdir -p figures/generated/ch07_logistic_regression
mkdir -p figures/generated/ch08_naive_bayes
mkdir -p figures/generated/ch09_svm
mkdir -p figures/generated/ch10_decision_trees
mkdir -p figures/generated/ch11_kmeans_clustering
mkdir -p figures/generated/ch12_gaussian_mixture
mkdir -p figures/generated/ch14_neural_networks
mkdir -p figures/generated/ch15_pinn
mkdir -p figures/generated/ch16_pipelines_production
mkdir -p figures/generated/ch17_production_software
mkdir -p figures/book/ch02_prerequisites
mkdir -p figures/book/ch03_gradient_descent
# ... etc for all chapters
mkdir -p figures/assets
```

Move existing generated figures:
```bash
# Example mapping
mv figures/generated/cost_history.png figures/generated/ch04_linear_regression/
mv figures/generated/large_lr_cost_curve.png figures/generated/ch04_linear_regression/
mv figures/generated/ridge_alpha_sweep.png figures/generated/ch06_regularized_regression/
# ... etc for all figures
```

### Step 5: Remove data/raw/ (Duplicate Code)
```bash
rm -rf data/raw/
```
**Note:** All algorithm implementations already exist in `src/ml_from_scratch/`

### Step 6: Update regenerate_figures.py
Modify to output to `figures/generated/chXX/` structure.

### Step 7: Update check_book_routes.py
Update path mappings to point to `src/ml_from_scratch/` and `src/ml_from_scratch/notebooks/`.

### Step 8: Update pyproject.toml
Ensure package includes notebooks:
```toml
[tool.setuptools.packages.find]
where = ["src"]
include = ["ml_from_scratch*"]
```

### Step 9: Update Exercise Structure (Optional Enhancement)
Add notebook version to each exercise:
```
topics/ch03_gradient_descent/exercises/ex01_unified_optimizer/
├── starter.py
├── solution.py
├── starter.ipynb        # NEW: Notebook version of starter
├── solution.ipynb       # NEW: Notebook version of solution
├── test_ch03_ex01_unified_optimizer.py
└── README.md
```

---

## Migration Checklist

### Code Consolidation
- [ ] Verify `src/ml_from_scratch/` has all algorithms from `data/raw/`
- [ ] Move prerequisite notebooks to `src/ml_from_scratch/notebooks/`
- [ ] Create chapter teaching notebooks in `src/ml_from_scratch/notebooks/`
- [ ] Remove `data/raw/` directory
- [ ] Update imports in all exercise files to use `ml_from_scratch`

### Figure Organization
- [ ] Create chapter subdirectories in `figures/generated/`
- [ ] Move existing figures to appropriate chapter folders
- [ ] Create `figures/book/` for manuscript figures
- [ ] Update `regenerate_figures.py` output paths

### Tooling Updates
- [ ] Update `check_book_routes.py` path mappings
- [ ] Update `regenerate_figures.py` for new structure
- [ ] Update `scripts/run_exercise.py` if needed
- [ ] Verify all tests pass

### Validation
- [ ] Run `python scripts/check_book_routes.py` - all 46 routes resolved
- [ ] Run `python scripts/check_exercise_coverage.py` - 89 exercises
- [ ] Run `pytest topics/` - all 178 tests pass
- [ ] Run `pytest tests/` - all unit/integration tests pass
- [ ] Verify `python -c "from ml_from_scratch import *"` works

---

## Benefits After Migration

1. **Single Source of Truth**: All implementations in `src/ml_from_scratch/`
2. **Teaching + Library Unified**: `.py` for library, `.ipynb` for teaching
3. **Clean Separation**: Exercises in `topics/`, library in `src/`
4. **Organized Figures**: Chapter-based structure in `figures/`
5. **No Duplicates**: Removed `data/raw/` legacy code
6. **Raschka-Style**: Notebooks follow proven pedagogical pattern
7. **Maintainable**: Clear ownership of each component
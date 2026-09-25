#!/usr/bin/env python
"""
Self-assessment script to help users choose their learning path.

Usage:
    python scripts/self_assess.py
"""

import sys
from typing import Dict, List, Tuple


QUESTIONS = {
    "math": [
        ("Matrix multiplication & broadcasting", "Linear algebra"),
        ("Gradient descent intuition", "Calculus/optimization"),
        ("Gaussian distribution & log-likelihood", "Probability"),
        ("Eigenvalues/eigenvectors (PCA/SVD)", "Linear algebra"),
        ("Chain rule & partial derivatives", "Calculus"),
        ("Bayes' theorem", "Probability"),
        ("Convexity & stationary points", "Optimization"),
        ("Taylor series approximation", "Calculus"),
    ],
    "python": [
        ("NumPy array manipulation (broadcasting, slicing)", "NumPy"),
        ("Writing classes with __init__, fit, predict", "OOP"),
        ("Using type hints (PEP 484)", "Modern Python"),
        ("Running pytest and reading test output", "Testing"),
        ("Virtual environments (venv/conda)", "Environment"),
        ("Installing packages with pip install -e .", "Packaging"),
        ("Jupyter notebooks for exploration", "Notebooks"),
        ("Basic Git (clone, commit, push)", "Version control"),
    ],
    "ml": [
        ("Supervised vs unsupervised learning", "ML concepts"),
        ("Bias-variance tradeoff", "ML theory"),
        ("Cross-validation purpose", "Evaluation"),
        ("Regularization (L1/L2)", "ML theory"),
        ("Decision boundaries", "ML concepts"),
        ("Clustering vs classification", "ML concepts"),
        ("Neural network forward/backward pass", "Deep learning"),
        ("Hyperparameter tuning", "ML practice"),
    ],
}


def ask_questions(category: str, questions: List[Tuple[str, str]]) -> int:
    """Ask questions for a category, return score 0-5 per question."""
    print(f"\n{'='*60}")
    print(f"  {category.upper()} PREREQUISITES")
    print(f"{'='*60}")
    print("Rate yourself 1-5 on each (1=unfamiliar, 5=comfortable):\n")
    
    total = 0
    for i, (question, topic) in enumerate(questions, 1):
        while True:
            try:
                score = int(input(f"  [{i}/{len(questions)}] {question} [{topic}]: "))
                if 1 <= score <= 5:
                    total += score
                    break
                print("    Please enter 1-5")
            except ValueError:
                print("    Please enter a number 1-5")
            except KeyboardInterrupt:
                print("\n\nAssessment cancelled.")
                sys.exit(0)
    return total


def recommend_path(math_score: int, python_score: int, ml_score: int) -> str:
    """Recommend learning path based on scores."""
    total = math_score + python_score + ml_score
    max_possible = 5 * (8 + 8 + 8)  # 120
    
    pct = (total / max_possible) * 100
    
    if pct < 30:
        return "BEGINNER"
    elif pct < 60:
        return "BEGINNER_WITH_REVIEW"
    elif pct < 80:
        return "INTERMEDIATE"
    else:
        return "EXPERT"


def print_recommendation(path: str, math_score: int, python_score: int, ml_score: int):
    """Print detailed recommendation."""
    print(f"\n{'='*60}")
    print(f"  ASSESSMENT RESULTS")
    print(f"{'='*60}")
    print(f"  Math Score:      {math_score}/40")
    print(f"  Python Score:    {python_score}/40")
    print(f"  ML Score:        {ml_score}/40")
    print(f"  Total:           {math_score + python_score + ml_score}/120")
    print(f"  Recommended Path: {path}")
    print(f"{'='*60}")
    
    paths = {
        "BEGINNER": """
  RECOMMENDED: BEGINNER PATH (12 weeks)
  
  Start with Chapter 2 (Prerequisites) thoroughly, then:
  • Part I: Ch1-3 (Foundations) - 2 weeks
  • Part II: Ch4-7 (Supervised Learning) - 6 weeks  
  • Capstone: House Price Prediction - 1 week
  
  Focus: Build intuition, understand every line of code
  Run: python scripts/beginner_tour.py
        """,
        "BEGINNER_WITH_REVIEW": """
  RECOMMENDED: BEGINNER PATH WITH MATH REVIEW (14 weeks)
  
  • Appendix A: Mathematical Refresher - 2 weeks
  • Then follow Beginner Path above
  
  Focus: Strengthen math foundations first
  Run: python scripts/beginner_tour.py (after review)
        """,
        "INTERMEDIATE": """
  RECOMMENDED: INTERMEDIATE PATH (10 weeks)
  
  Skip Part I (review Ch3 only if needed), start at:
  • Part II: Ch8-12 (Unsupervised) - 4 weeks
  • Part III: Ch13, 16 (Neural Networks + Pipelines) - 3 weeks
  • Capstone: Customer Churn Prediction - 1 week
  
  Focus: Algorithm internals, comparison, production patterns
  Run: python scripts/algorithm_comparison.py --dataset iris
        """,
        "EXPERT": """
  RECOMMENDED: EXPERT PATH (8 weeks)
  
  Skip to advanced topics:
  • Part IV: Ch14-15 (PINN) - 2 weeks
  • Part V: Ch17 (Production Software) - 2 weeks
  • Capstone: Production ML System - 2 weeks
  
  Focus: Advanced architectures, MLOps, extensibility
  Run: python scripts/production_demo.py --config config/production.yaml
        """,
    }
    
    print(paths.get(path, ""))


def print_next_steps(path: str):
    """Print concrete next steps."""
    print("\nNEXT STEPS:")
    print("-" * 40)
    
    if path in ("BEGINNER", "BEGINNER_WITH_REVIEW"):
        print("1. Read Chapter 2 (Prerequisites) completely")
        print("2. Run: python scripts/run_exercise.py --chapter 3 --exercise 1 --variant starter")
        print("3. Compare with solution: python scripts/run_exercise.py --chapter 3 --exercise 1 --variant solution")
        print("4. Join the discussion: GitHub Discussions #learning-path")
    elif path == "INTERMEDIATE":
        print("1. Quick review: Ch3 Gradient Descent (run ex01)")
        print("2. Start: python scripts/run_exercise.py --chapter 8 --exercise 1 --variant starter")
        print("3. Try: python scripts/algorithm_comparison.py --dataset iris")
        print("4. Explore: topics/ch10_decision_trees/exercises/ex03_manual_tree_trace/")
    else:
        print("1. Jump to: topics/ch14_pinn/exercises/ex01_different_forcing/")
        print("2. Review: topics/ch16_production_software/exercises/ex01_interchangeable_interfaces/")
        print("3. Try: python scripts/production_demo.py")
        print("4. Contribute: See CONTRIBUTING.md for advanced contributions")


def main():
    print("ML FROM SCRATCH - LEARNING PATH SELF-ASSESSMENT")
    print("=" * 60)
    print("This assessment helps you choose the right learning path.")
    print("Answer honestly - there are no wrong answers!")
    
    math_score = ask_questions("math", QUESTIONS["math"])
    python_score = ask_questions("python", QUESTIONS["python"])
    ml_score = ask_questions("ml", QUESTIONS["ml"])
    
    path = recommend_path(math_score, python_score, ml_score)
    print_recommendation(path, math_score, python_score, ml_score)
    print_next_steps(path)
    
    print("\nUSEFUL COMMANDS:")
    print("  List all exercises:     python scripts/list_exercises.py")
    print("  Run any exercise:       python scripts/run_exercise.py --chapter 3 --exercise 1 --variant starter")
    print("  Check coverage:         python scripts/check_exercise_coverage.py")
    print("  View learning paths:    cat docs/LEARNING_PATHS.md")
    print("  View book routes:       python scripts/check_book_routes.py")


if __name__ == "__main__":
    main()
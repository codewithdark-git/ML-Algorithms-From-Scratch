#!/usr/bin/env python
"""
Beginner guided tour - runs key starter exercises with explanations.

Usage:
    python scripts/beginner_tour.py
"""

import subprocess
import sys
from pathlib import Path


EXERCISES = [
    (3, 1, "Gradient Descent - Unified Optimizer"),
    (4, 1, "Linear Regression - Large Learning Rate"),
    (5, 1, "Multiple Regression - Third Feature"),
    (6, 1, "Regularization - Ridge Alpha Sweep"),
    (7, 1, "Logistic Regression - Threshold Exploration"),
]


def run_exercise(chapter: int, exercise: int, description: str) -> bool:
    """Run a starter exercise."""
    print(f"\n{'='*60}")
    print(f"  EXERCISE: Chapter {chapter}, Exercise {exercise}")
    print(f"  TOPIC: {description}")
    print(f"{'='*60}")
    
    # Show the starter file first
    topics_dir = Path("topics")
    topic_dirs = list(topics_dir.glob(f"ch{chapter:02d}_*"))
    if not topic_dirs:
        print(f"  Chapter {chapter} not found")
        return False
    
    topic_dir = topic_dirs[0]
    ex_dirs = list((topic_dir / "exercises").glob(f"ex{exercise:02d}_*"))
    if not ex_dirs:
        print(f"  Exercise {exercise} not found in chapter {chapter}")
        return False
    
    ex_dir = ex_dirs[0]
    starter = ex_dir / "starter.py"
    
    print(f"\nLocation: {ex_dir}")
    print(f"Starter: {starter.name}")
    print(f"\n--- Starter Code Preview (first 50 lines) ---")
    try:
        lines = starter.read_text().split('\n')
        for i, line in enumerate(lines[:50], 1):
            print(f"  {i:3}: {line}")
        if len(lines) > 50:
            print(f"  ... ({len(lines) - 50} more lines)")
    except Exception as e:
        print(f"  Could not read starter: {e}")
    
    # Ask to run
    print(f"\nRun this exercise? [Y/n]: ", end="")
    try:
        response = input().strip().lower()
        if response in ('n', 'no'):
            print("  Skipped")
            return True
    except KeyboardInterrupt:
        print("\n\nTour cancelled.")
        return False
    
    # Run it
    print(f"\nRunning: python scripts/run_exercise.py --chapter {chapter} --exercise {exercise} --variant starter")
    print("-" * 60)
    
    result = subprocess.run([
        sys.executable, "scripts/run_exercise.py",
        "--chapter", str(chapter),
        "--exercise", str(exercise),
        "--variant", "starter"
    ], capture_output=False)
    
    if result.returncode == 0:
        print(f"\nExercise completed successfully!")
    else:
        print(f"\nExercise failed (exit code: {result.returncode})")
        print("   This is expected for starters - they have TODOs to complete!")
        print("   Check the solution with: --variant solution")
    
    return True


def main():
    print("ML FROM SCRATCH - BEGINNER GUIDED TOUR")
    print("=" * 60)
    print("This tour walks you through 5 foundational exercises.")
    print("Each demonstrates a core concept with a starter file you can explore.")
    print("\nYou'll see the code, then run it (expecting TODO errors),")
    print("then you can compare with the solution.")
    
    input("\nPress Enter to begin...")
    
    completed = 0
    for chapter, exercise, description in EXERCISES:
        if run_exercise(chapter, exercise, description):
            completed += 1
        else:
            break
    
    print(f"\n{'='*60}")
    print(f"  TOUR COMPLETE: {completed}/{len(EXERCISES)} exercises explored")
    print(f"{'='*60}")
    
    print("""
YOUR NEXT STEPS:

1. PICK AN EXERCISE TO COMPLETE
   python scripts/run_exercise.py --chapter 3 --exercise 1 --variant starter
   # Edit the TODO sections, then run again

2. COMPARE WITH SOLUTION
   python scripts/run_exercise.py --chapter 3 --exercise 1 --variant solution

3. READ THE EXPLANATION
   cat topics/ch03_gradient_descent/exercises/ex01_unified_optimizer/README.md

4. CONTINUE THE PATH
   • Beginner: Ch3→Ch4→Ch5→Ch6→Ch7 (12 weeks)
   • See docs/LEARNING_PATHS.md for full schedule

5. SELF-ASSESS ANYTIME
   python scripts/self_assess.py
""")


if __name__ == "__main__":
    main()
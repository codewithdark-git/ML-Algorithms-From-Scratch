#!/usr/bin/env python
"""
Verify exercise coverage - checks that all 83 exercises are present exactly once.

Usage:
    python scripts/check_exercise_coverage.py
"""

import sys
from pathlib import Path


# Required files pattern - test file name depends on chapter and exercise
REQUIRED_FILES_BASE = ["starter.py", "solution.py", "README.md"]

EXPECTED_EXERCISES = {
    # Chapter 3: Gradient Descent (5 exercises)
    3: 5,
    # Chapter 4: Linear Regression (6 exercises)
    4: 6,
    # Chapter 5: Multiple/Polynomial Regression (6 exercises)
    5: 6,
    # Chapter 6: Regularized Regression (6 exercises)
    6: 6,
    # Chapter 7: Logistic Regression (7 exercises)
    7: 7,
    # Chapter 8: Naive Bayes (7 exercises)
    8: 7,
    # Chapter 9: SVM (7 exercises)
    9: 7,
    # Chapter 10: Decision Trees (6 exercises)
    10: 6,
    # Chapter 11: K-Means (6 exercises)
    11: 6,
    # Chapter 12: Gaussian Mixture Models (6 exercises)
    12: 6,
    # Chapter 13: Neural Networks (6 exercises)
    13: 6,
    # Chapter 14: PINN (7 exercises)
    14: 7,
    # Chapter 15: Pipelines/Production (7 exercises)
    15: 7,
    # Chapter 16: Production/Software Engineering (7 exercises)
    16: 7,
}

TOTAL_EXPECTED = 89


def check_coverage(topics_dir: Path):
    """Check exercise coverage."""
    errors = []
    warnings = []
    found_exercises = {}
    
    # Find all exercises
    for topic_dir in sorted(topics_dir.iterdir()):
        if not topic_dir.is_dir() or not topic_dir.name.startswith("ch"):
            continue
        
        try:
            ch_num = int(topic_dir.name[2:4])
        except ValueError:
            continue
        
        exercises_dir = topic_dir / "exercises"
        if not exercises_dir.exists():
            continue
        
        for ex_dir in sorted(exercises_dir.iterdir()):
            if not ex_dir.is_dir() or not ex_dir.name.startswith("ex"):
                continue
            
            try:
                ex_num = int(ex_dir.name[2:4])
            except ValueError:
                continue
            
            key = (ch_num, ex_num)
            if key in found_exercises:
                errors.append(f"Duplicate exercise: Chapter {ch_num}, Exercise {ex_num} found at {ex_dir} and {found_exercises[key]}")
            else:
                found_exercises[key] = ex_dir
            
            # Check required files
            for req_file in REQUIRED_FILES_BASE:
                if not (ex_dir / req_file).exists():
                    errors.append(f"Missing {req_file} in {ex_dir}")
            # Check test file with new naming pattern
            test_file = f"test_ch{ch_num:02d}_{ex_dir.name}.py"
            if not (ex_dir / test_file).exists():
                errors.append(f"Missing {test_file} in {ex_dir}")
    
    # Check expected counts
    chapter_counts = {}
    for (ch, ex) in found_exercises:
        chapter_counts[ch] = chapter_counts.get(ch, 0) + 1
    
    for ch, expected in EXPECTED_EXERCISES.items():
        actual = chapter_counts.get(ch, 0)
        if actual != expected:
            errors.append(f"Chapter {ch}: expected {expected} exercises, found {actual}")
    
    total_found = len(found_exercises)
    if total_found != TOTAL_EXPECTED:
        errors.append(f"Total exercises: expected {TOTAL_EXPECTED}, found {total_found}")
    
    return errors, warnings, total_found


def main():
    topics_dir = Path("topics")
    if not topics_dir.exists():
        print(f"Error: Topics directory '{topics_dir}' not found")
        return 1
    
    print("Checking exercise coverage...")
    print("=" * 60)
    
    errors, warnings, total = check_coverage(topics_dir)
    
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    
    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  - {e}")
        print(f"\nResult: FAILED ({len(errors)} errors)")
        return 1
    else:
        print("\nAll checks passed!")
        print(f"   Total exercises found: {total}")
        print(f"   Expected: {TOTAL_EXPECTED}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
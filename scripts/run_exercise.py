#!/usr/bin/env python
"""
Run an exercise (starter or solution).

Usage:
    python scripts/run_exercise.py --chapter 3 --exercise 1 --variant starter
    python scripts/run_exercise.py --chapter 3 --exercise 1 --variant solution
"""

import argparse
import importlib.util
import sys
from pathlib import Path


def run_exercise(chapter: int, exercise: int, variant: str, topics_dir: Path = Path("topics")):
    """Run a specific exercise variant."""
    
    # Find the exercise directory
    topic_dirs = list(topics_dir.glob(f"ch{chapter:02d}_*"))
    if not topic_dirs:
        print(f"Error: No topic directory found for chapter {chapter}")
        return 1
    
    topic_dir = topic_dirs[0]
    exercises_dir = topic_dir / "exercises"
    
    ex_dirs = list(exercises_dir.glob(f"ex{exercise:02d}_*"))
    if not ex_dirs:
        print(f"Error: No exercise directory found for chapter {chapter}, exercise {exercise}")
        return 1
    
    ex_dir = ex_dirs[0]
    
    # Determine file to run
    if variant == "starter":
        file_path = ex_dir / "starter.py"
    elif variant == "solution":
        file_path = ex_dir / "solution.py"
    else:
        print(f"Error: Unknown variant '{variant}'. Use 'starter' or 'solution'")
        return 1
    
    if not file_path.exists():
        print(f"Error: {variant}.py not found at {file_path}")
        return 1
    
    print(f"Running {variant} for Chapter {chapter}, Exercise {exercise}")
    print(f"File: {file_path}")
    print("-" * 60)
    
    # Load and execute the module
    spec = importlib.util.spec_from_file_location("exercise", file_path)
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
        print("-" * 60)
        print(f"{variant.capitalize()} executed successfully!")
        return 0
    except Exception as e:
        print(f"Error executing {variant}: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run an exercise")
    parser.add_argument("--chapter", "-c", type=int, required=True, help="Chapter number")
    parser.add_argument("--exercise", "-e", type=int, required=True, help="Exercise number")
    parser.add_argument("--variant", "-v", choices=["starter", "solution"], required=True, help="Variant to run")
    parser.add_argument("--topics-dir", default="topics", help="Path to topics directory")
    args = parser.parse_args()
    
    return run_exercise(args.chapter, args.exercise, args.variant, Path(args.topics_dir))


if __name__ == "__main__":
    sys.exit(main())
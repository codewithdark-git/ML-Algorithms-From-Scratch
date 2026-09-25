#!/usr/bin/env python
"""
List all exercises in the topics/ directory.

Usage:
    python scripts/list_exercises.py
    python scripts/list_exercises.py --chapter 3
    python scripts/list_exercises.py --format json
"""

import argparse
import json
from pathlib import Path


def find_exercises(topics_dir: Path, chapter: int = None):
    """Find all exercises in the topics directory."""
    exercises = []
    
    for topic_dir in sorted(topics_dir.iterdir()):
        if not topic_dir.is_dir():
            continue
        
        # Extract chapter number from directory name (e.g., ch03_gradient_descent)
        dir_name = topic_dir.name
        if not dir_name.startswith("ch"):
            continue
        
        try:
            ch_num = int(dir_name[2:4])
        except ValueError:
            continue
        
        if chapter is not None and ch_num != chapter:
            continue
        
        exercises_dir = topic_dir / "exercises"
        if not exercises_dir.exists():
            continue
        
        for ex_dir in sorted(exercises_dir.iterdir()):
            if not ex_dir.is_dir():
                continue
            
            # Extract exercise number from directory name (e.g., ex01_unified_optimizer)
            ex_name = ex_dir.name
            if not ex_name.startswith("ex"):
                continue
            
            try:
                ex_num = int(ex_name[2:4])
            except ValueError:
                continue
            
            # Check for required files
            # Test file follows pattern: test_ch{ch:02d}_{ex_slug}.py
            test_file = f"test_ch{ch_num:02d}_{ex_name}.py"
            files = {
                "starter": (ex_dir / "starter.py").exists(),
                "solution": (ex_dir / "solution.py").exists(),
                "test": (ex_dir / test_file).exists(),
                "readme": (ex_dir / "README.md").exists(),
            }
            
            slug = ex_name[5:] if len(ex_name) > 5 else ""
            
            exercises.append({
                "chapter": ch_num,
                "topic": dir_name[5:],
                "exercise": ex_num,
                "slug": slug,
                "name": slug.replace("_", " ").title(),
                "path": str(ex_dir.relative_to(topics_dir)),
                "files": files,
                "complete": all(files.values()),
            })
    
    return exercises


def main():
    parser = argparse.ArgumentParser(description="List all exercises")
    parser.add_argument("--chapter", "-c", type=int, help="Filter by chapter number")
    parser.add_argument("--format", "-f", choices=["table", "json", "csv"], default="table")
    parser.add_argument("--topics-dir", default="topics", help="Path to topics directory")
    parser.add_argument("--incomplete-only", action="store_true", help="Show only incomplete exercises")
    args = parser.parse_args()
    
    topics_dir = Path(args.topics_dir)
    if not topics_dir.exists():
        print(f"Error: Topics directory '{topics_dir}' not found")
        return 1
    
    exercises = find_exercises(topics_dir, args.chapter)
    
    if args.incomplete_only:
        exercises = [e for e in exercises if not e["complete"]]
    
    if args.format == "json":
        print(json.dumps(exercises, indent=2))
    elif args.format == "csv":
        print("chapter,exercise,slug,name,complete,path")
        for ex in exercises:
            print(f"{ex['chapter']},{ex['exercise']},{ex['slug']},{ex['name']},{ex['complete']},{ex['path']}")
    else:
        # Table format
        print(f"{'Ch':>3} {'Ex':>3} {'Slug':<30} {'Name':<40} {'Files':<15} {'Complete'}")
        print("-" * 100)
        for ex in exercises:
            file_status = "".join(["+" if v else "-" for v in ex["files"].values()])
            complete_str = "OK" if ex["complete"] else "NO"
            print(f"{ex['chapter']:>3} {ex['exercise']:>3} {ex['slug']:<30} {ex['name']:<40} {file_status:<15} {complete_str}")
        
        print(f"\nTotal: {len(exercises)} exercises")
        complete_count = sum(1 for e in exercises if e["complete"])
        print(f"Complete: {complete_count}/{len(exercises)}")
    
    return 0


if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python
"""
Regenerate all figures from versioned scripts.

Usage:
    python scripts/regenerate_figures.py
    python scripts/regenerate_figures.py --chapter 3
"""

import argparse
import importlib.util
import sys
from pathlib import Path


def find_figure_scripts(figures_dir: Path, chapter: int = None):
    """Find all figure generation scripts."""
    scripts = []
    
    pattern = "ch*_fig*.py" if chapter is None else f"ch{chapter:02d}_fig*.py"
    
    for script in figures_dir.glob(pattern):
        scripts.append(script)
    
    return sorted(scripts)


def run_figure_script(script_path: Path):
    """Execute a figure generation script."""
    print(f"Running {script_path.name}...")
    
    spec = importlib.util.spec_from_file_location("figure", script_path)
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
        print("  ✓ Generated figures")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Regenerate figures")
    parser.add_argument("--chapter", "-c", type=int, help="Regenerate only specific chapter")
    parser.add_argument("--figures-dir", default="Book/figures", help="Figures directory")
    args = parser.parse_args()
    
    figures_dir = Path(args.figures_dir)
    if not figures_dir.exists():
        print(f"Error: Figures directory not found at {figures_dir}")
        return 1
    
    scripts = find_figure_scripts(figures_dir, args.chapter)
    
    if not scripts:
        print("No figure scripts found")
        return 0
    
    print(f"Found {len(scripts)} figure scripts")
    print("=" * 60)
    
    success = 0
    failed = 0
    
    for script in scripts:
        if run_figure_script(script):
            success += 1
        else:
            failed += 1
    
    print("=" * 60)
    print(f"Results: {success} succeeded, {failed} failed")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
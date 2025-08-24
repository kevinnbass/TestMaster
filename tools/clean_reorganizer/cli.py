"""
Clean Reorganizer CLI
=====================

Usage (PowerShell):
  python -m tools.clean_reorganizer.cli --mode preview
  python -m tools.clean_reorganizer.cli --mode apply

Modes:
  preview  - build and save plan only
  apply    - execute plan with moves and rollback script
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .planner import Planner
from .executor import Executor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Reorganizer")
    parser.add_argument("--root", type=str, default=".", help="project root directory")
    parser.add_argument("--mode", type=str, choices=["preview", "apply"], default="preview")
    parser.add_argument("--config", type=str, default="", help="optional config path")
    parser.add_argument('--include-cats', type=str, default='', help='comma-separated categories to include')
    parser.add_argument('--include-subcats', type=str, default='', help='comma-separated subcategories to include')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    cfg: Optional[Path] = Path(args.config).resolve() if args.config else None

    planner = Planner(root, cfg)
    plan = planner.build_plan()
    
    # Filter moves by category/subcategory if specified
    include_cats = set([s.strip() for s in (args.include_cats or '').split(',') if s.strip()])
    include_subcats = set([s.strip() for s in (args.include_subcats or '').split(',') if s.strip()])
    
    if include_cats or include_subcats:
        filtered = []
        for m in plan.moves:
            if include_cats and m.category not in include_cats:
                continue
            # infer subcategory from target path parts
            parts = str(m.target).replace('\\','/').split('/')
            sub = parts[parts.index('organized_codebase')+2] if 'organized_codebase' in parts and len(parts) > parts.index('organized_codebase')+2 else ''
            if include_subcats and sub not in include_subcats:
                continue
            filtered.append(m)
        plan.moves = filtered
        plan.summary.total = len(filtered)
        # Recalculate category counts
        from collections import defaultdict
        counts = defaultdict(int)
        for m in filtered:
            counts[m.category] += 1
        plan.summary.by_category = dict(counts)

    plan_path = root / "tools" / "clean_reorganizer" / "plan.json"
    planner.save_plan(plan, plan_path)

    print("\n=== CLEAN REORGANIZER PLAN ===")
    print(f"Total moves: {plan.summary.total}")
    for cat, count in sorted(plan.summary.by_category.items()):
        print(f"  {cat}: {count}")
    print(f"Plan saved: {plan_path}")

    execu = Executor(root)
    apply = args.mode == "apply"
    result = execu.execute(plan, apply=apply)

    print("\n=== EXECUTION ===")
    print(f"Executed: {result.executed}")
    print(f"Failed:   {result.failed}")
    if result.rollback_script:
        print(f"Rollback: {result.rollback_script}")
    if result.errors:
        print("Errors:")
        for e in result.errors[:10]:
            print(f"  - {e}")

    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())



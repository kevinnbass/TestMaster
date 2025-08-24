"""
Clean Reorganizer: Executor
===========================

Executes a plan with move operations and rollback scripting.
Default mode is preview-only; moves must be explicitly enabled by the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import shutil

from .planner import Plan, PlanItem


@dataclass
class ExecutionResult:
    executed: int
    failed: int
    errors: List[str]
    rollback_script: Optional[Path]


class Executor:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()

    def write_rollback(self, plan: Plan, out_path: Path) -> None:
        lines: List[str] = []
        for item in plan.moves[:10000]:
            # Inverse operation: move target back to source
            src = str(item.target)
            dst = str(item.source)
            lines.append(f"Move-Item -LiteralPath \"{src}\" -Destination \"{dst}\" -Force")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")

    def execute(self, plan: Plan, apply: bool = False) -> ExecutionResult:
        executed = 0
        failed = 0
        errors: List[str] = []

        # Always emit rollback script
        rollback = self.root_dir / "tools" / "clean_reorganizer" / "rollback.ps1"
        try:
            self.write_rollback(plan, rollback)
        except Exception as e:
            errors.append(f"failed to write rollback: {e}")

        if not apply:
            return ExecutionResult(executed=0, failed=0, errors=errors, rollback_script=rollback)

        for item in plan.moves[:10000]:
            try:
                item.target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item.source), str(item.target))
                executed += 1
            except Exception as e:
                failed += 1
                errors.append(f"move failed: {item.source} -> {item.target}: {e}")

        return ExecutionResult(executed=executed, failed=failed, errors=errors, rollback_script=rollback)



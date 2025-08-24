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
from .web_rewriter import rewrite_web_imports
import ast
import re
import json


@dataclass
class ExecutionResult:
    executed: int
    failed: int
    errors: List[str]
    rollback_script: Optional[Path]
    updated_imports: int = 0
    inits_created: int = 0
    duplicates_removed: int = 0


class Executor:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        # stats
        self._updated_imports = 0
        self._inits_created = 0
        self._duplicates_removed = 0

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

        # Deduplicate by content hash among targets; if two targets have same hash and identical content, keep first
        seen_hashes: Dict[str, Path] = {}
        for item in plan.moves[:10000]:
            try:
                item.target.parent.mkdir(parents=True, exist_ok=True)
                if item.analysis.file_hash and item.analysis.file_hash in seen_hashes:
                    # Skip move; file content duplicate exists; record as duplicate
                    self._duplicates_removed += 1
                    continue
                shutil.move(str(item.source), str(item.target))
                executed += 1
                if item.analysis.file_hash:
                    seen_hashes[item.analysis.file_hash] = item.target
            except Exception as e:
                failed += 1
                errors.append(f"move failed: {item.source} -> {item.target}: {e}")

        # Post-move: create minimal __init__.py in new packages where needed
        self._inits_created += self._ensure_package_inits(plan)

        # AST-based import updater (best-effort): only update files that were moved into organized_codebase
        self._updated_imports += self._update_imports_ast(plan)
        
        # Optional web import updates
        try:
            cfg = json.loads((self.root_dir / 'tools' / 'clean_reorganizer' / 'config' / 'config.json').read_text(encoding='utf-8'))
            if cfg.get('operations', {}).get('update_web_imports', False):
                moved_targets = [item.target for item in plan.moves[:10000]]
                self._updated_imports += rewrite_web_imports(self.root_dir, moved_targets)
            # Optional: remove duplicate sources that were skipped
            if cfg.get('operations', {}).get('remove_duplicate_sources', False):
                self._duplicates_removed += self._remove_duplicate_sources(plan, seen_hashes)
        except Exception:
            pass

        return ExecutionResult(
            executed=executed,
            failed=failed,
            errors=errors,
            rollback_script=rollback,
            updated_imports=self._updated_imports,
            inits_created=self._inits_created,
            duplicates_removed=self._duplicates_removed,
        )

    def _ensure_package_inits(self, plan: Plan) -> int:
        created = 0
        for item in plan.moves[:10000]:
            p = item.target.parent
            try:
                if (p.exists() and p.is_dir()):
                    init_path = p / "__init__.py"
                    if not init_path.exists():
                        init_path.write_text("", encoding="utf-8")
                        created += 1
            except Exception:
                continue
        return created

    def _update_imports_ast(self, plan: Plan) -> int:
        """Update import statements that refer to moved modules.

        Strategy: for each moved file, compute its new module path under organized_codebase and rewrite
        import statements in other moved files accordingly. Scope limited to moved files to avoid touching
        external areas on first pass.
        """
        # Prefer full old module path mapping; fall back to basename only
        mapping: Dict[str, str] = {}
        basename_fallback: Dict[str, str] = {}
        root = self.root_dir / "organized_codebase"
        
        # Load origin manifest to access old_module
        import json
        origin_path = self.root_dir / "tools" / "clean_reorganizer" / "origin_manifest.json"
        origins = []
        try:
            if origin_path.exists():
                origins = json.loads(origin_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        
        for m in origins[:50000]:
            tgt = m.get("target", "")
            old_mod = m.get("old_module", "")
            if not tgt:
                continue
            new_mod = str(Path(tgt).with_suffix("")).replace("\\", ".").replace("/", ".")
            if old_mod:
                mapping[old_mod] = new_mod
            # Also record basename fallback
            base = Path(tgt).stem
            if base:
                basename_fallback[base] = new_mod

        updated = 0
        for item in plan.moves[:10000]:
            path = item.target
            if path.suffix.lower() != ".py":
                continue
            try:
                src = path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(src)
            except Exception:
                continue

            changed = False
            lines = src.splitlines()
            for i, line in enumerate(lines[:5000]):
                l = line.strip()
                if l.startswith("import ") or l.startswith("from "):
                    # Exact module match first
                    for old, new in list(mapping.items())[:500]:
                        if re.search(rf"\b{re.escape(old)}\b", l):
                            new_line = line
                            if l.startswith("from "):
                                new_line = re.sub(rf"\b{re.escape(old)}\b", new, line)
                            elif l.startswith("import "):
                                # keep last token imported as alias if present
                                tail = new.split('.')[-1]
                                new_line = re.sub(rf"\b{re.escape(old)}\b", tail, line)
                            if new_line != line:
                                lines[i] = new_line
                                changed = True
                                break
                    if changed:
                        continue
                    # Basename fallback
                    for base, new_mod in list(basename_fallback.items())[:500]:
                        if re.search(rf"\b{re.escape(base)}\b", l):
                            new_line = re.sub(rf"\b{re.escape(base)}\b", new_mod.split('.')[-1] if l.startswith("import ") else new_mod, line)
                            if new_line != line:
                                lines[i] = new_line
                                changed = True
                                break
            if changed:
                try:
                    path.write_text("\n".join(lines), encoding="utf-8")
                    updated += 1
                except Exception:
                    continue
        return updated
    
    def _remove_duplicate_sources(self, plan: Plan, seen_hashes: Dict[str, Path]) -> int:
        """Optional: remove duplicate sources that were skipped"""
        removed = 0
        for item in plan.moves[:10000]:
            if item.analysis.file_hash and item.analysis.file_hash in seen_hashes:
                try:
                    if item.source.exists():
                        item.source.unlink()
                        removed += 1
                except Exception:
                    continue
        return removed



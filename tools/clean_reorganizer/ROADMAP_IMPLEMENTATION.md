Clean Reorganizer – Exhaustive Implementation Roadmap
=====================================================

Purpose: Provide step-by-step, precise edits to extend coverage and robustness across the entire codebase reorganization, suitable for execution by Claude Code.

Scope already implemented and validated
- Broad file inclusion (.py, .md/.txt, .ps1/.bat/.cmd/.sh, .ipynb, .json/.yaml/.toml/.ini, .csv/.tsv/.db/.sqlite/.parquet/.sql, .log, common images/fonts, .js/.ts/.tsx/.jsx/.html/.css/.scss)
- Categorization + subcategories + cluster grouping
- Manifests: plan.json, origin_manifest.json, relationships.json, duplicates.json, import_resolve_report.json
- Safe apply (Python): deduplication skip, __init__.py creation, AST import updater (moved Python files only)
- Preview/dry-run with rollback script

This roadmap focuses on:
- A) Reduce unresolved imports and improve mapping
- B) JS/TS rewriter (preview-ready now; enable on demand)
- C) Batch applies by domain/subcategory
- D) Robustness, logging, and test harness


Step 1 — Improve import mapping (avoid basename collisions)
----------------------------------------------------------
Goal: Map old dotted module paths → new dotted paths, not just basenames.

1.1 Edit `tools/clean_reorganizer/planner.py` to compute each file’s “old dotted module path” for Python.

Add helper near other private helpers:
```python
def _old_python_module_path(self, facts: FileFacts) -> str:
    rel = str(facts.path.relative_to(self.root_dir)).replace('\\', '/').rstrip('/')
    if rel.endswith('.py'):
        rel = rel[:-3]
    # Drop common non-package roots if desired (optional tuning)
    return rel.replace('/', '.')
```

1.2 In `Planner.build_plan`, when assembling `origins`, add `old_module` for Python files:
```python
old_mod = self._old_python_module_path(facts) if facts.path.suffix.lower() == '.py' else ''
origins.append({
    "source": rel,
    "target": str(target),
    "category": decision.category,
    "cluster": cluster_name,
    "confidence": decision.confidence,
    "old_module": old_mod
})
```

Step 2 — Strengthen Python AST import updater to use full-path mapping
----------------------------------------------------------------------
File: `tools/clean_reorganizer/executor.py`

2.1 Build a mapping old_module → new_module from `origin_manifest.json` items that have `old_module` set.

Replace inside `_update_imports_ast` mapping construction:
```python
# Build mapping from old module basename -> new dotted path
mapping: Dict[str, str] = {}
root = self.root_dir / "organized_codebase"
for item in plan.moves[:10000]:
    try:
        rel = item.target.relative_to(root)
        new_mod = str(rel.with_suffix("")).replace("\\", ".").replace("/", ".")
        basename = item.source.stem
        mapping[basename] = new_mod
    except Exception:
        continue
```

with this:
```python
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
```

2.2 Use full-path mapping when rewriting import lines; only fallback to basename if full path not found:
```python
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
```

Step 3 — JS/TS import rewriter (separate pass; configurable)
------------------------------------------------------------
3.1 Add `tools/clean_reorganizer/web_rewriter.py`:
```python
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json, re

def _load_origins(root: Path) -> List[dict]:
    p = root / "tools" / "clean_reorganizer" / "origin_manifest.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _build_web_map(root: Path) -> Dict[str, str]:
    """Map old relative specifiers (best-effort) -> new specifiers.
    For now, we only handle local (./ or ../) style specifiers by name."""
    m: Dict[str, str] = {}
    origins = _load_origins(root)
    for o in origins[:50000]:
        src = o.get("source", "")
        tgt = o.get("target", "")
        if not src or not tgt:
            continue
        # Heuristic: map basename to new relative-like token; real path resolution left for future
        base = Path(src).stem
        new_base = Path(tgt).with_suffix("").name
        if base and new_base:
            m[base] = new_base
    return m

def rewrite_web_imports(root: Path, moved_targets: List[Path]) -> int:
    mapping = _build_web_map(root)
    updated = 0
    for path in moved_targets[:20000]:
        if path.suffix.lower() not in {'.js', '.ts', '.tsx', '.jsx', '.css', '.scss', '.html'}:
            continue
        try:
            src = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        lines = src.splitlines()
        changed = False
        for i, line in enumerate(lines[:10000]):
            l = line.strip()
            m = re.search(r"import\s+(?:[\w\{\},\s\*]+\s+from\s+)?['\"]([^'\"]+)['\"]", l)
            if not m:
                continue
            spec = m.group(1)
            base = Path(spec).stem
            if base in mapping:
                new_spec = spec.replace(base, mapping[base])
                lines[i] = line.replace(spec, new_spec)
                changed = True
        if changed:
            try:
                path.write_text("\n".join(lines), encoding='utf-8')
                updated += 1
            except Exception:
                continue
    return updated
```

3.2 Wire optional web rewrite in `tools/clean_reorganizer/executor.py` after Python updates:
```python
from .web_rewriter import rewrite_web_imports  # add at top
```

In `execute()` just before returning the `ExecutionResult`:
```python
# Optional web import updates
try:
    cfg = json.loads((self.root_dir / 'tools' / 'clean_reorganizer' / 'config' / 'config.json').read_text(encoding='utf-8'))
    if cfg.get('operations', {}).get('update_web_imports', False):
        moved_targets = [item.target for item in plan.moves[:10000]]
        self._updated_imports += rewrite_web_imports(self.root_dir, moved_targets)
except Exception:
    pass
```

Step 4 — Batch apply by category/subcategory
--------------------------------------------
4.1 Update CLI `tools/clean_reorganizer/cli.py` to accept include filters:
```python
parser.add_argument('--include-cats', type=str, default='', help='comma-separated categories to include')
parser.add_argument('--include-subcats', type=str, default='', help='comma-separated subcategories to include')
```

4.2 Filter moves before execution in CLI:
```python
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
```

Step 5 — Reduce unresolved imports via targeted category tuning
----------------------------------------------------------------
Use `tools/clean_reorganizer/import_resolve_report.json` → `top_unresolved_bases` to refine config.

5.1 Edit `tools/clean_reorganizer/config/config.json`:
- For frequent unresolved names that are actually domain hints (e.g., `scheduler`, `router`, `metrics`), add to relevant category `keywords`/`path_patterns`.
- Avoid adding Python stdlib tokens (`typing`, `datetime`, etc.).

Example additions:
```json
{
  "categories": {
    "core/orchestration": {
      "keywords": ["orchestrator", "workflow", "agent", "scheduler", "router"],
      "path_patterns": [".*orchestrat.*", ".*workflow.*", ".*agent.*", ".*schedule.*", ".*router.*"]
    },
    "monitoring": {
      "keywords": ["monitor", "metric", "alert", "telemetry", "tracing"],
      "path_patterns": [".*metric.*", ".*telemetry.*", ".*trace.*"]
    }
  }
}
```

Step 6 — Import resolve dry-run enhancement (optional)
------------------------------------------------------
Add a resolver script to attempt actual dynamic imports of moved Python modules in an isolated process (no execution of module code beyond import).

6.1 Create `tools/clean_reorganizer/resolver.py`:
```python
from __future__ import annotations
from pathlib import Path
import json, sys, importlib

def main() -> int:
    root = Path.cwd()
    plan_path = root / 'tools' / 'clean_reorganizer' / 'origin_manifest.json'
    if not plan_path.exists():
        print('origin_manifest.json missing')
        return 1
    items = json.loads(plan_path.read_text(encoding='utf-8'))
    # Add target root to sys.path
    tgt_root = root / 'organized_codebase'
    sys.path.insert(0, str(tgt_root))
    failed = []
    checked = 0
    for it in items[:5000]:
        tgt = it.get('target', '')
        if not tgt.endswith('.py'):
            continue
        mod = str(Path(tgt).with_suffix('')).replace('\\','/').split('organized_codebase/')[-1].replace('/', '.')
        try:
            importlib.import_module(mod)
            checked += 1
        except Exception as e:
            failed.append((mod, str(e)))
    out = root / 'tools' / 'clean_reorganizer' / 'import_dynamic_report.json'
    out.write_text(json.dumps({
        'checked': checked,
        'failed': failed[:200]
    }, indent=2), encoding='utf-8')
    print(f'Checked={checked}, Failed={len(failed)}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
```

Step 7 — Duplicates apply-time policy (optional)
-----------------------------------------------
Current: duplicates are skipped when moving. Optional: remove duplicates in source tree post-apply.

7.1 In `tools/clean_reorganizer/executor.py`, after moves, add:
```python
# Optional: remove duplicate sources that were skipped
def _remove_duplicate_sources(self, plan: Plan, seen_hashes: Dict[str, Path]) -> int:
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
```

Invoke it under a new config flag (e.g., `operations.remove_duplicate_sources`) guarded by try/except.

Step 8 — CLI recipes for staged apply
-------------------------------------
- Docs only:
```powershell
python -m tools.clean_reorganizer.cli --mode apply --include-cats documentation
```
- Data + Scripts:
```powershell
python -m tools.clean_reorganizer.cli --mode apply --include-cats data,scripts
```
- Selected Python clusters (example: orchestration):
```powershell
python -m tools.clean_reorganizer.cli --mode apply --include-cats core/orchestration
```
Keep `operations.update_web_imports=false` until the JS/TS rewriter is vetted.

Step 9 — Logging and performance (optional)
-------------------------------------------
- Increase `performance.max_workers` and `bounds.max_files` as needed.
- Consider chunking manifest writes for very large repos.

Step 10 — Update documentation
-------------------------------
- Update main `README.md` to document CLI, manifests, category/subcategory structure, and staged apply procedures.

Step 11 — Windows long-path support (optional)
----------------------------------------------
- If needed, enable long paths in Windows Group Policy or registry to exceed MAX_PATH.

Validation checklist
--------------------
1) Run preview
```powershell
python -m tools.clean_reorganizer.cli --mode preview
```
2) Verify manifests: `origin_manifest.json`, `relationships.json`, `duplicates.json`, `import_resolve_report.json`
3) Optionally run dynamic import test (Python only)
```powershell
python tools\clean_reorganizer\resolver.py
```
4) Apply safe subset (docs/data/scripts) and confirm rollback script is present
5) Inspect `organized_codebase` structure for subcategories and clusters (`cc_*`)
6) For Python batches, import critical modules from `organized_codebase` path

Notes
-----
- Keep web moves preview-only until `update_web_imports=true` is validated.
- Avoid over-fitting config; prefer resolving unresolved bases via real module mapping and cluster grouping.



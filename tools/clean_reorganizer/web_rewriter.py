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
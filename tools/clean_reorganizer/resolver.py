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
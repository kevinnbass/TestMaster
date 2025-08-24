Clean Reorganizer
=================

A small, testable core for codebase reorganization with:
- Bounded traversal and AST-based analysis
- Config-driven categorization
- Previewable plan generation
- Safe execution with rollback script

Quick start (PowerShell):
```
python -m tools.clean_reorganizer.cli --mode preview
```

Then to apply:
```
python -m tools.clean_reorganizer.cli --mode apply
```

Notes:
- Keeps a single canonical tree under `organized_codebase/`.
- Does not rewrite imports yet; preview and review first.
- Update main README.md with new usage and rationale.



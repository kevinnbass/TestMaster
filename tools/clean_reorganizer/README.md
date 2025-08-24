# Clean Reorganizer

Intelligent, configurable codebase reorganization with import resolution.

## Features

- **Broad file support**: Python, JS/TS, documentation, configs, data files, assets
- **Smart categorization**: ML-driven categorization with subcategories and clustering
- **Import resolution**: Automatic Python import updates, optional JS/TS rewriting
- **Batch operations**: Apply by category or subcategory for staged reorganization
- **Duplicate handling**: Content-based deduplication with optional source removal
- **Rollback support**: Generated PowerShell scripts for easy rollback
- **Extensive manifests**: origin_manifest.json, relationships.json, duplicates.json, import_resolve_report.json

## Quick Start

### Preview Mode (Default)
```powershell
python -m tools.clean_reorganizer.cli --mode preview
```

### Apply Changes
```powershell
python -m tools.clean_reorganizer.cli --mode apply
```

### Batch Apply by Category
```powershell
# Documentation only
python -m tools.clean_reorganizer.cli --mode apply --include-cats documentation

# Multiple categories
python -m tools.clean_reorganizer.cli --mode apply --include-cats data,scripts,assets

# Core orchestration components
python -m tools.clean_reorganizer.cli --mode apply --include-cats core/orchestration
```

### Batch Apply by Subcategory
```powershell
# Specific subcategories
python -m tools.clean_reorganizer.cli --mode apply --include-subcats agents,workflows

# Combined category and subcategory filtering
python -m tools.clean_reorganizer.cli --mode apply --include-cats frontend --include-subcats web,assets
```

## Configuration

Edit `tools/clean_reorganizer/config/config.json` to customize:

### Categories
Define categories with keywords, class patterns, and path patterns:
```json
{
  "categories": {
    "core/orchestration": {
      "keywords": ["orchestrator", "workflow", "agent", "scheduler", "router"],
      "class_patterns": [".*Orchestrator.*", ".*Agent.*"],
      "path_patterns": [".*orchestrat.*", ".*workflow.*", ".*agent.*"]
    }
  }
}
```

### Operations
Control reorganization behavior:
```json
{
  "operations": {
    "create_backups": true,
    "update_imports": false,           // Python AST-based import updates
    "update_web_imports": false,       // JS/TS import rewriting
    "remove_duplicate_sources": false, // Remove duplicate source files after move
    "validate_after": true
  }
}
```

## Manifests

After running, check these generated manifests:

### origin_manifest.json
Maps source paths to target paths with categorization:
- `source`: Original file path
- `target`: New organized path
- `category`: Assigned category
- `cluster`: Cluster group (cc_N for related files)
- `confidence`: Categorization confidence score
- `old_module`: Original Python module path (for imports)

### relationships.json
Captures file relationships:
- `file`: File path
- `imports`: Python imports detected
- `classes`: Class definitions found
- `functions`: Function definitions found

### duplicates.json
Lists files with identical content:
- `hash`: Content hash
- `sources`: List of duplicate file paths

### import_resolve_report.json
Import resolution analysis:
- `total_imports`: Total import statements found
- `unresolved`: Count of unresolved imports
- `top_unresolved_bases`: Most common unresolved module names
- `unresolved_by_file`: File-specific unresolved imports

## Import Resolution

### Python Import Updates
The reorganizer automatically updates Python imports when `update_imports` is enabled:
1. Maps old module paths to new organized paths
2. Updates import statements in moved files
3. Prefers full module path matching over basename matching

### JS/TS Import Updates (Experimental)
Enable with `update_web_imports: true` in config:
1. Maps old specifiers to new ones
2. Updates import/require statements
3. Currently best-effort, preview recommended

### Dynamic Import Validation
Test import resolution after reorganization:
```powershell
python tools\clean_reorganizer\resolver.py
```
This attempts to import moved Python modules and reports failures.

## Directory Structure

After reorganization:
```
organized_codebase/
├── core/
│   ├── intelligence/      # ML/AI components
│   ├── orchestration/      # Workflow and agent coordination
│   │   ├── agents/        # Agent implementations
│   │   ├── workflows/     # Pipeline definitions
│   │   └── cc_N/          # Clustered related files
│   └── security/          # Security and compliance
├── frontend/              # UI components
├── monitoring/            # Metrics and telemetry
├── testing/              # Test files
├── documentation/        # Docs and guides
├── configuration/        # Config files
├── utilities/           # Helper functions
├── data/               # Data files
├── scripts/            # Automation scripts
└── assets/             # Images, fonts, etc.
```

## Clustering

Files with strong relationships (imports, similar names) are grouped into clusters:
- Cluster naming: `cc_N` where N is the cluster ID
- Minimum cluster size: 3 related files
- Only applies to files with confidence >= 0.6

## Rollback

A PowerShell rollback script is generated for each apply operation:
```powershell
.\rollback.ps1
```

This script reverses all file moves and restores the original structure.

## Validation Checklist

1. **Run preview first**
   ```powershell
   python -m tools.clean_reorganizer.cli --mode preview
   ```

2. **Review manifests**
   - Check `origin_manifest.json` for correct categorization
   - Review `duplicates.json` for unintended duplicates
   - Examine `import_resolve_report.json` for import issues

3. **Test with safe categories**
   ```powershell
   python -m tools.clean_reorganizer.cli --mode apply --include-cats documentation,data,assets
   ```

4. **Run dynamic import test** (Python only)
   ```powershell
   python tools\clean_reorganizer\resolver.py
   ```

5. **Verify structure**
   - Check `organized_codebase/` for proper organization
   - Confirm clusters (cc_*) contain related files
   - Validate subcategory structure

## Advanced Usage

### Custom Config
```powershell
python -m tools.clean_reorganizer.cli --config path/to/custom/config.json
```

### Performance Tuning
Adjust in config.json:
```json
{
  "bounds": {
    "max_files": 50000,
    "max_directories": 20000
  }
}
```

## Troubleshooting

### Import Resolution Issues
- Check `import_resolve_report.json` for unresolved imports
- Add missing keywords to category definitions in config
- Enable `update_imports` for Python files

### Duplicate Files
- Review `duplicates.json` for content duplicates
- Enable `remove_duplicate_sources` to clean up originals
- Duplicates are automatically skipped during moves

### Windows Long Path Support
If encountering path length errors:
1. Enable long paths in Windows
2. Use `git config --global core.longpaths true`

## Notes

- Keeps a single canonical tree under `organized_codebase/`
- Does not modify files outside the target directory
- Preserves file content and encoding
- Creates `__init__.py` files for Python packages
- Respects exclusion patterns in config
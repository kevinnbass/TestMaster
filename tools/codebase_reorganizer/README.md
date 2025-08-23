# Intelligent Codebase Reorganization System

A comprehensive Python codebase reorganization tool that intelligently preserves relationships and maintains semantic context while improving organization.

## Key Innovation: Relationship-Aware Reorganization

**Traditional Approach**: Analyze individual files → categorize broadly → move everything
```
Result: core/intelligence/neural_network.py
        core/intelligence/training.py
        core/intelligence/inference.py  ← All ML files dumped together
```

**Intelligent Approach**: Analyze directory structures as units → preserve relationships → reorganize thoughtfully
```
Result: core/intelligence/neural_network.py      ← Related files stay together
        core/intelligence/training.py
        core/intelligence/inference.py
        + subdirectory relationships maintained
        + semantic context preserved
```

## System Architecture

### Core Components

1. **`intelligent_reorganizer.py`** - Main intelligent reorganization engine
2. **`reorganizer_engine.py`** - Core file analysis and categorization
3. **`validation_module.py`** - Comprehensive validation and bounds checking
4. **`launcher.py`** - Clean launcher system with proper error handling
5. **`config.json`** - Configuration with exclusions and categories

### Key Features

- **Relationship Preservation**: Maintains meaningful subdirectory hierarchies
- **Smart Categorization**: Analyzes file content and context for accurate placement
- **Import Dependency Tracking**: Understands how modules relate to each other
- **Safe Operations**: Comprehensive validation and error handling
- **Configurable Exclusions**: Skip research repos, archives, and system files
- **Backup System**: Creates backups before making changes

## Installation & Usage

### Quick Start

1. **Run the launcher**:
   ```bash
   python tools/codebase_reorganizer/launcher.py
   ```

2. **Or use the simple runner**:
   ```bash
   python tools/codebase_reorganizer/run_reorganizer.py
   ```

3. **Run intelligent reorganization**:
   ```bash
   python tools/codebase_reorganizer/demo_intelligent_reorg.py
   ```

### Configuration

Edit `config.json` to customize:
- **Research repo exclusions**: Add repositories to skip
- **System directory exclusions**: Control what gets ignored
- **Category definitions**: Define how files are categorized

## How It Works

### 1. Intelligent Analysis
- Analyzes directory structures as meaningful units
- Tracks import relationships between modules
- Preserves semantic context and relationships

### 2. Smart Categorization
- Content-based analysis of Python files
- Keyword and pattern matching
- Confidence scoring for categorization decisions

### 3. Safe Reorganization
- Creates backups before making changes
- Validates all operations
- Comprehensive error handling
- Preserves existing well-organized structures

### 4. Relationship Preservation
- Maintains subdirectory hierarchies that make sense
- Keeps related modules together
- Respects existing package boundaries
- Only reorganizes when it adds clear value

## Directory Structure After Reorganization

```
organized_codebase/
├── core/
│   ├── intelligence/           # AI/ML modules
│   ├── orchestration/          # Workflow coordination
│   ├── security/               # Authentication & security
│   ├── services/               # Core services
│   └── foundation/             # Base utilities
├── monitoring/                 # Dashboards & metrics
├── testing/                    # Test suites
├── deployment/                 # Deployment tools
├── documentation/              # Docs & guides
└── utilities/                  # General utilities
```

## Configuration Options

### Exclusion Categories

- **Research Repositories**: Skip cloned research projects
- **System Directories**: Ignore `__pycache__`, `.git`, etc.
- **Archive Directories**: Skip backup and archive folders
- **Test Files**: Optionally reorganize test files separately

### Category Definitions

Each category has:
- **Keywords**: Content patterns to match
- **Priority**: Order of matching
- **Description**: Human-readable explanation

## Validation & Error Handling

The system includes comprehensive validation:
- **File Size Limits**: Prevents processing extremely large files
- **Path Length Checks**: Ensures filesystem compatibility
- **Import Validation**: Verifies module dependencies
- **Bounds Checking**: Fixed limits on all data structures
- **Error Recovery**: Graceful handling of failures

## Backup & Recovery

- **Automatic Backups**: Created before any file moves
- **Incremental Operations**: Process files one by one
- **Error Recovery**: Can restore from backup on failure
- **Logging**: Comprehensive operation logs

## Performance Considerations

- **Memory Bounded**: Fixed memory usage limits
- **File Size Limits**: Skip extremely large files
- **Processing Limits**: Configurable maximum files to process
- **Time Limits**: Execution time bounds for reliability

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure write access to target directories
2. **Memory Issues**: Reduce `MAX_FILES_TO_PROCESS` in config
3. **Long Execution**: Check for very large files or directories
4. **Import Errors**: Verify Python environment is properly set up

### Logs & Debugging

- Check `logs/` directory for detailed operation logs
- Review `backups/` for any created backup files
- Run with verbose mode for detailed output

## Contributing

To extend the system:

1. **Add Categories**: Update `config.json` with new categories
2. **Improve Analysis**: Enhance keyword matching in analysis modules
3. **Add Validations**: Extend validation checks in `validation_module.py`
4. **Optimize Performance**: Improve file processing efficiency

## Version History

- **v3.0**: Intelligent reorganization with relationship preservation
- **v2.0**: Core reorganization engine with validation
- **v1.0**: Basic file categorization and moving

---

**Note**: This system is designed to be safe and conservative. It will never delete files, always creates backups, and includes comprehensive validation at every step.
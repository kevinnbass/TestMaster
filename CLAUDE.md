# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Auto-commit Policy

**IMPORTANT**: After completing any coding task that modifies files:
1. Automatically run `git add .`
2. Create a descriptive commit message based on the changes made
3. Commit using `git commit -m "message"`
4. Push to GitHub origin with `git push origin main` (or current branch)

This should happen WITHOUT asking for permission, as part of the natural workflow after completing each task. All changes must be pushed to the GitHub remote repository, not just committed locally.

## Commands

### Running Tests
```bash
# Run all tests with coverage
python3 -m pytest --cov=. --cov-report=html

# Run a specific test file
python3 -m pytest path/to/test_file.py

# Run tests with verbose output
python3 -m pytest -v

# Run intelligent test suite
python3 run_intelligent_tests.py

# Quick test summary
python3 quick_test_summary.py
```

### Test Generation
```bash
# Generate tests for a single module
python3 intelligent_test_builder.py --module path/to/module.py

# Generate tests for all modules in a directory
python3 intelligent_test_builder.py --directory path/to/modules/

# Fix broken tests automatically
python3 enhanced_self_healing_verifier.py --fix path/to/broken_test.py

# Batch fix all tests
python3 enhanced_self_healing_verifier.py --batch-all

# Generate integration tests
python3 integration_test_generator.py
```

### Monitoring & Coverage
```bash
# Monitor for changes continuously (2-hour intervals)
python3 agentic_test_monitor.py --mode continuous --interval 120

# Run after idle (perfect for breaks)
python3 agentic_test_monitor.py --mode after-idle --idle 10

# Monitor progress to 100% coverage
python3 monitor_to_100.py

# Quick coverage analysis
python3 scripts/measure_final_coverage.py
```

### Parallel Processing
```bash
# Convert multiple tests in parallel
python3 parallel_converter.py --input modules.txt --workers 4

# Accelerated conversion with caching
python3 accelerated_converter.py --batch --cache

# Turbo converter with optimizations
python3 turbo_converter.py
```

### Import & Error Fixing
```bash
# Fix import paths in tests
python3 fix_import_paths.py

# Fix failing tests
python3 fix_failing_tests.py

# Fix all imports batch mode
python3 scripts/fix_all_imports.py
```

## Architecture

### Core Test Generation Pipeline
The system follows a multi-layer approach:
1. **Initial Generation** - Creates syntactically correct tests using Gemini AI
2. **Self-Healing** - Automatically fixes syntax/import errors (max 5 iterations)
3. **Verification** - Scores test quality and suggests improvements
4. **Enhancement** - Refines tests based on verification feedback

### Key Components

**Test Generators:**
- `intelligent_test_builder.py` - Main intelligent test generator with self-healing
- `intelligent_test_builder_v2.py` - Enhanced version with better error handling
- `enhanced_context_aware_test_generator.py` - Analyzes code context for better tests
- `specialized_test_generators.py` - Domain-specific test generation

**Self-Healing System:**
- `enhanced_self_healing_verifier.py` - Automatic test repair with 5 iteration limit
- `independent_test_verifier.py` - Quality verification and scoring (0-100)
- Handles syntax errors, import issues, and incomplete code automatically

**Batch Processing:**
- `parallel_converter.py` / `parallel_converter_fixed.py` - Parallel test conversion
- `accelerated_converter.py` - High-speed batch processing with caching
- Uses concurrent processing with configurable worker threads (default: 4)

**Monitoring:**
- `agentic_test_monitor.py` - Continuous monitoring with configurable intervals
- Detects code changes and triggers test generation/updates
- Supports continuous, after-idle, and one-time modes

### Configuration
The system requires Gemini API key configuration:
```bash
# Set in environment
export GEMINI_API_KEY=your_key_here
export GOOGLE_API_KEY=your_key_here  # Same as GEMINI_API_KEY
```

Test generation uses Gemini 2.5 Pro model with configurable parameters:
- Max healing iterations: 5
- Default quality threshold: 80
- Parallel workers: 4
- Response caching enabled by default

### Test Quality Metrics
- Target: 100% code coverage across all modules
- Current: ~55% coverage (144/262 files)
- Quality scoring: 0-100 scale based on completeness, assertions, and edge cases
- Minimum acceptable quality score: 70

### Refactoring Detection
The system automatically detects and handles:
- Module renames (updates imports in tests)
- Module splits (generates tests for each new module)
- Module merges (combines existing tests)
- Module moves (updates import paths)
- Module deletions (archives associated tests)

### Important Notes
- Tests use real imports, no mocking by default
- Self-healing has 5 iteration limit to prevent infinite loops
- Import path resolution has ~85% success rate
- Execution time: ~10s per test, 2.6 files/minute conversion rate
- Parallel processing significantly improves batch performance

## Improvement Roadmap
A comprehensive improvement roadmap is available at `C:\Users\kbass\OneDrive\Documents\testmaster\TestMaster\IMPROVEMENT_ROADMAP.md`

## 🚨 CRITICAL: Comprehensive Reorganization Required
**REORGANIZATION ROADMAP**: `C:\Users\kbass\OneDrive\Documents\testmaster\TestMaster\COMPREHENSIVE_REORGANIZATION_ROADMAP.md`

**URGENT ISSUES IDENTIFIED:**
- 68 scattered scripts across root and subdirectories
- Massive code duplication and redundancy  
- Poor directory organization
- No clear separation of concerns
- Debug scripts mixed with production code

**REORGANIZATION PLAN:**
8-week systematic consolidation with multiple failsafe layers:
1. **Foundation & Safety** - Backup, archive, safety mechanisms
2. **Analysis & Planning** - Code analysis, consolidation mapping
3. **Core Consolidation** - Generators, converters, verification
4. **Systems Integration** - Monitoring, analysis, maintenance
5. **Interface Unification** - CLI, API standardization
6. **Testing & Validation** - Comprehensive functionality testing
7. **Documentation** - Complete documentation overhaul
8. **Migration Completion** - Final migration and optimization

**TARGET STRUCTURE:**
- 68 scripts → 15 organized modules (78% reduction)
- Root directory: 25 files → 5 files (80% reduction)
- Complete package structure with proper separation of concerns
- 100% functionality preservation with multiple failsafe checks

The roadmap outlines a 12-week transformation plan with 5 phases:
1. **Foundation** - Pipeline orchestrator, unified config, smart caching
2. **Intelligence** - Test deduplication, incremental generation, prioritization
3. **Analytics** - Quality dashboard, failure analysis, comprehensive reporting
4. **Optimization** - Execution optimization, cross-module testing, profiling
5. **Advanced** - ML integration, multi-model support, plugin architecture

Key improvements target:
- 95%+ code coverage
- 50% reduction in API calls
- 40% faster test execution
- 30% reduction in test writing time
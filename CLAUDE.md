# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Auto-commit Policy

**CRITICAL**: After completing any coding task that modifies files:
1. **ALWAYS UPDATE README.md FIRST** - Document all new features, components, and capabilities in README.md
2. Automatically run `git add .`
3. Create a descriptive commit message based on the changes made
4. Commit using `git commit -m "message"`
5. Push to GitHub origin with `git push origin main` (or current branch)

**README.md UPDATE REQUIREMENT**: Every commit MUST include updated README.md documentation that reflects:
- New features and capabilities added
- System architecture changes
- New commands and usage examples
- Configuration options and profiles
- Integration test results and status

This should happen WITHOUT asking for permission, as part of the natural workflow after completing each task. All changes must be pushed to the GitHub remote repository, not just committed locally.

## Commands

### Running Tests
```bash
# Run all tests with coverage
python -m pytest --cov=. --cov-report=html    # Windows
python3 -m pytest --cov=. --cov-report=html   # Unix

# Run a specific test file
python -m pytest path/to/test_file.py         # Windows
python3 -m pytest path/to/test_file.py        # Unix

# Run tests with verbose output
python -m pytest -v                           # Windows
python3 -m pytest -v                          # Unix

# Run intelligent test suite
py run_intelligent_tests.py                   # Windows
python3 run_intelligent_tests.py              # Unix

# Quick test summary
py quick_test_summary.py                      # Windows
python3 quick_test_summary.py                 # Unix
```

### Test Generation
```bash
# Generate tests for a single module
py intelligent_test_builder.py --module path/to/module.py         # Windows
python3 intelligent_test_builder.py --module path/to/module.py    # Unix

# Generate tests for all modules in a directory
py intelligent_test_builder.py --directory path/to/modules/       # Windows
python3 intelligent_test_builder.py --directory path/to/modules/  # Unix

# Fix broken tests automatically
py enhanced_self_healing_verifier.py --fix path/to/broken_test.py         # Windows
python3 enhanced_self_healing_verifier.py --fix path/to/broken_test.py    # Unix

# Batch fix all tests
py enhanced_self_healing_verifier.py --batch-all                  # Windows
python3 enhanced_self_healing_verifier.py --batch-all             # Unix

# Generate integration tests
py integration_test_generator.py                                  # Windows
python3 integration_test_generator.py                             # Unix
```

### Monitoring & Coverage
```bash
# Monitor for changes continuously (2-hour intervals)
py agentic_test_monitor.py --mode continuous --interval 120       # Windows
python3 agentic_test_monitor.py --mode continuous --interval 120  # Unix

# Run after idle (perfect for breaks)
py agentic_test_monitor.py --mode after-idle --idle 10           # Windows
python3 agentic_test_monitor.py --mode after-idle --idle 10      # Unix

# Monitor progress to 100% coverage
py monitor_to_100.py                                             # Windows
python3 monitor_to_100.py                                        # Unix

# Quick coverage analysis
py scripts/measure_final_coverage.py                             # Windows
python3 scripts/measure_final_coverage.py                        # Unix
```

### Parallel Processing
```bash
# Convert multiple tests in parallel
py parallel_converter.py --input modules.txt --workers 4         # Windows
python3 parallel_converter.py --input modules.txt --workers 4    # Unix

# Accelerated conversion with caching
py accelerated_converter.py --batch --cache                      # Windows
python3 accelerated_converter.py --batch --cache                 # Unix

# Turbo converter with optimizations
py turbo_converter.py                                            # Windows
python3 turbo_converter.py                                       # Unix
```

### Import & Error Fixing
```bash
# Fix import paths in tests
py fix_import_paths.py                  # Windows
python3 fix_import_paths.py             # Unix

# Fix failing tests
py fix_failing_tests.py                 # Windows
python3 fix_failing_tests.py            # Unix

# Fix all imports batch mode
py scripts/fix_all_imports.py           # Windows
python3 scripts/fix_all_imports.py      # Unix
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

## Hybrid Intelligence System - FULLY OPERATIONAL

TestMaster is now a complete unified hybrid intelligence platform with 16 intelligent agents and 5 bridge components working in perfect coordination:

**Core Intelligence Components:**
- **Configuration Intelligence Agent** - Dynamic environment adaptation with 4 smart profiles
- **Hierarchical Test Planning Engine** - Advanced multi-strategy reasoning with 5 planning strategies  
- **Multi-Agent Consensus System** - Democratic decision-making with 6 voting methods
- **Security Intelligence Agent** - Comprehensive OWASP vulnerability scanning and compliance
- **Multi-Objective Optimization Agent** - NSGA-II algorithm for coverage, quality, and performance
- **Performance Monitoring Agents** - Real-time bottleneck detection and adaptive resource management
- **Bridge Communication System** - 5 specialized bridges for protocol, event, session, SOP, and context management

**Deep Integration Architecture:**
- **Phase 1A**: Configuration Intelligence (3/3 agents) ✓
- **Phase 2**: Intelligence Layer Integration (3/3 agents) ✓  
- **Phase 3**: Flow Optimization & Monitoring (3/3 agents) ✓
- **Phase 4**: Bridge Implementation (5/5 bridge agents) ✓
- **Phase 5**: Final Integration & Testing ✓

**Advanced Capabilities:**
- DAG-based workflow orchestration with parallel task execution
- Multi-source configuration hierarchy (default → file → env → runtime → override)
- Event-driven architecture with unified event bus and correlation
- Session management with checkpoint and recovery capabilities
- Universal LLM provider management with fallback chains
- Security intelligence with vulnerability scanning and compliance checking
- Universal framework adaptation supporting any programming language

**Essential Commands:**
```bash
# Run hybrid intelligence orchestration (comprehensive mode - all 16 agents + 5 bridges)
python -m testmaster orchestrate --target path/to/code --mode comprehensive

# Security-focused hybrid intelligence
python -m testmaster orchestrate --target path/to/code --mode security_focused

# Compliance checking with multi-agent consensus
python -m testmaster compliance --target path/to/code --standard OWASP_ASVS

# Hierarchical planning test generation
python -m testmaster intelligence-test --target path/to/code --reasoning-depth 5

# Security Intelligence Agent scanning
python -m testmaster security-scan --target path/to/code --detailed

# Codebase analysis with hybrid intelligence
python -m testmaster analyze --target path/to/code

# Configuration Intelligence testing
python test_config_intelligence.py                              # Windows
python3 test_config_intelligence.py                             # Unix

# Integration testing (all components)
python run_integration_test.py                                  # Windows  
python3 run_integration_test.py                                 # Unix

# Bridge system testing
python -c "from testmaster.intelligence.bridges import get_protocol_bridge; print('Bridge ready!')"

# Multi-agent consensus testing
python -c "from testmaster.intelligence.consensus import ConsensusEngine; print('Consensus ready!')"
```


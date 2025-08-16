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

## 🚀 Production-Ready TestMaster Roadmap

### **Current Status: Foundation Complete, Production Development Phase**
- ✅ **Phase 0**: Foundation & Safety (backup, archive, safety mechanisms)
- ✅ **Consolidation Started**: Base generator classes and intelligent test builder consolidated
- 🟡 **Current Production Readiness**: 25/100 - Good architecture, missing production infrastructure

## **Phase 1: Complete Core Consolidation (2-3 weeks) - IN PROGRESS**
**Objective**: Finish consolidating all 65 remaining scattered scripts into organized modules

### Critical Consolidation Tasks (Foundation Phase Completion):
- **CONV-001**: Consolidate all *converter*.py scripts → testmaster/converters/
- **VER-001**: Consolidate verification scripts → testmaster/verification/  
- **COV-001**: Consolidate coverage scripts → testmaster/analysis/
- **FIX-001**: Consolidate maintenance scripts → testmaster/maintenance/
- **MON-001**: Consolidate monitoring scripts → testmaster/monitoring/
- **EXEC-001**: Consolidate execution scripts → testmaster/execution/
- **API-001**: Preserve 100% API compatibility with legacy aliases

**Target**: 68 scripts → 15 organized modules (78% reduction)

## **Phase 2: Production Infrastructure (2-3 weeks)**
**Objective**: Build production-grade infrastructure and core systems

### 2.1 Configuration & Core Systems
- **CONFIG-001**: Unified configuration system (YAML/JSON + env overrides)
- **LOG-001**: Structured logging framework with levels and rotation
- **ERROR-001**: Comprehensive exception handling with recovery mechanisms
- **CLI-001**: Unified command-line interface with subcommands

### 2.2 API & Integration Layer
- **API-002**: RESTful API design with OpenAPI documentation
- **AUTH-001**: Authentication and authorization system
- **SECRETS-001**: Secure secrets management for API keys
- **VALID-001**: Input validation and sanitization

## **Phase 3: Quality & Operations (2-3 weeks)**  
**Objective**: Implement testing, monitoring, and operational excellence

### 3.1 Testing Infrastructure
- **TEST-001**: Comprehensive unit test suite for TestMaster itself
- **TEST-002**: Integration tests for end-to-end workflows
- **TEST-003**: Performance tests and benchmarking
- **CICD-001**: Automated CI/CD pipeline

### 3.2 Monitoring & Observability
- **MONITOR-001**: Health checks and service monitoring
- **METRICS-001**: Performance metrics and SLA tracking
- **ALERT-001**: Alerting system for failures and performance issues
- **LOG-002**: Centralized logging aggregation and analysis

### 3.3 Documentation & Security
- **DOC-001**: Complete API documentation and user guides
- **DOC-002**: Deployment and operational guides
- **SEC-001**: Security audit and hardening
- **SEC-002**: Rate limiting and quota management

## **Phase 4: Enterprise & Scale (2-3 weeks)**
**Objective**: Enterprise features and horizontal scalability

### 4.1 Scalability & Performance
- **SCALE-001**: Asynchronous processing with job queues
- **SCALE-002**: Distributed test execution
- **SCALE-003**: Database layer for persistent storage
- **PERF-001**: Performance optimization and caching

### 4.2 Enterprise Features
- **UI-001**: Web dashboard for test management
- **MULTI-001**: Multi-tenant support and organization management
- **INTEGR-001**: CI/CD integrations (GitHub Actions, Jenkins, etc.)
- **WEBHOOK-001**: Webhook support for external integrations

## **Phase 5: Advanced Intelligence (2-3 weeks)**
**Objective**: Advanced AI features and extensibility

### 5.1 AI Enhancement
- **AI-001**: Multi-model support (Gemini, GPT-4, Claude)
- **AI-002**: Test result analysis and pattern recognition
- **AI-003**: Intelligent test prioritization and scheduling
- **ML-001**: Machine learning for failure prediction

### 5.2 Ecosystem & Extensions
- **PLUGIN-001**: Plugin architecture for extensibility
- **ECOSYS-001**: Package manager integration
- **CLOUD-001**: Cloud deployment and scaling
- **MARKET-001**: Marketplace for test templates and plugins

## **Production Readiness Targets**

### **End of Phase 2 (6 weeks): Beta Production Ready (70/100)**
- ✅ Complete consolidation and architecture
- ✅ Production infrastructure and monitoring
- ✅ Basic security and authentication
- ✅ Unified CLI and API

### **End of Phase 3 (9 weeks): Production Ready (85/100)**
- ✅ Comprehensive testing and CI/CD
- ✅ Full monitoring and alerting
- ✅ Complete documentation
- ✅ Security audit passed

### **End of Phase 4 (12 weeks): Enterprise Ready (95/100)**
- ✅ Horizontal scalability
- ✅ Enterprise features and multi-tenancy
- ✅ Web interface and integrations
- ✅ SLA compliance

### **End of Phase 5 (15 weeks): Market Leading (100/100)**
- ✅ Advanced AI capabilities
- ✅ Ecosystem and plugin support
- ✅ Cloud-native deployment
- ✅ Complete feature parity with market leaders

## **Key Success Metrics**
- **Code Quality**: 95%+ test coverage, <5% technical debt
- **Performance**: <2s test generation, 10x faster than manual
- **Reliability**: 99.9% uptime, <1% failure rate
- **Security**: Zero critical vulnerabilities, SOC2 compliance
- **Usability**: <5min setup time, 90%+ user satisfaction
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

## 🎯 TestMaster: Test-First Orchestration Platform

### **Vision**: TestMaster as an orchestration platform built on rock-solid test infrastructure
- **Core Foundation**: Test generation, monitoring, supervision (must be bulletproof)
- **Orchestration Layer**: Leverages test insights for broader coordination capabilities
- **Evolution Path**: Test orchestration → Development orchestration → Agent guidance
- **Communication**: File-based messaging with Claude Code and other agents
- **Real-time Intelligence**: Live dashboard showing codebase health and workflow status

### **Why Test-First Orchestration?**
- Tests reveal **codebase structure** and dependencies
- Test monitoring provides **real-time change awareness**
- Test coverage shows **risk areas** and priorities
- Test failures indicate **where help is needed**
- Test mappings enable **intelligent work distribution**

### **Current Status**
- ✅ **Foundation Complete**: Base generator and verification classes created
- ✅ **Consolidation Started**: 2/68 script groups consolidated (converters, verifiers)
- 🟡 **Core Stability**: 40/100 - Good architecture, needs bulletproofing
- 📍 **Next Priority**: Rock-solid test generation and monitoring

## **Layer Control System (Foundation for All Layers)**
**Objective**: Make each layer independently toggleable via configuration

### Configuration Architecture
```yaml
# testmaster_config.yaml
layers:
  layer1_test_foundation:
    enabled: true  # Always on - this is the core
    features:
      test_generation: true
      test_verification: true
      test_mapping: true
      failure_detection: true
  
  layer2_monitoring:
    enabled: false  # Toggle on when ready
    requires: ["layer1_test_foundation"]
    features:
      file_monitoring: true
      idle_detection: true
      claude_communication: true
      dashboard_ui: false  # Can disable UI separately
  
  layer3_orchestration:
    enabled: false  # Toggle on when ready
    requires: ["layer1_test_foundation", "layer2_monitoring"]
    features:
      auto_tagging: true
      work_distribution: true
      smart_handoffs: true
      codebase_intelligence: true
```

### Implementation Pattern
```python
# testmaster/core/layer_manager.py
class LayerManager:
    def __init__(self, config_path: str = "testmaster_config.yaml"):
        self.config = load_config(config_path)
        self.active_layers = {}
        
    def is_enabled(self, layer: str, feature: str = None) -> bool:
        """Check if a layer or specific feature is enabled"""
        if layer not in self.config['layers']:
            return False
        layer_config = self.config['layers'][layer]
        
        # Check dependencies
        for dep in layer_config.get('requires', []):
            if not self.is_enabled(dep):
                raise ConfigError(f"{layer} requires {dep} to be enabled")
        
        if feature:
            return layer_config.get('enabled', False) and \
                   layer_config.get('features', {}).get(feature, False)
        return layer_config.get('enabled', False)
    
    def get_active_features(self) -> Dict[str, List[str]]:
        """Return all active features by layer"""
        active = {}
        for layer, config in self.config['layers'].items():
            if config.get('enabled'):
                active[layer] = [f for f, enabled in 
                                config.get('features', {}).items() if enabled]
        return active
```

## **Layer 1: Rock-Solid Test Foundation (Weeks 1-3)**
**Objective**: Bulletproof the core test lifecycle - generation, evaluation, and repair
**Toggleable**: Always enabled (this is the core foundation)

### 1.1 Test Generation & Evaluation Consolidation
- **TESTGEN-001**: Consolidate all test generators → testmaster/generators/
  - ✅ Base generator classes completed
  - ✅ Intelligent test builder consolidated
  - 🔄 Remaining: specialized, context-aware, integration generators
  - **NEW Pattern**: Pydantic validation (Agency-Swarm) for test completeness
  - **NEW Pattern**: Performance decorators (PraisonAI) for quality scoring
- **TESTEVAL-001**: Robust verification → testmaster/verification/
  - ✅ Self-healing verifier completed
  - ✅ Quality analyzer implemented
  - 🔄 Add: failure pattern recognition, improvement suggestions
  - **NEW Pattern**: Error categorization with statistics (PraisonAI)

### 1.2 Test-Module Mapping System (NEW)
- **MAP-001**: Bidirectional test ↔ module mapping → testmaster/mapping/
  - Track which tests cover which modules
  - Update mappings on file changes
  - Integration test → multiple module relationships
  - **Pattern**: Agency-Swarm's hierarchical thread mapping adapted for tests
- **MAP-002**: Dependency tracking system
  - Module dependency graphs
  - Test impact analysis on changes
  - Cascade testing for dependent modules
  - **Pattern**: Graph-based workflows (LangGraph) for dependency chains

### 1.3 Breaking Detection & Analysis (NEW)
- **BREAK-001**: Real-time failure detection → testmaster/breaking/
  - Immediate test failure notifications
  - Failure categorization (syntax, logic, import, etc.)
  - Root cause analysis
- **BREAK-002**: Regression identification
  - Track test history and patterns
  - Identify new vs recurring failures
  - Suggest fixes based on past resolutions

## **Layer 2: Active Monitoring & Communication (Weeks 4-6)**
**Objective**: Real-time monitoring with Claude Code integration
**Toggleable**: `layer2_monitoring.enabled: true` in config

### 2.1 Real-time Monitoring System
- **MONITOR-001**: Continuous codebase watcher → testmaster/monitoring/
  - File system event monitoring (file changes, additions, deletions)
  - Git commit and branch tracking
  - Real-time test execution triggers
  - **Pattern**: Callback-based file monitoring (Agency-Swarm SettingsCallbacks)
- **MONITOR-002**: Idle detection system
  - 2-hour idle threshold for unchanged modules
  - Stale code identification
  - Coverage decay tracking
  - **Pattern**: Performance statistics tracking (PraisonAI telemetry)
- **MONITOR-003**: Periodic test scheduler
  - Configurable test execution intervals
  - Priority-based test scheduling
  - Resource-aware execution
  - **Pattern**: Queue-based task management (Agency-Swarm Gradio)

### 2.2 Claude Code Communication Layer
- **COMM-001**: File-based messaging → testmaster/communication/
  - Write status files for CLAUDE.md directives
  - Structured YAML/JSON message format
  - Priority and urgency indicators
  - **Pattern**: SharedState key-value store (Agency-Swarm) for status
- **COMM-002**: Tag reading system
  - Read Claude Code tags from source files
  - Parse directives and metadata
  - Synchronize tagging systems
  - **Pattern**: SendMessage validation (Agency-Swarm) for message integrity
- **COMM-003**: Bidirectional queue
  - Message queue via filesystem
  - Acknowledgment and response tracking
  - Error handling and retry logic
  - **Pattern**: Thread-based conversation management (Agency-Swarm)

### 2.3 Live Dashboard UI
- **UI-001**: Real-time web dashboard → testmaster/ui/
  - WebSocket-based live updates
  - Module health visualization
  - Test execution status
- **UI-002**: Coverage and quality metrics
  - Coverage heatmaps
  - Quality score displays
  - Trend analysis graphs
- **UI-003**: Alert and notification system
  - Breaking change alerts
  - Idle module warnings
  - Fix suggestion displays

## **Layer 3: Intelligent Orchestration (Weeks 7-9)**
**Objective**: Smart coordination between TestMaster and Claude Code
**Toggleable**: `layer3_orchestration.enabled: true` in config

### 3.1 File Tagging & Classification
- **TAG-001**: Automatic file tagging → testmaster/orchestrator/
  - Module type classification (core, utility, test, config)
  - Status tags (stable, breaking, needs-attention, idle)
  - Priority levels for Claude Code attention
- **TAG-002**: Dynamic tag updates
  - Real-time tag changes based on test results
  - Automatic priority escalation
  - Tag history and rollback
- **TAG-003**: Claude Code directive generation
  - Generate CLAUDE.md instructions based on tags
  - Prioritized work queues
  - Context-aware fix suggestions

### 3.2 Orchestration Engine
- **ORCH-001**: Work distribution logic
  - Decide: TestMaster fix vs Claude Code fix
  - Complexity assessment for handoff decisions
  - Batch similar issues for efficiency
  - **Pattern**: OpenAI Swarm's function-based handoff for dynamic routing
  - **Pattern**: Agent-Squad's configuration-driven classification
- **ORCH-002**: Automated investigation
  - Investigate 2-hour idle modules
  - Analyze test coverage gaps
  - Generate improvement recommendations
  - **Pattern**: LangGraph supervisor delegation for task distribution
- **ORCH-003**: Smart handoff system
  - Package context for Claude Code
  - Track handoff status and responses
  - Learn from resolution patterns
  - **Pattern**: Context preservation in handoffs (OpenAI Swarm)

### 3.3 Codebase Intelligence
- **OVERVIEW-001**: Functional structure mapping → testmaster/overview/
  - Module relationship graphs
  - API surface tracking
  - Business logic identification
- **OVERVIEW-002**: Coverage intelligence
  - Identify critical uncovered paths
  - Risk assessment for low coverage
  - Test priority recommendations
- **OVERVIEW-003**: Regression tracking
  - Historical failure patterns
  - Regression frequency analysis
  - Predictive failure detection

## **Communication Protocol: TestMaster ↔ Claude Code**

### TestMaster → Claude Code Messages
```yaml
# TESTMASTER_STATUS.yaml - Written by TestMaster for Claude Code
breaking_tests:
  - module: src/auth/login.py
    test: tests/test_login.py
    failure: "AssertionError line 45"
    last_working: "2024-01-15 14:30"
    priority: HIGH
    suggested_action: "Check recent authentication logic changes"
    
modules_need_attention:
  - path: src/payment/processor.py
    status: "idle_2_hours"
    coverage: 45%
    risks: ["Uncovered error paths", "No edge case tests"]
    recommendation: "Add payment failure handling tests"

coverage_gaps:
  - module: src/api/endpoints.py
    uncovered_lines: [45, 67-89, 102]
    critical_paths: true
    suggested_tests: ["Error response tests", "Rate limit tests"]
```

### Claude Code → TestMaster Directives
```yaml
# CLAUDE_DIRECTIVES.yaml - Written by Claude Code for TestMaster
monitor_priority:
  - path: src/new_feature/*
    level: HIGH
    test_frequency: "on_every_change"
    reason: "Active development sprint"
    
temporary_ignore:
  - path: src/experimental/*
    until: "2024-01-20"
    reason: "Prototype code, not production"
    
test_preferences:
  - module_pattern: "*/api/*"
    test_style: "integration_first"
    coverage_target: 95
    
  - module_pattern: "*/utils/*"
    test_style: "unit_only"
    coverage_target: 80
```

## **File Tagging System**

### Module Tags (In source files as comments)
```python
# TESTMASTER_TAGS: core, critical, needs_90_coverage
# TESTMASTER_OWNER: auth_team
# TESTMASTER_LAST_TESTED: 2024-01-15T10:30:00Z
# TESTMASTER_STATUS: stable

class AuthenticationManager:
    """Critical authentication logic"""
    pass
```

### Test Tags (In test files)
```python
# TESTMASTER_COVERS: src/auth/login.py, src/auth/session.py
# TESTMASTER_TYPE: integration
# TESTMASTER_PRIORITY: high
# TESTMASTER_LAST_PASSED: 2024-01-15T14:30:00Z

def test_login_flow():
    """Integration test for login"""
    pass
```

## **Layer 4: Extended Orchestration Capabilities (Future - Built on Test Foundation)**
**Objective**: Leverage test infrastructure for broader orchestration

### 4.1 Development Workflow Orchestration
- **WORKFLOW-001**: Feature development coordination
  - Track feature branches and their test status
  - Coordinate multiple Claude Code agents on large features
  - Manage code review and test approval workflows
- **WORKFLOW-002**: Refactoring orchestration
  - Identify refactoring opportunities through test patterns
  - Coordinate safe refactoring with test validation
  - Track refactoring impact across codebase

### 4.2 Multi-Agent Coordination (Inspired by Framework Analysis)
- **COORD-001**: Claude Code agent task distribution
  - Distribute work based on test coverage insights
  - Route complex issues to specialized agents
  - Load balance across multiple Claude Code instances
- **COORD-002**: Agent communication hub
  - Central message broker for agent coordination
  - Context sharing between agents
  - Conflict resolution for parallel changes

### 4.3 Intelligent Automation
- **AUTO-001**: Predictive issue detection
  - Use test patterns to predict future failures
  - Proactive fix suggestions before breaks occur
  - Automated minor fix attempts
- **AUTO-002**: Smart dependency management
  - Track cascading effects of changes
  - Suggest update sequences
  - Coordinate multi-module updates

### 4.4 Advanced Insights (Building on Test Data)
- **INSIGHT-001**: Codebase health analytics
  - Technical debt identification through test complexity
  - Performance bottleneck detection via test timing
  - Architecture quality metrics
- **INSIGHT-002**: Development velocity tracking
  - Sprint progress based on test completion
  - Team productivity metrics
  - Code quality trends

## **Implementation Timeline & Milestones**

### **Weeks 1-3: Rock-Solid Foundation (Layer 1)**
- ✅ All test generation scripts consolidated
- ✅ Bulletproof test evaluation and self-healing
- ✅ Test-module mapping system operational
- ✅ Breaking detection with root cause analysis
- **Deliverable**: Reliable test generation that never fails

### **Weeks 4-6: Active Monitoring (Layer 2)**
- ✅ Real-time file system monitoring
- ✅ 2-hour idle detection working
- ✅ Claude Code communication protocol active
- ✅ Live dashboard showing codebase health
- **Deliverable**: Always-aware test monitoring system

### **Weeks 7-9: Smart Orchestration (Layer 3)**
- ✅ Automatic file tagging and classification
- ✅ Intelligent TestMaster ↔ Claude Code handoffs
- ✅ Codebase intelligence and overview
- ✅ Predictive failure detection
- **Deliverable**: Self-coordinating test orchestrator

## **Key Success Metrics**
- **Test Reliability**: 100% success rate for test generation and repair
- **Monitoring Coverage**: Real-time awareness of all codebase changes
- **Response Time**: <30s from breaking change to alert
- **Idle Detection**: 100% identification of 2-hour stale modules
- **Communication**: Seamless TestMaster ↔ Claude Code coordination
- **Dashboard**: Live view of test health across entire codebase

## **Layer Control CLI Commands**

```bash
# Check current layer status
testmaster status
# Output: Layer 1: ✅ (test_generation: ✅, test_verification: ✅, test_mapping: ❌)
#         Layer 2: ❌ (disabled)
#         Layer 3: ❌ (disabled)

# Enable/disable layers
testmaster enable layer2  # Enables monitoring & communication
testmaster disable layer2 # Disables monitoring features
testmaster enable layer3  # Enables orchestration (requires layer2)

# Enable/disable specific features
testmaster enable layer2.dashboard_ui
testmaster disable layer3.auto_tagging

# Run with specific layer configuration
testmaster run --layer1-only          # Just test generation/verification
testmaster run --layer2               # Layer 1 + monitoring
testmaster run --all-layers           # Full orchestration mode

# Validate configuration
testmaster validate-config            # Check dependencies and conflicts
```

### **Progressive Usage Patterns:**

**Week 1-3: Layer 1 Only**
```bash
testmaster run --layer1-only
# - Generate tests for existing code
# - Verify test quality and self-heal
# - Map tests to modules
# - Detect failures immediately
```

**Week 4-6: Add Layer 2**
```bash
testmaster enable layer2
testmaster run --layer2
# - All Layer 1 features +
# - File monitoring and idle detection
# - Communication with Claude Code
# - Real-time dashboard (optional)
```

**Week 7-9: Full Orchestration**
```bash
testmaster enable layer3
testmaster run --all-layers
# - All previous features +
# - Smart work distribution
# - Automated investigation
# - Intelligent handoffs to Claude Code
```

## **Implementation Patterns from Framework Analysis**

### **Quick Win Implementations (Start Here):**

1. **SharedState Pattern** (Agency-Swarm) → testmaster/core/shared_state.py
```python
class TestMasterState:
    def set_test_status(self, test_name: str, status: str): ...
    def get_overall_health(self) -> Dict[str, Any]: ...
```

2. **Performance Monitoring Decorators** (PraisonAI) → testmaster/monitoring/decorators.py
```python
@monitor_test_execution("test_quality")
def run_test(): ...  # Automatic timing and quality tracking
```

3. **Queue-based Updates** (Agency-Swarm Gradio) → testmaster/ui/queue.py
```python
test_status_queue = queue.Queue()
# Real-time dashboard updates via queue
```

4. **Simple Handoff Functions** (OpenAI Swarm) → testmaster/orchestrator/handoff.py
```python
def route_test_execution(test_type: str, complexity: int) -> TestExecutor:
    if complexity > 8: return HeavyTestExecutor()
    elif test_type == "integration": return IntegrationTestExecutor()
    else: return StandardTestExecutor()
```

## **Architecture Overview**

```
TestMaster Orchestrator
├── Layer 1: Test Foundation
│   ├── generators/         # Unified test generation
│   ├── verification/       # Test evaluation & repair
│   ├── mapping/           # Test-module relationships
│   └── breaking/          # Failure detection
│
├── Layer 2: Active Monitoring
│   ├── monitoring/        # Real-time watchers
│   ├── communication/     # Claude Code messaging
│   └── ui/               # Live dashboard
│
└── Layer 3: Orchestration
    ├── orchestrator/     # Smart coordination
    ├── tagging/         # File classification
    └── overview/        # Codebase intelligence
```

## **Next Steps After Roadmap Completion**

### Potential Enhancements (Based on actual needs):
1. **Multi-repo Support**: Monitor multiple codebases simultaneously
2. **Test Deduplication**: Identify and merge redundant tests
3. **Performance Optimization**: Parallel test execution, caching
4. **Advanced Analytics**: Test flakiness detection, coverage trends
5. **Integration Options**: GitHub Actions, CI/CD webhooks
6. **Learning System**: Pattern recognition from historical fixes

### Focus Areas:
- Keep the system **simple and reliable**
- Prioritize **developer experience**
- Maintain **clear separation** between TestMaster and Claude Code roles
- Build **incrementally** based on real usage feedback
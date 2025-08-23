# LLM Intelligence System - API Reference

This document provides comprehensive technical documentation for the LLM Intelligence System API, including all classes, methods, configuration options, and data structures.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Data Structures](#data-structures)
3. [Configuration Options](#configuration-options)
4. [LLM Providers](#llm-providers)
5. [Integration Methods](#integration-methods)
6. [Error Handling](#error-handling)
7. [Performance Tuning](#performance-tuning)

## Core Classes

### LLMIntelligenceScanner

The main scanner class that analyzes Python files using LLM and static analysis.

#### Constructor
```python
LLMIntelligenceScanner(
    root_dir: Path,
    config: Optional[Dict[str, Any]] = None
)
```

**Parameters:**
- `root_dir`: Root directory to scan (Path object)
- `config`: Configuration dictionary (optional)

#### Key Methods

##### scan_and_analyze(max_files: Optional[int] = None, output_file: Optional[Path] = None) -> LLMIntelligenceMap
Performs complete intelligence scanning and analysis.

**Parameters:**
- `max_files`: Maximum number of files to analyze (optional)
- `output_file`: Path to save results (optional)

**Returns:**
- `LLMIntelligenceMap`: Complete intelligence map

**Example:**
```python
scanner = LLMIntelligenceScanner(Path("/path/to/codebase"))
intelligence_map = scanner.scan_and_analyze(max_files=50)
```

##### run_single_step(step: str, **kwargs) -> Dict[str, Any]
Runs a single step of the analysis pipeline.

**Parameters:**
- `step`: Step name ("scan", "integrate", "plan", "execute")
- `**kwargs`: Step-specific parameters

**Returns:**
- Dictionary with step results

**Example:**
```python
results = scanner.run_single_step("scan", max_files=20, provider="openai")
```

### IntelligenceIntegrationEngine

Combines LLM analysis with static analysis to create integrated intelligence.

#### Constructor
```python
IntelligenceIntegrationEngine(
    root_dir: Path,
    config: Optional[Dict[str, Any]] = None
)
```

#### Key Methods

##### integrate_intelligence(llm_intelligence_map: Dict[str, Any]) -> List[IntegratedIntelligence]
Integrates LLM analysis with static analysis.

**Parameters:**
- `llm_intelligence_map`: LLM intelligence map from scanner

**Returns:**
- List of integrated intelligence entries

### ReorganizationPlanner

Creates detailed reorganization plans with risk assessment.

#### Constructor
```python
ReorganizationPlanner(
    root_dir: Path,
    config: Optional[Dict[str, Any]] = None
)
```

#### Key Methods

##### create_reorganization_plan(llm_intelligence_map: Dict[str, Any], integrated_intelligence: List[IntegratedIntelligence]) -> DetailedReorganizationPlan
Creates a comprehensive reorganization plan.

**Parameters:**
- `llm_intelligence_map`: LLM intelligence map
- `integrated_intelligence`: Integrated intelligence entries

**Returns:**
- Detailed reorganization plan

##### execute_plan_batch(plan: DetailedReorganizationPlan, batch_id: str, dry_run: bool = True) -> Dict[str, Any]
Executes a specific batch from the reorganization plan.

**Parameters:**
- `plan`: Reorganization plan
- `batch_id`: Batch ID to execute
- `dry_run`: Whether to perform dry run

**Returns:**
- Execution results dictionary

## Data Structures

### LLMIntelligenceEntry

Represents analysis results for a single Python file.

```python
@dataclass
class LLMIntelligenceEntry:
    full_path: str                    # Absolute file path
    relative_path: str                # Path relative to root
    file_hash: str                    # SHA256 hash for caching
    analysis_timestamp: str           # ISO timestamp
    module_summary: str               # 2-3 sentence summary
    functionality_details: str        # Detailed functionality
    dependencies_analysis: str        # Import/dependency analysis
    security_implications: str        # Security considerations
    testing_requirements: str         # Testing needs
    architectural_role: str           # Architectural purpose
    primary_classification: str       # Main category
    secondary_classifications: List[str]  # Additional categories
    reorganization_recommendations: List[str]  # Move suggestions
    confidence_score: float           # 0.0-1.0 confidence
    key_features: List[str]           # Important functionality
    integration_points: List[str]     # Integration dependencies
    complexity_assessment: str        # low/medium/high/very_high
    maintainability_notes: str        # Maintenance considerations
    file_size: int                    # Bytes
    line_count: int                   # Lines of code
    class_count: int                  # Number of classes
    function_count: int               # Number of functions
    analysis_errors: List[str]        # Any errors encountered
```

### StaticAnalysisResult

Results from static analysis tools.

```python
@dataclass
class StaticAnalysisResult:
    semantic: Dict[str, Any] = field(default_factory=dict)      # Semantic analysis
    relationship: Dict[str, Any] = field(default_factory=dict)  # Relationship analysis
    pattern: Dict[str, Any] = field(default_factory=dict)       # Pattern detection
    quality: Dict[str, Any] = field(default_factory=dict)       # Quality metrics
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
```

### IntegratedIntelligence

Combined analysis from multiple sources with confidence scoring.

```python
@dataclass
class IntegratedIntelligence:
    file_path: str                           # File path
    relative_path: str                       # Relative path
    static_analysis: StaticAnalysisResult    # Static analysis results
    llm_analysis: LLMIntelligenceEntry        # LLM analysis results
    confidence_factors: Dict[str, float]     # Confidence from each source
    integrated_classification: str           # Final classification
    reorganization_priority: int             # 1-10 priority
    integration_confidence: float            # Overall confidence
    final_recommendations: List[str]         # Final recommendations
    synthesis_reasoning: str                 # Reasoning explanation
    analysis_timestamp: str                  # Analysis timestamp
```

### ConfidenceFactors

Confidence scores from different analysis sources.

```python
@dataclass
class ConfidenceFactors:
    llm_confidence: float = 0.0              # LLM confidence
    semantic_confidence: float = 0.0         # Semantic analysis confidence
    pattern_confidence: float = 0.0          # Pattern detection confidence
    quality_confidence: float = 0.0          # Quality analysis confidence
    relationship_confidence: float = 0.0     # Relationship analysis confidence
    agreement_confidence: float = 0.0        # Agreement between sources

    def calculate_overall_confidence(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted overall confidence"""
```

### ReorganizationPhase

A phase in the reorganization plan.

```python
@dataclass
class ReorganizationPhase:
    phase_number: int                        # Phase number (1, 2, 3, ...)
    phase_name: str                          # Descriptive name
    description: str                         # Detailed description
    modules: List[Dict[str, Any]]            # Modules in this phase
    risk_level: str                          # low/medium/high/critical
    estimated_time_minutes: int              # Estimated duration
    prerequisites: List[str]                 # Prerequisites for execution
    postconditions: List[str]                # Expected outcomes
    status: str = "pending"                  # pending/in_progress/completed
```

### ReorganizationTask

A specific reorganization task.

```python
@dataclass
class ReorganizationTask:
    task_id: str                             # Unique task identifier
    action: ReorganizationAction             # Action type
    source_path: Optional[str]               # Source file path
    target_path: Optional[str]               # Target file path
    rationale: str                           # Why this task is needed
    confidence: float                        # Confidence in the task
    priority: int                            # 1-10 priority
    risk_level: RiskLevel                    # Risk assessment
    dependencies: List[str]                  # Task dependencies
    prerequisites: List[str]                 # Prerequisites
    estimated_effort_minutes: int            # Estimated effort
    success_criteria: List[str]              # Success conditions
    rollback_plan: str                       # How to rollback
    status: str = "pending"                  # Task status
    error_message: str = ""                  # Error details if failed
```

### DetailedReorganizationPlan

Complete reorganization plan with executable tasks.

```python
@dataclass
class DetailedReorganizationPlan:
    plan_id: str                             # Unique plan identifier
    created_timestamp: str                   # Creation timestamp
    source_intelligence: str                 # Source intelligence ID
    total_tasks: int                         # Total number of tasks
    total_batches: int                       # Total number of batches
    batches: List[ReorganizationBatch]        # Task batches
    summary: Dict[str, Any]                  # Plan summary statistics
    execution_guidelines: List[str]          # How to execute
    risk_mitigation: Dict[str, Any]          # Risk mitigation strategies
    success_metrics: Dict[str, Any]          # Success measurement
```

## Configuration Options

### Core Configuration

```python
{
    # LLM Configuration
    "llm_provider": "openai",                # openai, ollama, mock
    "llm_model": "gpt-4",                    # Model name
    "api_key": "sk-...",                     # API key (for OpenAI)
    "llm_temperature": 0.0,                  # Temperature (0.0 for consistency)
    "llm_max_tokens": 2000,                  # Maximum tokens per request

    # Analysis Configuration
    "max_concurrent": 3,                     # Concurrent LLM requests
    "max_file_size": 50000,                  # Maximum file size (bytes)
    "max_lines_per_file": 1000,              # Maximum lines per file
    "enable_static_analysis": True,          # Enable static analyzers

    # Confidence and Risk
    "min_confidence_threshold": 0.7,         # Minimum confidence for operations
    "high_confidence_threshold": 0.85,       # High confidence threshold
    "integration_method": "weighted_voting", # Integration method
    "risk_thresholds": {                     # Risk thresholds
        "low": 0.8,
        "medium": 0.6,
        "high": 0.4
    },

    # Execution Control
    "auto_approve_risk_levels": ["low"],     # Auto-approve these risks
    "require_review_risk_levels": ["high", "critical"], # Require review
    "backup_enabled": True,                  # Create backups
    "dry_run_enabled": True,                 # Enable dry-run mode
    "import_validation_enabled": True,       # Validate imports

    # Performance and Caching
    "cache_enabled": True,                   # Enable result caching
    "preserve_directory_order": True,        # Maintain directory structure
    "chunk_size": 4000,                      # Text chunking size
    "chunk_overlap": 200                     # Chunk overlap
}
```

### Integration Methods

#### Weighted Voting (Default)
Combines confidence from all sources using weighted average.

```python
"integration_method": "weighted_voting"
```

#### Consensus with Fallback
Requires agreement between sources, falls back to LLM if no consensus.

```python
"integration_method": "consensus_with_fallback"
```

#### LLM Dominant
Uses LLM classification with static analysis for validation.

```python
"integration_method": "llm_dominant"
```

#### Static Dominant
Uses static analysis with LLM for enhancement.

```python
"integration_method": "static_dominant"
```

#### Adaptive Confidence
Dynamically chooses method based on confidence levels.

```python
"integration_method": "adaptive_confidence"
```

### Classification Categories

```python
CLASSIFICATION_CATEGORIES = [
    "security",           # Authentication, encryption, access control
    "intelligence",       # ML models, data analysis, predictive systems
    "frontend_dashboard", # UI components, visualization, user interfaces
    "documentation",      # Code documentation, help systems
    "testing",            # Unit tests, integration tests, test frameworks
    "utility",            # Helper functions, common utilities, shared code
    "api",                # REST endpoints, GraphQL, web services
    "database",           # Models, migrations, data persistence
    "data_processing",    # ETL, data cleaning, transformation
    "orchestration",      # Coordination, messaging, event handling
    "automation",         # Scripts, workflows, scheduled tasks
    "monitoring",         # Logging, metrics, alerting systems
    "analytics",          # Reporting, insights, statistical analysis
    "devops",             # Deployment, infrastructure, CI/CD
    "uncategorized"       # Code that doesn't fit other categories
]
```

## LLM Providers

### OpenAI Provider

```python
from llm_intelligence_system import OpenAIClient

client = OpenAIClient(
    api_key="sk-your-key",
    model="gpt-4"
)

response = client.analyze_code(prompt)
```

**Configuration:**
```python
{
    "llm_provider": "openai",
    "llm_model": "gpt-4",
    "api_key": "sk-your-key",
    "llm_temperature": 0.0,
    "llm_max_tokens": 2000
}
```

### Ollama Provider (Local)

```python
from llm_intelligence_system import OllamaClient

client = OllamaClient(model="llama2:7b")
response = client.analyze_code(prompt)
```

**Configuration:**
```python
{
    "llm_provider": "ollama",
    "llm_model": "llama2:7b"
}
```

### Mock Provider (Testing)

```python
from llm_intelligence_system import MockLLMClient

client = MockLLMClient()
response = client.analyze_code(prompt)
```

**Configuration:**
```python
{
    "llm_provider": "mock"
}
```

## Error Handling

### Exception Hierarchy

```python
class LLMIntelligenceError(Exception):
    """Base exception for LLM Intelligence System"""
    pass

class LLMProviderError(LLMIntelligenceError):
    """LLM provider-related errors"""
    pass

class ConfigurationError(LLMIntelligenceError):
    """Configuration-related errors"""
    pass

class AnalysisError(LLMIntelligenceError):
    """Analysis processing errors"""
    pass

class IntegrationError(LLMIntelligenceError):
    """Integration processing errors"""
    pass

class ExecutionError(LLMIntelligenceError):
    """Execution-related errors"""
    pass
```

### Error Handling Patterns

```python
from llm_intelligence_system import LLMIntelligenceError

try:
    scanner = LLMIntelligenceScanner(Path("/path/to/codebase"))
    intelligence_map = scanner.scan_and_analyze(max_files=50)
except LLMProviderError as e:
    print(f"LLM provider error: {e}")
    # Fallback to mock provider
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Check configuration file
except AnalysisError as e:
    print(f"Analysis error: {e}")
    # Check file permissions and content
```

### Graceful Degradation

The system handles errors gracefully:

1. **LLM Failures**: Falls back to static analysis only
2. **File Access Errors**: Skips problematic files with logging
3. **Configuration Issues**: Uses sensible defaults
4. **Network Issues**: Implements retry logic with exponential backoff

## Performance Tuning

### Concurrency Control

```python
# Adjust based on API limits and system resources
config = {
    "max_concurrent": 3,        # OpenAI tier limits
    "max_concurrent": 10,       # Ollama local models
    "max_concurrent": 1         # Very limited API keys
}
```

### Caching Strategy

```python
# File-based caching with hash validation
cache_config = {
    "cache_enabled": True,
    "cache_dir": "tools/codebase_reorganizer/llm_cache",
    "cache_validation": "hash",  # hash, timestamp, or none
    "cache_max_age_days": 30
}
```

### Memory Management

```python
# Large codebase handling
memory_config = {
    "chunk_size": 4000,         # Token chunks for large files
    "chunk_overlap": 200,       # Overlap to preserve context
    "max_file_size": 50000,     # Skip very large files
    "streaming_enabled": True   # Stream processing for large datasets
}
```

### Cost Optimization

```python
# API cost management
cost_config = {
    "budget_per_hour": 10.0,        # Maximum spend per hour
    "requests_per_minute": 60,      # Rate limiting
    "cache_first": True,            # Use cache when possible
    "triage_enabled": True,         # Prioritize high-impact files
    "progressive_analysis": True    # Start with summaries, then detailed
}
```

## Method Signatures

### LLMIntelligenceScanner Methods

```python
def scan_and_analyze(self, max_files: Optional[int] = None,
                    output_file: Optional[Path] = None) -> LLMIntelligenceMap:
    """Complete intelligence scanning pipeline"""

def run_single_step(self, step: str, **kwargs) -> Dict[str, Any]:
    """Execute single pipeline step"""

def _analyze_file_with_llm(self, file_path: Path) -> Optional[LLMIntelligenceEntry]:
    """Analyze single file with LLM"""

def _perform_static_analysis(self, file_path: Path, content: str) -> StaticAnalysisResult:
    """Perform static analysis on file"""

def _calculate_file_hash(self, file_path: Path) -> str:
    """Calculate file hash for caching"""
```

### IntelligenceIntegrationEngine Methods

```python
def integrate_intelligence(self, llm_intelligence_map: Dict[str, Any]) -> List[IntegratedIntelligence]:
    """Integrate LLM and static analysis"""

def _calculate_confidence_factors(self, llm_entry: LLMIntelligenceEntry,
                                static_analysis: StaticAnalysisResult) -> ConfidenceFactors:
    """Calculate confidence from all sources"""

def _determine_integrated_classification(self, llm_entry: LLMIntelligenceEntry,
                                       static_analysis: StaticAnalysisResult,
                                       confidence_factors: ConfidenceFactors) -> ClassificationResult:
    """Determine final classification"""
```

### ReorganizationPlanner Methods

```python
def create_reorganization_plan(self, llm_intelligence_map: Dict[str, Any],
                             integrated_intelligence: List[IntegratedIntelligence]) -> DetailedReorganizationPlan:
    """Create detailed reorganization plan"""

def execute_plan_batch(self, plan: DetailedReorganizationPlan,
                      batch_id: str, dry_run: bool = True) -> Dict[str, Any]:
    """Execute reorganization batch"""

def _execute_task(self, task: ReorganizationTask, dry_run: bool = True) -> Dict[str, Any]:
    """Execute individual reorganization task"""
```

## Constants and Enums

### ReorganizationAction Enum

```python
class ReorganizationAction(Enum):
    MOVE_FILE = "move_file"              # Move file to new location
    RENAME_FILE = "rename_file"          # Rename file
    CREATE_DIRECTORY = "create_directory" # Create new directory
    UPDATE_IMPORTS = "update_imports"    # Update import statements
    SPLIT_MODULE = "split_module"        # Split large module
    MERGE_MODULES = "merge_modules"      # Merge related modules
    MANUAL_REVIEW = "manual_review"      # Requires human review
    SKIP = "skip"                        # Skip this file
```

### RiskLevel Enum

```python
class RiskLevel(Enum):
    LOW = "low"                          # Safe operations
    MEDIUM = "medium"                    # Requires caution
    HIGH = "high"                        # Needs review
    CRITICAL = "critical"                # High-risk operations
```

### Classification Enum

```python
class Classification(Enum):
    SECURITY = "security"
    INTELLIGENCE = "intelligence"
    FRONTEND_DASHBOARD = "frontend_dashboard"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    UTILITY = "utility"
    API = "api"
    DATABASE = "database"
    DATA_PROCESSING = "data_processing"
    ORCHESTRATION = "orchestration"
    AUTOMATION = "automation"
    MONITORING = "monitoring"
    ANALYTICS = "analytics"
    DEVOPS = "devops"
    UNCATEGORIZED = "uncategorized"
```

## Performance Benchmarks

### Typical Performance

- **Small Project (1-50 files)**: 2-5 minutes complete analysis
- **Medium Project (50-500 files)**: 10-30 minutes complete analysis
- **Large Project (500+ files)**: Use batching, 30-120 minutes

### Resource Usage

- **Memory**: 200-500MB for typical projects
- **Disk**: 10-50MB for cache and output files
- **Network**: Variable based on LLM provider

### Optimization Tips

1. **Use Caching**: Enable caching for repeated analyses
2. **Batch Processing**: Process large codebases in batches
3. **Prioritize Files**: Start with high-impact files
4. **Local Models**: Use Ollama for cost-effective local processing
5. **Incremental Analysis**: Only analyze changed files

---

**This API reference provides the complete technical specification for the LLM Intelligence System. Use it as a guide for extending, customizing, or integrating with other systems.**


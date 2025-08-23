# LLM Intelligence System API Reference

This document provides comprehensive API documentation for all components of the LLM Intelligence System. It covers classes, methods, data structures, and configuration options.

## 📋 Table of Contents

- [Core Classes](#core-classes)
- [Data Structures](#data-structures)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [Examples](#examples)

## 🔧 Core Classes

### LLMIntelligenceScanner

Main scanner class that orchestrates the entire intelligence gathering process.

#### Constructor
```python
LLMIntelligenceScanner(root_dir: Path, config: Dict[str, Any] = None)
```

**Parameters:**
- `root_dir` (Path): Root directory to scan
- `config` (Dict[str, Any], optional): Configuration dictionary

#### Methods

##### scan_and_analyze
```python
def scan_and_analyze(self, output_file: Optional[Path] = None,
                    max_files: Optional[int] = None) -> LLMIntelligenceMap
```

Scans and analyzes Python files using LLM intelligence.

**Parameters:**
- `output_file` (Optional[Path]): Path to save results
- `max_files` (Optional[int]): Maximum files to analyze

**Returns:**
- `LLMIntelligenceMap`: Complete intelligence map

**Example:**
```python
scanner = LLMIntelligenceScanner(Path("/path/to/code"))
result = scanner.scan_and_analyze(max_files=50)
```

##### analyze_file_with_llm
```python
def analyze_file_with_llm(self, file_path: Path) -> Optional[LLMIntelligenceEntry]
```

Analyzes a single file using LLM.

**Parameters:**
- `file_path` (Path): File to analyze

**Returns:**
- `Optional[LLMIntelligenceEntry]`: Analysis result or None

### IntelligenceIntegrationEngine

Combines multiple analysis methods into unified intelligence.

#### Constructor
```python
IntelligenceIntegrationEngine(root_dir: Path, config: Optional[Dict[str, Any]] = None)
```

#### Methods

##### integrate_intelligence
```python
def integrate_intelligence(self, llm_intelligence_map: Dict[str, Any]) -> List[IntegratedIntelligence]
```

Integrates LLM analysis with static analysis.

**Parameters:**
- `llm_intelligence_map` (Dict[str, Any]): LLM intelligence map

**Returns:**
- `List[IntegratedIntelligence]`: Integrated intelligence entries

**Example:**
```python
engine = IntelligenceIntegrationEngine(Path("/path/to/code"))
integrated = engine.integrate_intelligence(llm_data)
```

### ReorganizationPlanner

Creates executable reorganization plans.

#### Constructor
```python
ReorganizationPlanner(root_dir: Path, config: Optional[Dict[str, Any]] = None)
```

#### Methods

##### create_reorganization_plan
```python
def create_reorganization_plan(self, llm_intelligence_map: Dict[str, Any],
                             integrated_intelligence: List[IntegratedIntelligence]) -> DetailedReorganizationPlan
```

Creates a detailed reorganization plan.

**Parameters:**
- `llm_intelligence_map` (Dict[str, Any]): LLM intelligence map
- `integrated_intelligence` (List[IntegratedIntelligence]): Integrated intelligence

**Returns:**
- `DetailedReorganizationPlan`: Complete reorganization plan

##### execute_plan_batch
```python
def execute_plan_batch(self, plan: DetailedReorganizationPlan,
                      batch_id: str, dry_run: bool = True) -> Dict[str, Any]
```

Executes a specific batch from the reorganization plan.

**Parameters:**
- `plan` (DetailedReorganizationPlan): Reorganization plan
- `batch_id` (str): Batch ID to execute
- `dry_run` (bool): Whether to perform dry run

**Returns:**
- `Dict[str, Any]`: Execution results

## 📊 Data Structures

### LLMIntelligenceEntry

Represents analysis results for a single file.

#### Attributes
```python
@dataclass
class LLMIntelligenceEntry:
    full_path: str                    # Full file system path
    relative_path: str               # Path relative to root
    file_hash: str                   # SHA256 hash for caching
    analysis_timestamp: str          # When analysis was performed
    module_summary: str = ""         # LLM-generated summary
    functionality_details: str = ""  # Detailed functionality
    dependencies_analysis: str = ""  # Import and dependency analysis
    security_implications: str = ""  # Security considerations
    testing_requirements: str = ""   # Testing needs
    architectural_role: str = ""     # Architectural classification
    primary_classification: str = "uncategorized"
    secondary_classifications: List[str] = field(default_factory=list)
    reorganization_recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.5    # 0.0 - 1.0
    key_features: List[str] = field(default_factory=list)
    integration_points: List[str] = field(default_factory=list)
    complexity_assessment: str = "unknown"
    maintainability_notes: str = ""  # Maintenance considerations
    file_size: int = 0               # Size in bytes
    line_count: int = 0              # Lines of code
    class_count: int = 0             # Number of classes
    function_count: int = 0          # Number of functions
    analysis_errors: List[str] = field(default_factory=list)
```

### IntegratedIntelligence

Combines multiple analysis sources with confidence scoring.

#### Attributes
```python
@dataclass
class IntegratedIntelligence:
    file_path: str
    relative_path: str
    static_analysis: StaticAnalysisResult
    llm_analysis: LLMIntelligenceEntry
    confidence_factors: ConfidenceFactors
    integrated_classification: str
    reorganization_priority: int      # 1-10
    integration_confidence: float     # 0.0 - 1.0
    final_recommendations: List[str]
    synthesis_reasoning: str
```

### ConfidenceFactors

Confidence scores from different analysis sources.

#### Attributes
```python
@dataclass
class ConfidenceFactors:
    llm_confidence: float = 0.0
    semantic_confidence: float = 0.0
    pattern_confidence: float = 0.0
    quality_confidence: float = 0.0
    relationship_confidence: float = 0.0
    agreement_confidence: float = 0.0

    def calculate_overall_confidence(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall confidence using weighted factors"""
```

### ReorganizationTask

Represents a single executable task.

#### Attributes
```python
@dataclass
class ReorganizationTask:
    task_id: str
    action: ReorganizationAction
    source_path: Optional[str]
    target_path: Optional[str]
    rationale: str
    confidence: float
    priority: int
    risk_level: RiskLevel
    dependencies: List[str]
    prerequisites: List[str]
    estimated_effort_minutes: int
    success_criteria: List[str]
    rollback_plan: str
    status: str = "pending"
    error_message: str = ""
```

### ReorganizationBatch

A group of related tasks with dependencies.

#### Attributes
```python
@dataclass
class ReorganizationBatch:
    batch_id: str
    batch_name: str
    description: str
    tasks: List[ReorganizationTask]
    risk_level: RiskLevel
    estimated_total_time: int
    prerequisites: List[str]
    postconditions: List[str]
    status: str = "pending"
```

### LLMIntelligenceMap

Complete scan results with directory structure.

#### Attributes
```python
@dataclass
class LLMIntelligenceMap:
    scan_timestamp: str
    scan_id: str
    total_files_scanned: int = 0
    total_lines_analyzed: int = 0
    directory_structure: Dict[str, Any] = field(default_factory=dict)
    intelligence_entries: List[LLMIntelligenceEntry] = field(default_factory=list)
    scan_metadata: Dict[str, Any] = field(default_factory=dict)
    scan_statistics: Dict[str, Any] = field(default_factory=dict)
```

## ⚙️ Configuration

### Configuration Schema

```python
config_schema = {
    # LLM Provider Configuration
    'llm_provider': str,              # 'openai', 'ollama', 'mock'
    'llm_model': str,                 # Model name
    'api_key': str,                   # API key (if required)
    'llm_temperature': float,         # 0.0 for deterministic
    'llm_max_tokens': int,            # Response length limit

    # Analysis Configuration
    'enable_static_analysis': bool,   # Use static analyzers
    'max_concurrent': int,            # Concurrent LLM requests
    'max_file_size': int,             # Max file size to analyze
    'max_lines_per_file': int,        # Max lines per file
    'chunk_size': int,                # LLM chunk size
    'chunk_overlap': int,             # Overlap between chunks
    'cache_enabled': bool,            # Enable caching
    'preserve_directory_order': bool, # Maintain file order

    # Integration Configuration
    'integration_method': str,        # Integration strategy
    'min_confidence_threshold': float, # Minimum confidence
    'high_confidence_threshold': float, # High confidence threshold
    'consensus_threshold': float,     # Agreement threshold
    'classification_weights': Dict[str, float],  # Confidence weights

    # Risk & Safety Configuration
    'auto_approve_risk_levels': List[str],  # Auto-approve these
    'require_review_risk_levels': List[str], # Need review
    'backup_enabled': bool,           # Create backups
    'dry_run_enabled': bool,          # Support dry runs
    'import_validation_enabled': bool, # Validate imports
    'max_batch_size': int,            # Max tasks per batch
    'rollback_timeout': int,          # Rollback timeout (seconds)
    'validation_timeout': int         # Validation timeout
}
```

### Default Configuration

```python
DEFAULT_CONFIG = {
    'llm_provider': 'mock',
    'llm_model': 'gpt-4',
    'api_key': None,
    'llm_temperature': 0.0,
    'llm_max_tokens': 2000,

    'enable_static_analysis': True,
    'max_concurrent': 3,
    'max_file_size': 50000,
    'max_lines_per_file': 1000,
    'chunk_size': 4000,
    'chunk_overlap': 200,
    'cache_enabled': True,
    'preserve_directory_order': True,

    'integration_method': 'weighted_voting',
    'min_confidence_threshold': 0.7,
    'high_confidence_threshold': 0.85,
    'consensus_threshold': 0.6,
    'classification_weights': {
        'llm_confidence': 0.35,
        'semantic_confidence': 0.20,
        'pattern_confidence': 0.15,
        'quality_confidence': 0.15,
        'relationship_confidence': 0.10,
        'agreement_confidence': 0.05
    },

    'auto_approve_risk_levels': ['low'],
    'require_review_risk_levels': ['high', 'critical'],
    'backup_enabled': True,
    'dry_run_enabled': True,
    'import_validation_enabled': True,
    'max_batch_size': 10,
    'rollback_timeout': 300,
    'validation_timeout': 60
}
```

### Configuration Methods

#### 1. Dictionary Configuration
```python
config = {
    'llm_provider': 'openai',
    'api_key': 'sk-...',
    'max_concurrent': 5
}

scanner = LLMIntelligenceScanner(root_dir, config)
```

#### 2. Environment Variables
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export MAX_CONCURRENT=5
```

#### 3. Configuration File
```python
# config.json
{
    "llm_provider": "openai",
    "api_key": "sk-...",
    "max_concurrent": 5
}
```

## 🚨 Error Handling

### Exception Hierarchy

```
LLMIntelligenceError
├── ConfigurationError          # Invalid configuration
├── ProviderError              # LLM provider issues
├── AnalysisError              # Analysis failures
├── IntegrationError           # Integration failures
├── PlanningError              # Planning failures
├── ExecutionError             # Execution failures
├── CacheError                 # Cache system errors
└── ValidationError            # Validation failures
```

### Common Error Patterns

```python
try:
    scanner = LLMIntelligenceScanner(root_dir, config)
    result = scanner.scan_and_analyze()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Fix configuration
except ProviderError as e:
    print(f"LLM provider error: {e}")
    # Check provider configuration
except AnalysisError as e:
    print(f"Analysis failed: {e}")
    # Check file permissions or content
```

### Error Recovery

```python
# Automatic retry with exponential backoff
scanner = LLMIntelligenceScanner(root_dir, {
    'retry_attempts': 3,
    'retry_delay': 1.0,
    'retry_backoff': 2.0
})

# Graceful degradation
scanner = LLMIntelligenceScanner(root_dir, {
    'enable_static_analysis': False,  # Disable if static analysis fails
    'continue_on_error': True         # Continue processing other files
})
```

## 📝 Examples

### Basic Usage

```python
from pathlib import Path
from llm_intelligence_system import LLMIntelligenceScanner

# Initialize scanner
root_dir = Path("/path/to/your/codebase")
scanner = LLMIntelligenceScanner(root_dir, {
    'llm_provider': 'mock',  # Use mock for testing
    'max_concurrent': 2
})

# Run complete analysis
result = scanner.scan_and_analyze(max_files=20)

print(f"Analyzed {result.total_files_scanned} files")
for entry in result.intelligence_entries[:5]:
    print(f"  {entry.relative_path}: {entry.primary_classification} ({entry.confidence_score:.2f})")
```

### Advanced Integration

```python
from intelligence_integration_engine import IntelligenceIntegrationEngine

# Initialize integration engine
engine = IntelligenceIntegrationEngine(root_dir, {
    'integration_method': 'weighted_voting',
    'min_confidence_threshold': 0.7
})

# Integrate LLM results with static analysis
integrated = engine.integrate_intelligence(result.__dict__)

# Filter high-confidence results
high_confidence = [
    intel for intel in integrated
    if intel.integration_confidence >= 0.8
]

print(f"High confidence files: {len(high_confidence)}")
```

### Reorganization Planning

```python
from reorganization_planner import ReorganizationPlanner

# Initialize planner
planner = ReorganizationPlanner(root_dir, {
    'min_confidence_threshold': 0.7,
    'backup_enabled': True
})

# Create reorganization plan
plan = planner.create_reorganization_plan(result.__dict__, integrated)

print(f"Reorganization plan created with {plan.total_batches} batches")
print(f"Total tasks: {plan.total_tasks}")
print(".1f")

# Execute a batch (dry run first)
results = planner.execute_plan_batch(plan, plan.batches[0].batch_id, dry_run=True)
print(f"Batch execution results: {results}")
```

### Custom Analysis Provider

```python
from llm_intelligence_system import LLMIntelligenceScanner
from typing import Dict, Any

class CustomLLMClient:
    def analyze_code(self, prompt: str) -> str:
        # Your custom LLM implementation
        return "Custom analysis result"

# Use custom provider
scanner = LLMIntelligenceScanner(root_dir, {
    'llm_provider': 'custom'
})

# Monkey patch the client
scanner.llm_client = CustomLLMClient()
```

### Batch Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def process_large_codebase():
    scanner = LLMIntelligenceScanner(large_root_dir, {
        'max_concurrent': 10,
        'max_file_size': 100000
    })

    # Process in batches
    batch_size = 100
    all_results = []

    for i in range(0, total_files, batch_size):
        batch_result = await scanner.scan_and_analyze_batch(
            start_index=i,
            batch_size=batch_size
        )
        all_results.append(batch_result)

    return all_results
```

### Custom Integration Method

```python
from intelligence_integration_engine import (
    IntelligenceIntegrationEngine,
    ClassificationResult,
    IntegrationMethod
)

class CustomIntegrationEngine(IntelligenceIntegrationEngine):
    def _determine_integrated_classification(self, *args, **kwargs):
        # Custom integration logic
        result = super()._determine_integrated_classification(*args, **kwargs)

        # Add custom post-processing
        if result.primary_classification == 'utility':
            result.primary_classification = 'shared_utility'

        return result

# Use custom integration
engine = CustomIntegrationEngine(root_dir)
```

### Event-Driven Processing

```python
from typing import Callable

def on_analysis_complete(entry: LLMIntelligenceEntry):
    """Callback for completed analysis"""
    print(f"Completed: {entry.relative_path}")

def on_batch_complete(batch_id: str, results: Dict[str, Any]):
    """Callback for completed batch"""
    print(f"Batch {batch_id} completed: {results}")

# Setup callbacks
scanner = LLMIntelligenceScanner(root_dir, {
    'on_analysis_complete': on_analysis_complete,
    'on_batch_complete': on_batch_complete
})
```

This API reference covers the complete interface for the LLM Intelligence System. For implementation details and advanced usage patterns, see the specific component documentation and examples in the system.


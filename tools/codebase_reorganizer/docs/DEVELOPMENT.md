# LLM Intelligence System - Development Guide

This guide is for developers who want to contribute to, extend, or modify the LLM Intelligence System. It covers the codebase architecture, development practices, testing, and contribution guidelines.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Codebase Architecture](#codebase-architecture)
3. [Development Workflow](#development-workflow)
4. [Adding New Features](#adding-new-features)
5. [Testing](#testing)
6. [Code Quality](#code-quality)
7. [Contributing](#contributing)
8. [Advanced Topics](#advanced-topics)

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- Virtual environment tool (venv, conda, etc.)
- LLM API access (OpenAI, Ollama) or use mock mode

### Initial Setup

```bash
# 1. Clone or navigate to the repository
cd tools/codebase_reorganizer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install development dependencies
pip install -r requirements-dev.txt

# 5. Set up pre-commit hooks
pre-commit install

# 6. Run initial tests
python -m pytest tests/
```

### Development Dependencies

Create `requirements-dev.txt`:

```
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
isort>=5.10.0
flake8>=4.0.0
mypy>=1.0.0
pre-commit>=2.20.0
sphinx>=5.0.0
sphinx-rtd-theme>=1.2.0
```

### IDE Configuration

#### VS Code Settings

`.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm Configuration

1. Set Python interpreter to project virtual environment
2. Enable flake8, mypy, and black integrations
3. Configure test runner to use pytest
4. Set up code style to match black formatter

## Codebase Architecture

### Core Modules

```
llm_intelligence_system/
├── __init__.py
├── llm_intelligence_system.py      # Main scanner class
├── intelligence_integration_engine.py # Integration engine
├── reorganization_planner.py        # Planning system
└── run_intelligence_system.py       # CLI interface

docs/                               # Documentation
├── OVERVIEW.md                     # System overview
├── USER_GUIDE.md                   # User instructions
├── API_REFERENCE.md                # API documentation
├── CONFIGURATION.md                # Configuration guide
├── TROUBLESHOOTING.md             # Troubleshooting
├── BEST_PRACTICES.md              # Best practices
├── INTEGRATION_GUIDE.md           # Integration guide
└── DEVELOPMENT.md                 # This file

tools/                             # Additional tools
├── test_intelligence_system.py    # Test suite
└── README_INTELLIGENCE_SYSTEM.md  # Main README

intelligence_output/               # Generated outputs (runtime)
llm_cache/                        # Caching directory (runtime)
logs/                             # Log files (runtime)
backups/                          # Backup files (runtime)
```

### Key Classes and Relationships

```python
# Main components and their relationships
LLMIntelligenceScanner
├── Configures analysis pipeline
├── Manages LLM providers
├── Handles file discovery and caching
└── Produces LLMIntelligenceEntry objects

IntelligenceIntegrationEngine
├── Receives LLMIntelligenceEntry objects
├── Performs static analysis
├── Combines multiple analysis sources
└── Produces IntegratedIntelligence objects

ReorganizationPlanner
├── Receives IntegratedIntelligence objects
├── Assesses risks and dependencies
├── Creates execution batches
└── Produces DetailedReorganizationPlan objects

Runner (run_intelligence_system.py)
├── Orchestrates the entire pipeline
├── Handles user interaction
├── Manages configuration
└── Provides CLI interface
```

### Data Flow

```
Input Files → LLMIntelligenceScanner → LLMIntelligenceEntry[]
                              ↓
                    Static Analysis → StaticAnalysisResult[]
                              ↓
           LLMIntelligenceEntry[] + StaticAnalysisResult[] → IntelligenceIntegrationEngine → IntegratedIntelligence[]
                              ↓
                    IntegratedIntelligence[] → ReorganizationPlanner → DetailedReorganizationPlan
                              ↓
                    DetailedReorganizationPlan → Execution Engine → Reorganized Files
```

## Development Workflow

### 1. Branching Strategy

```bash
# Create feature branch
git checkout -b feature/new-llm-provider

# Make changes
# ... development work ...

# Run tests
python -m pytest tests/

# Format code
black .
isort .

# Run linting
flake8 .
mypy .

# Commit changes
git add .
git commit -m "feat: add new LLM provider support"

# Push branch
git push origin feature/new-llm-provider

# Create pull request
```

### 2. Code Style

The project uses:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

```bash
# Format code
black .

# Sort imports
isort --profile black .

# Check style
flake8 --max-line-length 88 --extend-ignore E203 .

# Type checking
mypy --ignore-missing-imports .
```

### 3. Testing Strategy

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=llm_intelligence_system --cov-report=html tests/

# Run specific test
python -m pytest tests/test_llm_scanner.py::test_scan_single_file

# Run tests in parallel
python -m pytest -n auto tests/
```

### 4. Documentation

```bash
# Build documentation
cd docs/
make html

# Serve documentation locally
cd _build/html
python -m http.server 8000

# Update API documentation
sphinx-apidoc -f -o docs/ llm_intelligence_system/
```

## Adding New Features

### Adding a New LLM Provider

1. **Create Provider Class**

```python
# llm_intelligence_system.py
class NewLLMProvider:
    """New LLM provider implementation"""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        # Initialize client

    def analyze_code(self, prompt: str) -> str:
        """Analyze code with the new provider"""
        # Implement API call
        # Return response as string
        pass
```

2. **Update Provider Factory**

```python
# In LLMIntelligenceScanner class
def _initialize_llm_client(self) -> Any:
    """Initialize LLM client based on configuration"""
    provider = self.config.get('llm_provider', 'mock')

    if provider == 'new_provider':
        return NewLLMProvider(
            api_key=self.config.get('api_key'),
            model=self.config.get('llm_model', 'default-model')
        )
    # ... existing providers
```

3. **Add Configuration Support**

```python
# In _get_default_config method
def _get_default_config(self) -> Dict[str, Any]:
    return {
        # ... existing config
        'new_provider_base_url': 'https://api.newprovider.com',
        'new_provider_timeout': 30,
        # ...
    }
```

4. **Add Tests**

```python
# tests/test_new_provider.py
import pytest
from llm_intelligence_system import NewLLMProvider

def test_new_provider_initialization():
    provider = NewLLMProvider(api_key="test-key", model="test-model")
    assert provider.api_key == "test-key"

def test_new_provider_analyze_code():
    provider = NewLLMProvider(api_key="test-key", model="test-model")

    prompt = "Analyze this Python code: def hello(): pass"
    response = provider.analyze_code(prompt)

    assert isinstance(response, str)
    assert len(response) > 0
```

### Adding a New Analysis Method

1. **Create Analysis Class**

```python
# new_analysis.py
from typing import Dict, Any
from pathlib import Path

class NewAnalysis:
    """New analysis method implementation"""

    def analyze(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Perform new analysis on code"""

        # Implement analysis logic
        analysis_result = {
            'new_metric': calculate_new_metric(content),
            'new_insights': extract_new_insights(content),
            'confidence': calculate_confidence(content)
        }

        return analysis_result
```

2. **Integrate into Static Analysis**

```python
# intelligence_integration_engine.py
from new_analysis import NewAnalysis

class IntelligenceIntegrationEngine:
    def _initialize_analyzers(self) -> Dict[str, Any]:
        analyzers = super()._initialize_analyzers()
        analyzers['new_analysis'] = NewAnalysis()
        return analyzers

    def _perform_static_analysis(self, file_path: Path, content: str) -> StaticAnalysisResult:
        result = super()._perform_static_analysis(file_path, content)

        if 'new_analysis' in self.analyzers:
            result.new_analysis = self.analyzers['new_analysis'].analyze(content, file_path)

        return result
```

3. **Update Data Structures**

```python
# Update StaticAnalysisResult
@dataclass
class StaticAnalysisResult:
    # ... existing fields
    new_analysis: Dict[str, Any] = field(default_factory=dict)
```

4. **Integrate into Classification**

```python
def _calculate_confidence_factors(self, llm_entry: LLMIntelligenceEntry,
                                static_analysis: StaticAnalysisResult) -> ConfidenceFactors:

    factors = ConfidenceFactors()
    # ... existing calculations

    # Add new analysis confidence
    if static_analysis.new_analysis:
        factors.new_analysis_confidence = static_analysis.new_analysis.get('confidence', 0.0)

    return factors
```

### Adding a New Classification Category

1. **Update Classification Enum**

```python
# llm_intelligence_system.py
class Classification(Enum):
    # ... existing categories
    NEW_CATEGORY = "new_category"
    ANOTHER_CATEGORY = "another_category"
```

2. **Add Category Mappings**

```python
# intelligence_integration_engine.py
def _load_classification_taxonomy(self) -> Dict[str, Any]:
    return {
        'primary_categories': [c.value for c in Classification],
        'category_mappings': {
            # ... existing mappings
            'new_keyword': 'new_category',
            'another_keyword': 'another_category'
        },
        'category_hierarchies': {
            # ... existing hierarchies
            'new_category': ['new_subtype', 'another_subtype']
        }
    }
```

3. **Add Classification Logic**

```python
def _get_new_category_classification(self, analysis_result: Dict[str, Any]) -> str:
    """Determine new category classification"""
    if analysis_result.get('new_metric', 0) > 0.8:
        return 'new_category'
    return 'another_category'
```

### Adding a New Reorganization Action

1. **Update Action Enum**

```python
# reorganization_planner.py
class ReorganizationAction(Enum):
    # ... existing actions
    NEW_ACTION = "new_action"
    ANOTHER_ACTION = "another_action"
```

2. **Implement Action Handler**

```python
class ReorganizationPlanner:
    def _execute_task(self, task: ReorganizationTask, dry_run: bool = True) -> Dict[str, Any]:
        if task.action == ReorganizationAction.NEW_ACTION:
            return self._execute_new_action_task(task, dry_run)
        # ... existing handlers

    def _execute_new_action_task(self, task: ReorganizationTask, dry_run: bool = True) -> Dict[str, Any]:
        """Execute new action"""
        if dry_run:
            self.logger.info(f"[DRY RUN] Would perform new action: {task.source_path}")
            return {'success': True, 'action': 'dry_run_new_action'}

        try:
            # Implement new action logic
            success = self._perform_new_action(task.source_path, task.target_path)
            return {'success': success, 'action': 'new_action_performed'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
```

3. **Add Action to Task Creation**

```python
def _create_tasks_from_intelligence(self, intelligence: IntegratedIntelligence) -> List[ReorganizationTask]:
    # ... existing task creation

    # Add new action based on analysis
    if self._should_perform_new_action(intelligence):
        task = ReorganizationTask(
            task_id=f"new_action_{intelligence.relative_path.replace('/', '_')}",
            action=ReorganizationAction.NEW_ACTION,
            source_path=intelligence.file_path,
            target_path=self._get_new_action_target_path(intelligence),
            rationale="New action required based on analysis",
            confidence=intelligence.integration_confidence,
            priority=intelligence.reorganization_priority,
            risk_level=RiskLevel.LOW,
            dependencies=[],
            prerequisites=["New action prerequisites"],
            estimated_effort_minutes=15,
            success_criteria=["New action completed successfully"],
            rollback_plan="Revert new action"
        )
        tasks.append(task)
```

## Testing

### Unit Tests

```python
# tests/test_llm_scanner.py
import pytest
from pathlib import Path
from llm_intelligence_system import LLMIntelligenceScanner

@pytest.fixture
def sample_file(tmp_path):
    """Create a sample Python file for testing"""
    file_path = tmp_path / "sample.py"
    file_path.write_text("""
def hello_world():
    '''A simple function'''
    return "Hello, World!"

class SampleClass:
    def method(self):
        return True
""")
    return file_path

@pytest.fixture
def mock_scanner(sample_file):
    """Create scanner with mock configuration"""
    config = {
        'llm_provider': 'mock',
        'max_concurrent': 1
    }
    return LLMIntelligenceScanner(sample_file.parent, config)

def test_scanner_initialization(mock_scanner):
    """Test scanner initialization"""
    assert mock_scanner.root_dir.exists()
    assert mock_scanner.config['llm_provider'] == 'mock'

def test_scan_single_file(mock_scanner, sample_file):
    """Test scanning a single file"""
    # Mock the LLM response
    mock_scanner.llm_client.analyze_code = lambda prompt: """
{
    "summary": "Simple module with greeting function and sample class",
    "functionality": "Contains hello_world function and SampleClass",
    "primary_classification": "utility",
    "confidence": 0.9
}
"""

    result = mock_scanner._analyze_file_with_llm(sample_file)

    assert result is not None
    assert result.primary_classification == "utility"
    assert result.confidence_score == 0.9
    assert "greeting function" in result.module_summary
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
from pathlib import Path
from llm_intelligence_system import LLMIntelligenceScanner
from intelligence_integration_engine import IntelligenceIntegrationEngine

def test_full_analysis_pipeline(tmp_path):
    """Test the complete analysis pipeline"""

    # Create test files
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "__init__.py").write_text("")
    (tmp_path / "src" / "main.py").write_text("""
def main():
    return "Hello"

class App:
    pass
""")

    # Initialize scanner
    scanner = LLMIntelligenceScanner(tmp_path, {'llm_provider': 'mock'})

    # Run scan
    intelligence_map = scanner.scan_and_analyze(max_files=5)

    # Initialize integration engine
    integration_engine = IntelligenceIntegrationEngine(tmp_path, {
        'enable_static_analysis': False
    })

    # Run integration
    integrated = integration_engine.integrate_intelligence(intelligence_map.__dict__)

    # Verify results
    assert len(integrated) > 0
    assert all(isinstance(item, IntegratedIntelligence) for item in integrated)
    assert all(item.integration_confidence > 0 for item in integrated)
```

### End-to-End Tests

```python
# tests/test_e2e.py
import subprocess
import json
from pathlib import Path

def test_cli_interface(tmp_path):
    """Test the CLI interface end-to-end"""

    # Create test files
    (tmp_path / "test_module.py").write_text("def test(): pass")

    # Run CLI command
    result = subprocess.run([
        "python", "run_intelligence_system.py",
        "--step", "scan",
        "--root", str(tmp_path),
        "--provider", "mock",
        "--max-files", "1",
        "--output", str(tmp_path / "test_output.json")
    ], capture_output=True, text=True, cwd="tools/codebase_reorganizer")

    # Check command succeeded
    assert result.returncode == 0
    assert "✅ LLM scanning completed" in result.stdout

    # Check output file exists
    output_file = tmp_path / "test_output.json"
    assert output_file.exists()

    # Check output is valid JSON
    with open(output_file, 'r') as f:
        data = json.load(f)
        assert 'intelligence_entries' in data
```

### Performance Tests

```python
# tests/test_performance.py
import time
import pytest
from pathlib import Path
from llm_intelligence_system import LLMIntelligenceScanner

def test_scan_performance(tmp_path):
    """Test scanning performance"""

    # Create multiple test files
    for i in range(10):
        (tmp_path / f"module_{i}.py").write_text(f"""
def function_{i}():
    return {i}

class Class_{i}:
    def method(self):
        return True
""")

    scanner = LLMIntelligenceScanner(tmp_path, {'llm_provider': 'mock'})

    start_time = time.time()
    intelligence_map = scanner.scan_and_analyze(max_files=10)
    end_time = time.time()

    duration = end_time - start_time
    files_per_second = len(intelligence_map.intelligence_entries) / duration

    # Performance should be reasonable (adjust threshold as needed)
    assert duration < 30  # Should complete within 30 seconds
    assert files_per_second > 0.1  # At least 0.1 files per second
```

## Code Quality

### Code Style

```bash
# Format code with black
black .

# Sort imports with isort
isort --profile black .

# Check style with flake8
flake8 --max-line-length 88 --extend-ignore E203 .

# Type checking with mypy
mypy --ignore-missing-imports .
```

### Pre-commit Hooks

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 5.0.4
    hooks:
      - id: flake8
        args: ["--max-line-length", "88", "--extend-ignore", "E203"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
        args: ["--ignore-missing-imports"]
```

### Documentation

```bash
# Build documentation
cd docs/
sphinx-build -b html . _build/html

# Check documentation coverage
sphinx-build -b coverage . _build/coverage

# Serve documentation locally
python -m http.server 8000 --directory _build/html
```

## Contributing

### Contribution Guidelines

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run the test suite**: `python -m pytest tests/`
6. **Format code**: `black . && isort .`
7. **Check linting**: `flake8 . && mypy .`
8. **Update documentation** if needed
9. **Commit changes**: `git commit -m "feat: add amazing feature"`
10. **Push to branch**: `git push origin feature/amazing-feature`
11. **Open a Pull Request**

### Pull Request Requirements

- **Title**: Clear, descriptive title
- **Description**: Detailed explanation of changes
- **Tests**: Include tests for new functionality
- **Documentation**: Update docs if needed
- **Breaking Changes**: Clearly mark breaking changes
- **Screenshots**: Include screenshots for UI changes

### Code Review Process

1. **Automated Checks**: GitHub Actions run tests, linting, and type checking
2. **Peer Review**: At least one maintainer reviews the code
3. **Testing**: Reviewer tests the functionality
4. **Approval**: Approved by at least one maintainer
5. **Merge**: Squash merge with descriptive commit message

### Issue Reporting

When reporting bugs or requesting features:

1. **Use the issue template**
2. **Provide a clear title**
3. **Describe the problem** with steps to reproduce
4. **Include environment information**
5. **Attach relevant files or screenshots**
6. **Specify the expected behavior**

## Advanced Topics

### Extending the LLM Prompt System

```python
# custom_prompts.py
class CustomPromptGenerator:
    """Generate custom prompts for specific analysis types"""

    def generate_security_prompt(self, code: str, file_path: Path) -> str:
        """Generate security-focused analysis prompt"""
        return f"""
Analyze this Python code for security vulnerabilities and implications:

File: {file_path}
Code:
{code}

Provide a detailed security analysis including:
1. Potential security vulnerabilities
2. Authentication and authorization patterns
3. Data handling security
4. Input validation
5. Security best practices compliance

Return the analysis as a JSON object with these fields:
- security_vulnerabilities: array of identified issues
- security_score: 0.0-1.0 security rating
- recommendations: array of security recommendations
- authentication_patterns: array of identified patterns
- data_handling_security: security assessment of data operations
"""

    def generate_performance_prompt(self, code: str, file_path: Path) -> str:
        """Generate performance-focused analysis prompt"""
        return f"""
Analyze this Python code for performance characteristics:

File: {file_path}
Code:
{code}

Provide a detailed performance analysis including:
1. Time complexity assessment
2. Space complexity assessment
3. Potential performance bottlenecks
4. Optimization opportunities
5. Memory usage patterns

Return the analysis as a JSON object with these fields:
- time_complexity: O(1), O(n), O(n^2), etc.
- space_complexity: memory usage assessment
- bottlenecks: array of identified bottlenecks
- optimizations: array of optimization suggestions
- performance_score: 0.0-1.0 performance rating
"""
```

### Custom Analysis Pipeline

```python
# custom_pipeline.py
from llm_intelligence_system import LLMIntelligenceScanner
from intelligence_integration_engine import IntelligenceIntegrationEngine

class CustomAnalysisPipeline:
    """Custom analysis pipeline with specialized analyzers"""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.scanner = LLMIntelligenceScanner(root_dir, {
            'llm_provider': 'openai',
            'llm_model': 'gpt-4-turbo'
        })
        self.integration_engine = IntelligenceIntegrationEngine(root_dir, {
            'enable_static_analysis': True,
            'custom_analyzers': ['security', 'performance', 'maintainability']
        })

    def run_security_audit(self):
        """Run comprehensive security audit"""
        # Scan with security-focused prompts
        intelligence_map = self.scanner.scan_and_analyze(
            security_focus=True,
            max_files=50
        )

        # Integrate with security analyzers
        integrated = self.integration_engine.integrate_intelligence(
            intelligence_map.__dict__,
            analysis_types=['security']
        )

        # Generate security report
        return self.generate_security_report(integrated)

    def run_performance_analysis(self):
        """Run comprehensive performance analysis"""
        # Similar pattern for performance analysis
        pass

    def generate_security_report(self, integrated_intelligence):
        """Generate detailed security report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'modules_analyzed': len(integrated_intelligence),
            'security_issues': [],
            'high_risk_modules': [],
            'recommendations': []
        }

        for intelligence in integrated_intelligence:
            # Analyze security implications
            if 'security' in intelligence.llm_analysis.security_implications.lower():
                report['security_issues'].append({
                    'module': intelligence.relative_path,
                    'issues': intelligence.llm_analysis.security_implications,
                    'confidence': intelligence.integration_confidence
                })

        return report
```

### Plugin System

```python
# plugin_system.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

class AnalysisPlugin(ABC):
    """Base class for analysis plugins"""

    @abstractmethod
    def analyze(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Perform analysis on the given content"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name"""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version"""
        pass

class SecurityPlugin(AnalysisPlugin):
    """Security analysis plugin"""

    def get_name(self) -> str:
        return "security"

    def get_version(self) -> str:
        return "1.0.0"

    def analyze(self, content: str, file_path: Path) -> Dict[str, Any]:
        # Implement security analysis
        return {
            'vulnerabilities': self.find_vulnerabilities(content),
            'security_score': self.calculate_security_score(content),
            'recommendations': self.generate_recommendations(content)
        }

class PluginManager:
    """Manages analysis plugins"""

    def __init__(self):
        self.plugins: Dict[str, AnalysisPlugin] = {}

    def register_plugin(self, plugin: AnalysisPlugin):
        """Register a new plugin"""
        self.plugins[plugin.get_name()] = plugin

    def get_plugin(self, name: str) -> AnalysisPlugin:
        """Get a plugin by name"""
        return self.plugins.get(name)

    def list_plugins(self) -> List[str]:
        """List all registered plugins"""
        return list(self.plugins.keys())
```

### Custom Output Formats

```python
# custom_output.py
import json
from pathlib import Path
from typing import List
from llm_intelligence_system import LLMIntelligenceEntry

class MarkdownReportGenerator:
    """Generate markdown reports from analysis results"""

    def generate_module_report(self, entries: List[LLMIntelligenceEntry]) -> str:
        """Generate markdown report for modules"""
        report = "# Codebase Analysis Report\n\n"
        report += f"Generated: {datetime.now().isoformat()}\n\n"
        report += f"Total Modules: {len(entries)}\n\n"

        # Group by classification
        by_classification = {}
        for entry in entries:
            classification = entry.primary_classification
            if classification not in by_classification:
                by_classification[classification] = []
            by_classification[classification].append(entry)

        for classification, modules in by_classification.items():
            report += f"## {classification.title()} Modules ({len(modules)})\n\n"

            for module in sorted(modules, key=lambda x: x.relative_path):
                report += f"### {module.relative_path}\n\n"
                report += f"**Classification Confidence:** {module.confidence_score:.2f}\n\n"
                report += f"**Summary:** {module.module_summary}\n\n"
                report += f"**Key Features:**\n"
                for feature in module.key_features:
                    report += f"- {feature}\n"
                report += "\n"

                if module.reorganization_recommendations:
                    report += f"**Reorganization Suggestions:**\n"
                    for suggestion in module.reorganization_recommendations:
                        report += f"- {suggestion}\n"
                    report += "\n"

                report += "---\n\n"

        return report

    def save_report(self, entries: List[LLMIntelligenceEntry], output_path: Path):
        """Save markdown report to file"""
        report = self.generate_module_report(entries)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

class JSONSchemaGenerator:
    """Generate JSON schemas for analysis results"""

    def generate_entry_schema(self) -> Dict[str, Any]:
        """Generate JSON schema for LLMIntelligenceEntry"""
        return {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "title": "LLMIntelligenceEntry",
            "type": "object",
            "properties": {
                "full_path": {"type": "string"},
                "relative_path": {"type": "string"},
                "file_hash": {"type": "string"},
                "analysis_timestamp": {"type": "string", "format": "date-time"},
                "module_summary": {"type": "string"},
                "functionality_details": {"type": "string"},
                "dependencies_analysis": {"type": "string"},
                "security_implications": {"type": "string"},
                "testing_requirements": {"type": "string"},
                "architectural_role": {"type": "string"},
                "primary_classification": {"type": "string"},
                "secondary_classifications": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "reorganization_recommendations": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "key_features": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "integration_points": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "complexity_assessment": {"type": "string"},
                "maintainability_notes": {"type": "string"},
                "file_size": {"type": "integer"},
                "line_count": {"type": "integer"},
                "class_count": {"type": "integer"},
                "function_count": {"type": "integer"},
                "analysis_errors": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": [
                "full_path", "relative_path", "file_hash", "analysis_timestamp",
                "module_summary", "functionality_details", "dependencies_analysis",
                "security_implications", "testing_requirements", "architectural_role",
                "primary_classification", "secondary_classifications",
                "reorganization_recommendations", "confidence_score",
                "key_features", "integration_points", "complexity_assessment",
                "maintainability_notes", "file_size", "line_count",
                "class_count", "function_count", "analysis_errors"
            ]
        }
```

### Custom Metrics and Monitoring

```python
# custom_metrics.py
import time
from typing import Dict, Any, List
from dataclasses import dataclass, field

@dataclass
class AnalysisMetrics:
    """Custom metrics for analysis operations"""
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    files_processed: int = 0
    llm_requests: int = 0
    llm_tokens_used: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: List[str] = field(default_factory=list)
    performance_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get analysis duration in seconds"""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def files_per_second(self) -> float:
        """Get processing rate"""
        if self.duration == 0:
            return 0.0
        return self.files_processed / self.duration

    @property
    def cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

class MetricsCollector:
    """Collect and report custom metrics"""

    def __init__(self):
        self.metrics = AnalysisMetrics()

    def start_operation(self, operation: str):
        """Start timing an operation"""
        self.metrics.performance_data[f"{operation}_start"] = time.time()

    def end_operation(self, operation: str):
        """End timing an operation"""
        start_key = f"{operation}_start"
        if start_key in self.metrics.performance_data:
            start_time = self.metrics.performance_data[start_key]
            duration = time.time() - start_time
            self.metrics.performance_data[f"{operation}_duration"] = duration

    def increment_counter(self, counter: str, amount: int = 1):
        """Increment a counter"""
        current = self.metrics.performance_data.get(counter, 0)
        self.metrics.performance_data[counter] = current + amount

    def add_error(self, error: str):
        """Add an error to the metrics"""
        self.metrics.errors.append(error)

    def finalize(self):
        """Finalize metrics collection"""
        self.metrics.end_time = time.time()

    def generate_report(self) -> Dict[str, Any]:
        """Generate metrics report"""
        return {
            'duration_seconds': self.metrics.duration,
            'files_processed': self.metrics.files_processed,
            'files_per_second': self.metrics.files_per_second,
            'llm_requests': self.metrics.llm_requests,
            'llm_tokens_used': self.metrics.llm_tokens_used,
            'cache_hit_rate': self.metrics.cache_hit_rate,
            'errors_count': len(self.metrics.errors),
            'errors': self.metrics.errors,
            'performance_breakdown': self.metrics.performance_data
        }
```

---

This development guide provides comprehensive information for developers who want to contribute to, extend, or customize the LLM Intelligence System. The system is designed to be extensible and hackable, with clear patterns for adding new features, providers, and analysis methods. Whether you're fixing bugs, adding features, or completely customizing the system for your needs, this guide should provide the foundation you need to be successful.


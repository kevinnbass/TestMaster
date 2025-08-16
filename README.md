# TestMaster

Advanced automated test generation and management framework with self-healing capabilities.

## Features

### 🚀 Automated Test Generation
- **AI-Powered Generation**: Uses Gemini and GPT models to generate comprehensive tests
- **100% Coverage Achievement**: Systematically generates tests to reach full coverage
- **Multiple Strategies**: Edge cases, boundary testing, property-based testing

### 🔧 Self-Healing Tests
- **Automatic Syntax Fixing**: Iteratively fixes broken tests (up to 5 iterations)
- **Intelligent Error Correction**: Passes errors back to LLM for resolution
- **Parallel Processing**: Generate and fix multiple tests simultaneously

### 📊 Coverage Analysis
- **Branch Coverage**: Identifies uncovered branches and conditions
- **Missing Test Detection**: Finds modules without test coverage
- **Coverage Improvement**: Targeted test generation for coverage gaps

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate tests for a module
python scripts/ai_test_generator.py --module path/to/module.py

# Self-healing test conversion
python scripts/self_healing_converter.py

# Achieve 100% coverage
python scripts/achieve_100_percent.py
```

## Core Components

### Test Generators
- `ai_test_generator.py` - Main AI-powered test generator
- `gemini_test_generator.py` - Gemini-specific generation
- `smart_test_generator.py` - Pattern-based intelligent generation
- `simple_test_generator.py` - Basic test generation

### Self-Healing System
- `self_healing_converter.py` - Fixes syntax errors automatically
- `intelligent_converter.py` - Smart test conversion
- `parallel_coverage_converter.py` - Parallel processing

### Coverage Tools
- `achieve_100_percent.py` - Reaches 100% test coverage
- `coverage_improver.py` - Improves existing coverage
- `branch_coverage_analyzer.py` - Analyzes branch coverage
- `measure_final_coverage.py` - Measures actual coverage

### Framework Components
- `automated_test_generation.py` - Core test generation system
- `comprehensive_test_framework.py` - Complete testing framework
- `integration_test_matrix.py` - Integration testing matrix
- `coverage_analysis.py` - Coverage analysis tools

## Configuration

Set up your API keys in `.env`:
```env
GOOGLE_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_key
```

## Examples

### Generate tests with self-healing
```python
from scripts.self_healing_converter import SelfHealingConverter

converter = SelfHealingConverter()
converter.generate_and_fix_tests("src/my_module.py")
```

### Achieve 100% coverage
```python
from scripts.achieve_100_percent import achieve_full_coverage

achieve_full_coverage(
    source_dir="src",
    test_dir="tests",
    use_ai=True
)
```

## Architecture

```
TestMaster/
├── src/                    # Core framework modules
│   ├── automated_test_generation.py
│   ├── comprehensive_test_framework.py
│   ├── coverage_analysis.py
│   └── integration_test_matrix.py
├── scripts/               # Test generation scripts
│   ├── self_healing_converter.py
│   ├── ai_test_generator.py
│   ├── achieve_100_percent.py
│   └── ...
├── docs/                  # Documentation
├── examples/              # Usage examples
└── tests/                 # Tests for TestMaster itself
```

## Success Story

This framework successfully achieved **100% test coverage** for a complex codebase:
- Generated **466 test files** 
- Covered **115 source modules**
- Used multiple AI models and strategies
- Self-healed thousands of syntax errors

## Contributing

Contributions welcome! Please read our contributing guidelines.

## License

MIT License - See LICENSE file for details

## Credits

Originally developed as part of the regex_gen project's comprehensive reorganization.

🤖 Powered by AI test generation technology
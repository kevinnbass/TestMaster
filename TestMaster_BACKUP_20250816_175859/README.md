# TestMaster - Intelligent Test Generation & Maintenance System

An autonomous, self-healing test generation and maintenance system that uses AI to create, fix, and maintain comprehensive test coverage as code evolves.

## 🚀 Features

### Intelligent Test Generation
- **Automatic test creation** for any Python module
- **Real functionality testing** - no mocks, tests actual code
- **Self-healing capabilities** - automatically fixes syntax and import errors
- **Quality verification** - scores tests and suggests improvements
- **Continuous monitoring** - watches for code changes and updates tests

### Advanced Capabilities
- **Parallel processing** for batch test generation
- **Import path resolution** with 85% success rate
- **Refactoring detection** (renames, splits, merges, moves)
- **Coverage gap analysis** and automatic filling
- **Integration with Gemini AI** for intelligent test creation

## 📁 Core Components

### Test Generators
- `intelligent_test_builder.py` - Main intelligent test generator
- `intelligent_test_builder_v2.py` - Enhanced version with better error handling
- `intelligent_test_builder_offline.py` - Offline mode for cached responses
- `enhanced_context_aware_test_generator.py` - Context-aware test generation
- `integration_test_generator.py` - Integration test creator
- `specialized_test_generators.py` - Domain-specific test generators

### Self-Healing & Verification
- `enhanced_self_healing_verifier.py` - Automatic test repair system
- `independent_test_verifier.py` - Quality verification and scoring
- `fix_import_paths.py` - Automatic import path resolution

### Converters & Batch Processing
- `parallel_converter.py` - Parallel test conversion
- `accelerated_converter.py` - High-speed batch processing
- `turbo_converter.py` - Optimized converter with caching
- `convert_with_genai_sdk.py` - Gemini AI SDK integration
- `convert_batch_small.py` - Small batch converter

### Monitoring & Analysis
- `agentic_test_monitor.py` - Continuous test monitoring
- `monitor_progress.py` - Progress tracking
- `monitor_to_100.py` - Coverage completion monitor
- `run_intelligent_tests.py` - Test execution framework
- `simple_test_runner.py` - Lightweight test runner
- `quick_test_summary.py` - Test summary generator

## 🎯 Quick Start

### Basic Test Generation
```python
# Generate tests for a single module
python intelligent_test_builder.py --module path/to/module.py

# Generate tests for all modules in a directory
python intelligent_test_builder.py --directory path/to/modules/
```

### Self-Healing Mode
```python
# Fix broken tests automatically
python enhanced_self_healing_verifier.py --fix path/to/broken_test.py

# Batch fix all tests
python enhanced_self_healing_verifier.py --batch-all
```

### Continuous Monitoring
```python
# Monitor for changes every 2 hours
python agentic_test_monitor.py --mode continuous --interval 120

# Run after idle (perfect for breaks)
python agentic_test_monitor.py --mode after-idle --idle 10
```

### Parallel Batch Processing
```python
# Convert multiple tests in parallel
python parallel_converter.py --input modules.txt --workers 4

# Accelerated conversion with caching
python accelerated_converter.py --batch --cache
```

## 🔧 Configuration

### Environment Variables
```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here  # Same as GEMINI_API_KEY
```

### Test Generation Settings
```python
# In intelligent_test_builder.py
CONFIG = {
    "max_iterations": 5,      # Self-healing iterations
    "quality_threshold": 80,  # Minimum quality score
    "parallel_workers": 4,     # Parallel processing threads
    "cache_responses": True,   # Cache AI responses
}
```

## 📊 Architecture

See `AGENTIC_TEST_ARCHITECTURE.md` for detailed architecture documentation.

### Generation Pipeline
```
Code Module → Initial Generator → Self-Healer → Verifier → Enhanced Test
                     ↓                ↓            ↓
                 Basic Test    Syntax Fixed   Quality Score
```

### Quality Layers
1. **Basic Generation** - Syntactically correct tests with real imports
2. **Self-Healing** - Automatic error fixing (5 iterations max)
3. **Verification** - Quality scoring and improvement suggestions

## 🎪 Use Cases

### New Feature Development
```
Developer writes new module → 
  Monitor detects (within 2 hours) → 
    Generates comprehensive tests → 
      Verifies quality → 
        Notifies if gaps exist
```

### Major Refactoring
```
Developer refactors codebase →
  Tracker detects splits/merges →
    Updates affected tests →
      Maintains coverage →
        Reports success/issues
```

### Continuous Integration
```python
# Add to CI/CD pipeline
python independent_test_verifier.py --fail-under 80
python run_intelligent_tests.py --coverage-report
```

## 📈 Success Metrics

- **55% test coverage** achieved (144/262 files)
- **800+ test methods** generated automatically
- **85% import fix** success rate
- **~10s per test** execution time
- **2.6 files/minute** conversion rate

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with Gemini 2.5 Pro AI
- Inspired by self-healing systems architecture
- Part of the tot_branch_minimal project ecosystem

## 📞 Contact

Project Link: [https://github.com/kevinnbass/TestMaster](https://github.com/kevinnbass/TestMaster)

---

**Note**: This is an active project with continuous improvements. Check back regularly for updates and new features!
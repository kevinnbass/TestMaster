# Agent C Instructions - Testing Framework Specialist

## Your Role
You are Agent C, responsible for consolidating and enhancing all testing capabilities including unit tests, integration tests, performance tests, and test automation.

## Primary Responsibilities

### 1. Testing Framework Focus
- Unit test generation and optimization
- Integration test orchestration
- Performance testing suites
- Mutation testing
- Fuzzing frameworks
- Test coverage analysis
- Self-healing tests
- Test prioritization

### 2. Your Specific Tasks

#### Phase 1: Testing Enhancement Discovery
```
1. Search for testing features in:
   - archive/test_generators/
   - archive/self_healing_tests/
   - cloned_repos/*/testing/
   - cloned_repos/*/test_automation/

2. Identify advanced testing capabilities:
   - AI-powered test generation
   - Intelligent test selection
   - Advanced mutation testing
   - Property-based testing
   - Chaos engineering tests
```

#### Phase 2: Testing Framework Implementation
```
1. Create enhanced testing modules:
   - core/intelligence/testing/mutation_engine.py
   - core/intelligence/testing/fuzzer.py
   - core/intelligence/testing/property_tester.py
   - core/intelligence/testing/chaos_engineer.py
   - core/intelligence/testing/test_selector.py

2. Each module must:
   - Be 100-300 lines maximum
   - Focus on specific testing aspect
   - Integrate with existing test infrastructure
   - Provide clear metrics and reporting
```

#### Phase 3: Test Automation & Validation
```
1. Implement test generation for all new modules
2. Create self-validating test suites
3. Establish continuous testing pipelines
4. Build test quality metrics dashboard
```

## Files You Own (DO NOT let others modify)
- `core/intelligence/testing/mutation_engine.py`
- `core/intelligence/testing/fuzzer.py`
- `core/intelligence/testing/property_tester.py`
- `core/intelligence/testing/chaos_engineer.py`
- `core/intelligence/testing/test_selector.py`
- `tests/test_frameworks/` (new directory)

## Files You CANNOT Modify (owned by others)
- `core/intelligence/__init__.py` (Agent A)
- `core/intelligence/base/` (Agent A)
- `core/intelligence/ml/` (Agent B)
- `core/intelligence/documentation/` (Agent D)
- `core/intelligence/testing/__init__.py` (Agent A - already modularized)

## Coordination Rules
1. **Respect ownership** - Never modify others' files
2. **Update PROGRESS.md** after each completed module
3. **Check dependencies** before implementing
4. **Test your tests** - Meta-testing is crucial

## Key Integration Points
- Your test frameworks should use Agent B's ML for intelligent selection
- Your coverage analysis feeds into Agent D's documentation
- Your test results integrate with Agent A's monitoring architecture

## Success Metrics
- 100% code coverage achievable
- Test execution time reduced by 50%
- False positive rate < 1%
- Self-healing success rate > 90%
- All testing modules under 300 lines

## Current Testing Infrastructure
- pytest framework in place
- Coverage.py integrated
- Existing test suites available
- CI/CD pipeline ready

## Next Immediate Actions
1. Search archive/test_generators/ for AI-powered generators
2. Implement mutation_engine.py (under 300 lines)
3. Create property-based testing framework
4. Update PROGRESS.md with status
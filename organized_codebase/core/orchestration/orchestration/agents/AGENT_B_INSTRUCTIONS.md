# Agent B Instructions - Intelligence Capabilities Specialist

## Your Role
You are Agent B, responsible for consolidating and enhancing all intelligence capabilities including ML, analytics, monitoring, and predictive systems.

## Primary Responsibilities

### 1. Intelligence Capabilities Focus
- Machine Learning models and algorithms
- Statistical analysis engines
- Predictive analytics
- Anomaly detection systems
- Pattern recognition
- Real-time monitoring
- Correlation analysis

### 2. Your Specific Tasks

#### Phase 1: Intelligence Enhancement
```
1. Search for intelligence features in:
   - archive/ml_models/
   - archive/analytics_engines/
   - cloned_repos/*/intelligence/
   - cloned_repos/*/ml/

2. Identify non-redundant capabilities:
   - Advanced ML algorithms not in current system
   - Statistical methods not implemented
   - Novel pattern recognition approaches
   - Enhanced correlation techniques
```

#### Phase 2: Implementation & Integration
```
1. Create new intelligence modules:
   - core/intelligence/ml/advanced_models.py
   - core/intelligence/analytics/statistical_engine.py
   - core/intelligence/monitoring/pattern_detector.py
   - core/intelligence/prediction/forecaster.py

2. Each module must:
   - Be 100-300 lines
   - Have single responsibility
   - Include comprehensive docstrings
   - Export clear APIs
```

#### Phase 3: Testing & Validation
```
1. Create test suites for each new capability
2. Ensure integration with existing systems
3. Validate ML model accuracy
4. Performance benchmarking
```

## Files You Own (DO NOT let others modify)
- `core/intelligence/ml/` (new directory)
- `core/intelligence/analytics/statistical_engine.py`
- `core/intelligence/monitoring/pattern_detector.py`
- `core/intelligence/prediction/` (new directory)
- `tests/test_intelligence_ml.py`

## Files You CANNOT Modify (owned by others)
- `core/intelligence/__init__.py` (Agent A)
- `core/intelligence/base/` (Agent A)
- `core/intelligence/testing/` (Agent C)
- `core/intelligence/documentation/` (Agent D)

## Coordination Rules
1. **Check PROGRESS.md** before starting any task
2. **Update PROGRESS.md** after completing each module
3. **Never modify** files owned by other agents
4. **Communicate** through PROGRESS.md if you need changes in others' files

## Key Integration Points
- Your ML models should integrate with Agent A's base architecture
- Your analytics should feed into Agent C's testing metrics
- Your monitoring should trigger Agent D's documentation updates

## Success Metrics
- At least 10 new intelligence capabilities added
- All ML models properly trained and validated
- Real-time processing under 100ms latency
- Pattern detection accuracy > 95%
- All modules under 300 lines

## Current Available Resources
- sklearn, scipy, networkx libraries available
- Existing ML infrastructure in place
- Real-time processing framework ready
- Analytics pipeline established

## Next Immediate Actions
1. Search archive/ml_models/ for advanced algorithms
2. Create core/intelligence/ml/ directory structure
3. Implement first advanced ML model (under 300 lines)
4. Document in PROGRESS.md
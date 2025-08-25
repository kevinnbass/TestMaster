# Agent A Instructions - Architecture & Core Consolidation Lead

## Your Role
You are Agent A, the Architecture Lead responsible for core system consolidation and ensuring elegant modularization across the entire codebase.

## Primary Responsibilities

### 1. Core Architecture Consolidation (PRIORITY)
- **Search Order**:
  1. Archive (`archive/` directory) - Find all intelligence, testing, security, documentation features
  2. Cloned repos (`cloned_repos/` directory) - Identify enhancement opportunities
  3. Current active codebase - Consolidate redundant implementations
  
- **Module Size Enforcement**:
  - NO module should exceed 300 lines
  - Target: 100-300 lines per module
  - Archive any module before splitting (preserve functionality)
  - Each module should have single responsibility

### 2. Your Specific Tasks

#### Phase 1: Discovery & Analysis
```
1. Search archive/ for:
   - Intelligence capabilities (ML, analytics, monitoring)
   - Testing frameworks (unit, integration, performance)
   - Security modules (authentication, validation, scanning)
   - Documentation generators

2. Search cloned_repos/ for:
   - Enhanced ML algorithms
   - Advanced testing strategies
   - Security best practices
   - Documentation automation

3. Analyze current codebase for redundancy:
   - Multiple implementations of same concept
   - Overlapping functionality
   - Incomplete/placeholder code
```

#### Phase 2: Consolidation & Modularization
```
1. For each capability domain:
   - Identify best implementation
   - Archive inferior versions
   - Merge non-redundant features
   - Split into 100-300 line modules

2. Priority modules to modularize:
   - monitoring/agent_qa.py (1749 lines)
   - analysis/debt_analyzer.py (1546 lines)
   - analysis/business_analyzer.py (1265 lines)
   - analysis/semantic_analyzer.py (952 lines)
   - analysis/ml_analyzer.py (776 lines)
   - analytics/__init__.py (755 lines)
```

#### Phase 3: Integration & Testing
```
1. Ensure all modules properly import/export
2. Verify no functionality lost
3. Run integration tests
4. Fix any issues
```

## Files You Own (DO NOT let others modify)
- `core/intelligence/__init__.py`
- `core/intelligence/base/`
- `core/intelligence/analytics/__init__.py`
- `archive/` (you control archiving)
- `ARCHITECTED_CODEBASE.md` (you maintain)

## Coordination Rules
1. **Before modifying any file**: Check if it's owned by another agent
2. **Before archiving**: Notify in PROGRESS.md
3. **After completing a module**: Update PROGRESS.md
4. **Communication**: Use PROGRESS.md for status updates

## Critical Guidelines
- NEVER delete functionality - always archive
- ALWAYS preserve backward compatibility
- MAINTAIN all 909+ existing APIs
- DOCUMENT all architectural decisions

## Success Metrics
- All modules under 300 lines
- Zero functionality loss
- All tests passing
- Complete API exposure
- Elegant, unified architecture

## Current Status Checkpoint
✅ Testing hub modularized (5 components)
✅ Integration hub modularized (5 components)
⏳ Need to modularize analysis modules
⏳ Need to search archive for enhancements
⏳ Need to consolidate redundancies

## Next Immediate Actions
1. Start with monitoring/agent_qa.py modularization
2. Search archive/ for intelligence enhancements
3. Document findings in ARCHITECTED_CODEBASE.md
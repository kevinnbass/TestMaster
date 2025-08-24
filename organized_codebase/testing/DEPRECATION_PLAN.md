# Phase 1C Consolidation - Deprecation Plan

## Files to Remove (Redundant/Duplicate)

### 1. Observability Duplicates
**REMOVE** (Single source of truth: `observability/unified_monitor.py`)
- `monitoring/enhanced_monitor.py` - 645 lines (consolidated into unified_monitor.py)
- `core/observability/agent_ops.py` - 549 lines (consolidated into unified_monitor.py) 

### 2. State Management Duplicates
**REMOVE** (Single source of truth: `state/unified_state_manager.py`)
- `testmaster/async_processing/async_state_manager.py` - Async state management (superseded)
- `testmaster/core/shared_state.py` - Basic shared state (superseded)

### 3. Cross-System Integration (Mock/Redundant)
**REMOVE** (Functionality exists in individual unified systems)
- `integration/cross_system_apis.py` - 618 lines of mock implementations
- `integration/workflow_framework.py` - Redundant workflow definitions
- `integration/visual_workflow_designer.py` - Not integrated with actual systems

### 4. Redundant Monitoring Infrastructure  
**REMOVE** (Consolidated into unified_monitor.py)
- Multiple dashboard monitoring files (analytics_*_monitor.py)
- Redundant real-time monitoring implementations

## Files to Preserve & Modularize

### 1. Core Unified Systems (Keep but modularize)
- `observability/unified_monitor.py` → Break into 4 focused modules
- `state/unified_state_manager.py` → Break into 4 focused modules
- `orchestration/unified_orchestrator.py` → Review for modularization
- `ui/unified_dashboard.py` → Review for modularization

## Migration Strategy

### Phase 1: Archive & Backup
1. Create archive folder: `archive/phase1c_consolidation_YYYYMMDD_HHMMSS/`
2. Copy all files to be removed into archive
3. Document what each file provided and where functionality now lives

### Phase 2: Update Import Dependencies
1. Find all imports of deprecated files
2. Update to use unified systems
3. Test that functionality is preserved

### Phase 3: Remove Deprecated Files
1. Delete redundant files one by one
2. Validate system still works after each deletion
3. Update documentation

### Phase 4: Modularization
1. Break large unified files into focused components
2. Implement proper dependency injection
3. Restore clear domain boundaries

## Success Criteria

- [ ] File count reduced from 670 to ~300 files (55% reduction)
- [ ] Code duplication reduced from ~40% to <10%
- [ ] All functionality preserved (zero feature loss)
- [ ] Clear separation of concerns restored
- [ ] Import dependencies cleaned up
- [ ] Documentation updated

## Rollback Plan

If any issues are discovered:
1. All deprecated files are preserved in archive
2. Can restore from archive and revert import changes
3. Git history provides additional safety net

---
**Status**: Ready to execute
**Risk Level**: LOW (comprehensive archival strategy)
**Estimated Duration**: 2-3 hours
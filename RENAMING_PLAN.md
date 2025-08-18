# Tree-of-Thought to Hierarchical Planning Renaming Plan

## Name Mappings

### Class Names
- `UniversalToTTestGenerator` → `UniversalHierarchicalTestGenerator`
- `ToTGenerationConfig` → `HierarchicalPlanningConfig`
- `ToTGenerationResult` → `HierarchicalPlanningResult`
- `TreeOfThoughtReasoner` → `HierarchicalTestPlanner`
- `ThoughtNode` → `PlanningNode`
- `ThoughtTree` → `PlanningTree`
- `ThoughtGenerator` → `PlanGenerator`
- `ThoughtEvaluator` → `PlanEvaluator`
- `TestThoughtGenerator` → `TestPlanGenerator`
- `TestThoughtEvaluator` → `TestPlanEvaluator`
- `SimpleThoughtGenerator` → `SimplePlanGenerator`
- `SimpleThoughtEvaluator` → `SimplePlanEvaluator`

### Variable/Parameter Names
- `tot_generator` → `htp_generator` or `hierarchical_generator`
- `tot_config` → `htp_config` or `planning_config`
- `tot_result` → `htp_result` or `planning_result`
- `enable_tot_reasoning` → `enable_hierarchical_planning`
- `thought_tree` → `planning_tree`
- `thought_node` → `planning_node`
- `reasoning_strategy` → `planning_strategy`
- `max_reasoning_depth` → `max_planning_depth`
- `reasoning_depth` → `planning_depth`
- `reasoning_time` → `planning_time`

### Module/File Names
- `tree_of_thought/` → Keep for now, mark as deprecated
- `tot_reasoning.py` → Keep internal, use through unified_integration
- `test_thought_generator.py` → Keep internal
- `universal_tot_integration.py` → Keep internal

## Files to Update (in order)

### Phase 1: Update Core Orchestration
1. **testmaster/orchestration/universal_orchestrator.py**
   - Change import from `tree_of_thought` to `hierarchical_planning`
   - Rename `tot_generator` to `htp_generator`
   - Rename `enable_tot_reasoning` to `enable_hierarchical_planning`
   - Update all ToT class names to HTP names

2. **testmaster/orchestration/__init__.py**
   - Export new names alongside old (for compatibility)

### Phase 2: Update Main Module
3. **testmaster/main.py**
   - Update imports to use new names
   - Update `enable_tot` to `enable_htp` or `enable_hierarchical_planning`
   - Update config creation

### Phase 3: Update Examples
4. **unified_orchestration_example.py**
   - Update all references to use new names

### Phase 4: Update Tests
5. **test_tot_integration.py** → Rename to `test_htp_integration.py`
   - Update all class names and imports

6. **test_tot_output.py** → Rename to `test_htp_output.py`
   - Update all class names and imports

### Phase 5: Update Documentation
7. **README.md**
   - Already updated to use "Hierarchical Test Planning"

8. **CLAUDE.md**
   - Update any references to ToT

## Implementation Strategy

### Step 1: Keep Aliases Active
- Maintain backward compatibility aliases in `unified_integration.py`
- This allows gradual migration

### Step 2: Update Each File
- Use MultiEdit tool for bulk changes in each file
- Test after each file update

### Step 3: Create New Tests
- Create `test_hierarchical_planning.py` with new names
- Ensure all functionality works with new names

### Step 4: Deprecation Warnings (Future)
- Add deprecation warnings to old names
- Give users time to migrate

### Step 5: Final Cleanup (Future)
- Remove tree_of_thought directory
- Remove backward compatibility aliases

## Risk Mitigation

1. **Keep aliases during transition** - Don't break existing code
2. **Test after each change** - Catch issues immediately
3. **Update incrementally** - One module at a time
4. **Document changes** - Clear migration guide
5. **Version appropriately** - Major version bump if breaking changes
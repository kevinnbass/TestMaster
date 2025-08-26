# Handoff Protocol for Dashboard Consolidation

## Purpose
This directory facilitates asynchronous communication between Agents X, Y, and Z during the dashboard consolidation project.

## File Naming Standard
```
AGENT_[X|Y|Z]_[PHASE][NUMBER]_[STATUS]_[YYYYMMDD_HHMMSS_UTC].md
```

### Components:
- **AGENT**: X, Y, or Z
- **PHASE**: PHASE1, PHASE2, PHASE3, PHASE4, PHASE5, PHASE6
- **STATUS**: COMPLETE, IN_PROGRESS, BLOCKED, CRITICAL
- **TIMESTAMP**: YYYYMMDD_HHMMSS_UTC format

## Status Types

### COMPLETE
Used when a phase or major milestone is finished.

### IN_PROGRESS
Used for progress updates during long phases.

### BLOCKED
Used when waiting on another agent or encountering issues.

### CRITICAL
Used for urgent issues requiring immediate attention.

## Standard Template

```markdown
# AGENT_[X|Y|Z]_[PHASE]_[STATUS]_[TIMESTAMP].md

## Agent: [X|Y|Z]
## Phase: [1-6]
## Status: [COMPLETE|IN_PROGRESS|BLOCKED|CRITICAL]
## Timestamp: [YYYYMMDD_HHMMSS_UTC]

### Summary
[One-line summary of status]

### Completed Tasks
- [Specific completed items]
- [Include file names and line counts]

### Files Created/Modified
- [Path]: [Description] ([Line count])

### Features/Services Identified
- [Feature name]: [Source file] - [Description]

### Dependencies on Other Agents
- From Agent [X|Y|Z]: [What is needed]

### Providing to Other Agents
- For Agent [X|Y|Z]: [What is available]

### Next Steps
- [Immediate next actions]

### Blockers/Issues
- [Any blocking issues]
- [Estimated resolution time]

### Code Snippets/Examples
```python
# Any relevant code examples
```

### Notes
[Any additional context or information]
```

## Reading Protocol

### Frequency
- Check every 30 minutes during active phases
- Check immediately when blocked
- Check after completing major tasks

### Priority Order
1. CRITICAL status files (immediate)
2. BLOCKED status files (within 15 min)
3. COMPLETE status files (within 30 min)
4. IN_PROGRESS status files (regular check)

## Inter-Agent Dependencies

### Agent X → Others
- Provides: Core architecture, base dashboard structure
- Needs: Feature modules (Y), Service interfaces (Z)

### Agent Y → Others
- Provides: Feature modules, ML/Security components
- Needs: Core interfaces (X), Service endpoints (Z)

### Agent Z → Others
- Provides: Service layer, WebSocket, APIs
- Needs: Core hooks (X), Feature services (Y)

## Coordination Points

### Phase 1 → Phase 2
All agents must complete analysis before consolidation begins.

### Phase 2 → Phase 3
X must have core ready before Y and Z integrate.

### Phase 3 → Phase 4
Integration must be complete before modularization.

### Phase 4 → Phase 5
All modules ready before final unification.

## Conflict Resolution

### Decision Priority
1. Agent with file ownership
2. Agent with most context
3. Architectural consistency (X has precedence)
4. Performance impact (Z has precedence)
5. Feature completeness (Y has precedence)

## Example Files

### Success Example
```
AGENT_X_PHASE1_COMPLETE_20250823_120000_UTC.md
```

### Blocker Example
```
AGENT_Y_PHASE2_BLOCKED_20250823_143000_UTC.md
```

### Critical Example
```
AGENT_Z_PHASE3_CRITICAL_20250823_160000_UTC.md
```

## Best Practices

1. **Be Specific**: Include file names, line numbers, function names
2. **Be Timely**: Post updates as soon as milestones are reached
3. **Be Clear**: State exactly what you need from others
4. **Be Available**: Provide clear interfaces/examples for others
5. **Be Proactive**: Alert on potential issues before they block

## Archive Policy

After project completion:
- All handoff files move to `handoff/archive/`
- Maintain chronological order
- Keep for future reference

## Emergency Escalation

If blocked > 2 hours:
1. Post CRITICAL status
2. Include specific blocker details
3. Suggest potential solutions
4. Tag specific agent needed

## Success Metrics

- Average response time < 30 minutes
- Zero features lost in handoffs
- All dependencies resolved
- Clean integration achieved

---

Remember: Clear communication ensures successful consolidation!
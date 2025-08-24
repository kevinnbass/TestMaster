# Multi-Agent Coordination Protocol

## Agent Roles & Responsibilities

### Agent A - Architecture Lead
- Core system consolidation
- Module size enforcement (max 300 lines)
- Archive management
- Main architecture decisions

### Agent B - Intelligence Specialist  
- ML and AI capabilities
- Analytics engines
- Predictive systems
- Pattern recognition

### Agent C - Testing Specialist
- Testing frameworks
- Test automation
- Coverage analysis
- Test optimization

### Agent D - Documentation & Security
- Documentation generation
- API documentation
- Security scanning
- Compliance checking

## File Ownership Matrix

| Directory/File | Owner | Others Can Read | Others Can Modify |
|---------------|-------|-----------------|-------------------|
| core/intelligence/__init__.py | A | Yes | No |
| core/intelligence/base/ | A | Yes | No |
| core/intelligence/ml/ | B | Yes | No |
| core/intelligence/testing/*.py (new) | C | Yes | No |
| core/intelligence/documentation/ | D | Yes | No |
| core/intelligence/security/ | D | Yes | No |
| archive/ | A | Yes | No (only A archives) |
| ARCHITECTED_CODEBASE.md | A | Yes | No |
| PROGRESS.md | All | Yes | Yes (append only) |
| tests/ | C | Yes | Yes (with notice) |

## Coordination Protocol

### 1. Before Starting Work
```
1. Check PROGRESS.md for current status
2. Check file ownership matrix
3. Announce your work in PROGRESS.md
4. Proceed if no conflicts
```

### 2. During Work
```
1. Only modify files you own
2. If you need changes in others' files:
   - Document need in PROGRESS.md
   - Wait for owner to make changes
   - Or create interface/adapter in your space
3. Archive before deleting anything
4. Keep modules under 300 lines
```

### 3. After Completing Work
```
1. Update PROGRESS.md with:
   - What was completed
   - Any new files created
   - Any dependencies on other agents
   - Next steps needed
2. Run your specific tests
3. Document any API changes
```

## Communication Channels

### PROGRESS.md Format
```markdown
## Agent [A/B/C/D] Update - [timestamp]
### Completed
- [List completed tasks]

### Created Files
- [List new files]

### Needs From Other Agents
- [Agent X]: [Specific need]

### Next Steps
- [What you'll do next]
```

## Conflict Resolution

1. **File Conflict**: If two agents need same file
   - Agent A (Architecture Lead) decides
   - Create adapter/interface if needed
   - Document resolution in PROGRESS.md

2. **Feature Conflict**: If features overlap
   - Identify common interface
   - Agent A makes architectural decision
   - Implement in respective domains

3. **Test Conflict**: If tests fail due to others' changes
   - Agent C coordinates test fixes
   - Original author fixes their component
   - Document root cause

## Critical Rules

1. **NEVER DELETE** - Always archive
2. **NEVER EXCEED 300 LINES** - Split modules
3. **NEVER MODIFY OTHERS' FILES** - Without permission
4. **ALWAYS UPDATE PROGRESS.md** - Communication is key
5. **ALWAYS TEST** - Before marking complete
6. **ALWAYS DOCUMENT** - APIs and changes

## Parallel Work Guidelines

### Safe Parallel Tasks
- Different directories
- Different test suites
- Documentation generation
- Archive searching

### Sequential Tasks (Require Coordination)
- Core architecture changes
- API modifications
- Test framework changes
- Security policy updates

## Success Criteria

### Individual Success
- All assigned modules under 300 lines
- All tests passing
- Documentation complete
- No security vulnerabilities

### Collective Success
- Zero functionality loss
- All 909+ APIs preserved
- Seamless integration
- Complete documentation
- Unified architecture

## Emergency Protocol

If you encounter:
1. **Blocking Issue**: Document in PROGRESS.md with "BLOCKED:" prefix
2. **Critical Bug**: Document with "CRITICAL:" prefix
3. **Architecture Question**: Tag Agent A in PROGRESS.md
4. **Integration Issue**: Document all involved agents

## Current System State

### Completed
- Testing hub modularized (Agent A)
- Integration hub modularized (Agent A)
- API layer created (Agent A)
- 7/9 integration tests passing

### In Progress
- Large module modularization
- Archive feature discovery
- Redundancy elimination

### Pending
- ML enhancement integration
- Advanced testing frameworks
- Documentation automation
- Security scanning

## Remember
The goal is the ULTIMATE COMPANION to Claude Code - comprehensive, elegant, and powerful. Every decision should support this vision.
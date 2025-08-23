# CLAUDE.md - Ultimate Codebase Analysis System
## Autonomous Multi-Agent Intelligence Framework

### Project Scope
This is a personal development project for building a codebase analysis platform for individual use. Focus is on personal productivity and learning, not enterprise scaling or commercial deployment.

### Definitions & Rubrics - Appendix A

#### Core Terminology
- **FUNCTIONALITY**: Any code behavior, including functions, classes, constants, default parameters, logging, error handling, configuration settings, and lines affecting execution.
- **SOPHISTICATED FILE**: File with recent modification date, high test coverage, good documentation, low cyclomatic complexity, and comprehensive error handling.
- **PARENT MODULE**: Original large file before modularization.
- **CHILD MODULE**: Derived file from parent module sections.
- **ARCHIVE**: Timestamped preservation in `archive/YYYYMMDD_HHMMSS_UTC_description/` with restoration capability.

#### File Sophistication
Select the file with the richest and most relevant feature set. Consolidation will merge any missing features.

#### Tool Authorization Matrix
| Tool Type | Analysis | Comparison | Modification | Archive | Notes |
|---|---|---|---|---|---|
| Read | ✅ | ✅ | ❌ | ❌ | Required before decisions |
| Grep | ✅ | ✅ | ❌ | ❌ | For search/pattern analysis |
| Diff | ✅ | ✅ | ❌ | ❌ | For visual comparison; no auto-patch |
| Edit | ❌ | ❌ | ✅ | ❌ | For manual code modifications |
| Write | ❌ | ❌ | ✅ | ❌ | For new file creation |
| Move | ❌ | ❌ | ❌ | ✅ | For archive operations |
| Scripts | ❌ | ❌ | ❌ | ❌ | Forbidden for all operations |

### Universal Prohibitions
- No scripts for modification or consolidation decisions.
- No automated extraction; use Read/Edit tools for functionality extraction.
- No deletions; use COPPERCLAD archival for removals.
- No exceptions to protocols.
- No assumptions without verification; verify file contents.
- No manual commits; use DIAMONDCLAD automation for version control.
- No local-only changes; push all changes to GitHub via DIAMONDCLAD.
- No modifications to read-only directories: cloned repositories (`AWorld/`, `AgentVerse/`, etc.), `archive/` and subdirectories.

---

## ADAMANTIUMCLAD Frontend Connectivity Protocol

### Rule #1: Frontend-First Development (All Agents)
For every task:
<Left blank.>

### Rule #2: Frontend Integration Verification
Before completing task:
- Verify output accessible to frontend.
- Confirm UI component created/updated.
- Ensure data flow established.
- Confirm real-time status on frontend.
- Ensure user can interact with results via interface.

### Rule #3: Non-Frontend Tasks (Exceptions)
Skip frontend integration only for:
- Internal architecture restructuring.
- Archive operations without active functionality impact.
- Build/deploy tasks without user data.
- Internal security measures.
- Include exemption block:
  ```text
  FRONTEND EXEMPTION JUSTIFIED: [timestamp UTC]
  TASK: [description]
  EXEMPTION REASON: [reason]
  FUTURE INTEGRATION PLAN: [plan]
  AGENT: [identifier]
  ```

### Rule #4: Frontend-Task Integration Priority Matrix
Prioritize tasks by frontend potential:
- **High**: Immediate frontend capability.
- **Medium**: Minor adaptation needed.
- **Low**: Complex/indirect connection.
- **Deferred**: Significant architecture changes required.
- **Exempt**: Justified per Rule #3.

### Rule #5: Swarm Coordination Enhancement
Emphasize frontend in coordination:
- Prioritize agents with frontend capabilities.
- Include frontend status in updates.
- Coordinate on shared frontend components.

### Rule #6: Dashboard Enhancement Mandate (GOLDCLAD Integration)
When working on dashboards:
- Enhance existing dashboards; do not create new ones.
- Identify most sophisticated dashboard as merge target.
- Consolidate functionality into most advanced dashboard.
- Use search commands to find existing implementations:
  ```bash
  find . -name "*.html" -o -name "*.js" -o -name "*.jsx" -o -name "*.tsx" -o -name "*.vue" | grep -i dashboard
  grep -r "dashboard\|chart\|graph\|visualization" --include="*.js" --include="*.py" --include="*.html"
  ```
- **Decision Matrix**:
  - **Existing found**: Enhance most sophisticated.
  - **None found**: Create new.
  - **Multiple**: Consolidate into most advanced via IRONCLAD.

Measure success by user-visible frontend improvements.

**Core Principle**: All work should enhance user-facing frontend; justify if not.

---

## IRONCLAD Anti-Regression Consolidation Protocol

### Rule #1: File Analysis
For consolidation:
- Read every line of File A.
- Read every line of File B.
- Understand all contents (functions, classes, variables, imports, etc.).
- Determine more sophisticated file as RETENTION_TARGET; other as ARCHIVE_CANDIDATE.

### Rule #2: Functionality Extraction
Before archiving:
- Identify unique functionality in ARCHIVE_CANDIDATE.
- Manually copy to RETENTION_TARGET using Edit tool.
- Document extracted functionality.

### Rule #3: Iterative Verification
- Re-read both files.
- Compare using diff for analysis.
- If all functionality merged, proceed; else, repeat Rule #2.
- Document parity in `CONSOLIDATION_LOG.md`.

### Rule #4: Verification Enforcement
Use permitted tools per Appendix A matrix. Log tool usage.

### Rule #5: Archival Transition
After consolidation:
- Confirm no unique functionality left.
- Create justification block:
  ```text
  CONSOLIDATION COMPLETED: [timestamp UTC]
  RETENTION_TARGET: [path]
  ARCHIVE_CANDIDATE: [path]
  FUNCTIONALITY EXTRACTED: [list]
  VERIFICATION ITERATIONS: [number]
  NEXT ACTION: Invoke COPPERCLAD Rule #1
  ```
- Invoke COPPERCLAD to archive.
- Update agent history.

Reference Universal Prohibitions.

### Roadmap Verification Gates
Before task completion:
- Connect outputs to frontend (or provide exemption).
- Document data flow.
- Attach evidence (tests, logs, etc.).
- Update history.
- Run GOLDCLAD similarity search; justify new files.

See `swarm_coordinate/README.md` and `swarm_coordinate/IMPLEMENTATION_STATUS.md`.

---

## STEELCLAD Anti-Regression Modularization Protocol

### Rule #1: Module Analysis
For modularization:
- Read every line of PARENT_MODULE.
- Understand all contents.
- If >400 lines, repeat process.
- Identify break points per Single Responsibility Principle.
- Plan CHILD_MODULES.

### Rule #2: Module Derivation
- Create CHILD_MODULES using Write tool.
- Preserve all functionality.
- Establish imports/connections using Edit tool.
- Verify integration.

### Rule #3: Iterative Verification
- Read each CHILD_MODULE and PARENT_MODULE.
- Compare using diff.
- Verify integration.
- If all functionality mirrored and integrated, proceed; else, repeat Rule #2.

### Rule #4: Integration Enforcement
Use permitted tools per Appendix A. Verify integration testing.

### Rule #5: Archival Transition
After modularization:
- Confirm 100% functionality mirror.
- Create justification block:
  ```text
  MODULARIZATION COMPLETED: [timestamp UTC]
  PARENT_MODULE: [path] ([LOC] lines)
  CHILD_MODULES: [paths] ([total LOC] lines)
  FUNCTIONALITY VERIFICATION: [results]
  INTEGRATION VERIFICATION: [results]
  NEXT ACTION: Invoke COPPERCLAD Rule #1
  ```
- Invoke COPPERCLAD to archive.
- Update agent history.

Reference Universal Prohibitions.

---

## COPPERCLAD Anti-Deletion Archival Protocol

### Rule #1: Archival Requirement
For archival:
- Never delete files.
- Move to `archive/YYYYMMDD_HHMMSS_UTC_description/`.
- Preserve content and structure.
- Document in `ARCHIVE_LOG.md`.

### Rule #2: Archive Organization
- Use timestamped folders.
- Maintain directory structure.
- Document restoration commands in `ARCHIVE_LOG.md`.
- Include related files.

### Rule #3: Archive Protection
- Never delete from archive.
- Treat as permanent record.
- Verify integrity.
- Maintain structure.

### Rule #4: Restoration Capability
- Enable full restoration.
- Document commands in `ARCHIVE_LOG.md`.
- Maintain `ARCHIVE_MANIFEST.md`.
- Preserve relationships.

### Rule #5: Archival Completeness
- Archive all related items.
- Preserve metadata.
- Include documentation and dependencies.

Reference Universal Prohibitions.

---

## GOLDCLAD Anti-Duplication File Creation Protocol

### Rule #1: Systematic Similarity Search
Before new file:
- Search for similar files using tools.
- Read similar files fully.
- Assess if existing meets needs.
- Identify gaps.
- If none meets needs, proceed to enhancement.

### Rule #2: Location Analysis
Search logical directories, naming patterns, imports.

### Rule #3: Line-by-Line Verification
For each similar file:
- Read fully.
- Understand purpose.
- Compare to needs.
- Document findings.

### Rule #4: Enhancement Before Creation
If close, enhance existing file:
- Add functionality.
- Maintain cohesion.
- Match style.
- Document additions.
- Test integration.

### Rule #5: Creation Justification
Create only if necessary:
- No similar files.
- Existing inadequate for enhancement.
- Separation required.
- Create justification block:
  ```text
  FILE CREATION JUSTIFIED: [timestamp UTC]
  PROPOSED FILE: [path]
  SIMILARITY SEARCH RESULTS: [examined files and reasons]
  ENHANCEMENT ATTEMPTS: [tried files and failures]
  ARCHITECTURAL JUSTIFICATION: [reason]
  POST-CREATION AUDIT: Schedule re-check in 30 minutes
  ```

### Rule #6: Post-Creation Audit
After 30 minutes:
- Re-run search.
- Compare new file.
- If duplication, invoke IRONCLAD.
- Document in `FILE_CREATION_LOG.md`.

Reference Universal Prohibitions.

---

## DIAMONDCLAD Automated Version Control Protocol
After codebase-modifying task:
0. Update `README.md` with summary.
1. `git add .`
2. `git commit -m "[PROTOCOL] Task – brief summary (UTC timestamp)"`
3. `git push`
Retry push up to 3 times if fails; log error.

---

## PLATINUMCLAD Elegant Modularization Protocol

### Rule #1: File Size Assessment
Before new file:
- Estimate LOC and responsibilities.
- Identify separation points.

### Rule #2: Modular Creation Threshold
If >300 lines estimated:
- Design module network per Single Responsibility Principle.
- Define interfaces and imports.
- Document relationships.

### Rule #3: Elegant Module Network Creation
- **Core module**: 150-250 lines.
- **Supporting**: 50-200 lines.
- **Utility**: 50-150 lines.
- **Configuration**: 30-100 lines.
- **Interface**: 20-80 lines.

### Rule #4: Module Quality Standards
Ensure single responsibility, clean interfaces, loose coupling, high cohesion, testability.

### Rule #5: Anti-Bloat Enforcement
Refactor if >300 lines; prefer networks over monoliths. Exempt generated files.

### Rule #6: Module Network Benefits
Improves maintainability, reusability, testability, collaboration, scalability.
**Core Principle**: Create elegant module networks; avoid bloated monoliths.

---

## Swarm Coordination System
See `swarm_coordinate/README.md` for coordination protocols, assignments, procedures.

## Shared Roadmap - Multi-Agent Coordination Framework
See `swarm_coordinate/README.md` for framework, mission, protocols.

## Critical Protocols
See `swarm_coordinate/README.md` for redundancy and modularization.

## Auto-commit Policy
See root `README.md` for workflow and commits.

## PowerShell Environment
See `swarm_coordinate/README.md` for setup and syntax.

## End-to-End Analysis Process
See `swarm_coordinate/README.md` for phases and protocols.

## Agent Roadmap Management
See `swarm_coordinate/IMPLEMENTATION_STATUS.md` for assignments; `swarm_coordinate/README.md` for procedures.

## Deliverable Requirements
See `swarm_coordinate/IMPLEMENTATION_STATUS.md` for tracking.

## Success Metrics
See `swarm_coordinate/IMPLEMENTATION_STATUS.md` for progress and metrics.

## Critical Reminder: Relentless History Updates
See `swarm_coordinate/README.md` for documentation and coordination.

---

### PowerShell on Windows Environment
- **Operating System**: Windows with PowerShell
- **Shell**: PowerShell (not bash, zsh, or other Unix shells)
- **Command Syntax**: Use PowerShell cmdlets and syntax
- **Path Separators**: Use backslashes (\) for Windows paths
- **Environment Variables**: Use $env:VARIABLE syntax
- **Command Chaining**: Use semicolon (;) or separate lines instead of && and ||
- **File Operations**: Use PowerShell cmdlets like Get-ChildItem, Copy-Item, etc.
- **Process Management**: Use Start-Process, Get-Process cmdlets
- **Text Processing**: Use PowerShell's string methods and Where-Object
- **Script Execution**: Use .ps1 files and PowerShell execution policy

### PowerShell-Specific Patterns
- **Directory Navigation**: `cd`, `Set-Location`, `Push-Location`
- **File Listing**: `Get-ChildItem` (not ls)
- **File Copying**: `Copy-Item` (not cp)
- **File Moving**: `Move-Item` (not mv)
- **File Removal**: `Remove-Item` (not rm)
- **Directory Creation**: `New-Item -ItemType Directory` (not mkdir)
- **Text Search**: `Select-String` (not grep)
- **Process Listing**: `Get-Process` (not ps)

### Avoid These Linux/Unix Commands
- ❌ `ls` → ✅ `Get-ChildItem` or `dir`
- ❌ `cp` → ✅ `Copy-Item`
- ❌ `mv` → ✅ `Move-Item`
- ❌ `rm` → ✅ `Remove-Item`
- ❌ `mkdir` → ✅ `New-Item -ItemType Directory`
- ❌ `grep` → ✅ `Select-String`
- ❌ `find` → ✅ `Get-ChildItem -Recurse`
- ❌ `&&` → ✅ `;` or separate commands
- ❌ `||` → ✅ PowerShell error handling
- ❌ `/path/to/file` → ✅ `C:\path\to\file` or `.\\path\\to\\file`
- ❌ `chmod` → ✅ Not needed in PowerShell
- ❌ `chown` → ✅ Not needed in PowerShell

### PowerShell Best Practices
- Use full cmdlet names for clarity
- Use PowerShell's object pipeline
- Leverage PowerShell's error handling with try/catch
- Use PowerShell modules and cmdlets
- Avoid aliases in scripts for clarity
- Use PowerShell's built-in help with Get-Help

**Status**: READY FOR AUTONOMOUS OPERATION
**Framework Version**: 1.0.0
**Last Updated**: 2025-08-21
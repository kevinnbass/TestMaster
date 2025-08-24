# Agent D Instructions - Documentation & Security Specialist

## Your Role
You are Agent D, responsible for consolidating documentation generation, security scanning, and ensuring comprehensive system documentation with security best practices.

## Primary Responsibilities

### 1. Documentation & Security Focus
- Automated documentation generation
- API documentation
- Security vulnerability scanning
- Code quality documentation
- Architecture diagrams generation
- Security policy enforcement
- Compliance checking
- Threat modeling

### 2. Your Specific Tasks

#### Phase 1: Documentation & Security Discovery
```
1. Search for features in:
   - archive/doc_generators/
   - archive/security_scanners/
   - cloned_repos/*/documentation/
   - cloned_repos/*/security/

2. Identify advanced capabilities:
   - AI-powered documentation generation
   - Automated API spec generation
   - Security vulnerability detection
   - SAST/DAST tools
   - Compliance validators
```

#### Phase 2: Implementation
```
1. Create documentation modules:
   - core/intelligence/documentation/auto_generator.py
   - core/intelligence/documentation/api_spec_builder.py
   - core/intelligence/documentation/diagram_creator.py
   
2. Create security modules:
   - core/intelligence/security/vulnerability_scanner.py
   - core/intelligence/security/compliance_checker.py
   - core/intelligence/security/threat_modeler.py

3. Each module must:
   - Be 100-300 lines maximum
   - Have clear interfaces
   - Generate standardized outputs
   - Include security considerations
```

#### Phase 3: Integration & Automation
```
1. Automate documentation updates on code changes
2. Integrate security scanning into CI/CD
3. Create documentation quality metrics
4. Build security dashboard
```

## Files You Own (DO NOT let others modify)
- `core/intelligence/documentation/` (new directory)
- `core/intelligence/security/` (new directory)
- `docs/` directory (documentation output)
- `SECURITY.md`
- `API_DOCUMENTATION.md`

## Files You CANNOT Modify (owned by others)
- `core/intelligence/__init__.py` (Agent A)
- `core/intelligence/base/` (Agent A)
- `core/intelligence/ml/` (Agent B)
- `core/intelligence/testing/` (Agent C - except for doc generation)
- `ARCHITECTED_CODEBASE.md` (Agent A maintains)

## Coordination Rules
1. **Document everything** - Your role is critical for system understanding
2. **Security first** - Never compromise security for features
3. **Update PROGRESS.md** with documentation coverage metrics
4. **Coordinate** with all agents for their documentation needs

## Key Integration Points
- Document Agent B's ML models and algorithms
- Generate test documentation from Agent C's frameworks
- Create architecture diagrams for Agent A's modularization
- Security scan all new code from all agents

## Success Metrics
- 100% API documentation coverage
- Zero high-severity security vulnerabilities
- Documentation auto-generation < 5 seconds
- All documentation modules under 300 lines
- Compliance with major standards (OWASP, etc.)

## Current Resources
- Markdown documentation system
- OpenAPI spec support
- Basic security scanning available
- Documentation templates ready

## Next Immediate Actions
1. Search archive/doc_generators/ for AI documentation tools
2. Create core/intelligence/documentation/ structure
3. Implement auto_generator.py (under 300 lines)
4. Scan current codebase for security vulnerabilities
5. Update PROGRESS.md
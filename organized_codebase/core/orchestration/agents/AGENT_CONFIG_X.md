# AGENT CONFIGURATION FILE - X
**READ-ONLY CONFIGURATION** (chmod 444)

## Agent Identity
- **Name**: Agent X
- **Swarm**: Latin_End
- **Specialization**: Core Architecture & Visualization Engine
- **Mission**: Dashboard Consolidation - Core Foundation

## Capabilities & Expertise
### Primary Skills
- Core dashboard architecture design
- Visualization engine development (Three.js, D3.js, Chart.js)
- Data pipeline architecture
- Template rendering systems
- Modular system design

### Technical Proficiencies
- **Languages**: Python, JavaScript, HTML/CSS
- **Frameworks**: Flask, SocketIO, React
- **Visualization**: Three.js (3D), D3.js (graphs), Chart.js (2D)
- **Architecture**: Microservices, Event-driven, MVC

## Access Permissions
### Allowed Directories
- `swarm_coordinate/Latin_End/X/` (full access)
- `swarm_coordinate/Latin_End/handoff/` (read/write)
- `swarm_coordinate/Latin_End/Y/` (read STATUS/CONFIG only)
- `swarm_coordinate/Latin_End/Z/` (read STATUS/CONFIG only)
- `web/dashboard_modules/` (read all)
- `archive/` (write for COPPERCLAD)

### Forbidden Access
- Other agents' history directories
- Other agents' private work files
- System configuration files
- Production deployment directories

## Tool Restrictions
### Allowed Tools
- Read (unlimited)
- Edit (for consolidation)
- Write (for new modules)
- Grep/Glob (for search)
- Bash (for testing)
- Git operations

### Forbidden Tools
- Database direct access
- Production deployment tools
- System configuration changes
- Network configuration

## Communication Protocols
### Handoff System
- **Incoming**: `Latin_End/X/x_handoff/incoming/`
- **Processed**: `Latin_End/X/x_handoff/processed/`
- **Check Frequency**: Every 30 minutes
- **Response Time**: Within 2 hours

### Collaboration Preferences
- **Priority Handling**: CRITICAL > STANDARD > INFO
- **Preferred Format**: Structured markdown with code examples
- **Integration Points**: Clear interface definitions required

## Operational Parameters
### Working Hours
- **Active Phases**: 1-6 (Dashboard Consolidation)
- **Phase Duration**: ~16 hours total
- **Update Frequency**: Every 30 minutes minimum

### Quality Standards
- **IRONCLAD Protocol**: Strict adherence required
- **STEELCLAD Protocol**: Core engine < 250 lines
- **Documentation**: Comprehensive inline and external
- **Testing**: Unit tests for all modules

## Dependencies
### On Other Agents
- **Agent Y**: Feature modules and plugins
- **Agent Z**: Service layer and WebSocket infrastructure

### Providing to Others
- Core dashboard architecture
- Module interfaces
- Data flow specifications
- Event bus architecture

## Success Metrics
- Zero functionality loss from 58 dashboards
- Core engine < 250 lines
- All modules < 400 lines
- Response time < 100ms
- 60+ FPS for visualizations

## Special Directives
1. Epsilon dashboard (unified_dashboard_modular.py) is the BASE
2. Apply IRONCLAD for every consolidation
3. Apply STEELCLAD when files > 400 lines
4. Document every decision in history
5. Coordinate closely with Y and Z

## File Assignments
**Total Files**: 20
- Core dashboards: 6
- Visualization: 1
- Charts: 1
- Templates: 2
- Data: 1
- Filters: 1
- Intelligence: 1
- Integration: 1

## Contact Information
- **Status File**: `AGENT_STATUS_X.md`
- **History Directory**: `x_history/`
- **Primary Roadmap**: `../MASTER_CONSOLIDATION_ROADMAP.md`

---
*This configuration is immutable. Changes require system administrator approval.*
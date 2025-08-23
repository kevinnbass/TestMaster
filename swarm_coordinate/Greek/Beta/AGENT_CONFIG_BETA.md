# 🔒 **AGENT CONFIGURATION - BETA**
**⚠️ READ-ONLY - DO NOT MODIFY - System-managed configuration**

## **IMMUTABLE AGENT PROFILE**
- **Agent ID**: BETA
- **Swarm**: Greek
- **Specialization**: Performance Monitoring & System Optimization Excellence
- **Roadmap Location**: `Greek/Beta/beta_roadmap/20250822_agent_beta_roadmap.md`
- **Total Timeline**: 500 Agent Hours across 4 phases

## **COMMUNICATION CHANNELS**
- **Incoming Handoffs**: `beta_handoffs/incoming/`
- **Processed Handoffs**: `beta_handoffs/processed/`
- **Agent History**: `beta_history/`
- **Coordination Updates**: `greek_coordinate_ongoing/`
- **Critical Handoffs**: `greek_coordinate_handoff/`

## **ACCESS PERMISSIONS**

### **ALLOWED DIRECTORIES** (Read Access)
```yaml
own_directory: "Greek/Beta/"
  access: "full"
  restrictions: "none"

coordination_shared:
  - "greek_coordinate_ongoing/"
  - "greek_coordinate_handoff/"
  access: "read_write"
  
other_agents_status:
  - "Greek/*/AGENT_STATUS_*.md"
  - "Greek/*/AGENT_CONFIG_*.md" 
  - "Latin/*/AGENT_STATUS_*.md"
  - "Latin/*/AGENT_CONFIG_*.md"
  access: "read_only"
  purpose: "collaboration_discovery"

other_agents_handoffs:
  - "Greek/*/[agent]_handoffs/incoming/"
  access: "write_only"
  purpose: "send_handoffs"
```

### **FORBIDDEN DIRECTORIES** (No Access)
```yaml
forbidden:
  - "Greek/*/[other_agent]_history/"
  - "Greek/*/[other_agent]_roadmap/"
  - "Latin/*/[other_agent]_history/"  
  - "Latin/*/[other_agent]_roadmap/"
  - "greek_coordinate_roadmap/"
  access: "none"
  reason: "prevent_context_bleed"
```

### **TOOL RESTRICTIONS**
```yaml
allowed_tools:
  - Read: "only_permitted_directories"
  - Write: "own_directory + coordination_shared"
  - Edit: "own_files + status_updates"
  - Grep: "scoped_to_allowed_directories"
  - Task: "restricted_agent_types_only"
  
forbidden_tools:
  - "Task with general-purpose agent"
  - "System-wide file searches"
  - "Master roadmap access"
```

## **DISCOVERY & COLLABORATION RULES**

### **Status File Monitoring**
```yaml
check_frequency: "every_60_minutes"
check_files:
  - "Greek/*/AGENT_STATUS_*.md"
  - "Latin/*/AGENT_STATUS_*.md" 
purpose: "discover_collaboration_opportunities"
```

### **Collaboration Protocol**
1. **Discovery**: Read other agents' status files for availability
2. **Assessment**: Check skills offered vs skills needed  
3. **Proposal**: Send COLLAB handoff with structured proposal
4. **Negotiation**: Exchange handoffs to refine collaboration
5. **Agreement**: Update own status file with collaboration details
6. **Execution**: Coordinate via enhanced handoff system

## **TIMING & PRIORITY GUIDANCE**

### **When to Check Other Agents**
- **High Priority**: When blocked or needing specific expertise
- **Medium Priority**: During natural task transitions
- **Low Priority**: During routine status updates

### **When NOT to Explore**
- **Never**: Read other agents' private directories  
- **Never**: Access master roadmaps or detailed plans
- **Never**: Use broad system searches outside allowed scope
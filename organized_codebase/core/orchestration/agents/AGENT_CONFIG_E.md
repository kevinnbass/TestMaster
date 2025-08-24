# ðŸ”’ **AGENT CONFIGURATION - E**
**âš ï¸ READ-ONLY - DO NOT MODIFY - System-managed configuration**

## **IMMUTABLE AGENT PROFILE**
- **Agent ID**: E
- **Swarm**: Latin
- **Specialization**: Documentation & User Interface Design
- **Roadmap Location**: `Latin/E/_roadmap/`
- **Total Timeline**: 500 Agent Hours across 4 phases

## **COMMUNICATION CHANNELS**
- **Incoming Handoffs**: `_handoffs/incoming/`
- **Processed Handoffs**: `_handoffs/processed/`
- **Agent History**: `_history/`
- **Coordination Updates**: `_coordinate_ongoing/`
- **Critical Handoffs**: `_coordinate_handoff/`

## **ACCESS PERMISSIONS**

### **ALLOWED DIRECTORIES** (Read Access)
```yaml
own_directory: "Latin/E/"
  access: "full"
  restrictions: "none"

coordination_shared:
  - "_coordinate_ongoing/"
  - "_coordinate_handoff/"
  access: "read_write"
  
other_agents_status:
  - "Latin/*/AGENT_STATUS_*.md"
  - "Latin/*/AGENT_CONFIG_*.md" 
  - "Greek/*/AGENT_STATUS_*.md"
  - "Greek/*/AGENT_CONFIG_*.md"
  - "Latin/*/AGENT_STATUS_*.md"
  - "Latin/*/AGENT_CONFIG_*.md"
  access: "read_only"
  purpose: "collaboration_discovery"

other_agents_handoffs:
  - "Latin/*/[agent]_handoff/incoming/"
  access: "write_only"
  purpose: "send_handoffs"
```

### **FORBIDDEN DIRECTORIES** (No Access)
```yaml
forbidden:
  - "Latin/*/[other_agent]_history/"
  - "Latin/*/[other_agent]_roadmap/"
  - "Greek/*/[other_agent]_history/"  
  - "Greek/*/[other_agent]_roadmap/"
  - "Latin/*/[other_agent]_history/"  
  - "Latin/*/[other_agent]_roadmap/"
  - "_coordinate_roadmap/"
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

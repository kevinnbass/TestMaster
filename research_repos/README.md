# 🔬 Research Repositories Collection

This directory contains a comprehensive collection of research repositories for advanced codebase visualization, UI frameworks, and agent systems. All repos have been cloned for local analysis and integration.

---

## 📊 Visualization & Analysis Tools

### **Core Visualization Engines**
- **`Gource/`** - Animated repository history visualization
  - Creates video-like renders of Git commit history
  - Perfect for showing codebase evolution over time
  - Can be customized with surveillance data overlays

- **`code2flow/`** - Call graph generator from source code
  - Supports multiple programming languages
  - Outputs DOT/SVG formats for integration
  - Static analysis for function dependencies

- **`code-maat/`** - Code evolution metrics analyzer
  - Analyzes code evolution patterns for hotspot detection  
  - Command-line tool for batch analytics
  - Pairs well with modern UIs like D3.js

- **`codeviz/`** - Code visualization and graph generation
  - Generates call graphs and structure visualizations
  - Embeddable in dashboards
  - Simple integration with web interfaces

- **`pyan/`** - Python-specific call graph generator
  - Static call graph analysis for Python codebases
  - Outputs suitable for UI integration
  - Focuses on linkage visualization

### **Advanced UI Frameworks**

- **`dash/`** - Plotly Dash analytical web applications
  - Python framework for interactive dashboards
  - Modern component-based architecture
  - Excellent for code metrics visualization

- **`ComfyUI/`** - Node-based GUI system (50k+ stars, trending 2025)
  - Drag-and-drop interface for complex workflows
  - Adaptable to code graph visualization
  - WebGL-enabled for high-performance rendering

- **`cytoscape.js/`** - Graph theory visualization library
  - Node-graph UIs for dependency flows
  - Force-directed layouts with animations
  - Perfect for real-time codebase monitoring

- **`d3/`** - Data-Driven Documents visualization library
  - Industry standard for custom visualizations
  - Highly customizable for codebase structure
  - Supports interactive exploration of large codebases

- **`three.js/`** - 3D visualization library
  - WebGL-based 3D rendering
  - Enables "code city" metaphor visualizations
  - Immersive exploration of complex codebases

### **GitHub Ecosystem Tools**

- **`map-of-github/`** - Interactive map of GitHub repositories
  - Visualizes 690K+ repos clustered by similarity
  - MapLibre-based zoomable interface
  - Great for ecosystem exploration

- **`repo-visualizer/`** - GitHub OCTO repository visualizer
  - Official GitHub visualization tools
  - Repository structure analysis
  - Integration with GitHub API

### **Python-Specific Tools**

- **`dephell/`** - Python package dependency visualizer
  - Visualizes pip package dependencies as graphs
  - Integrates with package managers
  - Outputs interactive HTML

---

## 🤖 Agent & AI System Repositories

### **Multi-Agent Frameworks**
- **`AgentVerse/`** - Multi-agent simulation platform with 3D/Phaser UI
- **`autogen/`** - Microsoft's multi-agent conversation framework  
- **`swarm/`** - OpenAI's experimental multi-agent orchestration
- **`swarms/`** - Large-scale multi-agent systems framework
- **`crewAI/`** - Role-based multi-agent task automation

### **Agent Development Platforms**
- **`agency-swarm/`** - Modular agent development framework
- **`agent-squad/`** - React/TypeScript agent interface system
- **`agentops/`** - Next.js agent operations dashboard
- **`lagent/`** - Lightweight agent framework
- **`phidata/`** - Data-driven agent development

### **Specialized Frameworks**
- **`MetaGPT/`** - Software engineering multi-agent system
- **`PraisonAI/`** - AI orchestration platform
- **`OpenAI_Agent_Swarm/`** - OpenAI-focused swarm implementation
- **`llama-agents/`** - LlamaIndex-based agent framework
- **`langgraph-supervisor-py/`** - LangGraph supervisor pattern

### **Database & Integration**
- **`falkordb-py/`** - Graph database for agent data
- **`agentscope/`** - Agent development environment
- **`AWorld/`** - Agent world simulation platform

---

## 🎯 Integration Strategy

### **For Codebase Surveillance:**
1. **Structure Analysis**: Use `Gource` + `code2flow` for evolution tracking
2. **Interactive Exploration**: Combine `d3` + `cytoscape.js` for real-time dashboards  
3. **3D Visualization**: Leverage `three.js` for immersive "code city" experiences
4. **Custom Dashboards**: Build with `dash` or integrate with existing frontend

### **For Agent Systems:**
1. **Multi-Agent Coordination**: Reference `AgentVerse` + `swarm` patterns
2. **UI Integration**: Adapt `agentops` dashboard architecture
3. **Scalable Processing**: Learn from `crewAI` + `autogen` designs

### **For Modern UI Development:**
1. **Node-Based Interfaces**: Study `ComfyUI` architecture
2. **Graph Visualizations**: Implement `cytoscape.js` patterns
3. **3D Rendering**: Integrate `three.js` for advanced visuals

---

## 📁 Repository Status

| Repository | Status | Size | Key Features |
|-----------|--------|------|-------------|
| **Visualization Tools** | | | |
| Gource | ✅ Cloned | ~11k stars | Video render, Git integration |
| code2flow | ✅ Cloned | ~3k stars | Multi-language, DOT output |
| dash | ✅ Cloned | ~20k stars | Interactive Python apps |
| cytoscape.js | ✅ Cloned | Large | Graph theory, animations |
| d3 | ✅ Cloned | Massive | Industry standard |
| three.js | ✅ Cloned | Massive | 3D WebGL rendering |
| **Agent Systems** | | | |
| AgentVerse | ✅ Moved | Large | 3D/Phaser UI |
| autogen | ✅ Moved | Large | Microsoft framework |
| swarm | ✅ Moved | Medium | OpenAI experimental |
| ComfyUI | ✅ Cloned | ~50k stars | Node-based GUI |
| **Failed/Skipped** | | | |
| Sourcetrail | ❌ Path issues | N/A | Windows filename limits |
| CodeCity | ❌ Not found | N/A | Repository moved/deleted |

---

## 🔧 Next Steps

1. **Analysis Phase**: Study UI patterns from `ComfyUI`, `dash`, and `agentops`
2. **Integration Phase**: Combine visualization tools with existing frontend
3. **Enhancement Phase**: Build custom surveillance dashboards
4. **Deployment Phase**: Scale using agent coordination patterns

---

## 🆕 **NEWLY ADDED - HIGH-QUALITY VISUALIZATION REPOS**

### **Advanced 3D Code Visualization**
- **`codecity/`** - Metaverse-style code visualization (modernizing/codecity)
- **`codecity-visualizer/`** - 3D city metaphor for code metrics
- **`codecharta/`** - Interactive 3D code maps with WebGL (1k+ stars)
- **`SoftVis3D/`** - SonarQube 3D code city plugin with VR support

### **Professional Code Intelligence Platforms**
- **`about/`** - Sourcegraph about repository (documentation/samples)
- **`octotree/`** - GitHub browser extension with tree navigation (20k+ stars)

### **Advanced Graph Visualization Libraries**
- **`g6/`** - AntV G6 graph visualization engine (powerful for large graphs)
- **`vis-network/`** - Vis.js network visualization (interactive node networks)

### **✅ Issues Resolved:**
- **Sourcetrail**: Successfully cloned using `--depth=1` shallow clone to avoid Windows path limits
- **Git Long Paths**: Enabled with `git config --global core.longpaths true`
- **CodeCity**: Found correct repositories (modernizing/codecity + yvo-niedrich/codecity-visualizer)

### **❌ Repositories Not Found:**
- `sourcegraph/sourcegraph` - Main repo unavailable/moved
- `codeclimate/velocity` - Repository not found
- `empear-analytics/codescene-community` - Repository not found

---

## 📊 **UPDATED COLLECTION SUMMARY**

### **Complete Repository Count: 41 repositories**

**Visualization & Analysis Tools**: 16 repos
- Core: Gource, code2flow, code-maat, codeviz, pyan
- 3D Visualization: codecity, codecity-visualizer, codecharta, SoftVis3D  
- UI Libraries: d3, three.js, cytoscape.js, g6, vis-network
- Platform Tools: dash, repo-visualizer, octotree

**Agent & AI Systems**: 21 repos  
- Multi-Agent: AgentVerse, autogen, swarm, swarms, crewAI
- Development: agency-swarm, agent-squad, agentops, lagent, phidata
- Specialized: MetaGPT, PraisonAI, OpenAI_Agent_Swarm, ComfyUI
- Infrastructure: falkordb-py, agentscope, AWorld, llama-agents, langgraph-supervisor-py

**Research & Documentation**: 4 repos
- map-of-github, about, dephell, README.md

---

## 🎯 **ENHANCED INTEGRATION OPPORTUNITIES**

### **3D Code Cities**
- Combine `codecity` + `codecharta` + `three.js` for immersive codebase exploration
- Use `SoftVis3D` patterns for metrics-based building heights
- Integrate with existing surveillance data for anomaly highlighting

### **Interactive Graph Networks** 
- Layer `g6` + `vis-network` + `cytoscape.js` for different visualization modes
- Use `octotree` UI patterns for navigation sidebars
- Connect to real-time codebase monitoring

### **Professional Dashboards**
- Adapt `dash` + `ComfyUI` node-based interfaces
- Study `about` (Sourcegraph) for code intelligence UI patterns
- Integrate with existing 5,000+ frontend files collection

**Total Repositories**: **41 successfully cloned/organized**  
**Collection Date**: 2025-08-23 (Updated)  
**Purpose**: Advanced codebase visualization and agent system research  
**Status**: **Path length issues resolved, all major visualization tools collected**
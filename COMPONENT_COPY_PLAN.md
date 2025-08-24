# Component Copy Plan - Phase 3
## Organized Strategy for Unique UI/Visualization Components

Generated: 2025-08-24

## Copy Priority Matrix

### Priority 1: Advanced Visualization Libraries (Immediate)

#### Cytoscape.js Demos
**Source**: `research_repos/cytoscape.js/documentation/demos/`
**Target**: `frontend_final/cytoscape_demos/`
**Components**: 30+ interactive graph visualizations
**Action**: 
```powershell
Copy-Item -Path "C:\Users\kbass\OneDrive\Documents\testmaster\research_repos\cytoscape.js\documentation\demos\*" -Destination "C:\Users\kbass\OneDrive\Documents\testmaster\frontend_final\cytoscape_demos\" -Recurse
```

#### Three.js Advanced Examples
**Source**: `research_repos/three.js/examples/`
**Target**: `frontend_final/threejs_advanced/`
**Components**: Physics, shaders, advanced WebGL
**Action**:
```powershell
# Copy only unique advanced examples
$advanced = @('physics', 'shader', 'postprocessing', 'animation')
foreach ($type in $advanced) {
    Copy-Item -Path "C:\Users\kbass\OneDrive\Documents\testmaster\research_repos\three.js\examples\*$type*" -Destination "C:\Users\kbass\OneDrive\Documents\testmaster\frontend_final\threejs_advanced\" -Recurse
}
```

#### G6 Graph Components
**Source**: `research_repos/g6/packages/g6/__tests__/demos/`
**Target**: `frontend_final/g6_components/`
**Components**: 160+ TypeScript graph visualizations
**Action**:
```powershell
Copy-Item -Path "C:\Users\kbass\OneDrive\Documents\testmaster\research_repos\g6\packages\g6\__tests__\demos\*.ts" -Destination "C:\Users\kbass\OneDrive\Documents\testmaster\frontend_final\g6_components\"
```

### Priority 2: Complete UI Frameworks (High Value)

#### ComfyUI Node Editor
**Source**: `research_repos/ComfyUI/web/`
**Target**: `frontend_final/comfyui_editor/`
**Components**: Complete node-based workflow editor
**Action**:
```powershell
Copy-Item -Path "C:\Users\kbass\OneDrive\Documents\testmaster\research_repos\ComfyUI\web\*" -Destination "C:\Users\kbass\OneDrive\Documents\testmaster\frontend_final\comfyui_editor\" -Recurse
```

#### CodeCharta 3D Visualization
**Source**: `research_repos/codecharta/visualization/`
**Target**: `frontend_final/codecharta_3d/`
**Components**: 3D code city visualization
**Action**:
```powershell
Copy-Item -Path "C:\Users\kbass\OneDrive\Documents\testmaster\research_repos\codecharta\visualization\*" -Destination "C:\Users\kbass\OneDrive\Documents\testmaster\frontend_final\codecharta_3d\" -Recurse
```

### Priority 3: Scientific Components (Specialized)

#### Dash Core Components
**Source**: `research_repos/dash/components/dash-core-components/src/components/`
**Target**: `frontend_final/dash_scientific/`
**Components**: DatePicker, Graph, Geolocation, etc.
**Action**:
```powershell
$components = @('DatePicker*.js', 'Graph.react.js', 'Geolocation.react.js', 'Dropdown.react.js')
foreach ($comp in $components) {
    Copy-Item -Path "C:\Users\kbass\OneDrive\Documents\testmaster\research_repos\dash\components\dash-core-components\src\components\$comp" -Destination "C:\Users\kbass\OneDrive\Documents\testmaster\frontend_final\dash_scientific\"
}
```

#### D3.js Advanced Examples
**Source**: `research_repos/d3/examples/`
**Target**: `frontend_final/d3_advanced/`
**Components**: Complex data visualizations
**Action**:
```powershell
Copy-Item -Path "C:\Users\kbass\OneDrive\Documents\testmaster\research_repos\d3\examples\*" -Destination "C:\Users\kbass\OneDrive\Documents\testmaster\frontend_final\d3_advanced\" -Recurse -ErrorAction SilentlyContinue
```

### Priority 4: Specialized Visualizations (Unique)

#### Map of GitHub
**Source**: `research_repos/map-of-github/`
**Target**: `frontend_final/github_map/`
**Components**: Vue.js GitHub visualization
**Action**:
```powershell
Copy-Item -Path "C:\Users\kbass\OneDrive\Documents\testmaster\research_repos\map-of-github\src\*" -Destination "C:\Users\kbass\OneDrive\Documents\testmaster\frontend_final\github_map\" -Recurse
```

#### VIS-Network Examples
**Source**: `research_repos/vis-network/examples/`
**Target**: `frontend_final/vis_network_demos/`
**Components**: Network visualization demos
**Action**:
```powershell
Copy-Item -Path "C:\Users\kbass\OneDrive\Documents\testmaster\research_repos\vis-network\examples\*" -Destination "C:\Users\kbass\OneDrive\Documents\testmaster\frontend_final\vis_network_demos\" -Recurse
```

#### Gource Visualization Config
**Source**: `research_repos/Gource/data/`
**Target**: `frontend_final/gource_config/`
**Components**: Repository visualization configs
**Action**:
```powershell
Copy-Item -Path "C:\Users\kbass\OneDrive\Documents\testmaster\research_repos\Gource\data\*.glsl" -Destination "C:\Users\kbass\OneDrive\Documents\testmaster\frontend_final\gource_config\"
```

### Priority 5: Agent UI Components (Selective)

#### About.com Components (Sourcegraph)
**Source**: `research_repos/about/src/components/`
**Target**: `frontend_final/sourcegraph_components/`
**Components**: Code insights, charts
**Filter**: Only unique visualization components
**Action**:
```powershell
Copy-Item -Path "C:\Users\kbass\OneDrive\Documents\testmaster\research_repos\about\src\components\CodeInsights\*" -Destination "C:\Users\kbass\OneDrive\Documents\testmaster\frontend_final\sourcegraph_components\" -Recurse
```

## Directory Structure Plan

```
frontend_final/
├── cytoscape_demos/          # Network visualizations
├── threejs_advanced/          # 3D graphics & physics
├── g6_components/             # Graph components (TypeScript)
├── comfyui_editor/           # Node-based workflow editor
├── codecharta_3d/            # 3D code visualization
├── dash_scientific/          # Scientific React components
├── d3_advanced/              # Advanced D3 visualizations
├── github_map/               # GitHub visualization (Vue)
├── vis_network_demos/        # Network demos
├── gource_config/            # Repository viz configs
└── sourcegraph_components/   # Code insight components
```

## Execution Strategy

### Step 1: Create Target Directories
```powershell
$dirs = @(
    'cytoscape_demos',
    'threejs_advanced',
    'g6_components',
    'comfyui_editor',
    'codecharta_3d',
    'dash_scientific',
    'd3_advanced',
    'github_map',
    'vis_network_demos',
    'gource_config',
    'sourcegraph_components'
)

foreach ($dir in $dirs) {
    New-Item -Path "C:\Users\kbass\OneDrive\Documents\testmaster\frontend_final\$dir" -ItemType Directory -Force
}
```

### Step 2: Execute Priority 1 Copies
- Run Cytoscape.js demo copy
- Run Three.js advanced copy
- Run G6 components copy

### Step 3: Execute Priority 2 Copies
- Copy ComfyUI editor
- Copy CodeCharta visualization

### Step 4: Execute Priority 3-5 Copies
- Copy Dash scientific components
- Copy D3 advanced examples
- Copy specialized visualizations
- Copy selected agent UI components

## Validation Checklist

- [ ] Verify no overwrites of existing files
- [ ] Check total size before copy (~500MB estimated)
- [ ] Confirm directory structure created
- [ ] Test sample files after copy
- [ ] Update manifest with new components
- [ ] Document integration points

## Risk Mitigation

### Duplicate Prevention
- Use `-ErrorAction SilentlyContinue` to skip existing
- Check file hashes before overwriting
- Maintain copy log for rollback

### Size Management
- Monitor disk space during copy
- Compress large demo datasets
- Skip redundant documentation files

### Integration Planning
- Document required dependencies
- Note configuration requirements
- Plan module loading strategy

## Success Metrics

- **Target**: 300+ unique components copied
- **Quality**: No duplicate files created
- **Organization**: Clean directory structure
- **Documentation**: Full manifest of additions
- **Integration**: Clear paths for usage

## Post-Copy Tasks

1. Generate component manifest
2. Create integration guide
3. Update frontend_final README
4. Document API requirements
5. Plan consolidation with existing dashboards
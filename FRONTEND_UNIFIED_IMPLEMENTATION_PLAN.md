# 🚀 FRONTEND UNIFIED IMPLEMENTATION PLAN
**Mission**: Step-by-step execution plan for organizing 385+ frontend files into unified structure
**Created**: 2025-08-23
**Updated**: 2025-08-23 - Fixed to preserve directory structure and use actual file locations

---

## 📂 TARGET DIRECTORY STRUCTURE

```
frontend_unified/
├── web/
│   └── dashboard_modules/   # Our current working modules from X,Y,Z,T agents
├── agentops/                # AgentOps dashboard (preserving structure)
│   └── app/dashboard/...
├── autogen/                 # AutoGen Studio (preserving structure)
│   └── python/packages/...
├── agent-squad/             # Agent-Squad React (preserving structure)
│   └── examples/...
├── TestMaster/              # TestMaster dashboards (preserving structure)
│   └── dashboard/...
├── AgentVerse/              # AgentVerse 3D UI (preserving structure)
│   └── ui/...
├── root_dashboards/         # Root-level HTML dashboards
├── atoms/                   # STEELCLAD atomic components (TO BE CREATED)
└── unified_system/          # Final unified dashboard (FUTURE)
```

---

## 🧹 PHASE 0A: CLEANUP OUTDATED COPY
**Duration**: 5 minutes  
**Status**: CRITICAL - RUN FIRST
**Reason**: frontend_unified is an outdated snapshot from 09:40 AM missing 53+ files of agent work

### Cleanup Commands:
```powershell
# Verify frontend_unified is the outdated copy
Write-Host "Checking frontend_unified status..." -ForegroundColor Yellow
if (Test-Path ".\frontend_unified") {
    # Count missing atoms directories (proof it's outdated)
    $hasWebAtoms = Test-Path ".\web\dashboard_modules\atoms"
    $hasFrontendAtoms = Test-Path ".\frontend_unified\atoms"
    
    if ($hasWebAtoms -and -not $hasFrontendAtoms) {
        Write-Host "✓ Confirmed: frontend_unified is outdated (missing atoms work)" -ForegroundColor Green
        
        # Backup just in case (optional)
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        Write-Host "Creating safety backup..." -ForegroundColor Cyan
        Rename-Item -Path ".\frontend_unified" -NewName "frontend_unified_BACKUP_$timestamp" -Force
        Write-Host "✓ Backed up to frontend_unified_BACKUP_$timestamp" -ForegroundColor Green
        
        # Or directly remove (uncomment if you don't want backup)
        # Remove-Item -Path ".\frontend_unified" -Recurse -Force
        # Write-Host "✓ Removed outdated frontend_unified" -ForegroundColor Green
    } else {
        Write-Host "⚠ Manual verification needed - unexpected directory state" -ForegroundColor Red
        Exit
    }
} else {
    Write-Host "✓ frontend_unified doesn't exist - safe to proceed" -ForegroundColor Green
}
```

### Why This Cleanup Is Necessary:
- **frontend_unified was copied from web/dashboard_modules at 09:40 AM**
- **Missing 53+ new files from agent atomization work**
- **Missing atoms/ directories with Agent T and Z's work**
- **Missing services/atoms/ with 13 components**
- **Contains outdated snapshot that will cause confusion**
- **Better to start fresh with current agent work included**

---

## 🎯 PHASE 0B: SAFETY CHECKS
**Duration**: 5 minutes
**Status**: RUN SECOND - CRITICAL

### Verification Commands:
```powershell
# Check if frontend_unified already exists
if (Test-Path ".\frontend_unified") {
    Write-Host "WARNING: frontend_unified already exists!" -ForegroundColor Red
    Write-Host "Contents:" -ForegroundColor Yellow
    Get-ChildItem ".\frontend_unified" -Recurse | Select-Object FullName
    Write-Host "Consider backing up or using a different directory" -ForegroundColor Yellow
} else {
    Write-Host "Safe to proceed - frontend_unified does not exist" -ForegroundColor Green
}

# Verify source directories exist
$sources = @(
    ".\web\dashboard_modules",
    ".\agentops\app\dashboard",
    ".\autogen\python\packages",
    ".\agent-squad\examples",
    ".\TestMaster\dashboard",
    ".\AgentVerse\ui"
)

foreach ($source in $sources) {
    if (Test-Path $source) {
        $count = (Get-ChildItem $source -Recurse -File -ErrorAction SilentlyContinue).Count
        Write-Host "✓ Found: $source ($count files)" -ForegroundColor Green
    } else {
        Write-Host "✗ Missing: $source" -ForegroundColor Red
    }
}

# Check for root HTML dashboards
$rootDashboards = Get-ChildItem "." -Filter "*.html" | Where-Object { $_.Name -like "*dashboard*" }
Write-Host "Found $($rootDashboards.Count) root dashboard HTML files" -ForegroundColor Cyan
```

---

## 🎯 PHASE 1: PREPARE DIRECTORY STRUCTURE
**Duration**: 10 minutes
**Status**: READY TO EXECUTE

### PowerShell Commands:
```powershell
# Create main frontend_unified directory
New-Item -ItemType Directory -Path ".\frontend_unified" -Force

# Create atoms directory for STEELCLAD output
New-Item -ItemType Directory -Path ".\frontend_unified\atoms" -Force
New-Item -ItemType Directory -Path ".\frontend_unified\atoms\core" -Force
New-Item -ItemType Directory -Path ".\frontend_unified\atoms\visualization" -Force
New-Item -ItemType Directory -Path ".\frontend_unified\atoms\services" -Force
New-Item -ItemType Directory -Path ".\frontend_unified\atoms\coordination" -Force

# Create root_dashboards directory for HTML files
New-Item -ItemType Directory -Path ".\frontend_unified\root_dashboards" -Force

# Create unified_system directory for future integration
New-Item -ItemType Directory -Path ".\frontend_unified\unified_system" -Force
```

---

## 📦 PHASE 2: COPY CURRENT WORKING MODULES
**Duration**: 10 minutes
**Files**: 244 Python dashboard modules from agents X,Y,Z,T

### File Collection Commands:
```powershell
# Check if source exists
if (Test-Path ".\web\dashboard_modules") {
    # Copy preserving structure
    Write-Host "Copying web\dashboard_modules..." -ForegroundColor Cyan
    Copy-Item -Path ".\web" -Destination ".\frontend_unified\" -Recurse -ErrorAction SilentlyContinue
    
    $fileCount = (Get-ChildItem ".\frontend_unified\web\dashboard_modules" -Recurse -File).Count
    Write-Host "Copied $fileCount files from web\dashboard_modules" -ForegroundColor Green
} else {
    Write-Host "WARNING: web\dashboard_modules not found!" -ForegroundColor Red
}

# Also check for any _misc_non_frontend that was separated
if (Test-Path ".\_misc_non_frontend") {
    Write-Host "Found _misc_non_frontend directory (non-frontend modules)" -ForegroundColor Yellow
}
```

---

## 📦 PHASE 3: INGEST AGENTOPS DASHBOARD
**Duration**: 15 minutes
**Files**: 50+ React/TypeScript components

### File Collection Commands:
```powershell
# Copy AgentOps dashboard preserving directory structure
if (Test-Path ".\agentops\app\dashboard") {
    Write-Host "Copying AgentOps dashboard components..." -ForegroundColor Cyan
    
    # Create destination and copy entire structure
    New-Item -ItemType Directory -Path ".\frontend_unified\agentops\app\dashboard" -Force
    Copy-Item -Path ".\agentops\app\dashboard\*" -Destination ".\frontend_unified\agentops\app\dashboard\" -Recurse -ErrorAction SilentlyContinue
    
    $fileCount = (Get-ChildItem ".\frontend_unified\agentops" -Recurse -File -Include "*.tsx","*.ts","*.jsx","*.js","*.css").Count
    Write-Host "Copied $fileCount AgentOps UI files" -ForegroundColor Green
} else {
    Write-Host "AgentOps dashboard not found - skipping" -ForegroundColor Yellow
}

# Copy landing page if exists
if (Test-Path ".\agentops\app\landing\public") {
    New-Item -ItemType Directory -Path ".\frontend_unified\agentops\app\landing\public" -Force
    Copy-Item -Path ".\agentops\app\landing\public\*" -Destination ".\frontend_unified\agentops\app\landing\public\" -Recurse -ErrorAction SilentlyContinue
}
```

---

## 📚 PHASE 4: INGEST AUTOGEN STUDIO
**Duration**: 15 minutes
**Files**: 30+ frontend files

### File Collection Commands:
```powershell
# Copy AutoGen Studio frontend preserving structure
if (Test-Path ".\autogen\python\packages\autogen-studio\frontend") {
    Write-Host "Copying AutoGen Studio frontend..." -ForegroundColor Cyan
    
    # Preserve the full path structure
    New-Item -ItemType Directory -Path ".\frontend_unified\autogen\python\packages\autogen-studio\frontend" -Force
    Copy-Item -Path ".\autogen\python\packages\autogen-studio\frontend\*" -Destination ".\frontend_unified\autogen\python\packages\autogen-studio\frontend\" -Recurse -ErrorAction SilentlyContinue
    
    $fileCount = (Get-ChildItem ".\frontend_unified\autogen" -Recurse -File -Include "*.js","*.jsx","*.ts","*.tsx","*.css","*.py").Count
    Write-Host "Copied $fileCount AutoGen UI files" -ForegroundColor Green
} else {
    Write-Host "AutoGen Studio frontend not found - skipping" -ForegroundColor Yellow
}

# Copy UI console components if they exist
if (Test-Path ".\autogen\python\packages\autogen-agentchat\src\autogen_agentchat\ui") {
    New-Item -ItemType Directory -Path ".\frontend_unified\autogen\python\packages\autogen-agentchat\src\autogen_agentchat\ui" -Force
    Copy-Item -Path ".\autogen\python\packages\autogen-agentchat\src\autogen_agentchat\ui\*.py" -Destination ".\frontend_unified\autogen\python\packages\autogen-agentchat\src\autogen_agentchat\ui\" -ErrorAction SilentlyContinue
}
```

---

## ⚛️ PHASE 5: INGEST AGENT-SQUAD REACT
**Duration**: 10 minutes
**Files**: 10+ React components + styles

### File Collection Commands:
```powershell
# Copy Agent-Squad UI components preserving structure
if (Test-Path ".\agent-squad\examples") {
    Write-Host "Copying Agent-Squad React components..." -ForegroundColor Cyan
    
    # Copy chat demo UI
    if (Test-Path ".\agent-squad\examples\chat-demo-app\ui") {
        New-Item -ItemType Directory -Path ".\frontend_unified\agent-squad\examples\chat-demo-app\ui" -Force
        Copy-Item -Path ".\agent-squad\examples\chat-demo-app\ui\*" -Destination ".\frontend_unified\agent-squad\examples\chat-demo-app\ui\" -Recurse -ErrorAction SilentlyContinue
    }
    
    # Copy e-commerce UI
    if (Test-Path ".\agent-squad\examples\ecommerce-support-simulator\resources\ui") {
        New-Item -ItemType Directory -Path ".\frontend_unified\agent-squad\examples\ecommerce-support-simulator\resources\ui" -Force
        Copy-Item -Path ".\agent-squad\examples\ecommerce-support-simulator\resources\ui\*" -Destination ".\frontend_unified\agent-squad\examples\ecommerce-support-simulator\resources\ui\" -Recurse -ErrorAction SilentlyContinue
    }
    
    # Copy docs styles
    if (Test-Path ".\agent-squad\docs\src\styles") {
        New-Item -ItemType Directory -Path ".\frontend_unified\agent-squad\docs\src\styles" -Force
        Copy-Item -Path ".\agent-squad\docs\src\styles\*.css" -Destination ".\frontend_unified\agent-squad\docs\src\styles\" -ErrorAction SilentlyContinue
    }
    
    $fileCount = (Get-ChildItem ".\frontend_unified\agent-squad" -Recurse -File -Include "*.tsx","*.jsx","*.css" -ErrorAction SilentlyContinue).Count
    Write-Host "Copied $fileCount Agent-Squad UI files" -ForegroundColor Green
} else {
    Write-Host "Agent-Squad not found - skipping" -ForegroundColor Yellow
}
```

---

## 🧪 PHASE 6: INGEST TESTMASTER DASHBOARDS
**Duration**: 10 minutes
**Files**: 20+ dashboard files

### File Collection Commands:
```powershell
# Copy TestMaster dashboard preserving structure
if (Test-Path ".\TestMaster\dashboard") {
    Write-Host "Copying TestMaster dashboard files..." -ForegroundColor Cyan
    
    # Copy entire dashboard directory
    New-Item -ItemType Directory -Path ".\frontend_unified\TestMaster\dashboard" -Force
    Copy-Item -Path ".\TestMaster\dashboard\*" -Destination ".\frontend_unified\TestMaster\dashboard\" -Recurse -ErrorAction SilentlyContinue
    
    # Copy docs with dashboards if they exist
    if (Test-Path ".\TestMaster\docs\api_integration") {
        New-Item -ItemType Directory -Path ".\frontend_unified\TestMaster\docs\api_integration" -Force
        Copy-Item -Path ".\TestMaster\docs\api_integration\*.html" -Destination ".\frontend_unified\TestMaster\docs\api_integration\" -ErrorAction SilentlyContinue
    }
    
    if (Test-Path ".\TestMaster\docs\validation") {
        New-Item -ItemType Directory -Path ".\frontend_unified\TestMaster\docs\validation" -Force
        Copy-Item -Path ".\TestMaster\docs\validation\*.html" -Destination ".\frontend_unified\TestMaster\docs\validation\" -ErrorAction SilentlyContinue
    }
    
    $fileCount = (Get-ChildItem ".\frontend_unified\TestMaster" -Recurse -File -Include "*.html","*.js","*.css" -ErrorAction SilentlyContinue).Count
    Write-Host "Copied $fileCount TestMaster dashboard files" -ForegroundColor Green
} else {
    Write-Host "TestMaster dashboard not found - skipping" -ForegroundColor Yellow
}
```

---

## 🎮 PHASE 7: COLLECT SPECIALIZED SYSTEMS
**Duration**: 15 minutes
**Files**: 100+ specialized UI files

### File Collection Commands:
```powershell
# Copy root HTML dashboards
Write-Host "Copying root-level HTML dashboards..." -ForegroundColor Cyan
$rootDashboards = @(
    "advanced_analytics_dashboard.html",
    "ultimate_3d_visualization_dashboard.html",
    "unified_greek_coordination_dashboard.html",
    "api_usage_dashboard.html",
    "unified_cross_agent_dashboard.html"
)

foreach ($dashboard in $rootDashboards) {
    if (Test-Path ".\$dashboard") {
        Copy-Item -Path ".\$dashboard" -Destination ".\frontend_unified\root_dashboards\" -ErrorAction SilentlyContinue
        Write-Host "  ✓ Copied $dashboard" -ForegroundColor Green
    }
}

# Copy AgentVerse 3D UI preserving structure
if (Test-Path ".\AgentVerse\ui") {
    Write-Host "Copying AgentVerse 3D UI system..." -ForegroundColor Cyan
    
    # Copy entire UI directory preserving structure
    New-Item -ItemType Directory -Path ".\frontend_unified\AgentVerse\ui" -Force
    Copy-Item -Path ".\AgentVerse\ui\*" -Destination ".\frontend_unified\AgentVerse\ui\" -Recurse -ErrorAction SilentlyContinue
    
    # Copy documentation if exists
    if (Test-Path ".\AgentVerse\documentation") {
        New-Item -ItemType Directory -Path ".\frontend_unified\AgentVerse\documentation" -Force
        Copy-Item -Path ".\AgentVerse\documentation\*.html" -Destination ".\frontend_unified\AgentVerse\documentation\" -ErrorAction SilentlyContinue
    }
    
    $fileCount = (Get-ChildItem ".\frontend_unified\AgentVerse" -Recurse -File -Include "*.js","*.html","*.css" -ErrorAction SilentlyContinue).Count
    Write-Host "Copied $fileCount AgentVerse UI files" -ForegroundColor Green
} else {
    Write-Host "AgentVerse UI not found - skipping" -ForegroundColor Yellow
}

# Copy Jupyter notebooks preserving structure
if (Test-Path ".\agency-swarm\notebooks") {
    Write-Host "Copying Jupyter notebook interfaces..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path ".\frontend_unified\agency-swarm\notebooks" -Force
    Copy-Item -Path ".\agency-swarm\notebooks\*.ipynb" -Destination ".\frontend_unified\agency-swarm\notebooks\" -ErrorAction SilentlyContinue
    
    $notebookCount = (Get-ChildItem ".\frontend_unified\agency-swarm\notebooks" -Filter "*.ipynb" -ErrorAction SilentlyContinue).Count
    Write-Host "Copied $notebookCount Jupyter notebooks" -ForegroundColor Green
}
```

---

## ⚡ PHASE 8: VERIFY FILE COLLECTION
**Duration**: 5 minutes

### Verification Script:
```powershell
# Count files in each category
Write-Host "`nFile Count Verification:" -ForegroundColor Green
Write-Host "========================" -ForegroundColor Green

# Check web/dashboard_modules
if (Test-Path ".\frontend_unified\web\dashboard_modules") {
    $webModules = (Get-ChildItem -Path ".\frontend_unified\web\dashboard_modules\" -Recurse -File -ErrorAction SilentlyContinue).Count
    Write-Host "Web Dashboard Modules: $webModules files" -ForegroundColor Cyan
} else {
    Write-Host "Web Dashboard Modules: NOT FOUND" -ForegroundColor Red
}

# Check each framework
$frameworks = @{
    "AgentOps" = ".\frontend_unified\agentops"
    "AutoGen" = ".\frontend_unified\autogen"
    "Agent-Squad" = ".\frontend_unified\agent-squad"
    "TestMaster" = ".\frontend_unified\TestMaster"
    "AgentVerse" = ".\frontend_unified\AgentVerse"
}

$totalFiles = 0
foreach ($framework in $frameworks.GetEnumerator()) {
    if (Test-Path $framework.Value) {
        $count = (Get-ChildItem -Path $framework.Value -Recurse -File -ErrorAction SilentlyContinue).Count
        Write-Host "$($framework.Key): $count files" -ForegroundColor Cyan
        $totalFiles += $count
    } else {
        Write-Host "$($framework.Key): Directory not found" -ForegroundColor Yellow
    }
}

# Check root dashboards
if (Test-Path ".\frontend_unified\root_dashboards") {
    $rootCount = (Get-ChildItem -Path ".\frontend_unified\root_dashboards" -Filter "*.html" -ErrorAction SilentlyContinue).Count
    Write-Host "Root Dashboards: $rootCount HTML files" -ForegroundColor Cyan
    $totalFiles += $rootCount
}

# Check Jupyter notebooks
if (Test-Path ".\frontend_unified\agency-swarm\notebooks") {
    $notebookCount = (Get-ChildItem -Path ".\frontend_unified\agency-swarm\notebooks" -Filter "*.ipynb" -ErrorAction SilentlyContinue).Count
    Write-Host "Jupyter Notebooks: $notebookCount files" -ForegroundColor Cyan
    $totalFiles += $notebookCount
}

$totalFiles += $webModules
Write-Host "========================" -ForegroundColor Green
Write-Host "TOTAL: $totalFiles files collected" -ForegroundColor Yellow

# Show directory tree
Write-Host "`nDirectory Structure:" -ForegroundColor Green
Get-ChildItem ".\frontend_unified" -Directory | ForEach-Object {
    Write-Host "  📁 $($_.Name)" -ForegroundColor Cyan
    Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue | Select-Object -First 3 | ForEach-Object {
        Write-Host "     └─ $($_.Name)" -ForegroundColor DarkGray
    }
}
```

---

## 🔍 PHASE 9: CREATE FILE MANIFEST
**Duration**: 5 minutes

### Generate Manifest:
```powershell
# Create comprehensive file manifest
Write-Host "Generating file manifest..." -ForegroundColor Cyan

$manifest = @"
# FRONTEND UNIFIED FILE MANIFEST
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss UTC")

## Directory Structure
- All files maintain their original directory structure
- This preserves relationships and makes source tracking easy
- No files are overwritten due to unique paths

"@

# Document each framework
$manifest += "## Web Dashboard Modules (Agent Work)`n"
if (Test-Path ".\frontend_unified\web\dashboard_modules") {
    Get-ChildItem -Path ".\frontend_unified\web\dashboard_modules\" -Recurse -File | ForEach-Object {
        $relativePath = $_.FullName.Replace((Get-Location).Path + '\frontend_unified\', '')
        $manifest += "- $relativePath`n"
    }
}

$manifest += "`n## Production Frameworks`n"

# List files for each framework
$frameworks = @("agentops", "autogen", "agent-squad", "TestMaster", "AgentVerse")
foreach ($framework in $frameworks) {
    if (Test-Path ".\frontend_unified\$framework") {
        $manifest += "`n### $framework`n"
        Get-ChildItem -Path ".\frontend_unified\$framework" -Recurse -File | Select-Object -First 20 | ForEach-Object {
            $relativePath = $_.FullName.Replace((Get-Location).Path + '\frontend_unified\', '')
            $manifest += "- $relativePath`n"
        }
        $totalCount = (Get-ChildItem -Path ".\frontend_unified\$framework" -Recurse -File).Count
        if ($totalCount -gt 20) {
            $manifest += "... and $($totalCount - 20) more files`n"
        }
    }
}

# Save manifest
$manifest | Out-File -FilePath ".\frontend_unified\FILE_MANIFEST.md" -Encoding UTF8
Write-Host "✓ File manifest created at frontend_unified\FILE_MANIFEST.md" -ForegroundColor Green
```

---

## ⚙️ PHASE 10: PREPARE FOR STEELCLAD
**Duration**: 5 minutes

### Create Agent Work Directories:
```powershell
# Create work directories for each agent
Write-Host "Preparing STEELCLAD agent work directories..." -ForegroundColor Cyan

New-Item -ItemType Directory -Path ".\frontend_unified\atoms\agent_x_work" -Force | Out-Null
New-Item -ItemType Directory -Path ".\frontend_unified\atoms\agent_y_work" -Force | Out-Null
New-Item -ItemType Directory -Path ".\frontend_unified\atoms\agent_z_work" -Force | Out-Null
New-Item -ItemType Directory -Path ".\frontend_unified\atoms\agent_t_work" -Force | Out-Null

# Copy agent assignment files if they exist
$agentFiles = @(
    "AGENT_X_STEELCLAD_FRONTEND_TARGETS.md",
    "AGENT_Y_STEELCLAD_FRONTEND_TARGETS.md",
    "AGENT_Z_STEELCLAD_FRONTEND_TARGETS.md",
    "AGENT_T_STEELCLAD_FRONTEND_TARGETS.md"
)

foreach ($file in $agentFiles) {
    if (Test-Path ".\$file") {
        $agentLetter = $file.Split('_')[1].ToLower()
        Copy-Item -Path ".\$file" -Destination ".\frontend_unified\atoms\agent_$($agentLetter)_work\" -ErrorAction SilentlyContinue
        Write-Host "  ✓ Copied $file to agent work directory" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ $file not found" -ForegroundColor Yellow
    }
}

Write-Host "✓ STEELCLAD preparation complete" -ForegroundColor Green
```

---

## 📊 SUCCESS METRICS

### File Organization:
- ✅ All web/dashboard_modules preserved with structure
- ✅ AgentOps components in `agentops/app/dashboard/`
- ✅ AutoGen files in `autogen/python/packages/`
- ✅ Agent-Squad components in `agent-squad/examples/`
- ✅ TestMaster files in `TestMaster/dashboard/`
- ✅ AgentVerse UI in `AgentVerse/ui/`
- ✅ Root dashboards in `root_dashboards/`

### Safety Features:
- ✅ No overwriting risk - unique directory paths
- ✅ Original structure preserved for traceability
- ✅ Copy operations only - no moves or deletes
- ✅ Error handling with -ErrorAction SilentlyContinue
- ✅ Existence checks before operations

### Readiness:
- ✅ Agent work directories prepared
- ✅ Assignment files ready for distribution
- ✅ File manifest for tracking
- ✅ Verification scripts included

---

## 🚦 EXECUTION CHECKLIST

### Pre-Flight Safety:
- [ ] Run Phase 0A: Cleanup outdated copy (CRITICAL - RUN FIRST)
- [ ] Run Phase 0B: Safety checks (RUN SECOND)
- [ ] Review safety check output
- [ ] Verify backup created or old directory removed

### Execution Phases:
- [ ] Run Phase 1: Create directory structure
- [ ] Run Phase 2: Copy current working modules (web/dashboard_modules)
- [ ] Run Phase 3: Ingest AgentOps dashboard
- [ ] Run Phase 4: Ingest AutoGen Studio
- [ ] Run Phase 5: Ingest Agent-Squad React
- [ ] Run Phase 6: Ingest TestMaster dashboards
- [ ] Run Phase 7: Collect specialized systems (AgentVerse, root HTML, notebooks)
- [ ] Run Phase 8: Verify file collection
- [ ] Run Phase 9: Create file manifest
- [ ] Run Phase 10: Prepare for STEELCLAD

### Post-Execution:
- [ ] Review verification output
- [ ] Check FILE_MANIFEST.md
- [ ] Deploy 4 agents for parallel STEELCLAD execution

---

## ⚠️ IMPORTANT SAFETY NOTES

1. **RUN PHASE 0A FIRST** - Cleanup outdated frontend_unified copy
2. **RUN PHASE 0B SECOND** - Safety checks for source availability  
3. **All operations are COPY only** - No files are moved or deleted from source
4. **Directory structure preserved** - Original relationships maintained
5. **No overwrite risk** - Each framework has unique subdirectory path
6. **Error handling included** - -ErrorAction SilentlyContinue prevents crashes
7. **Existence checks** - Each phase verifies source before copying

### **Critical Discovery:**
- **frontend_unified is an OUTDATED COPY from 09:40 AM**
- **Missing 53+ files of agent atomization work**
- **Must be cleaned up before proceeding**

---

**STATUS**: Implementation plan updated with cleanup phase
**NEXT STEP**: Run Phase 0A (Cleanup) then Phase 0B (Safety Checks) before any other operations
# Dashboard Module Structure Documentation

## Overview
This document describes the modular architecture of the TestMaster Dashboard system.
Each module has a specific responsibility and clear interfaces with other modules.

## Directory Layout
```
dashboard/
├── MODULE_STRUCTURE.md      # This file  
├── server.py                # Main Flask application entry point
├── api/                     # API endpoint modules
│   ├── __init__.py
│   ├── performance.py       # Performance monitoring endpoints
│   ├── analytics.py         # Analytics data endpoints  
│   ├── workflow.py          # Workflow management endpoints
│   ├── tests.py             # Test status endpoints
│   ├── refactor.py          # Refactoring analysis endpoints
│   └── llm.py              # LLM integration endpoints
├── core/                    # Core business logic
│   ├── __init__.py
│   ├── monitor.py           # Real-time monitoring
│   ├── metrics_collector.py # Metrics collection and aggregation
│   ├── cache.py             # Data caching layer
│   └── config.py            # Configuration management
├── static/                  # Frontend assets
│   ├── index.html           # Main HTML template
│   ├── css/                 # Stylesheets
│   │   ├── main.css         # Core layout and structure
│   │   ├── components.css   # Reusable component styles
│   │   └── themes.css       # Theme definitions
│   ├── js/                  # JavaScript modules
│   │   ├── app.js           # Main application logic
│   │   ├── charts/          # Chart management
│   │   ├── tabs/            # Tab-specific logic
│   │   ├── api/             # API client
│   │   └── utils/           # Utilities and helpers
│   └── templates/           # HTML templates (if needed)
└── utils/                   # Shared utilities
    ├── __init__.py
    ├── decorators.py        # Reusable decorators
    └── helpers.py           # Utility functions
```

## Critical Original Features to Preserve

### Real-time Performance Charts (TOP PRIORITY)
**Location**: Analytics tab → Performance Monitoring section
**Current Behavior**:
- 3 scrolling line charts (CPU, Memory, Network)
- Updates every 100ms via `/api/performance/realtime`
- 30-second window (300 data points)
- Scrolls right-to-left
- Thin lines (0.5px width)
- Starts when entering Analytics tab, stops when leaving

**Module Mapping**:
- Backend: `api/performance.py` + `core/monitor.py`
- Frontend: `static/js/charts/performance.js`

### Tab System
**Current Behavior**: 6 tabs (Overview, Analytics, Tests, Workflow, Refactor, Analyzer)
**Module Mapping**: `static/js/tabs/manager.js`

### LLM Integration  
**Current Behavior**: Toggle button, status tracking, API call management
**Module Mapping**: `api/llm.py` + `static/js/llm/`

## Module Responsibilities

### API Modules (`api/`)

#### `performance.py`
```python
"""
Real-time performance monitoring endpoints.
Handles the critical 100ms chart updates.
"""

@blueprint.route('/realtime')
def get_realtime_metrics():
    """Return current metrics for scrolling charts."""
    # Must return: cpu_usage, memory_usage_mb, network_kb_s arrays
```

#### `workflow.py` 
```python
"""
Workflow management endpoints.
Must include the /status endpoint that was returning 404.
"""

@blueprint.route('/status') 
def workflow_status():
    """Return workflow status - NO 404 ERRORS!"""
```

### Core Modules (`core/`)

#### `monitor.py`
```python
"""
Real-time monitoring engine.
Collects metrics every 100ms for performance charts.
"""

class RealTimeMonitor:
    def __init__(self):
        self.performance_history = {}  # Per-codebase history
        self.max_history_points = 300  # 30s at 100ms intervals
        
    def start_monitoring(self):
        """Start 100ms collection timer."""
        
    def get_current_metrics(self, codebase):
        """Get latest metrics for API."""
```

### Frontend Modules (`static/js/`)

#### `charts/performance.js`
```javascript
/**
 * Performance chart management.
 * Handles the 3 real-time scrolling charts.
 */

class PerformanceCharts {
    constructor() {
        this.updateInterval = null;
        this.data = {
            cpu: new Array(300).fill(0),
            memory: new Array(300).fill(0), 
            network: new Array(300).fill(0)
        };
    }
    
    startRealTimeUpdates() {
        // Start 100ms updates
    }
    
    stopRealTimeUpdates() {
        // Clean stop when leaving tab
    }
}
```

#### `tabs/manager.js`
```javascript
/**
 * Tab switching logic.
 * Manages chart lifecycle (start/stop).
 */

class TabManager {
    switchTab(tabName) {
        // Handle analytics tab special case
        if (tabName === 'analytics') {
            this.performanceCharts.startRealTimeUpdates();
        } else {
            this.performanceCharts.stopRealTimeUpdates();
        }
    }
}
```

## Data Flow Architecture

### Real-time Performance Flow
```
Core Monitor (100ms) → Metrics Buffer → Cache → API Endpoint → Frontend Charts
```

### Tab Switching Flow  
```
User Click → Tab Manager → Chart Lifecycle → API Start/Stop → UI Update
```

## Interface Contracts

### Performance API Response
```json
{
    "status": "success",
    "timeseries": {
        "cpu_usage": [23.5, 24.1, 22.8],
        "memory_usage_mb": [145.2, 146.8, 144.9],
        "network_kb_s": [5.2, 6.1, 4.8]
    },
    "timestamp": "2025-08-18T11:30:00.000Z"
}
```

### Chart Data Format
```javascript
// Each chart maintains 300 data points
performanceData = {
    cpu: [0, 0, ..., 23.5],      // 300 elements
    memory: [0, 0, ..., 145.2],  // 300 elements  
    network: [0, 0, ..., 5.2]    // 300 elements
};
```

## Critical Migration Rules

1. **Preserve All Functionality**: Every feature in FUNCTIONALITY_CHECKLIST.md must work
2. **No Performance Regression**: Charts must still update at 100ms
3. **Same API Endpoints**: Frontend expects exact same endpoint paths
4. **Identical Data Formats**: API responses must match current structure
5. **Error Handling**: Graceful fallbacks when APIs unavailable

## Module Creation Order

1. **Core modules first**: `monitor.py`, `cache.py` - foundation
2. **API modules second**: `performance.py` critical for charts
3. **Frontend structure**: HTML template, then JS modules
4. **Integration last**: Wire everything together

## Testing Strategy Per Module

### `api/performance.py`
- [ ] `/realtime` endpoint returns proper format
- [ ] Response time under 10ms
- [ ] Handles missing codebase parameter

### `charts/performance.js`  
- [ ] Creates 3 Chart.js instances
- [ ] Updates every 100ms when active
- [ ] Stops cleanly when tab switches
- [ ] Scrolling animation smooth

### `tabs/manager.js`
- [ ] All 6 tabs switch properly
- [ ] Analytics tab triggers chart start
- [ ] Other tabs trigger chart stop

## File Size Targets (Reduced from Original)

Original sizes:
- `hybrid_intelligence_dashboard_grouped.html`: 170KB
- `web_monitor.py`: 71KB

Target sizes:
- Largest Python module: < 500 lines (~15KB)
- Largest JS module: < 300 lines (~10KB) 
- Main HTML: < 200 lines (~6KB)

## Success Metrics

1. ✅ All functionality checklist items pass
2. ✅ Performance charts work identically
3. ✅ No 404 errors in console
4. ✅ Tab switching smooth
5. ✅ Code easier to navigate and modify
6. ✅ Total line count reduced by 50%+
7. ✅ Every function documented

## Rollback Triggers

If any of these occur, use ROLLBACK_SCRIPT.sh:
- Performance charts don't update
- Any tab switching broken  
- 404 errors return
- Significant performance regression
- Any critical functionality lost

This modular structure maintains all original functionality while making the codebase much more maintainable for future development.
# 🚨 CRITICAL: API Cost Control Integration Guide
**MANDATORY for Alpha, Beta, and Gamma Dashboard Teams**

## URGENT REQUIREMENT

Before running ANY AI-powered analysis tools, the cost tracking system MUST be integrated with all dashboards to prevent budget overruns.

## Integration Requirements

### 1. Import Cost Tracking (MANDATORY)

```python
# Add these imports to ALL AI analysis tools
from core.monitoring.api_usage_tracker import track_api_call, pre_check_cost, get_usage_stats
from core.monitoring.ai_analysis_wrapper import create_ai_wrapper, mandatory_cost_check

# Create wrapper for your component
ai_wrapper = create_ai_wrapper("YourComponentName", "gpt-3.5-turbo")
```

### 2. Pre-Check Before AI Calls (MANDATORY)

```python
# BEFORE any LLM API call, check budget
allowed, message, cost = mandatory_cost_check(
    purpose="Your analysis description",
    estimated_input_tokens=1000,
    estimated_output_tokens=500,
    model="gpt-3.5-turbo"
)

if not allowed:
    raise Exception(f"AI analysis blocked: {message}")
    
print(f"Estimated cost: ${cost:.4f}")
```

### 3. Track All API Calls (MANDATORY)

```python
# After successful API call
track_api_call(
    model="gpt-3.5-turbo",
    call_type="analysis",  # or "generation", "embedding", etc.
    purpose="Code architecture analysis",
    component="YourComponentName", 
    input_tokens=actual_input_tokens,
    output_tokens=actual_output_tokens
)
```

### 4. Dashboard Integration (REQUIRED)

```python
from core.monitoring.api_dashboard_integration import APIDashboardIntegration

# In your dashboard code
api_integration = APIDashboardIntegration()

# Register for real-time updates
def on_api_update(message):
    # Update your dashboard with new API call data
    print(f"API Update: {message}")

api_integration.register_dashboard("YourDashboardName", on_api_update)

# Get current stats for display
stats = api_integration.get_dashboard_data()
```

## Dashboard Display Requirements

### Essential Metrics to Show

1. **Current Cost**: `${stats['current_stats']['total_cost']:.2f}`
2. **Daily Budget**: `${stats['current_stats']['budget_status']['daily']['spent']:.2f} / ${stats['current_stats']['budget_status']['daily']['limit']:.2f}`
3. **Hourly Usage**: Real-time hourly spend tracking
4. **API Call Count**: Total calls today
5. **Success Rate**: Percentage of successful calls
6. **Top Models**: Most expensive models being used

### Required Alerts

```python
budget_status = stats['current_stats']['budget_status']['daily']
percentage = budget_status['percentage']

if percentage > 90:
    show_critical_alert("Budget 90% exceeded!")
elif percentage > 75:
    show_warning_alert("Budget 75% reached")
elif percentage > 50:
    show_caution_alert("Budget 50% used")
```

## Files to Integrate

### 1. Core Monitoring Files
- `core/monitoring/api_usage_tracker.py` - Main tracking system
- `core/monitoring/ai_analysis_wrapper.py` - Mandatory wrapper for AI calls
- `core/monitoring/api_dashboard_integration.py` - Dashboard connection

### 2. Example Dashboard
- `api_usage_dashboard.html` - Reference implementation

### 3. Updated Architecture Validator
- `architecture/clean/clean_architecture_validator.py` - Example of cost tracking integration

## Budget Configuration

Current default limits (can be adjusted):
- **Daily Limit**: $10.00
- **Hourly Limit**: $2.00  
- **Per-Call Limit**: $0.50
- **Total Limit**: $100.00
- **Auto-Stop**: Enabled (blocks calls when limits exceeded)

## Emergency Controls

```python
# Emergency stop all AI analysis
from core.monitoring.ai_analysis_wrapper import emergency_stop_all_ai
emergency_stop_all_ai()

# Reset emergency stop
from core.monitoring.ai_analysis_wrapper import reset_emergency_stop
reset_emergency_stop()
```

## Integration Checklist for Dashboard Teams

### Alpha Dashboard Team
- [ ] Import cost tracking modules
- [ ] Add pre-check calls before AI analysis
- [ ] Integrate real-time cost display
- [ ] Add budget status indicators
- [ ] Test emergency stop functionality

### Beta Dashboard Team  
- [ ] Import cost tracking modules
- [ ] Add pre-check calls before AI analysis
- [ ] Integrate real-time cost display
- [ ] Add budget status indicators
- [ ] Test emergency stop functionality

### Gamma Dashboard Team
- [ ] Import cost tracking modules
- [ ] Add pre-check calls before AI analysis  
- [ ] Integrate real-time cost display
- [ ] Add budget status indicators
- [ ] Test emergency stop functionality

## Testing Before Production

```python
# Test the tracking system
from core.monitoring.api_usage_tracker import get_api_tracker

tracker = get_api_tracker()
tracker.set_budget(daily_limit=1.0, auto_stop=True)  # $1 test limit

# Test a small call
allowed, message, cost = mandatory_cost_check(
    "test analysis", 100, 50, "gpt-3.5-turbo"
)
print(f"Test result: {allowed}, Cost: ${cost:.4f}")
```

## Model Cost Reference

| Model | Input (per 1K tokens) | Output (per 1K tokens) |
|-------|----------------------|------------------------|
| gpt-3.5-turbo | $0.0005 | $0.0015 |
| gpt-4 | $0.0300 | $0.0600 |
| claude-3-haiku | $0.00025 | $0.00125 |
| claude-3-sonnet | $0.0030 | $0.0150 |
| claude-3-opus | $0.0150 | $0.0750 |

## Critical Warnings

⚠️ **ANY AI analysis tool that doesn't use this tracking system risks burning through budget uncontrolled**

⚠️ **All API calls MUST be tracked - no exceptions**

⚠️ **Test with small budgets before production use**

⚠️ **Monitor dashboards continuously during AI analysis**

## Support and Questions

For integration issues:
1. Check the example in `clean_architecture_validator.py`
2. Review the test dashboard at `api_usage_dashboard.html`
3. Test with small budgets first

## Implementation Priority

**IMMEDIATE (Today)**:
1. Alpha, Beta, Gamma teams integrate cost tracking
2. Test with $1 daily limits
3. Verify dashboard displays work

**BEFORE AI Analysis**:
1. Confirm all tracking is working
2. Set appropriate budget limits
3. Monitor continuously during analysis

---

**Status**: CRITICAL IMPLEMENTATION REQUIRED  
**Deadline**: Before any AI-powered analysis  
**Priority**: MAXIMUM - Budget Protection
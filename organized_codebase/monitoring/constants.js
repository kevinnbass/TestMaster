/**
 * Application Constants
 * =====================
 * 
 * Global constants and configuration values.
 * 
 * @author TestMaster Team
 */

// Application Info
window.APP_CONFIG = {
    NAME: 'TestMaster Dashboard',
    VERSION: '2.0.0',
    BUILD: 'modular-refactor'
};

// API Configuration
window.API_CONFIG = {
    DEFAULT_TIMEOUT: 10000,
    PERFORMANCE_UPDATE_INTERVAL: 100, // 100ms for real-time charts
    WORKFLOW_UPDATE_INTERVAL: 10000,  // 10s for workflow status
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 1000
};

// Chart Configuration
window.CHART_CONFIG = {
    PERFORMANCE_POINTS: 300, // 30 seconds at 100ms intervals
    CHART_COLORS: {
        CPU: '#3b82f6',
        MEMORY: '#8b5cf6', 
        NETWORK: '#10b981',
        SUCCESS: '#10b981',
        WARNING: '#f59e0b',
        ERROR: '#ef4444',
        INFO: '#3b82f6'
    },
    CHART_OPTIONS: {
        RESPONSIVE: true,
        MAINTAIN_ASPECT_RATIO: false,
        ANIMATION: false,
        BORDER_WIDTH: 0.5 // Thin lines
    }
};

// UI Configuration
window.UI_CONFIG = {
    DEFAULT_TAB: 'overview',
    AVAILABLE_TABS: ['overview', 'analytics', 'tests', 'workflow', 'refactor', 'analyzer'],
    ANIMATION_DURATION: 300,
    DEBOUNCE_DELAY: 250
};

// Status Indicators
window.STATUS = {
    ONLINE: 'online',
    OFFLINE: 'offline', 
    LOADING: 'loading',
    ERROR: 'error'
};

// Event Names
window.EVENTS = {
    TAB_CHANGED: 'tab:changed',
    DATA_UPDATED: 'data:updated',
    ERROR_OCCURRED: 'error:occurred',
    LLM_TOGGLED: 'llm:toggled'
};

// Storage Keys
window.STORAGE_KEYS = {
    LAST_TAB: 'testmaster:last-tab',
    LLM_STATE: 'testmaster:llm-enabled',
    THEME: 'testmaster:theme',
    CODEBASES: 'testmaster:codebases'
};

// Default Values
window.DEFAULTS = {
    CODEBASE: '/testmaster',
    THEME: 'dark',
    REFRESH_INTERVAL: 5000
};

console.log('Constants loaded:', window.APP_CONFIG);
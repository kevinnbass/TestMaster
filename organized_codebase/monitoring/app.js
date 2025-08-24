/**
 * Main Application Module
 * =======================
 * 
 * Application initialization and global event handling.
 * This replaces the massive JavaScript section in the original HTML.
 * 
 * @author TestMaster Team
 */

class DashboardApp {
    /**
     * Initialize the dashboard application.
     */
    constructor() {
        this.initialized = false;
        this.currentCodebase = window.DEFAULTS.CODEBASE;
        this.connectionStatus = window.STATUS.OFFLINE;
        this.llmEnabled = false;
        
        console.log('Dashboard App v2.0 initializing...');
    }
    
    /**
     * Initialize the application.
     */
    async init() {
        if (this.initialized) {
            console.warn('App already initialized');
            return;
        }
        
        try {
            // Initialize core components
            this._initializeGlobals();
            this._initializeEventHandlers();
            
            // Initialize managers
            if (window.tabManager) {
                window.tabManager.init();
            }
            
            // Load initial data
            await this._loadInitialData();
            
            // Start periodic updates
            this._startPeriodicUpdates();
            
            // Request notification permission
            window.Utils.requestNotificationPermission();
            
            this.initialized = true;
            console.log('Dashboard App initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize app:', error);
            this._showInitializationError(error);
        }
    }
    
    /**
     * Initialize global variables and state.
     */
    _initializeGlobals() {
        // Set global current codebase
        window.currentCodebase = this.currentCodebase;
        
        // Initialize codebase management
        window.activeCodebases = new Set([this.currentCodebase]);
        
        // Add initial codebase tab
        this._updateCodebaseTabs();
        
        console.log('Global state initialized');
    }
    
    /**
     * Initialize event handlers.
     */
    _initializeEventHandlers() {
        // Window events
        window.addEventListener('beforeunload', () => this._cleanup());
        window.addEventListener('online', () => this._handleConnectionChange(true));
        window.addEventListener('offline', () => this._handleConnectionChange(false));
        
        // LLM toggle button
        const llmToggle = document.getElementById('llm-toggle-btn');
        if (llmToggle) {
            llmToggle.addEventListener('click', () => this._toggleLlmMode());
        }
        
        // Custom events
        document.addEventListener(window.EVENTS.TAB_CHANGED, (e) => {
            console.log('Tab changed:', e.detail.tab);
        });
        
        console.log('Event handlers initialized');
    }
    
    /**
     * Load initial application data.
     */
    async _loadInitialData() {
        console.log('Loading initial data...');
        
        try {
            // Health check
            const health = await window.apiClient.getHealth();
            if (health.status === 'healthy') {
                this._updateConnectionStatus(window.STATUS.ONLINE);
            }
            
            // Load LLM status
            await this._updateLlmStatus();
            
            // Load initial metrics
            await this._updateSystemMetrics();
            
        } catch (error) {
            console.error('Error loading initial data:', error);
            this._updateConnectionStatus(window.STATUS.ERROR);
        }
    }
    
    /**
     * Start periodic updates for dashboard data.
     */
    _startPeriodicUpdates() {
        // Workflow status updates (every 10 seconds)
        setInterval(async () => {
            try {
                await this._updateWorkflowStatus();
            } catch (error) {
                console.error('Error updating workflow status:', error);
            }
        }, window.API_CONFIG.WORKFLOW_UPDATE_INTERVAL);
        
        // System metrics updates (every 5 seconds)
        setInterval(async () => {
            try {
                await this._updateSystemMetrics();
            } catch (error) {
                console.error('Error updating system metrics:', error);
            }
        }, window.DEFAULTS.REFRESH_INTERVAL);
        
        console.log('Periodic updates started');
    }
    
    /**
     * Update connection status indicator.
     * @param {string} status - Connection status
     */
    _updateConnectionStatus(status) {
        this.connectionStatus = status;
        
        const indicator = document.getElementById('connection-status');
        if (indicator) {
            indicator.className = `status-indicator ${window.Utils.getStatusColor(status)}`;
            indicator.title = `Status: ${status}`;
        }
    }
    
    /**
     * Handle connection state changes.
     * @param {boolean} isOnline - Whether connection is online
     */
    _handleConnectionChange(isOnline) {
        const status = isOnline ? window.STATUS.ONLINE : window.STATUS.OFFLINE;
        this._updateConnectionStatus(status);
        
        const message = isOnline ? 'Connection restored' : 'Connection lost';
        window.Utils.showNotification('Dashboard', message, isOnline ? 'success' : 'warning');
    }
    
    /**
     * Update LLM status and toggle button.
     */
    async _updateLlmStatus() {
        try {
            const status = await window.apiClient.getLlmStatus();
            
            if (status.status === 'success') {
                this.llmEnabled = status.api_enabled || false;
                
                // Update toggle button
                const toggleBtn = document.getElementById('llm-toggle-btn');
                const statusText = document.getElementById('llm-status-text');
                const statusIndicator = document.getElementById('llm-status-indicator');
                
                if (toggleBtn) {
                    toggleBtn.classList.toggle('active', this.llmEnabled);
                }
                
                if (statusText) {
                    statusText.textContent = this.llmEnabled ? 'LLM ON' : 'LLM OFF';
                }
                
                if (statusIndicator) {
                    statusIndicator.classList.toggle('active', this.llmEnabled);
                }
            }
            
        } catch (error) {
            console.error('Error updating LLM status:', error);
        }
    }
    
    /**
     * Toggle LLM mode.
     */
    async _toggleLlmMode() {
        try {
            const newState = !this.llmEnabled;
            const result = await window.apiClient.toggleLlmMode(newState);
            
            if (result.status === 'success') {
                this.llmEnabled = result.enabled;
                await this._updateLlmStatus();
                
                const message = `LLM ${this.llmEnabled ? 'enabled' : 'disabled'}`;
                window.Utils.showNotification('LLM Status', message, 'info');
                
                // Dispatch custom event
                document.dispatchEvent(new CustomEvent(window.EVENTS.LLM_TOGGLED, {
                    detail: { enabled: this.llmEnabled }
                }));
            }
            
        } catch (error) {
            console.error('Error toggling LLM mode:', error);
            window.Utils.showNotification('Error', 'Failed to toggle LLM mode', 'error');
        }
    }
    
    /**
     * Update workflow status display.
     */
    async _updateWorkflowStatus() {
        try {
            const status = await window.apiClient.getWorkflowStatus();
            
            if (status.status === 'success') {
                // Update workflow metrics in UI
                const runningEl = document.getElementById('workflow-running');
                const queuedEl = document.getElementById('workflow-queued');
                const completedEl = document.getElementById('workflow-completed');
                
                if (runningEl) runningEl.textContent = status.running_tasks || 0;
                if (queuedEl) queuedEl.textContent = status.pending_tasks || 0;
                if (completedEl) completedEl.textContent = status.completed_tasks || 0;
            }
            
        } catch (error) {
            console.debug('Workflow status update failed (expected if not on workflow tab)');
        }
    }
    
    /**
     * Update system metrics display.
     */
    async _updateSystemMetrics() {
        try {
            const metrics = await window.apiClient.getAnalyticsMetrics();
            
            if (metrics.status === 'success' && metrics.metrics) {
                // Update system metrics in UI
                const uptimeEl = document.getElementById('system-uptime');
                const processesEl = document.getElementById('active-processes');
                
                if (uptimeEl && metrics.metrics.uptime) {
                    uptimeEl.textContent = window.Utils.formatRelativeTime(metrics.metrics.uptime);
                }
                
                if (processesEl && metrics.metrics.active_processes) {
                    processesEl.textContent = metrics.metrics.active_processes;
                }
            }
            
        } catch (error) {
            console.debug('System metrics update failed');
        }
    }
    
    /**
     * Update codebase tabs in UI.
     */
    _updateCodebaseTabs() {
        const container = document.getElementById('codebase-tabs-container');
        if (!container) return;
        
        // Clear existing tabs
        container.innerHTML = '';
        
        // Add tabs for each active codebase
        window.activeCodebases.forEach(codebase => {
            const tab = document.createElement('div');
            tab.className = `codebase-tab ${codebase === this.currentCodebase ? 'active' : ''}`;
            tab.setAttribute('data-codebase-path', codebase);
            
            const name = codebase.split('/').pop() || codebase;
            tab.innerHTML = `
                <span>${name}</span>
                ${window.activeCodebases.size > 1 ? '<span class="remove-btn" title="Remove codebase">×</span>' : ''}
            `;
            
            // Add click handler for switching
            tab.addEventListener('click', (e) => {
                if (e.target.classList.contains('remove-btn')) {
                    this._removeCodebase(codebase);
                } else {
                    this._switchCodebase(codebase);
                }
            });
            
            container.appendChild(tab);
        });
    }
    
    /**
     * Switch to a different codebase.
     * @param {string} codebase - Codebase path
     */
    _switchCodebase(codebase) {
        if (this.currentCodebase === codebase) return;
        
        console.log(`Switching codebase: ${this.currentCodebase} -> ${codebase}`);
        
        this.currentCodebase = codebase;
        window.currentCodebase = codebase;
        
        // Update current codebase display
        const currentDisplay = document.getElementById('current-codebase');
        if (currentDisplay) {
            currentDisplay.textContent = codebase;
        }
        
        // Update tabs
        this._updateCodebaseTabs();
        
        // Refresh data for new codebase
        this._loadInitialData();
    }
    
    /**
     * Remove a codebase from active list.
     * @param {string} codebase - Codebase path
     */
    _removeCodebase(codebase) {
        if (window.activeCodebases.size <= 1) {
            window.Utils.showNotification('Error', 'Cannot remove the last codebase', 'warning');
            return;
        }
        
        window.activeCodebases.delete(codebase);
        
        // If removing current codebase, switch to first available
        if (this.currentCodebase === codebase) {
            const firstCodebase = Array.from(window.activeCodebases)[0];
            this._switchCodebase(firstCodebase);
        } else {
            this._updateCodebaseTabs();
        }
        
        console.log(`Removed codebase: ${codebase}`);
    }
    
    /**
     * Show initialization error.
     * @param {Error} error - Initialization error
     */
    _showInitializationError(error) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-error';
        errorDiv.innerHTML = `
            <h3>Initialization Error</h3>
            <p>Failed to initialize dashboard: ${error.message}</p>
            <button onclick="location.reload()">Reload Page</button>
        `;
        
        document.body.insertBefore(errorDiv, document.body.firstChild);
    }
    
    /**
     * Cleanup resources before page unload.
     */
    _cleanup() {
        console.log('Cleaning up dashboard app...');
        
        // Stop performance chart updates
        if (window.performanceCharts) {
            window.performanceCharts.destroy();
        }
        
        // Clear any intervals/timeouts if needed
        // (Most are handled by their respective modules)
    }
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.dashboardApp = new DashboardApp();
        window.dashboardApp.init();
    });
} else {
    window.dashboardApp = new DashboardApp();
    window.dashboardApp.init();
}
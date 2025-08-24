/**
 * Tab Manager Module
 * ==================
 * 
 * Manages tab switching and lifecycle.
 * Critical for starting/stopping performance chart updates.
 * 
 * @author TestMaster Team
 */

class TabManager {
    /**
     * Initialize tab manager.
     */
    constructor() {
        this.currentTab = 'overview';
        this.tabButtons = null;
        this.tabContents = null;
        
        console.log('TabManager initialized');
    }
    
    /**
     * Initialize tab manager with DOM elements.
     */
    init() {
        this.tabButtons = document.querySelectorAll('.tab-button');
        this.tabContents = document.querySelectorAll('.tab-content');
        
        if (!this.tabButtons.length) {
            console.error('No tab buttons found');
            return;
        }
        
        // Add click handlers
        this.tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.getAttribute('data-tab');
                this.switchTab(tabName);
            });
        });
        
        console.log(`Tab manager initialized with ${this.tabButtons.length} tabs`);
    }
    
    /**
     * Switch to a specific tab.
     * Manages performance chart lifecycle.
     * 
     * @param {string} tabName - Name of tab to switch to
     */
    switchTab(tabName) {
        console.log(`Switching to tab: ${tabName}`);
        
        if (!tabName) {
            console.error('Tab name is required');
            return;
        }
        
        // Update current tab
        const previousTab = this.currentTab;
        this.currentTab = tabName;
        
        // Update tab buttons
        this.tabButtons.forEach(button => {
            const buttonTab = button.getAttribute('data-tab');
            if (buttonTab === tabName) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });
        
        // Update tab contents
        this.tabContents.forEach(content => {
            const contentId = `${tabName}-tab`;
            if (content.id === contentId) {
                content.classList.add('active');
                console.log(`Activated tab: ${contentId}`);
            } else {
                content.classList.remove('active');
            }
        });
        
        // Handle tab-specific logic
        this._handleTabSwitch(tabName, previousTab);
    }
    
    /**
     * Handle tab-specific initialization and cleanup.
     * 
     * @param {string} newTab - Tab being switched to
     * @param {string} previousTab - Tab being switched from
     */
    _handleTabSwitch(newTab, previousTab) {
        // Analytics tab: Start performance charts
        if (newTab === 'analytics') {
            this._initializeAnalyticsTab();
        }
        
        // Stop performance charts when leaving analytics
        if (previousTab === 'analytics' && newTab !== 'analytics') {
            this._cleanupAnalyticsTab();
        }
        
        // Initialize other tabs as needed
        switch (newTab) {
            case 'tests':
                this._initializeTestsTab();
                break;
            case 'workflow':
                this._initializeWorkflowTab();
                break;
            case 'refactor':
                this._initializeRefactorTab();
                break;
            case 'analyzer':
                this._initializeAnalyzerTab();
                break;
        }
    }
    
    /**
     * Initialize Analytics tab with performance charts.
     * CRITICAL: This starts the real-time performance monitoring.
     */
    _initializeAnalyticsTab() {
        console.log('Initializing Analytics tab');
        
        // Initialize performance charts if not already done
        if (!window.performanceCharts) {
            window.performanceCharts = new PerformanceCharts();
        }
        
        // Always reinitialize to ensure fresh start
        if (typeof window.performanceCharts.initializeCharts === 'function') {
            window.performanceCharts.initializeCharts();
        } else {
            console.error('Performance charts not properly initialized');
        }
    }
    
    /**
     * Cleanup Analytics tab.
     * Stops performance chart updates to save resources.
     */
    _cleanupAnalyticsTab() {
        console.log('Cleaning up Analytics tab');
        
        if (window.performanceCharts && typeof window.performanceCharts.stopRealTimeUpdates === 'function') {
            window.performanceCharts.stopRealTimeUpdates();
        }
    }
    
    /**
     * Initialize Tests tab.
     */
    _initializeTestsTab() {
        console.log('Initializing Tests tab');
        
        // Load test status
        if (typeof window.loadTestsStatus === 'function') {
            window.loadTestsStatus();
        }
    }
    
    /**
     * Initialize Workflow tab.
     */
    _initializeWorkflowTab() {
        console.log('Initializing Workflow tab');
        
        // Update workflow status
        if (typeof window.updateWorkflowStatus === 'function') {
            window.updateWorkflowStatus();
        }
    }
    
    /**
     * Initialize Refactor tab.
     */
    _initializeRefactorTab() {
        console.log('Initializing Refactor tab');
        
        // Load refactor analysis
        if (typeof window.loadRefactorAnalysis === 'function') {
            window.loadRefactorAnalysis();
        }
    }
    
    /**
     * Initialize Analyzer tab.
     */
    _initializeAnalyzerTab() {
        console.log('Initializing Analyzer tab');
        
        // Load analyzer data
        if (typeof window.loadAnalyzerData === 'function') {
            window.loadAnalyzerData();
        }
    }
    
    /**
     * Get current active tab.
     * @returns {string} Current tab name
     */
    getCurrentTab() {
        return this.currentTab;
    }
    
    /**
     * Check if a specific tab is active.
     * @param {string} tabName - Tab name to check
     * @returns {boolean} True if tab is active
     */
    isTabActive(tabName) {
        return this.currentTab === tabName;
    }
    
    /**
     * Get all available tabs.
     * @returns {Array<string>} Array of tab names
     */
    getAvailableTabs() {
        if (!this.tabButtons) return [];
        
        return Array.from(this.tabButtons).map(button => 
            button.getAttribute('data-tab')
        ).filter(Boolean);
    }
}

// Global tab manager instance
window.tabManager = new TabManager();
/**
 * API Client Module
 * =================
 * 
 * Provides centralized API communication with error handling.
 * 
 * @author TestMaster Team
 */

class ApiClient {
    /**
     * Initialize API client.
     */
    constructor() {
        this.baseUrl = '';
        this.defaultTimeout = 10000;
        this.requestCount = 0;
        
        console.log('ApiClient initialized');
    }
    
    /**
     * Make a GET request.
     * 
     * @param {string} endpoint - API endpoint
     * @param {Object} params - Query parameters
     * @param {Object} options - Request options
     * @returns {Promise<Object>} API response
     */
    async get(endpoint, params = {}, options = {}) {
        const url = this._buildUrl(endpoint, params);
        return this._request(url, { method: 'GET', ...options });
    }
    
    /**
     * Make a POST request.
     * 
     * @param {string} endpoint - API endpoint  
     * @param {Object} data - Request body
     * @param {Object} options - Request options
     * @returns {Promise<Object>} API response
     */
    async post(endpoint, data = {}, options = {}) {
        const url = this._buildUrl(endpoint);
        return this._request(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            body: JSON.stringify(data),
            ...options
        });
    }
    
    /**
     * Build URL with parameters.
     * 
     * @param {string} endpoint - API endpoint
     * @param {Object} params - Query parameters
     * @returns {string} Complete URL
     */
    _buildUrl(endpoint, params = {}) {
        const url = new URL(endpoint, window.location.origin);
        
        Object.keys(params).forEach(key => {
            if (params[key] !== null && params[key] !== undefined) {
                url.searchParams.append(key, params[key]);
            }
        });
        
        return url.toString();
    }
    
    /**
     * Make HTTP request with error handling.
     * 
     * @param {string} url - Request URL
     * @param {Object} options - Fetch options
     * @returns {Promise<Object>} Response data
     */
    async _request(url, options = {}) {
        this.requestCount++;
        const requestId = this.requestCount;
        
        try {
            console.debug(`[${requestId}] API Request: ${options.method || 'GET'} ${url}`);
            
            // Set timeout
            const controller = new AbortController();
            const timeout = options.timeout || this.defaultTimeout;
            const timeoutId = setTimeout(() => controller.abort(), timeout);
            
            const response = await fetch(url, {
                signal: controller.signal,
                ...options
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.debug(`[${requestId}] API Response:`, data);
            
            return data;
            
        } catch (error) {
            console.error(`[${requestId}] API Error:`, error);
            
            // Return error response in consistent format
            return {
                status: 'error',
                error: error.message || 'Request failed',
                timestamp: new Date().toISOString()
            };
        }
    }
    
    /**
     * Get API health status.
     * @returns {Promise<Object>} Health status
     */
    async getHealth() {
        return this.get('/api/health');
    }
    
    /**
     * Get performance data.
     * @param {string} codebase - Codebase identifier
     * @returns {Promise<Object>} Performance metrics
     */
    async getPerformanceRealtime(codebase = '/testmaster') {
        return this.get('/api/performance/realtime', { codebase });
    }
    
    /**
     * Get workflow status.
     * @returns {Promise<Object>} Workflow status
     */
    async getWorkflowStatus() {
        return this.get('/api/workflow/status');
    }
    
    /**
     * Get LLM status.
     * @returns {Promise<Object>} LLM status
     */
    async getLlmStatus() {
        return this.get('/api/llm/status');
    }
    
    /**
     * Toggle LLM mode.
     * @param {boolean} enabled - New LLM state
     * @returns {Promise<Object>} Toggle result
     */
    async toggleLlmMode(enabled) {
        return this.post('/api/llm/toggle-mode', { enabled });
    }
    
    /**
     * Get test status.
     * @returns {Promise<Object>} Test status
     */
    async getTestStatus() {
        return this.get('/api/tests/status');
    }
    
    /**
     * Get analytics metrics.
     * @returns {Promise<Object>} Analytics data
     */
    async getAnalyticsMetrics() {
        return this.get('/api/analytics/metrics');
    }
    
    /**
     * Get refactor analysis.
     * @param {string} codebase - Codebase identifier  
     * @returns {Promise<Object>} Refactor analysis
     */
    async getRefactorAnalysis(codebase = '/testmaster') {
        return this.get('/api/refactor/analysis', { codebase });
    }
}

// Global API client instance
window.apiClient = new ApiClient();
/**
 * Helper Functions
 * ================
 * 
 * Utility functions used across the application.
 * 
 * @author TestMaster Team
 */

/**
 * Format bytes to human readable string.
 * @param {number} bytes - Number of bytes
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted string
 */
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Format number with thousands separator.
 * @param {number} num - Number to format
 * @returns {string} Formatted number
 */
function formatNumber(num) {
    if (typeof num !== 'number') return '0';
    return num.toLocaleString();
}

/**
 * Format timestamp to readable string.
 * @param {string|Date} timestamp - Timestamp to format
 * @returns {string} Formatted time
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return '--';
    
    const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
    if (isNaN(date.getTime())) return '--';
    
    return date.toLocaleString();
}

/**
 * Format relative time (e.g., "2 minutes ago").
 * @param {string|Date} timestamp - Timestamp
 * @returns {string} Relative time string
 */
function formatRelativeTime(timestamp) {
    if (!timestamp) return 'Never';
    
    const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
    const now = new Date();
    const diffMs = now - date;
    
    const seconds = Math.floor(diffMs / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    if (seconds < 60) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
}

/**
 * Debounce function calls.
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function calls.
 * @param {Function} func - Function to throttle
 * @param {number} limit - Time limit in milliseconds
 * @returns {Function} Throttled function
 */
function throttle(func, limit) {
    let inThrottle;
    return function executedFunction(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Deep clone an object.
 * @param {Object} obj - Object to clone
 * @returns {Object} Cloned object
 */
function deepClone(obj) {
    if (obj === null || typeof obj !== 'object') return obj;
    if (obj instanceof Date) return new Date(obj.getTime());
    if (obj instanceof Array) return obj.map(item => deepClone(item));
    if (typeof obj === 'object') {
        const cloned = {};
        Object.keys(obj).forEach(key => {
            cloned[key] = deepClone(obj[key]);
        });
        return cloned;
    }
}

/**
 * Check if value is empty (null, undefined, empty string, empty array).
 * @param {*} value - Value to check
 * @returns {boolean} True if empty
 */
function isEmpty(value) {
    if (value == null) return true;
    if (typeof value === 'string') return value.trim().length === 0;
    if (Array.isArray(value)) return value.length === 0;
    if (typeof value === 'object') return Object.keys(value).length === 0;
    return false;
}

/**
 * Generate random ID.
 * @param {number} length - ID length
 * @returns {string} Random ID
 */
function generateId(length = 8) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < length; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
}

/**
 * Show notification (if supported).
 * @param {string} title - Notification title
 * @param {string} message - Notification message
 * @param {string} type - Notification type (info, success, warning, error)
 */
function showNotification(title, message, type = 'info') {
    // Try browser notifications first
    if ('Notification' in window && Notification.permission === 'granted') {
        new Notification(title, {
            body: message,
            icon: '/favicon.ico'
        });
    } else {
        // Fallback to console
        const prefix = `[${type.toUpperCase()}]`;
        console.log(`${prefix} ${title}: ${message}`);
    }
}

/**
 * Request notification permission.
 */
function requestNotificationPermission() {
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
}

/**
 * Get status color based on value.
 * @param {string} status - Status string
 * @returns {string} CSS color class
 */
function getStatusColor(status) {
    switch (status?.toLowerCase()) {
        case 'success':
        case 'healthy':
        case 'online':
        case 'active':
            return 'status-good';
        case 'warning':
        case 'degraded':
            return 'status-warning';
        case 'error':
        case 'failed':
        case 'offline':
        case 'inactive':
            return 'status-error';
        default:
            return 'status-info';
    }
}

/**
 * Smooth scroll to element.
 * @param {string|Element} target - CSS selector or element
 */
function scrollToElement(target) {
    const element = typeof target === 'string' ? document.querySelector(target) : target;
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

/**
 * Copy text to clipboard.
 * @param {string} text - Text to copy
 * @returns {Promise<boolean>} Success state
 */
async function copyToClipboard(text) {
    try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
            return true;
        } else {
            // Fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            const success = document.execCommand('copy');
            document.body.removeChild(textarea);
            return success;
        }
    } catch (error) {
        console.error('Failed to copy to clipboard:', error);
        return false;
    }
}

// Export functions to global scope
window.Utils = {
    formatBytes,
    formatNumber,
    formatTimestamp,
    formatRelativeTime,
    debounce,
    throttle,
    deepClone,
    isEmpty,
    generateId,
    showNotification,
    requestNotificationPermission,
    getStatusColor,
    scrollToElement,
    copyToClipboard
};

console.log('Helper functions loaded');
/* STEELCLAD EXTRACTED: Unified Gamma Dashboard JavaScript */
/* Original: unified_gamma_dashboard.html lines 434-953 (519 lines) */
/* Author: Agent X (STEELCLAD Frontend Atomization) */

// Unified Dashboard Controller
class UnifiedGammaDashboard {
    constructor() {
        this.socket = null;
        this.charts = {};
        this.graph3d = null;
        this.connected = false;
        
        this.init();
    }
    
    async init() {
        await this.setupSocketIO();
        this.setupCharts();
        this.setup3DVisualization();
        this.startPerformanceMonitoring();
        
        console.log('🚀 Unified Gamma Dashboard initialized');
    }
    
    async setupSocketIO() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            this.connected = true;
            document.getElementById('connection-status').textContent = 'Connected';
            console.log('✅ Connected to dashboard backend');
        });
        
        this.socket.on('disconnect', () => {
            this.connected = false;
            document.getElementById('connection-status').textContent = 'Disconnected';
            console.log('❌ Disconnected from dashboard backend');
        });
        
        this.socket.on('initial_data', (data) => {
            this.updateDashboard(data);
        });
        
        this.socket.on('data_update', (data) => {
            this.updateDashboard(data);
        });
        
        this.socket.on('agent_update', (data) => {
            this.updateAgentMetrics(data);
        });
        
        this.socket.on('performance_update', (data) => {
            this.updatePerformanceMetrics(data);
        });
    }
    
    setupCharts() {
        // EPSILON ENHANCEMENT: Advanced Intelligent Charts with AI-Powered Interactions
        
        // System Health Chart with Intelligence
        const healthCtx = document.getElementById('system-health-chart').getContext('2d');
        this.charts.systemHealth = new Chart(healthCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'CPU %',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4,
                    pointHoverRadius: 8,
                    pointHoverBackgroundColor: '#3b82f6'
                }, {
                    label: 'Memory %',
                    data: [],
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    tension: 0.4,
                    pointHoverRadius: 8,
                    pointHoverBackgroundColor: '#8b5cf6'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: { 
                        display: true,
                        labels: { 
                            color: 'white',
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        enabled: true,
                        backgroundColor: 'rgba(0, 0, 0, 0.9)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: '#3b82f6',
                        borderWidth: 1,
                        displayColors: true,
                        callbacks: {
                            // EPSILON: Enhanced tooltips with intelligent context
                            beforeTitle: function(tooltipItems) {
                                return 'System Health Analysis';
                            },
                            afterBody: function(tooltipItems) {
                                const cpu = tooltipItems[0]?.parsed?.y || 0;
                                const memory = tooltipItems[1]?.parsed?.y || 0;
                                const analysis = [];
                                
                                if (cpu > 80) analysis.push('⚠️ High CPU usage detected');
                                if (memory > 85) analysis.push('⚠️ High memory consumption');
                                if (cpu < 30 && memory < 40) analysis.push('✅ System running efficiently');
                                
                                return analysis.length ? analysis : ['📊 System performance normal'];
                            }
                        }
                    }
                },
                scales: {
                    y: { 
                        display: true,
                        ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        beginAtZero: true,
                        max: 100
                    },
                    x: { 
                        display: true,
                        ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                },
                // EPSILON: Advanced chart interactions
                onHover: (event, elements) => {
                    event.native.target.style.cursor = elements.length > 0 ? 'pointer' : 'default';
                },
                onClick: (event, elements) => {
                    if (elements.length > 0) {
                        this.handleChartDrillDown('system-health', elements[0], event);
                    }
                }
            }
        });
        
        // API Usage Chart
        const apiCtx = document.getElementById('api-usage-chart').getContext('2d');
        this.charts.apiUsage = new Chart(apiCtx, {
            type: 'bar',
            data: {
                labels: ['Alpha', 'Beta', 'Gamma', 'D', 'E'],
                datasets: [{
                    label: 'API Calls',
                    data: [],
                    backgroundColor: [
                        '#10b981', '#f59e0b', '#8b5cf6', 
                        '#06b6d4', '#84cc16'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: { 
                        display: true,
                        ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    x: { 
                        display: true,
                        ticks: { color: 'rgba(255, 255, 255, 0.7)' },
                        grid: { display: false }
                    }
                }
            }
        });
    }
    
    async setup3DVisualization() {
        try {
            const response = await fetch('/api/3d-visualization-data');
            const data = await response.json();
            
            const container = document.getElementById('3d-visualization');
            container.innerHTML = ''; // Remove loading spinner
            
            this.graph3d = ForceGraph3D()(container)
                .graphData(data.graph_data)
                .nodeLabel('id')
                .nodeColor('color')
                .nodeVal('size')
                .linkColor(() => 'rgba(255, 255, 255, 0.2)')
                .backgroundColor('rgba(0, 0, 0, 0)')
                .showNavInfo(false);
                
            console.log('✅ 3D Visualization initialized');
        } catch (error) {
            console.error('❌ Failed to initialize 3D visualization:', error);
        }
    }
    
    updateDashboard(data) {
        if (data.system_health) {
            document.getElementById('cpu-usage').textContent = 
                data.system_health.cpu_usage?.toFixed(1) + '%' || '--';
            document.getElementById('memory-usage').textContent = 
                data.system_health.memory_usage?.toFixed(1) + '%' || '--';
            document.getElementById('health-score').textContent = 
                data.system_health.system_health || '--';
                
            // Update charts
            this.updateSystemHealthChart(data.system_health);
        }
        
        if (data.api_usage) {
            document.getElementById('daily-requests').textContent = 
                data.api_usage.total_calls || '0';
            document.getElementById('daily-cost').textContent = 
                '$' + (data.api_usage.daily_spending?.toFixed(2) || '0.00');
            document.getElementById('api-budget').textContent = 
                data.api_usage.daily_spending?.toFixed(2) || '0.00';
            document.getElementById('budget-status').textContent = 
                data.api_usage.budget_status || 'OK';
        }
    }
    
    updateAgentMetrics(data) {
        document.getElementById('active-agents').textContent = 
            data.active_agents || '--';
        document.getElementById('coordination-health').textContent = 
            data.coordination_health || '--';
        document.getElementById('total-tasks').textContent = 
            Object.values(data.agents || {}).reduce((sum, agent) => 
                sum + (agent.tasks || 0), 0);
    }
    
    updatePerformanceMetrics(data) {
        document.getElementById('load-time').textContent = 
            data.load_time?.toFixed(1) + 's' || '--';
        document.getElementById('bundle-size').textContent = 
            data.bundle_size?.toFixed(1) + 'MB' || '--';
        document.getElementById('lighthouse-score').textContent = 
            data.lighthouse_score || '--';
    }
    
    updateSystemHealthChart(healthData) {
        const chart = this.charts.systemHealth;
        const now = new Date().toLocaleTimeString();
        
        chart.data.labels.push(now);
        chart.data.datasets[0].data.push(healthData.cpu_usage || 0);
        chart.data.datasets[1].data.push(healthData.memory_usage || 0);
        
        // Keep only last 20 data points
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
            chart.data.datasets[1].data.shift();
        }
        
        chart.update('none');
    }
    
    // EPSILON ENHANCEMENT: Advanced Chart Intelligence Methods
    // =====================================================
    
    handleChartDrillDown(chartType, element, event) {
        const dataIndex = element.index;
        const datasetIndex = element.datasetIndex;
        
        // Context-aware drill-down based on chart type and user role
        switch(chartType) {
            case 'system-health':
                this.showSystemHealthDetails(dataIndex, datasetIndex, event);
                break;
            case 'api-usage':
                this.showApiUsageDetails(dataIndex, datasetIndex, event);
                break;
            default:
                this.showGenericDetails(chartType, dataIndex, datasetIndex, event);
        }
    }
    
    showSystemHealthDetails(dataIndex, datasetIndex, event) {
        const chart = this.charts.systemHealth;
        const labels = chart.data.labels;
        const datasets = chart.data.datasets;
        
        if (!labels[dataIndex]) return;
        
        const timestamp = labels[dataIndex];
        const cpuValue = datasets[0].data[dataIndex] || 0;
        const memoryValue = datasets[1].data[dataIndex] || 0;
        
        // Create intelligent analysis popup
        const analysis = this.generateHealthAnalysis(cpuValue, memoryValue, timestamp);
        this.showIntelligentPopup('System Health Analysis', analysis, event);
    }
    
    generateHealthAnalysis(cpu, memory, timestamp) {
        const analysis = {
            timestamp: timestamp,
            metrics: {
                cpu: `${cpu.toFixed(1)}%`,
                memory: `${memory.toFixed(1)}%`,
                health_score: this.calculateHealthScore(cpu, memory)
            },
            insights: [],
            recommendations: []
        };
        
        // AI-powered insights
        if (cpu > 85) {
            analysis.insights.push('🔴 Critical CPU usage - immediate attention required');
            analysis.recommendations.push('Review active processes and consider load balancing');
        } else if (cpu > 70) {
            analysis.insights.push('🟡 High CPU usage detected');
            analysis.recommendations.push('Monitor CPU-intensive processes');
        } else if (cpu < 20) {
            analysis.insights.push('🟢 Low CPU utilization - system resources available');
        }
        
        if (memory > 90) {
            analysis.insights.push('🔴 Critical memory usage - risk of system instability');
            analysis.recommendations.push('Check for memory leaks and restart services if needed');
        } else if (memory > 75) {
            analysis.insights.push('🟡 High memory consumption');
            analysis.recommendations.push('Review memory usage patterns');
        } else if (memory < 30) {
            analysis.insights.push('🟢 Efficient memory utilization');
        }
        
        // Correlation analysis
        const correlation = this.analyzeCorrelation(cpu, memory);
        if (correlation.strength > 0.7) {
            analysis.insights.push(`📊 Strong correlation detected: ${correlation.description}`);
        }
        
        return analysis;
    }
    
    calculateHealthScore(cpu, memory) {
        const cpuScore = Math.max(0, 100 - cpu);
        const memoryScore = Math.max(0, 100 - memory);
        const composite = (cpuScore + memoryScore) / 2;
        
        if (composite >= 80) return { score: composite.toFixed(1), status: '🟢 Excellent' };
        if (composite >= 60) return { score: composite.toFixed(1), status: '🟡 Good' };
        if (composite >= 40) return { score: composite.toFixed(1), status: '🟠 Fair' };
        return { score: composite.toFixed(1), status: '🔴 Poor' };
    }
    
    analyzeCorrelation(cpu, memory) {
        const ratio = cpu / Math.max(memory, 1);
        
        if (ratio > 1.5) {
            return {
                strength: 0.8,
                description: 'CPU-bound workload detected - CPU usage exceeds memory pressure'
            };
        } else if (ratio < 0.5) {
            return {
                strength: 0.7,
                description: 'Memory-intensive workload - high memory usage with low CPU'
            };
        } else {
            return {
                strength: 0.9,
                description: 'Balanced system load - proportional CPU and memory usage'
            };
        }
    }
    
    showIntelligentPopup(title, analysis, event) {
        // Remove existing popup
        const existingPopup = document.getElementById('intelligent-popup');
        if (existingPopup) existingPopup.remove();
        
        // Create new popup
        const popup = document.createElement('div');
        popup.id = 'intelligent-popup';
        popup.style.cssText = `
            position: fixed;
            top: ${event.clientY + 10}px;
            left: ${event.clientX + 10}px;
            background: rgba(0, 0, 0, 0.95);
            border: 1px solid #3b82f6;
            border-radius: 12px;
            padding: 20px;
            min-width: 320px;
            max-width: 500px;
            color: white;
            font-family: 'SF Pro Text', sans-serif;
            z-index: 10000;
            backdrop-filter: blur(20px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
        `;
        
        let content = `
            <div style="font-size: 16px; font-weight: 600; margin-bottom: 12px; color: #3b82f6;">
                ${title}
            </div>
            <div style="font-size: 12px; color: rgba(255, 255, 255, 0.7); margin-bottom: 16px;">
                ${analysis.timestamp}
            </div>
        `;
        
        // Metrics section
        content += `<div style="margin-bottom: 16px;">`;
        for (const [key, value] of Object.entries(analysis.metrics)) {
            if (typeof value === 'object') {
                content += `<div style="margin-bottom: 8px;">
                    <span style="color: rgba(255, 255, 255, 0.8);">${key.replace('_', ' ').toUpperCase()}:</span>
                    <span style="margin-left: 8px; font-weight: 600;">${value.score}</span>
                    <span style="margin-left: 8px;">${value.status}</span>
                </div>`;
            } else {
                content += `<div style="margin-bottom: 8px;">
                    <span style="color: rgba(255, 255, 255, 0.8);">${key.replace('_', ' ').toUpperCase()}:</span>
                    <span style="margin-left: 8px; font-weight: 600;">${value}</span>
                </div>`;
            }
        }
        content += `</div>`;
        
        // Insights section
        if (analysis.insights.length) {
            content += `<div style="margin-bottom: 16px;">
                <div style="font-weight: 600; margin-bottom: 8px; color: #8b5cf6;">AI Insights:</div>`;
            analysis.insights.forEach(insight => {
                content += `<div style="margin-bottom: 6px; font-size: 14px;">${insight}</div>`;
            });
            content += `</div>`;
        }
        
        // Recommendations section
        if (analysis.recommendations.length) {
            content += `<div style="margin-bottom: 16px;">
                <div style="font-weight: 600; margin-bottom: 8px; color: #10b981;">Recommendations:</div>`;
            analysis.recommendations.forEach(rec => {
                content += `<div style="margin-bottom: 6px; font-size: 14px;">• ${rec}</div>`;
            });
            content += `</div>`;
        }
        
        // Close button
        content += `
            <div style="text-align: right; margin-top: 16px;">
                <button onclick="document.getElementById('intelligent-popup').remove()" 
                        style="background: #3b82f6; color: white; border: none; padding: 8px 16px; 
                               border-radius: 6px; cursor: pointer; font-size: 14px;">
                    Close
                </button>
            </div>
        `;
        
        popup.innerHTML = content;
        document.body.appendChild(popup);
        
        // Auto-remove after 15 seconds
        setTimeout(() => {
            if (document.getElementById('intelligent-popup')) {
                document.getElementById('intelligent-popup').remove();
            }
        }, 15000);
    }
    
    startPerformanceMonitoring() {
        // Request performance updates every 5 seconds
        setInterval(() => {
            if (this.connected) {
                this.socket.emit('request_update', { component: 'performance' });
            }
        }, 5000);
    }
}

// Global functions for button interactions
function refreshSystemHealth() {
    if (window.dashboard && window.dashboard.connected) {
        window.dashboard.socket.emit('request_update', { component: 'all' });
    }
}

function refreshAgentStatus() {
    if (window.dashboard && window.dashboard.connected) {
        window.dashboard.socket.emit('request_update', { component: 'agents' });
    }
}

function showCostEstimator() {
    alert('Cost Estimator - Feature coming in next phase!');
}

function toggle3DFullscreen() {
    const container = document.getElementById('3d-visualization');
    if (container.requestFullscreen) {
        container.requestFullscreen();
    }
}

function reset3DView() {
    if (window.dashboard && window.dashboard.graph3d) {
        window.dashboard.graph3d.zoomToFit(1000);
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new UnifiedGammaDashboard();
});
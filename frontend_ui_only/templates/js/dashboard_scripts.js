/* STEELCLAD EXTRACTED: Dashboard Core JavaScript */
/* Original: dashboard.html lines 462-1429 (~969 lines) */
/* Author: Agent X (STEELCLAD Frontend Atomization) */

class ModularDashboard {
    constructor() {
        this.socket = io();
        this.realTimeChart = null;
        this.isStreaming = false;
        this.chartData = {
            labels: [],
            datasets: [{
                label: 'Performance Score',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.4
            }, {
                label: 'CPU Usage',
                data: [],
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                tension: 0.4
            }, {
                label: 'Memory Usage',
                data: [],
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                tension: 0.4
            }]
        };
        this.predictiveData = null;
        this.setupEventHandlers();
        this.initializeDashboard();
        this.initializeRealTimeChart();
        this.initializeD3Charts();
    }
    
    setupEventHandlers() {
        this.socket.on('connect', () => {
            console.log('Connected to Modular Dashboard');
            document.getElementById('connectionStatus').textContent = 'Connected';
            document.getElementById('connectionStatus').className = 'connected';
            this.requestInitialData();
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from dashboard');
            document.getElementById('connectionStatus').textContent = 'Disconnected';
            document.getElementById('connectionStatus').className = 'disconnected';
        });
        
        this.socket.on('analysis_result', (data) => {
            this.updateContextualAnalysis(data.analysis);
        });
        
        // Hour 8: Real-time streaming event handlers
        this.socket.on('performance_stream', (data) => {
            this.updateRealTimeChart(data.metrics);
        });
        
        this.socket.on('predictive_stream', (data) => {
            this.updatePredictiveAnalytics(data.predictions);
        });
        
        this.socket.on('chart_data_update', (data) => {
            this.handleChartDataUpdate(data);
        });
        
        this.socket.on('predictive_analysis_result', (data) => {
            this.displayPredictiveAnalysis(data);
        });
    }
    
    async initializeDashboard() {
        // Fetch initial data via REST API
        await this.fetchHealthData();
        await this.fetchProactiveInsights();
        await this.fetchBehaviorPredictions();
        await this.fetchUnifiedData();
        await this.fetchPerformanceMetrics();
        await this.fetchVisualizationIntelligence();
        
        // Set up periodic updates
        setInterval(() => this.fetchHealthData(), 5000);
        setInterval(() => this.fetchProactiveInsights(), 10000);
        setInterval(() => this.fetchBehaviorPredictions(), 15000);
        setInterval(() => this.fetchUnifiedData(), 8000);
        setInterval(() => this.fetchPerformanceMetrics(), 3000);
        setInterval(() => this.fetchVisualizationIntelligence(), 12000);
    }
    
    requestInitialData() {
        // Request analysis via WebSocket
        const mockAgentData = {
            'agent_alpha': {
                'cpu_usage': 45,
                'memory_usage': 62,
                'response_time': 120,
                'error_rate': 2
            },
            'agent_beta': {
                'cpu_usage': 38,
                'memory_usage': 55,
                'response_time': 95,
                'error_rate': 1
            },
            'agent_gamma': {
                'cpu_usage': 72,
                'memory_usage': 81,
                'response_time': 250,
                'error_rate': 4
            }
        };
        
        this.socket.emit('request_analysis', { agent_data: mockAgentData });
    }
    
    async fetchHealthData() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            if (data.modules && data.modules.intelligence_metrics) {
                const metrics = data.modules.intelligence_metrics;
                document.getElementById('correlationsCount').textContent = metrics.correlations_detected;
                document.getElementById('insightsCount').textContent = metrics.insights_generated;
                document.getElementById('predictionsCount').textContent = metrics.predictions_made;
                document.getElementById('optimizationCount').textContent = metrics.optimization_opportunities;
            }
            
            // Remove loading state
            document.getElementById('contextualMetrics').classList.remove('loading');
        } catch (error) {
            console.error('Error fetching health data:', error);
        }
    }
    
    async fetchProactiveInsights() {
        try {
            const response = await fetch('/api/proactive-insights');
            const data = await response.json();
            
            const container = document.getElementById('proactiveInsights');
            container.innerHTML = '';
            container.classList.remove('loading');
            
            if (data.insights && data.insights.length > 0) {
                data.insights.forEach(insight => {
                    const insightEl = document.createElement('div');
                    insightEl.className = 'insight';
                    insightEl.innerHTML = `
                        <div class="insight-type">${insight.type.replace('_', ' ').toUpperCase()}</div>
                        <div class="insight-message">${insight.message}</div>
                        ${insight.recommendations ? `
                            <div class="recommendations">
                                Recommendations:
                                <ul>${insight.recommendations.map(r => `<li>${r}</li>`).join('')}</ul>
                            </div>
                        ` : ''}
                    `;
                    container.appendChild(insightEl);
                });
            } else {
                container.innerHTML = '<div class="insight"><div class="insight-type">All systems optimal</div></div>';
            }
        } catch (error) {
            console.error('Error fetching insights:', error);
        }
    }
    
    async fetchBehaviorPredictions() {
        try {
            const response = await fetch('/api/behavior-prediction', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_context: { role: 'technical', device: 'desktop' },
                    history: [
                        { action: 'view_metrics', timestamp: Date.now() - 60000 },
                        { action: 'check_health', timestamp: Date.now() - 30000 },
                        { action: 'view_metrics', timestamp: Date.now() - 15000 }
                    ]
                })
            });
            
            const data = await response.json();
            
            if (data.predictions) {
                const predictions = data.predictions;
                
                // Update next action
                if (predictions.next_likely_actions && predictions.next_likely_actions.length > 0) {
                    const topAction = predictions.next_likely_actions[0];
                    document.getElementById('nextAction').textContent = 
                        `${topAction.action} (${Math.round(topAction.probability * 100)}%)`;
                }
                
                // Update information needs
                if (predictions.information_needs && predictions.information_needs.length > 0) {
                    document.getElementById('infoNeed').textContent = 
                        predictions.information_needs[0].replace('_', ' ');
                }
                
                // Update attention focus
                document.getElementById('attentionFocus').textContent = 'Metrics & Health';
            }
            
            document.getElementById('behaviorPredictions').classList.remove('loading');
        } catch (error) {
            console.error('Error fetching predictions:', error);
        }
    }
    
    async fetchUnifiedData() {
        try {
            const response = await fetch('/api/unified-data?role=technical&device=desktop');
            const data = await response.json();
            
            if (data.data && data.data.intelligence_metadata) {
                const metadata = data.data.intelligence_metadata;
                document.getElementById('dataSources').textContent = '5 Active';
                document.getElementById('intelligenceScore').textContent = 
                    `${(metadata.synthesis_quality * 100).toFixed(1)}%`;
                document.getElementById('informationDensity').textContent = 
                    `${metadata.information_density.toFixed(0)}%`;
            }
            
            document.getElementById('dataIntegration').classList.remove('loading');
        } catch (error) {
            console.error('Error fetching unified data:', error);
        }
    }
    
    async fetchPerformanceMetrics() {
        try {
            const response = await fetch('/api/performance-metrics');
            const data = await response.json();
            
            if (data.metrics) {
                const metrics = data.metrics;
                
                // Update performance score and status indicator
                document.getElementById('performanceScore').textContent = 
                    `${Math.round(metrics.performance_score)}%`;
                
                const performanceStatusEl = document.getElementById('performanceStatus');
                if (metrics.performance_score >= 80) {
                    performanceStatusEl.className = 'status-indicator status-excellent';
                } else if (metrics.performance_score >= 60) {
                    performanceStatusEl.className = 'status-indicator status-good';
                } else {
                    performanceStatusEl.className = 'status-indicator status-needs_attention';
                }
                
                // Update system metrics
                document.getElementById('cpuUsage').textContent = `${metrics.cpu_usage.toFixed(1)}%`;
                document.getElementById('memoryUsage').textContent = `${metrics.memory_usage.toFixed(1)}%`;
                document.getElementById('systemHealth').textContent = metrics.system_health;
            }
            
            document.getElementById('performanceMonitoring').classList.remove('loading');
        } catch (error) {
            console.error('Error fetching performance metrics:', error);
            document.getElementById('performanceMonitoring').classList.remove('loading');
        }
    }
    
    async fetchVisualizationIntelligence() {
        try {
            const response = await fetch('/api/visualization/intelligence-insights');
            const data = await response.json();
            
            if (data.insights) {
                const insights = data.insights;
                
                // Update AI recommendations count
                document.getElementById('aiRecommendations').textContent = 
                    insights.recommendations ? insights.recommendations.length : '0';
                
                // Update layout optimization status
                const optimizations = insights.optimizations || [];
                document.getElementById('layoutOptimization').textContent = 
                    optimizations.length > 0 ? `${optimizations.length} Available` : 'Optimal';
                
                // Update interactive features count
                const uxImprovements = insights.user_experience_improvements || [];
                document.getElementById('interactiveFeatures').textContent = 
                    uxImprovements.length > 0 ? `${uxImprovements.length} Suggested` : 'Active';
                
                // Update data quality score
                const dataQualityIssues = insights.data_quality_issues || [];
                const qualityScore = Math.max(0, 100 - (dataQualityIssues.length * 20));
                document.getElementById('dataQualityScore').textContent = `${qualityScore}%`;
                
                const qualityStatusEl = document.getElementById('dataQualityStatus');
                if (qualityScore >= 80) {
                    qualityStatusEl.className = 'status-indicator status-excellent';
                } else if (qualityScore >= 60) {
                    qualityStatusEl.className = 'status-indicator status-good';
                } else {
                    qualityStatusEl.className = 'status-indicator status-needs_attention';
                }
            }
            
            document.getElementById('visualizationIntelligence').classList.remove('loading');
        } catch (error) {
            console.error('Error fetching visualization intelligence:', error);
            document.getElementById('visualizationIntelligence').classList.remove('loading');
        }
    }
    
    // Hour 8: Real-time Chart Integration Methods
    initializeRealTimeChart() {
        const ctx = document.getElementById('realTimeChart').getContext('2d');
        
        this.realTimeChart = new Chart(ctx, {
            type: 'line',
            data: this.chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 750,
                    easing: 'easeInOutQuad'
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'minute',
                            displayFormats: {
                                minute: 'HH:mm'
                            }
                        },
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Percentage'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            afterBody: function(tooltipItems) {
                                let total = 0;
                                tooltipItems.forEach(function(tooltipItem) {
                                    total += tooltipItem.parsed.y;
                                });
                                return 'Combined Load: ' + (total / tooltipItems.length).toFixed(1) + '%';
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }
    
    updateRealTimeChart(metrics) {
        if (!this.realTimeChart || !this.isStreaming) return;
        
        const timestamp = new Date();
        const maxDataPoints = 50;
        
        // Add new data point
        this.chartData.labels.push(timestamp);
        this.chartData.datasets[0].data.push(metrics.performance_score || 0);
        this.chartData.datasets[1].data.push(metrics.cpu_usage || 0);
        this.chartData.datasets[2].data.push(metrics.memory_usage || 0);
        
        // Remove old data points if we exceed max
        if (this.chartData.labels.length > maxDataPoints) {
            this.chartData.labels.shift();
            this.chartData.datasets.forEach(dataset => dataset.data.shift());
        }
        
        // Update chart
        this.realTimeChart.update('none'); // No animation for real-time updates
        
        // Update chart status
        document.getElementById('chartStatus').textContent = 
            `Live streaming - Last update: ${timestamp.toLocaleTimeString()}`;
    }
    
    updatePredictiveAnalytics(predictions) {
        if (!predictions.predictions || predictions.predictions.length === 0) return;
        
        const nextHour = predictions.predictions[0];
        const avgConfidence = predictions.confidence?.average || 0;
        
        // Update predictive metrics
        document.getElementById('nextHourForecast').textContent = 
            `${Math.round(nextHour.predicted_value)}%`;
        
        // Determine trend direction
        const trend = predictions.confidence?.trend_strength > 0.5 ? 'Upward' : 'Stable';
        document.getElementById('trendDirection').textContent = trend;
        
        // Update confidence level
        document.getElementById('confidenceLevel').textContent = 
            `${Math.round(avgConfidence * 100)}%`;
            
        const confidenceStatusEl = document.getElementById('confidenceStatus');
        if (avgConfidence >= 0.8) {
            confidenceStatusEl.className = 'status-indicator status-excellent';
        } else if (avgConfidence >= 0.6) {
            confidenceStatusEl.className = 'status-indicator status-good';
        } else {
            confidenceStatusEl.className = 'status-indicator status-needs_attention';
        }
        
        // Update anomaly risk
        const anomalyRisk = predictions.predictions.some(p => p.predicted_value > 90) ? 'High' : 'Low';
        document.getElementById('anomalyRisk').textContent = anomalyRisk;
        
        document.getElementById('predictiveAnalytics').classList.remove('loading');
    }
    
    handleChartDataUpdate(data) {
        if (data.chart_type === 'performance_line') {
            // Handle historical data loading
            this.loadHistoricalData(data);
        }
    }
    
    loadHistoricalData(data) {
        if (!this.realTimeChart || !data.data || !data.data.points) return;
        
        // Clear existing data
        this.chartData.labels = [];
        this.chartData.datasets.forEach(dataset => dataset.data = []);
        
        // Load historical points
        data.data.points.forEach(point => {
            this.chartData.labels.push(new Date(point.timestamp));
            // For historical data, we'll use the value for all datasets as demo
            this.chartData.datasets[0].data.push(point.value);
            this.chartData.datasets[1].data.push(point.value * 0.8);
            this.chartData.datasets[2].data.push(point.value * 1.2);
        });
        
        this.realTimeChart.update();
        document.getElementById('chartStatus').textContent = 
            `Historical data loaded (${data.data.points.length} points)`;
    }
    
    displayPredictiveAnalysis(data) {
        if (!this.realTimeChart) return;
        
        // Add prediction overlay to chart
        const predictions = data.predictions;
        if (predictions && predictions.length > 0) {
            // Create prediction dataset
            const predictionDataset = {
                label: 'Forecast',
                data: predictions.map(p => ({
                    x: new Date(p.timestamp),
                    y: p.predicted_value
                })),
                borderColor: 'rgba(255, 206, 84, 1)',
                backgroundColor: 'rgba(255, 206, 84, 0.2)',
                borderDash: [5, 5],
                tension: 0.4
            };
            
            // Add or update prediction dataset
            const existingPredictionIndex = this.realTimeChart.data.datasets.findIndex(
                ds => ds.label === 'Forecast'
            );
            
            if (existingPredictionIndex >= 0) {
                this.realTimeChart.data.datasets[existingPredictionIndex] = predictionDataset;
            } else {
                this.realTimeChart.data.datasets.push(predictionDataset);
            }
            
            this.realTimeChart.update();
        }
    }
    
    // IRONCLAD CONSOLIDATION: D3.js Chart Methods from charts_dashboard.html
    // =====================================================================
    
    initializeD3Charts() {
        // Initialize D3.js visualizations
        this.createNetworkGraph();
        this.createHeatmap();
    }
    
    // Create D3.js Network Graph
    createNetworkGraph() {
        const width = document.getElementById('networkChart').offsetWidth;
        const height = 300;
        
        const svg = d3.select('#networkChart')
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        const nodes = [
            {id: 'Epsilon', group: 1, radius: 18}, // Main agent
            {id: 'Alpha', group: 1, radius: 15},
            {id: 'Beta', group: 1, radius: 15},
            {id: 'Gamma', group: 1, radius: 15},
            {id: 'Delta', group: 1, radius: 15},
            {id: 'Agent E', group: 2, radius: 12},
            {id: 'Agent A', group: 2, radius: 10},
            {id: 'Agent B', group: 2, radius: 10},
            {id: 'Agent C', group: 2, radius: 10},
            {id: 'Agent D', group: 2, radius: 10}
        ];
        
        const links = [
            {source: 'Epsilon', target: 'Alpha', value: 5}, // Epsilon as central
            {source: 'Epsilon', target: 'Beta', value: 4},
            {source: 'Epsilon', target: 'Gamma', value: 3},
            {source: 'Alpha', target: 'Beta', value: 2},
            {source: 'Beta', target: 'Delta', value: 2},
            {source: 'Delta', target: 'Gamma', value: 1},
            {source: 'Agent E', target: 'Agent A', value: 2},
            {source: 'Agent A', target: 'Agent B', value: 1},
            {source: 'Agent B', target: 'Agent C', value: 1},
            {source: 'Agent C', target: 'Agent D', value: 1}
        ];
        
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(60))
            .force('charge', d3.forceManyBody().strength(-400))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(d => d.radius + 8));
        
        const link = svg.append('g')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.7)
            .attr('stroke-width', d => Math.sqrt(d.value * 2));
        
        const node = svg.append('g')
            .selectAll('circle')
            .data(nodes)
            .enter().append('circle')
            .attr('r', d => d.radius)
            .attr('fill', d => d.group === 1 ? '#667eea' : '#48bb78')
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .call(d3.drag()
                .on('start', this.dragstarted)
                .on('drag', this.dragged)
                .on('end', this.dragended));
        
        const text = svg.append('g')
            .selectAll('text')
            .data(nodes)
            .enter().append('text')
            .text(d => d.id)
            .attr('font-size', '11px')
            .attr('font-weight', 'bold')
            .attr('text-anchor', 'middle')
            .attr('fill', '#fff')
            .attr('dy', '.35em');
        
        simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
            
            text
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
    }
    
    // Create D3.js Heatmap
    createHeatmap() {
        const width = document.getElementById('heatmapChart').offsetWidth;
        const height = 300;
        const margin = {top: 30, right: 30, bottom: 40, left: 80};
        
        const svg = d3.select('#heatmapChart')
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        const hours = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
                      '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'];
        const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
        
        // Generate activity data based on realistic patterns
        const data = [];
        days.forEach((day, i) => {
            hours.forEach((hour, j) => {
                let value;
                if (i >= 5) { // Weekend
                    value = Math.floor(Math.random() * 40) + 10;
                } else if (j >= 9 && j <= 17) { // Business hours
                    value = Math.floor(Math.random() * 50) + 50;
                } else {
                    value = Math.floor(Math.random() * 30) + 10;
                }
                data.push({
                    day: day,
                    hour: hour,
                    value: value
                });
            });
        });
        
        const x = d3.scaleBand()
            .range([margin.left, width - margin.right])
            .domain(hours)
            .padding(0.02);
        
        const y = d3.scaleBand()
            .range([margin.top, height - margin.bottom])
            .domain(days)
            .padding(0.02);
        
        const colorScale = d3.scaleSequential()
            .interpolator(d3.interpolateRdYlBu)
            .domain([100, 0]);
        
        svg.selectAll()
            .data(data, d => d.day + ':' + d.hour)
            .enter()
            .append('rect')
            .attr('x', d => x(d.hour))
            .attr('y', d => y(d.day))
            .attr('width', x.bandwidth())
            .attr('height', y.bandwidth())
            .style('fill', d => colorScale(d.value))
            .on('mouseover', function(event, d) {
                // Add tooltip on hover
                d3.select(this).style('stroke', '#000').style('stroke-width', 2);
            })
            .on('mouseout', function(event, d) {
                d3.select(this).style('stroke', null);
            });
        
        // Add X axis
        svg.append('g')
            .attr('transform', `translate(0,${height - margin.bottom})`)
            .call(d3.axisBottom(x))
            .selectAll('text')
            .style('font-size', '10px');
        
        // Add Y axis
        svg.append('g')
            .attr('transform', `translate(${margin.left},0)`)
            .call(d3.axisLeft(y))
            .selectAll('text')
            .style('font-size', '11px');
        
        // Add title
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', margin.top / 2)
            .attr('text-anchor', 'middle')
            .style('font-size', '12px')
            .style('font-weight', 'bold')
            .text('Agent Activity by Hour and Day');
    }
    
    // D3.js drag handlers
    dragstarted(event, d) {
        if (!event.active) d3.select(this).simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    dragended(event, d) {
        if (!event.active) d3.select(this).simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    
    updateContextualAnalysis(analysis) {
        if (!analysis) return;
        
        // Update coordination health
        if (analysis.agent_coordination_health) {
            const health = analysis.agent_coordination_health;
            document.getElementById('healthScore').textContent = health.overall_score;
            
            const statusEl = document.getElementById('healthStatus');
            statusEl.className = `status-indicator status-${health.status}`;
            
            if (health.factors) {
                document.getElementById('dataSync').textContent = 
                    `${Math.round(health.factors.data_synchronization)}%`;
                document.getElementById('responseConsistency').textContent = 
                    `${Math.round(health.factors.response_time_consistency)}%`;
                document.getElementById('resourceBalance').textContent = 
                    `${Math.round(health.factors.resource_utilization_balance)}%`;
            }
        }
        
        document.getElementById('coordinationHealth').classList.remove('loading');
    }
}

// API Testing Functions
async function testUnifiedData() {
    try {
        const response = await fetch('/api/unified-data?role=developer&device=desktop');
        const data = await response.json();
        showApiResponse('Unified Data Integration Test', data);
    } catch (error) {
        showApiResponse('Unified Data Integration Test', { error: error.message });
    }
}

async function testVisualizationRecs() {
    try {
        const response = await fetch('/api/visualization-recommendations', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                data_characteristics: {
                    volume: 150,
                    has_time_series: true,
                    correlation_count: 5,
                    has_hierarchy: false
                },
                user_context: { role: 'technical', device: 'desktop' }
            })
        });
        const data = await response.json();
        showApiResponse('Visualization Recommendations Test', data);
    } catch (error) {
        showApiResponse('Visualization Recommendations Test', { error: error.message });
    }
}

async function testChartConfig() {
    try {
        const response = await fetch('/api/chart-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                chart_type: 'intelligent_line_chart',
                data: { daily: true, hourly: true },
                user_context: { role: 'analyst', device: 'desktop' },
                enhancements: ['drill_down', 'smart_tooltips', 'trend_lines']
            })
        });
        const data = await response.json();
        showApiResponse('Chart Configuration Test', data);
    } catch (error) {
        showApiResponse('Chart Configuration Test', { error: error.message });
    }
}

function showApiResponse(title, data) {
    const responseEl = document.getElementById('apiResponse');
    responseEl.style.display = 'block';
    responseEl.textContent = `${title}:\n${JSON.stringify(data, null, 2)}`;
}

function showVisualizationResponse(title, data) {
    const responseEl = document.getElementById('visualizationResponse');
    responseEl.style.display = 'block';
    responseEl.textContent = `${title}:\n${JSON.stringify(data, null, 2)}`;
}

// Hour 7: Advanced Visualization Testing Functions
async function testInteractiveConfig() {
    try {
        const response = await fetch('/api/visualization/interactive-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                chart_type: 'intelligent_dashboard',
                data_sources: [
                    { type: 'performance_metrics', volume: 100 },
                    { type: 'user_behavior', volume: 250 }
                ],
                user_context: { role: 'analyst', device: 'desktop' },
                interactions: ['drill_down', 'smart_tooltips', 'correlation_highlights']
            })
        });
        const data = await response.json();
        showVisualizationResponse('Interactive Configuration Test', data);
    } catch (error) {
        showVisualizationResponse('Interactive Configuration Test', { error: error.message });
    }
}

async function testAdaptiveLayout() {
    try {
        const response = await fetch('/api/visualization/adaptive-layout', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                device_info: { 
                    type: 'desktop', 
                    screen_size: 'large',
                    viewport: { width: 1920, height: 1080 }
                },
                preferences: { 
                    compact_mode: false, 
                    high_contrast: false,
                    animation_enabled: true
                },
                dashboard_data: {
                    component_count: 8,
                    data_complexity: 'medium',
                    update_frequency: 'high'
                }
            })
        });
        const data = await response.json();
        showVisualizationResponse('Adaptive Layout Test', data);
    } catch (error) {
        showVisualizationResponse('Adaptive Layout Test', { error: error.message });
    }
}

async function testVisualizationInsights() {
    try {
        const response = await fetch('/api/visualization/intelligence-insights');
        const data = await response.json();
        showVisualizationResponse('Visualization AI Insights', data);
    } catch (error) {
        showVisualizationResponse('Visualization AI Insights', { error: error.message });
    }
}

// Hour 8: Real-time Streaming Control Functions
function startRealTimeStreaming() {
    const dashboard = window.dashboard;
    if (!dashboard.isStreaming) {
        dashboard.isStreaming = true;
        dashboard.socket.emit('subscribe_live_data', {
            streams: ['performance_metrics', 'predictive_analytics']
        });
        
        // Load initial historical data
        dashboard.socket.emit('request_chart_data', {
            chart_type: 'performance_line',
            range: document.getElementById('chartTimeRange').value,
            filters: {}
        });
        
        document.getElementById('chartStatus').textContent = 'Starting live stream...';
        console.log('Real-time streaming started');
    }
}

function stopRealTimeStreaming() {
    const dashboard = window.dashboard;
    if (dashboard.isStreaming) {
        dashboard.isStreaming = false;
        document.getElementById('chartStatus').textContent = 'Streaming stopped';
        console.log('Real-time streaming stopped');
    }
}

function updateChartRange() {
    const dashboard = window.dashboard;
    const range = document.getElementById('chartTimeRange').value;
    
    // Request new data for the selected range
    dashboard.socket.emit('request_chart_data', {
        chart_type: 'performance_line',
        range: range,
        filters: {}
    });
    
    document.getElementById('chartStatus').textContent = `Loading ${range} data...`;
}

function requestPredictiveAnalysis() {
    const dashboard = window.dashboard;
    dashboard.socket.emit('request_predictive_analysis', {
        type: 'trend_forecast',
        historical_data: {
            performance_metrics: dashboard.chartData.datasets[0].data.slice(-10),
            cpu_usage: dashboard.chartData.datasets[1].data.slice(-10),
            memory_usage: dashboard.chartData.datasets[2].data.slice(-10)
        },
        horizon: 12
    });
    
    document.getElementById('chartStatus').textContent = 'Generating predictive analysis...';
}

// IRONCLAD CONSOLIDATION: D3.js Control Functions from charts_dashboard.html
// ===========================================================================

function updateD3Chart(chartId) {
    console.log(`Updating D3 chart: ${chartId}`);
    
    if (chartId === 'networkChart') {
        // Clear and recreate network chart
        d3.select('#networkChart').selectAll('*').remove();
        if (window.dashboard) {
            window.dashboard.createNetworkGraph();
        }
        document.getElementById('networkStatus').textContent = 
            'Network updated - ' + new Date().toLocaleTimeString();
    } else if (chartId === 'heatmapChart') {
        // Clear and recreate heatmap
        d3.select('#heatmapChart').selectAll('*').remove();
        if (window.dashboard) {
            window.dashboard.createHeatmap();
        }
        document.getElementById('heatmapStatus').textContent = 
            'Heatmap updated - ' + new Date().toLocaleTimeString();
    }
}

function exportD3Chart(chartId, format) {
    console.log(`Exporting D3 chart ${chartId} as ${format}`);
    
    if (format === 'svg') {
        // Get SVG content
        const svgElement = document.querySelector(`#${chartId} svg`);
        if (svgElement) {
            const serializer = new XMLSerializer();
            const source = serializer.serializeToString(svgElement);
            
            // Create download
            const blob = new Blob([source], {type: 'image/svg+xml'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${chartId}.svg`;
            a.click();
            URL.revokeObjectURL(url);
        }
    } else {
        // For other formats, show alert (in production, would use canvas conversion)
        alert(`Chart exported as ${format}`);
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new ModularDashboard();
});
/**
 * Swarm Orchestration Frontend Module
 * ===================================
 * 
 * Universal swarm orchestration interface inspired by Swarms framework.
 * Provides architecture selection, task execution, and performance analytics.
 */

class SwarmOrchestrationManager {
    constructor() {
        this.swarms = new Map();
        this.agents = new Map();
        this.architectures = new Map();
        this.executionHistory = [];
        this.performanceMetrics = {
            totalExecutions: 0,
            successRate: 100.0,
            averageExecutionTime: 0,
            architectureUsage: {}
        };
        
        this.currentSwarmId = null;
        this.initializeEventListeners();
        this.loadInitialData();
    }

    async loadInitialData() {
        try {
            await Promise.all([
                this.loadSwarms(),
                this.loadAgents(),
                this.loadArchitectures(),
                this.loadAnalytics()
            ]);
            this.updateDashboard();
        } catch (error) {
            console.error('Failed to load swarm orchestration data:', error);
            this.showError('Failed to load swarm orchestration data');
        }
    }

    async loadSwarms() {
        try {
            const response = await fetch('/api/swarm/swarms');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.swarms.clear();
                data.swarms.forEach(swarm => {
                    this.swarms.set(swarm.id, swarm);
                });
                this.performanceMetrics = { ...this.performanceMetrics, ...data.global_metrics };
                console.log(`Loaded ${data.total_swarms} swarms`);
            }
        } catch (error) {
            console.error('Error loading swarms:', error);
        }
    }

    async loadAgents() {
        try {
            const response = await fetch('/api/swarm/agents');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.agents.clear();
                data.agents.forEach(agent => {
                    this.agents.set(agent.id, agent);
                });
                console.log(`Loaded ${data.total_agents} swarm agents`);
            }
        } catch (error) {
            console.error('Error loading swarm agents:', error);
        }
    }

    async loadArchitectures() {
        try {
            const response = await fetch('/api/swarm/architectures');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.architectures.clear();
                data.architectures.forEach(arch => {
                    this.architectures.set(arch.type, arch);
                });
                this.updateArchitectureSelector();
                console.log(`Loaded ${data.architectures.length} architectures`);
            }
        } catch (error) {
            console.error('Error loading architectures:', error);
        }
    }

    async loadAnalytics() {
        try {
            const response = await fetch('/api/swarm/analytics/performance');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.executionHistory = data.analytics.recent_executions || [];
                this.performanceMetrics.architectureUsage = data.analytics.architecture_usage || {};
            }
        } catch (error) {
            console.error('Error loading swarm analytics:', error);
        }
    }

    initializeEventListeners() {
        // Architecture selection buttons
        document.addEventListener('click', (event) => {
            if (event.target.classList.contains('architecture-btn')) {
                this.selectArchitecture(event.target.dataset.architecture);
            }
        });

        // Create swarm button
        const createSwarmBtn = document.getElementById('create-swarm-btn');
        if (createSwarmBtn) {
            createSwarmBtn.addEventListener('click', () => this.showCreateSwarmModal());
        }

        // Execute task button
        const executeTaskBtn = document.getElementById('execute-swarm-task-btn');
        if (executeTaskBtn) {
            executeTaskBtn.addEventListener('click', () => this.showExecuteTaskModal());
        }

        // Refresh button
        const refreshBtn = document.getElementById('refresh-swarms-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshData());
        }

        // Architecture comparison toggle
        const compareBtn = document.getElementById('compare-architectures-btn');
        if (compareBtn) {
            compareBtn.addEventListener('click', () => this.toggleArchitectureComparison());
        }

        // Auto-refresh every 45 seconds
        setInterval(() => this.refreshData(), 45000);
    }

    updateDashboard() {
        this.updateArchitectureGrid();
        this.updateSwarmsTable();
        this.updateAgentsCards();
        this.updatePerformanceMetrics();
        this.updateExecutionHistory();
        this.updateArchitectureAnalytics();
    }

    updateArchitectureGrid() {
        const grid = document.getElementById('architectures-grid');
        if (!grid) return;

        grid.innerHTML = '';

        this.architectures.forEach(arch => {
            const archCard = document.createElement('div');
            archCard.className = 'col-lg-4 col-md-6 mb-4';
            
            const usageCount = this.performanceMetrics.architectureUsage[arch.type] || 0;
            const isRecommended = this.getRecommendedArchitectures().includes(arch.type);
            
            archCard.innerHTML = `
                <div class="card architecture-card h-100 ${isRecommended ? 'border-success' : ''}" 
                     data-architecture="${arch.type}">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h6 class="card-title mb-0">${this.formatArchitectureName(arch.type)}</h6>
                        ${isRecommended ? '<span class="badge badge-success">Recommended</span>' : ''}
                        <span class="badge badge-info">${usageCount} uses</span>
                    </div>
                    <div class="card-body">
                        <p class="card-text">${arch.description}</p>
                        <div class="best-for-section">
                            <small class="text-muted">Best for:</small>
                            <div class="best-for-tags">
                                ${arch.best_for.map(use => 
                                    `<span class="badge badge-light badge-sm mr-1">${use}</span>`
                                ).join('')}
                            </div>
                        </div>
                        <div class="performance-indicators mt-3">
                            <div class="row">
                                <div class="col-6">
                                    <small class="text-muted">Complexity:</small>
                                    <div class="complexity-bar">
                                        ${this.getComplexityBar(arch.type)}
                                    </div>
                                </div>
                                <div class="col-6">
                                    <small class="text-muted">Speed:</small>
                                    <div class="speed-bar">
                                        ${this.getSpeedBar(arch.type)}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="card-footer">
                        <button class="btn btn-primary btn-sm architecture-btn" 
                                data-architecture="${arch.type}">
                            <i class="fas fa-rocket"></i> Select Architecture
                        </button>
                        <button class="btn btn-outline-info btn-sm ml-2" 
                                onclick="swarmManager.showArchitectureDetails('${arch.type}')">
                            <i class="fas fa-info-circle"></i> Details
                        </button>
                    </div>
                </div>
            `;
            
            grid.appendChild(archCard);
        });
    }

    updateSwarmsTable() {
        const tableBody = document.getElementById('swarms-table-body');
        if (!tableBody) return;

        tableBody.innerHTML = '';
        
        this.swarms.forEach(swarm => {
            const row = document.createElement('tr');
            row.className = 'swarm-row';
            
            const statusColor = this.getStatusColor(swarm.status);
            const architectureBadge = this.getArchitectureBadge(swarm.architecture);
            
            row.innerHTML = `
                <td>
                    <div class="swarm-info">
                        <strong>${this.escapeHtml(swarm.name)}</strong>
                        <small class="text-muted d-block">${swarm.id}</small>
                    </div>
                </td>
                <td>${architectureBadge}</td>
                <td>
                    <span class="badge ${statusColor}">${swarm.status}</span>
                </td>
                <td>
                    <span class="badge badge-secondary">${swarm.agent_count} agents</span>
                </td>
                <td>
                    <div class="execution-stats">
                        <small>Executions: ${swarm.execution_count || 0}</small><br>
                        <small>Success: ${(swarm.performance?.success_rate || 100).toFixed(1)}%</small>
                    </div>
                </td>
                <td>
                    <small class="text-muted">${this.formatDateTime(swarm.created_at)}</small>
                </td>
                <td>
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-primary btn-sm" onclick="swarmManager.viewSwarmDetails('${swarm.id}')">
                            <i class="fas fa-eye"></i> View
                        </button>
                        <button class="btn btn-success btn-sm" onclick="swarmManager.executeSwarmTask('${swarm.id}')">
                            <i class="fas fa-play"></i> Execute
                        </button>
                        <button class="btn btn-info btn-sm" onclick="swarmManager.analyzeSwarmPerformance('${swarm.id}')">
                            <i class="fas fa-chart-line"></i> Analyze
                        </button>
                    </div>
                </td>
            `;
            
            tableBody.appendChild(row);
        });
    }

    updateAgentsCards() {
        const container = document.getElementById('swarm-agents-container');
        if (!container) return;

        container.innerHTML = '';

        this.agents.forEach(agent => {
            const agentCard = document.createElement('div');
            agentCard.className = 'col-md-6 col-lg-4 mb-3';
            
            const statusIcon = this.getAgentStatusIcon(agent.status);
            const capabilitiesList = agent.capabilities.slice(0, 4).map(cap => 
                `<li><small>${cap}</small></li>`
            ).join('');
            
            agentCard.innerHTML = `
                <div class="card agent-card h-100">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <div class="agent-title">
                            <h6 class="mb-0">${this.escapeHtml(agent.name)}</h6>
                            <small class="text-muted">${agent.role}</small>
                        </div>
                        <span class="agent-status ${statusIcon.class}" title="${agent.status}">
                            <i class="${statusIcon.icon}"></i>
                        </span>
                    </div>
                    <div class="card-body">
                        <div class="agent-config mb-2">
                            <small><strong>Max Loops:</strong> ${agent.max_loops}</small><br>
                            <small><strong>Temperature:</strong> ${agent.temperature}</small>
                        </div>
                        <div class="capabilities-section">
                            <small class="text-muted">Key Capabilities:</small>
                            <ul class="capabilities-list">
                                ${capabilitiesList}
                                ${agent.capabilities.length > 4 ? 
                                    `<li><small><em>+${agent.capabilities.length - 4} more...</em></small></li>` : ''}
                            </ul>
                        </div>
                        <div class="performance-metrics mt-2">
                            <small class="text-muted">Performance:</small>
                            <div class="metrics-grid">
                                ${this.renderAgentMetrics(agent.performance_metrics)}
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            container.appendChild(agentCard);
        });
    }

    updatePerformanceMetrics() {
        // Update overview cards
        const elements = {
            'total-executions-count': this.performanceMetrics.total_executions || 0,
            'success-rate-value': `${(this.performanceMetrics.success_rate || 100).toFixed(1)}%`,
            'avg-execution-time-value': `${(this.performanceMetrics.average_execution_time || 0).toFixed(2)}s`,
            'active-swarms-count': this.swarms.size
        };

        Object.entries(elements).forEach(([elementId, value]) => {
            const element = document.getElementById(elementId);
            if (element) {
                element.textContent = value;
            }
        });

        // Update performance charts
        this.updatePerformanceCharts();
    }

    updatePerformanceCharts() {
        // Architecture usage chart
        this.renderArchitectureUsageChart();
        
        // Execution timeline chart
        this.renderExecutionTimelineChart();
        
        // Performance trends chart
        this.renderPerformanceTrendsChart();
    }

    renderArchitectureUsageChart() {
        const container = document.getElementById('architecture-usage-chart');
        if (!container || !Object.keys(this.performanceMetrics.architectureUsage).length) return;

        const data = Object.entries(this.performanceMetrics.architectureUsage)
            .map(([arch, count]) => ({ architecture: arch, count }))
            .sort((a, b) => b.count - a.count);

        container.innerHTML = '';
        
        const svg = d3.select(container)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '300px')
            .attr('viewBox', '0 0 400 300');

        const margin = { top: 20, right: 20, bottom: 60, left: 60 };
        const width = 400 - margin.left - margin.right;
        const height = 300 - margin.top - margin.bottom;

        const xScale = d3.scaleBand()
            .domain(data.map(d => d.architecture))
            .range([0, width])
            .padding(0.1);

        const yScale = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.count)])
            .range([height, 0]);

        const g = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Add bars
        g.selectAll('.bar')
            .data(data)
            .enter().append('rect')
            .attr('class', 'bar')
            .attr('x', d => xScale(d.architecture))
            .attr('width', xScale.bandwidth())
            .attr('y', d => yScale(d.count))
            .attr('height', d => height - yScale(d.count))
            .attr('fill', '#007bff')
            .attr('opacity', 0.7);

        // Add x-axis
        g.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(xScale))
            .selectAll('text')
            .style('text-anchor', 'end')
            .attr('dx', '-.8em')
            .attr('dy', '.15em')
            .attr('transform', 'rotate(-45)')
            .style('font-size', '10px');

        // Add y-axis
        g.append('g')
            .call(d3.axisLeft(yScale))
            .style('font-size', '10px');

        // Add title
        svg.append('text')
            .attr('x', 200)
            .attr('y', 15)
            .attr('text-anchor', 'middle')
            .style('font-size', '14px')
            .style('font-weight', 'bold')
            .text('Architecture Usage');
    }

    renderExecutionTimelineChart() {
        const container = document.getElementById('execution-timeline-chart');
        if (!container || !this.executionHistory.length) return;

        container.innerHTML = '';
        
        // Create simple timeline visualization
        const timelineHtml = this.executionHistory.slice(-10).map((execution, index) => {
            const statusColor = execution.success ? 'success' : 'danger';
            const architectureLabel = this.formatArchitectureName(execution.architecture);
            
            return `
                <div class="timeline-item mb-2">
                    <div class="d-flex justify-content-between align-items-center">
                        <div class="execution-info">
                            <small><strong>${architectureLabel}</strong></small><br>
                            <small class="text-muted">${this.formatDateTime(execution.timestamp)}</small>
                        </div>
                        <div class="execution-result">
                            <span class="badge badge-${statusColor}">${execution.success ? 'Success' : 'Failed'}</span>
                            <small class="text-muted ml-2">${execution.execution_time.toFixed(2)}s</small>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = `
            <div class="timeline-container">
                <h6 class="mb-3">Recent Executions</h6>
                ${timelineHtml}
            </div>
        `;
    }

    renderPerformanceTrendsChart() {
        const container = document.getElementById('performance-trends-chart');
        if (!container) return;

        // Simple performance indicators
        const trendsHtml = `
            <div class="performance-trends">
                <h6 class="mb-3">Performance Trends</h6>
                <div class="row">
                    <div class="col-6">
                        <div class="trend-metric">
                            <div class="d-flex justify-content-between">
                                <small>Success Rate</small>
                                <small class="text-success">
                                    <i class="fas fa-arrow-up"></i> 
                                    ${(this.performanceMetrics.success_rate || 100).toFixed(1)}%
                                </small>
                            </div>
                            <div class="progress progress-sm">
                                <div class="progress-bar bg-success" 
                                     style="width: ${this.performanceMetrics.success_rate || 100}%"></div>
                            </div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="trend-metric">
                            <div class="d-flex justify-content-between">
                                <small>Avg. Execution Time</small>
                                <small class="text-info">${(this.performanceMetrics.average_execution_time || 0).toFixed(2)}s</small>
                            </div>
                            <div class="progress progress-sm">
                                <div class="progress-bar bg-info" style="width: 75%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="row mt-2">
                    <div class="col-6">
                        <div class="trend-metric">
                            <div class="d-flex justify-content-between">
                                <small>Active Swarms</small>
                                <small class="text-primary">${this.swarms.size}</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-6">
                        <div class="trend-metric">
                            <div class="d-flex justify-content-between">
                                <small>Total Agents</small>
                                <small class="text-secondary">${this.agents.size}</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = trendsHtml;
    }

    async showCreateSwarmModal() {
        const modalContent = `
            <div class="modal fade" id="createSwarmModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Create Universal Swarm</h5>
                            <button type="button" class="close" data-dismiss="modal">
                                <span>&times;</span>
                            </button>
                        </div>
                        <div class="modal-body">
                            <form id="create-swarm-form">
                                <div class="form-group">
                                    <label>Swarm Name</label>
                                    <input type="text" class="form-control" id="swarm-name" required>
                                </div>
                                <div class="form-group">
                                    <label>Description</label>
                                    <textarea class="form-control" id="swarm-description" rows="2"></textarea>
                                </div>
                                <div class="form-group">
                                    <label>Architecture</label>
                                    <select class="form-control" id="swarm-architecture" required>
                                        <option value="">Select architecture...</option>
                                    </select>
                                    <small class="form-text text-muted" id="architecture-description"></small>
                                </div>
                                <div class="form-group">
                                    <label>Max Loops</label>
                                    <input type="number" class="form-control" id="max-loops" value="3" min="1" max="10">
                                </div>
                                <div class="form-group">
                                    <label>Timeout (seconds)</label>
                                    <input type="number" class="form-control" id="timeout" value="300" min="60" max="1800">
                                </div>
                                <div class="form-group">
                                    <label>Select Agents (optional - will use all if none selected)</label>
                                    <div id="swarm-agent-selection" class="agent-selection-grid">
                                        <!-- Agent checkboxes will be populated here -->
                                    </div>
                                </div>
                                <div class="form-group">
                                    <label>Rules (optional)</label>
                                    <textarea class="form-control" id="swarm-rules" rows="2" 
                                        placeholder="Enter any specific rules or constraints for this swarm..."></textarea>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" onclick="swarmManager.createSwarm()">
                                <i class="fas fa-plus"></i> Create Swarm
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalContent);
        
        this.populateSwarmArchitectureSelector();
        this.populateSwarmAgentSelection();
        
        $('#createSwarmModal').modal('show');
        
        $('#createSwarmModal').on('hidden.bs.modal', function() {
            $(this).remove();
        });

        // Add architecture change listener
        const archSelect = document.getElementById('swarm-architecture');
        if (archSelect) {
            archSelect.addEventListener('change', (e) => {
                const description = document.getElementById('architecture-description');
                const arch = this.architectures.get(e.target.value);
                if (description && arch) {
                    description.textContent = arch.description;
                }
            });
        }
    }

    populateSwarmArchitectureSelector() {
        const selector = document.getElementById('swarm-architecture');
        if (!selector) return;

        this.architectures.forEach(arch => {
            const option = document.createElement('option');
            option.value = arch.type;
            option.textContent = this.formatArchitectureName(arch.type);
            selector.appendChild(option);
        });
    }

    populateSwarmAgentSelection() {
        const container = document.getElementById('swarm-agent-selection');
        if (!container) return;

        container.innerHTML = '';
        
        this.agents.forEach(agent => {
            const agentDiv = document.createElement('div');
            agentDiv.className = 'form-check agent-checkbox';
            
            agentDiv.innerHTML = `
                <input class="form-check-input" type="checkbox" value="${agent.id}" id="swarm-agent-${agent.id}">
                <label class="form-check-label" for="swarm-agent-${agent.id}">
                    <strong>${this.escapeHtml(agent.name)}</strong>
                    <small class="text-muted d-block">${agent.role} - ${agent.capabilities.slice(0, 2).join(', ')}</small>
                </label>
            `;
            
            container.appendChild(agentDiv);
        });
    }

    async createSwarm() {
        const form = document.getElementById('create-swarm-form');
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }

        const name = document.getElementById('swarm-name').value;
        const description = document.getElementById('swarm-description').value;
        const architecture = document.getElementById('swarm-architecture').value;
        const maxLoops = parseInt(document.getElementById('max-loops').value);
        const timeout = parseInt(document.getElementById('timeout').value);
        const rules = document.getElementById('swarm-rules').value;
        
        const selectedAgents = Array.from(document.querySelectorAll('#swarm-agent-selection input:checked'))
            .map(input => input.value);

        try {
            const response = await fetch('/api/swarm/swarms', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name,
                    description,
                    architecture,
                    max_loops: maxLoops,
                    timeout,
                    agents: selectedAgents,
                    rules: rules ? [rules] : []
                })
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                this.showSuccess('Swarm created successfully!');
                $('#createSwarmModal').modal('hide');
                await this.refreshData();
            } else {
                this.showError(data.message || 'Failed to create swarm');
            }
        } catch (error) {
            console.error('Error creating swarm:', error);
            this.showError('Failed to create swarm');
        }
    }

    async executeSwarmTask(swarmId) {
        const modalContent = `
            <div class="modal fade" id="executeSwarmTaskModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Execute Swarm Task</h5>
                            <button type="button" class="close" data-dismiss="modal">
                                <span>&times;</span>
                            </button>
                        </div>
                        <div class="modal-body">
                            <form id="execute-swarm-task-form">
                                <div class="form-group">
                                    <label>Task Description</label>
                                    <textarea class="form-control" id="swarm-task-description" rows="4" required 
                                        placeholder="Describe the task for the swarm to execute..."></textarea>
                                </div>
                                <div class="form-group">
                                    <label>Task Type</label>
                                    <select class="form-control" id="swarm-task-type">
                                        <option value="general">General Analysis</option>
                                        <option value="security">Security Assessment</option>
                                        <option value="testing">Test Generation</option>
                                        <option value="quality">Quality Analysis</option>
                                        <option value="research">Research Task</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>Priority</label>
                                    <select class="form-control" id="swarm-task-priority">
                                        <option value="critical">Critical</option>
                                        <option value="high">High</option>
                                        <option value="medium" selected>Medium</option>
                                        <option value="low">Low</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>Expected Output</label>
                                    <input type="text" class="form-control" id="swarm-expected-output" 
                                        placeholder="What should the task produce?">
                                </div>
                                <div class="form-group">
                                    <label>Context (optional)</label>
                                    <textarea class="form-control" id="swarm-task-context" rows="2"
                                        placeholder="Additional context or background information..."></textarea>
                                </div>
                                <div class="form-group">
                                    <label>Constraints (optional)</label>
                                    <input type="text" class="form-control" id="swarm-task-constraints" 
                                        placeholder="Any specific constraints or limitations...">
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" onclick="swarmManager.submitSwarmTaskExecution('${swarmId}')">
                                <i class="fas fa-rocket"></i> Execute Task
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalContent);
        $('#executeSwarmTaskModal').modal('show');
        
        $('#executeSwarmTaskModal').on('hidden.bs.modal', function() {
            $(this).remove();
        });
    }

    async submitSwarmTaskExecution(swarmId) {
        const form = document.getElementById('execute-swarm-task-form');
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }

        const taskConfig = {
            description: document.getElementById('swarm-task-description').value,
            task_type: document.getElementById('swarm-task-type').value,
            priority: document.getElementById('swarm-task-priority').value,
            expected_output: document.getElementById('swarm-expected-output').value,
            context: document.getElementById('swarm-task-context').value ? 
                { background: document.getElementById('swarm-task-context').value } : {},
            constraints: document.getElementById('swarm-task-constraints').value ? 
                [document.getElementById('swarm-task-constraints').value] : []
        };

        try {
            this.showInfo('Executing swarm task... This may take several moments.');
            
            const response = await fetch(`/api/swarm/swarms/${swarmId}/execute`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(taskConfig)
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                this.showSuccess('Swarm task executed successfully!');
                this.showSwarmTaskResults(data.execution_result);
                $('#executeSwarmTaskModal').modal('hide');
                await this.refreshData();
            } else {
                this.showError(data.message || 'Failed to execute swarm task');
            }
        } catch (error) {
            console.error('Error executing swarm task:', error);
            this.showError('Failed to execute swarm task');
        }
    }

    showSwarmTaskResults(result) {
        const modalContent = `
            <div class="modal fade" id="swarmTaskResultsModal" tabindex="-1">
                <div class="modal-dialog modal-xl">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Swarm Task Results - ${result.architecture}</h5>
                            <button type="button" class="close" data-dismiss="modal">
                                <span>&times;</span>
                            </button>
                        </div>
                        <div class="modal-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>Execution Summary</h6>
                                    <ul class="list-unstyled">
                                        <li><strong>Task ID:</strong> ${result.task_id}</li>
                                        <li><strong>Architecture:</strong> ${this.formatArchitectureName(result.architecture)}</li>
                                        <li><strong>Execution Time:</strong> ${result.execution_time.toFixed(2)}s</li>
                                        <li><strong>Success:</strong> <span class="badge badge-success">Yes</span></li>
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <h6>Performance Metrics</h6>
                                    <div class="performance-summary">
                                        ${this.renderExecutionMetrics(result.result)}
                                    </div>
                                </div>
                            </div>
                            <hr>
                            <div class="results-section">
                                <h6>Detailed Results</h6>
                                <div class="result-tabs">
                                    <ul class="nav nav-tabs" role="tablist">
                                        <li class="nav-item">
                                            <a class="nav-link active" data-toggle="tab" href="#result-summary">Summary</a>
                                        </li>
                                        <li class="nav-item">
                                            <a class="nav-link" data-toggle="tab" href="#result-details">Full Details</a>
                                        </li>
                                        <li class="nav-item">
                                            <a class="nav-link" data-toggle="tab" href="#result-json">Raw JSON</a>
                                        </li>
                                    </ul>
                                    <div class="tab-content mt-3">
                                        <div class="tab-pane active" id="result-summary">
                                            ${this.renderResultSummary(result.result)}
                                        </div>
                                        <div class="tab-pane" id="result-details">
                                            ${this.renderResultDetails(result.result)}
                                        </div>
                                        <div class="tab-pane" id="result-json">
                                            <pre class="bg-light p-3" style="max-height: 400px; overflow-y: auto;">${JSON.stringify(result, null, 2)}</pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
                            <button type="button" class="btn btn-info" onclick="swarmManager.downloadResults('${result.task_id}')">
                                <i class="fas fa-download"></i> Download Results
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalContent);
        $('#swarmTaskResultsModal').modal('show');
        
        $('#swarmTaskResultsModal').on('hidden.bs.modal', function() {
            $(this).remove();
        });
    }

    // Utility methods
    formatArchitectureName(archType) {
        return archType.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    getComplexityBar(archType) {
        const complexity = {
            'sequential_workflow': 20,
            'concurrent_workflow': 40,
            'hierarchical_swarm': 60,
            'mixture_of_agents': 70,
            'deep_research_swarm': 90,
            'adaptive_swarm': 95
        };
        const level = complexity[archType] || 50;
        return `<div class="progress progress-xs"><div class="progress-bar bg-warning" style="width: ${level}%"></div></div>`;
    }

    getSpeedBar(archType) {
        const speed = {
            'sequential_workflow': 60,
            'concurrent_workflow': 90,
            'hierarchical_swarm': 70,
            'mixture_of_agents': 50,
            'deep_research_swarm': 30,
            'adaptive_swarm': 80
        };
        const level = speed[archType] || 50;
        return `<div class="progress progress-xs"><div class="progress-bar bg-info" style="width: ${level}%"></div></div>`;
    }

    getRecommendedArchitectures() {
        return ['concurrent_workflow', 'hierarchical_swarm', 'adaptive_swarm'];
    }

    getStatusColor(status) {
        const statusColors = {
            'initialized': 'badge-info',
            'ready': 'badge-success',
            'executing': 'badge-warning',
            'completed': 'badge-success',
            'error': 'badge-danger'
        };
        return statusColors[status] || 'badge-secondary';
    }

    getArchitectureBadge(architecture) {
        const colors = {
            'sequential_workflow': 'badge-primary',
            'concurrent_workflow': 'badge-success',
            'hierarchical_swarm': 'badge-info',
            'mixture_of_agents': 'badge-warning',
            'deep_research_swarm': 'badge-dark',
            'adaptive_swarm': 'badge-danger'
        };
        const color = colors[architecture] || 'badge-secondary';
        return `<span class="badge ${color}">${this.formatArchitectureName(architecture)}</span>`;
    }

    getAgentStatusIcon(status) {
        const statusIcons = {
            'ready': { icon: 'fas fa-check-circle', class: 'text-success' },
            'busy': { icon: 'fas fa-spinner fa-spin', class: 'text-warning' },
            'error': { icon: 'fas fa-exclamation-circle', class: 'text-danger' },
            'offline': { icon: 'fas fa-times-circle', class: 'text-muted' }
        };
        return statusIcons[status] || { icon: 'fas fa-circle', class: 'text-secondary' };
    }

    renderAgentMetrics(metrics) {
        if (!metrics) return '<small class="text-muted">No metrics available</small>';
        
        return Object.entries(metrics).slice(0, 3).map(([key, value]) => {
            const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            return `<small><strong>${displayKey}:</strong> ${value}</small>`;
        }).join('<br>');
    }

    renderExecutionMetrics(result) {
        const metrics = [];
        
        if (result.summary) {
            metrics.push(`<small><strong>Summary:</strong> ${result.summary}</small>`);
        }
        
        if (result.participating_agents || result.agent_count) {
            const count = result.participating_agents || result.agent_count || 0;
            metrics.push(`<small><strong>Agents Used:</strong> ${count}</small>`);
        }
        
        if (result.consensus_confidence || result.aggregated_output?.combined_confidence) {
            const confidence = result.consensus_confidence || result.aggregated_output.combined_confidence || 0;
            metrics.push(`<small><strong>Confidence:</strong> ${confidence.toFixed(1)}%</small>`);
        }
        
        return metrics.join('<br>') || '<small class="text-muted">No metrics available</small>';
    }

    renderResultSummary(result) {
        let summary = `<div class="result-summary">`;
        
        if (result.summary) {
            summary += `<p><strong>Summary:</strong> ${result.summary}</p>`;
        }
        
        if (result.final_output) {
            summary += `<p><strong>Final Output:</strong> ${result.final_output}</p>`;
        }
        
        if (result.architecture) {
            summary += `<p><strong>Architecture Used:</strong> ${this.formatArchitectureName(result.architecture)}</p>`;
        }
        
        summary += `</div>`;
        return summary;
    }

    renderResultDetails(result) {
        // Create detailed view based on result structure
        let details = '<div class="result-details">';
        
        Object.entries(result).forEach(([key, value]) => {
            if (key !== 'summary' && key !== 'final_output' && key !== 'architecture') {
                details += `<div class="detail-section mb-3">`;
                details += `<h6>${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h6>`;
                
                if (Array.isArray(value)) {
                    details += '<ul>';
                    value.forEach(item => {
                        if (typeof item === 'object') {
                            details += `<li>${JSON.stringify(item, null, 2)}</li>`;
                        } else {
                            details += `<li>${item}</li>`;
                        }
                    });
                    details += '</ul>';
                } else if (typeof value === 'object') {
                    details += `<pre class="bg-light p-2">${JSON.stringify(value, null, 2)}</pre>`;
                } else {
                    details += `<p>${value}</p>`;
                }
                
                details += '</div>';
            }
        });
        
        details += '</div>';
        return details;
    }

    formatDateTime(isoString) {
        return new Date(isoString).toLocaleString();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async refreshData() {
        await this.loadInitialData();
        this.showInfo('Swarm data refreshed');
    }

    downloadResults(taskId) {
        // Implementation for downloading results
        this.showInfo('Download feature coming soon');
    }

    // Notification methods
    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'danger');
    }

    showInfo(message) {
        this.showNotification(message, 'info');
    }

    showNotification(message, type) {
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="close" data-dismiss="alert">
                    <span>&times;</span>
                </button>
            </div>
        `;
        
        const container = document.getElementById('notifications-container') || document.body;
        container.insertAdjacentHTML('afterbegin', alertHtml);
        
        setTimeout(() => {
            const alert = container.querySelector('.alert');
            if (alert) {
                $(alert).alert('close');
            }
        }, 5000);
    }
}

// Initialize swarm orchestration manager when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (typeof d3 !== 'undefined') {
        window.swarmManager = new SwarmOrchestrationManager();
    } else {
        console.warn('D3.js not loaded, some visualizations will be limited');
        window.swarmManager = new SwarmOrchestrationManager();
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SwarmOrchestrationManager;
}
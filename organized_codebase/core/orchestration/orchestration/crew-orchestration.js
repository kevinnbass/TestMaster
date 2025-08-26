/**
 * Crew Orchestration Frontend Module
 * ===================================
 * 
 * Enhanced multi-agent crew management interface inspired by CrewAI patterns.
 * Provides visual crew creation, task execution, and performance monitoring.
 */

class CrewOrchestrationManager {
    constructor() {
        this.crews = new Map();
        this.agents = new Map();
        this.activeExecutions = new Map();
        this.performanceMetrics = {
            totalCrews: 0,
            totalTasks: 0,
            successRate: 100.0,
            averageExecutionTime: 0
        };
        
        this.initializeEventListeners();
        this.loadInitialData();
    }

    async loadInitialData() {
        try {
            await Promise.all([
                this.loadCrews(),
                this.loadAgents(),
                this.loadSwarmTypes()
            ]);
            this.updateDashboard();
        } catch (error) {
            console.error('Failed to load initial crew data:', error);
            this.showError('Failed to load crew orchestration data');
        }
    }

    async loadCrews() {
        try {
            const response = await fetch('/api/crew/crews');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.crews.clear();
                data.crews.forEach(crew => {
                    this.crews.set(crew.id, crew);
                });
                this.performanceMetrics.totalCrews = data.total_crews;
                console.log(`Loaded ${data.total_crews} crews`);
            }
        } catch (error) {
            console.error('Error loading crews:', error);
        }
    }

    async loadAgents() {
        try {
            const response = await fetch('/api/crew/agents');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.agents.clear();
                data.agents.forEach(agent => {
                    this.agents.set(agent.id, agent);
                });
                console.log(`Loaded ${data.total_agents} agents`);
            }
        } catch (error) {
            console.error('Error loading agents:', error);
        }
    }

    async loadSwarmTypes() {
        try {
            const response = await fetch('/api/crew/swarm-types');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.updateSwarmTypeSelector(data.swarm_types);
            }
        } catch (error) {
            console.error('Error loading swarm types:', error);
        }
    }

    initializeEventListeners() {
        // Crew creation form
        const createCrewBtn = document.getElementById('create-crew-btn');
        if (createCrewBtn) {
            createCrewBtn.addEventListener('click', () => this.showCreateCrewModal());
        }

        // Task execution button
        const executeTaskBtn = document.getElementById('execute-task-btn');
        if (executeTaskBtn) {
            executeTaskBtn.addEventListener('click', () => this.showExecuteTaskModal());
        }

        // Refresh button
        const refreshBtn = document.getElementById('refresh-crews-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshData());
        }

        // Auto-refresh every 30 seconds
        setInterval(() => this.refreshData(), 30000);
    }

    updateDashboard() {
        this.updateCrewsTable();
        this.updateAgentsGrid();
        this.updatePerformanceMetrics();
        this.updateCrewVisualization();
    }

    updateCrewsTable() {
        const tableBody = document.getElementById('crews-table-body');
        if (!tableBody) return;

        tableBody.innerHTML = '';
        
        this.crews.forEach(crew => {
            const row = document.createElement('tr');
            row.className = 'crew-row';
            
            const statusColor = this.getStatusColor(crew.status);
            const swarmTypeBadge = this.getSwarmTypeBadge(crew.swarm_type);
            
            row.innerHTML = `
                <td>
                    <div class="crew-info">
                        <strong>${this.escapeHtml(crew.name)}</strong>
                        <small class="text-muted d-block">${crew.id}</small>
                    </div>
                </td>
                <td>
                    <span class="badge ${statusColor}">${crew.status}</span>
                </td>
                <td>${swarmTypeBadge}</td>
                <td>
                    <span class="badge badge-secondary">${crew.agent_count} agents</span>
                </td>
                <td>
                    <div class="performance-mini">
                        <small>Tasks: ${crew.performance?.tasks_completed || 0}</small><br>
                        <small>Success: ${(crew.performance?.success_rate || 100).toFixed(1)}%</small>
                    </div>
                </td>
                <td>
                    <small class="text-muted">${this.formatDateTime(crew.created_at)}</small>
                </td>
                <td>
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-primary btn-sm" onclick="crewManager.viewCrewDetails('${crew.id}')">
                            <i class="fas fa-eye"></i> View
                        </button>
                        <button class="btn btn-success btn-sm" onclick="crewManager.executeCrewTask('${crew.id}')">
                            <i class="fas fa-play"></i> Execute
                        </button>
                    </div>
                </td>
            `;
            
            tableBody.appendChild(row);
        });
    }

    updateAgentsGrid() {
        const agentsGrid = document.getElementById('agents-grid');
        if (!agentsGrid) return;

        agentsGrid.innerHTML = '';

        this.agents.forEach(agent => {
            const agentCard = document.createElement('div');
            agentCard.className = 'col-md-4 mb-3';
            
            const statusIcon = this.getAgentStatusIcon(agent.status);
            const capabilitiesBadges = agent.capabilities.slice(0, 3).map(cap => 
                `<span class="badge badge-info badge-sm mr-1">${cap}</span>`
            ).join('');
            
            agentCard.innerHTML = `
                <div class="card agent-card h-100">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h6 class="card-title mb-0">${this.escapeHtml(agent.role)}</h6>
                        <span class="agent-status ${statusIcon.class}" title="${agent.status}">
                            <i class="${statusIcon.icon}"></i>
                        </span>
                    </div>
                    <div class="card-body">
                        <p class="card-text small">${this.escapeHtml(agent.goal)}</p>
                        <div class="capabilities-section">
                            <small class="text-muted">Capabilities:</small><br>
                            ${capabilitiesBadges}
                            ${agent.capabilities.length > 3 ? 
                                `<span class="badge badge-light">+${agent.capabilities.length - 3} more</span>` : ''}
                        </div>
                        <div class="performance-section mt-2">
                            <small class="text-muted">Performance:</small>
                            <div class="performance-bars">
                                <div class="metric-bar">
                                    <small>Quality: ${(agent.performance_metrics?.quality_score || 90).toFixed(1)}%</small>
                                    <div class="progress progress-sm">
                                        <div class="progress-bar bg-success" style="width: ${agent.performance_metrics?.quality_score || 90}%"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            agentsGrid.appendChild(agentCard);
        });
    }

    updatePerformanceMetrics() {
        // Update overview cards
        const elements = {
            'total-crews-count': this.performanceMetrics.totalCrews,
            'total-tasks-count': this.performanceMetrics.totalTasks,
            'success-rate-percentage': `${this.performanceMetrics.successRate.toFixed(1)}%`,
            'avg-execution-time': `${this.performanceMetrics.averageExecutionTime.toFixed(2)}s`
        };

        Object.entries(elements).forEach(([elementId, value]) => {
            const element = document.getElementById(elementId);
            if (element) {
                element.textContent = value;
            }
        });
    }

    updateCrewVisualization() {
        const container = document.getElementById('crew-visualization');
        if (!container) return;

        // Create a network visualization of crews and agents
        const svg = d3.select(container).select('svg');
        if (svg.empty()) {
            const svgElement = d3.select(container)
                .append('svg')
                .attr('width', '100%')
                .attr('height', '400px')
                .attr('viewBox', '0 0 800 400');
            
            this.renderCrewNetwork(svgElement);
        }
    }

    renderCrewNetwork(svg) {
        // Clear existing content
        svg.selectAll('*').remove();

        const width = 800;
        const height = 400;

        // Create force simulation
        const simulation = d3.forceSimulation()
            .force('link', d3.forceLink().id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2));

        // Prepare data
        const nodes = [];
        const links = [];

        // Add crew nodes
        this.crews.forEach(crew => {
            nodes.push({
                id: crew.id,
                type: 'crew',
                name: crew.name,
                swarmType: crew.swarm_type,
                agentCount: crew.agent_count,
                status: crew.status
            });
        });

        // Add agent nodes and links
        this.agents.forEach(agent => {
            nodes.push({
                id: agent.id,
                type: 'agent',
                name: agent.role,
                status: agent.status,
                capabilities: agent.capabilities.length
            });

            // Link agents to crews (simplified - in reality this would be based on crew membership)
            if (this.crews.size > 0) {
                const randomCrew = Array.from(this.crews.keys())[0]; // Link to first crew for demo
                links.push({
                    source: agent.id,
                    target: randomCrew
                });
            }
        });

        // Create links
        const link = svg.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', 2);

        // Create nodes
        const node = svg.append('g')
            .attr('class', 'nodes')
            .selectAll('g')
            .data(nodes)
            .enter().append('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));

        // Add circles for nodes
        node.append('circle')
            .attr('r', d => d.type === 'crew' ? 20 : 12)
            .attr('fill', d => d.type === 'crew' ? '#007bff' : '#28a745')
            .attr('stroke', '#fff')
            .attr('stroke-width', 2);

        // Add labels
        node.append('text')
            .attr('dx', 25)
            .attr('dy', '.35em')
            .style('font-size', '12px')
            .text(d => d.name.length > 15 ? d.name.substring(0, 15) + '...' : d.name);

        // Add tooltips
        node.append('title')
            .text(d => d.type === 'crew' ? 
                `Crew: ${d.name}\nType: ${d.swarmType}\nAgents: ${d.agentCount}` :
                `Agent: ${d.name}\nStatus: ${d.status}\nCapabilities: ${d.capabilities}`);

        // Update simulation
        simulation.nodes(nodes).on('tick', ticked);
        simulation.force('link').links(links);

        function ticked() {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node
                .attr('transform', d => `translate(${d.x},${d.y})`);
        }

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    }

    async showCreateCrewModal() {
        // Create modal content
        const modalContent = `
            <div class="modal fade" id="createCrewModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Create New Crew</h5>
                            <button type="button" class="close" data-dismiss="modal">
                                <span>&times;</span>
                            </button>
                        </div>
                        <div class="modal-body">
                            <form id="create-crew-form">
                                <div class="form-group">
                                    <label>Crew Name</label>
                                    <input type="text" class="form-control" id="crew-name" required>
                                </div>
                                <div class="form-group">
                                    <label>Description</label>
                                    <textarea class="form-control" id="crew-description" rows="2"></textarea>
                                </div>
                                <div class="form-group">
                                    <label>Swarm Type</label>
                                    <select class="form-control" id="swarm-type" required>
                                        <option value="">Select swarm type...</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>Process Type</label>
                                    <select class="form-control" id="process-type">
                                        <option value="sequential">Sequential</option>
                                        <option value="hierarchical">Hierarchical</option>
                                        <option value="consensus">Consensus</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>Select Agents</label>
                                    <div id="agent-selection" class="agent-selection-grid">
                                        <!-- Agent checkboxes will be populated here -->
                                    </div>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" onclick="crewManager.createCrew()">Create Crew</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Add modal to DOM
        document.body.insertAdjacentHTML('beforeend', modalContent);
        
        // Populate agent selection
        this.populateAgentSelection();
        
        // Show modal
        $('#createCrewModal').modal('show');
        
        // Cleanup on hide
        $('#createCrewModal').on('hidden.bs.modal', function() {
            $(this).remove();
        });
    }

    populateAgentSelection() {
        const container = document.getElementById('agent-selection');
        if (!container) return;

        container.innerHTML = '';
        
        this.agents.forEach(agent => {
            const agentDiv = document.createElement('div');
            agentDiv.className = 'form-check agent-checkbox';
            
            agentDiv.innerHTML = `
                <input class="form-check-input" type="checkbox" value="${agent.id}" id="agent-${agent.id}">
                <label class="form-check-label" for="agent-${agent.id}">
                    <strong>${this.escapeHtml(agent.role)}</strong>
                    <small class="text-muted d-block">${agent.capabilities.slice(0, 2).join(', ')}</small>
                </label>
            `;
            
            container.appendChild(agentDiv);
        });
    }

    async createCrew() {
        const form = document.getElementById('create-crew-form');
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }

        const name = document.getElementById('crew-name').value;
        const description = document.getElementById('crew-description').value;
        const swarmType = document.getElementById('swarm-type').value;
        const processType = document.getElementById('process-type').value;
        
        const selectedAgents = Array.from(document.querySelectorAll('#agent-selection input:checked'))
            .map(input => input.value);

        if (selectedAgents.length === 0) {
            this.showError('Please select at least one agent for the crew');
            return;
        }

        try {
            const response = await fetch('/api/crew/crews', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name,
                    description,
                    swarm_type: swarmType,
                    process_type: processType,
                    agents: selectedAgents
                })
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                this.showSuccess('Crew created successfully!');
                $('#createCrewModal').modal('hide');
                await this.refreshData();
            } else {
                this.showError(data.message || 'Failed to create crew');
            }
        } catch (error) {
            console.error('Error creating crew:', error);
            this.showError('Failed to create crew');
        }
    }

    async executeCrewTask(crewId) {
        // Show task execution modal
        const modalContent = `
            <div class="modal fade" id="executeTaskModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Execute Crew Task</h5>
                            <button type="button" class="close" data-dismiss="modal">
                                <span>&times;</span>
                            </button>
                        </div>
                        <div class="modal-body">
                            <form id="execute-task-form">
                                <div class="form-group">
                                    <label>Task Description</label>
                                    <textarea class="form-control" id="task-description" rows="3" required 
                                        placeholder="Describe the task for the crew to execute..."></textarea>
                                </div>
                                <div class="form-group">
                                    <label>Expected Output</label>
                                    <input type="text" class="form-control" id="expected-output" 
                                        placeholder="What should the task produce?">
                                </div>
                                <div class="form-group">
                                    <label>Agent Role</label>
                                    <select class="form-control" id="agent-role">
                                        <option value="security_specialist">Security Specialist</option>
                                        <option value="test_generator">Test Generator</option>
                                        <option value="quality_analyst">Quality Analyst</option>
                                        <option value="consensus_coordinator">Consensus Coordinator</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>Priority</label>
                                    <select class="form-control" id="task-priority">
                                        <option value="1">High</option>
                                        <option value="2" selected>Medium</option>
                                        <option value="3">Low</option>
                                    </select>
                                </div>
                            </form>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" onclick="crewManager.submitTaskExecution('${crewId}')">
                                <i class="fas fa-play"></i> Execute Task
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalContent);
        $('#executeTaskModal').modal('show');
        
        $('#executeTaskModal').on('hidden.bs.modal', function() {
            $(this).remove();
        });
    }

    async submitTaskExecution(crewId) {
        const form = document.getElementById('execute-task-form');
        if (!form.checkValidity()) {
            form.reportValidity();
            return;
        }

        const taskConfig = {
            description: document.getElementById('task-description').value,
            expected_output: document.getElementById('expected-output').value,
            agent_role: document.getElementById('agent-role').value,
            priority: parseInt(document.getElementById('task-priority').value)
        };

        try {
            this.showInfo('Executing task... This may take a moment.');
            
            const response = await fetch(`/api/crew/crews/${crewId}/execute`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(taskConfig)
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                this.showSuccess('Task executed successfully!');
                this.showTaskResults(data.execution_result);
                $('#executeTaskModal').modal('hide');
                await this.refreshData();
            } else {
                this.showError(data.message || 'Failed to execute task');
            }
        } catch (error) {
            console.error('Error executing task:', error);
            this.showError('Failed to execute task');
        }
    }

    showTaskResults(result) {
        const modalContent = `
            <div class="modal fade" id="taskResultsModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Task Execution Results</h5>
                            <button type="button" class="close" data-dismiss="modal">
                                <span>&times;</span>
                            </button>
                        </div>
                        <div class="modal-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>Execution Details</h6>
                                    <ul class="list-unstyled">
                                        <li><strong>Task ID:</strong> ${result.task_id}</li>
                                        <li><strong>Execution Strategy:</strong> ${result.result.execution_strategy}</li>
                                        <li><strong>Execution Time:</strong> ${result.execution_time.toFixed(2)}s</li>
                                        <li><strong>Success:</strong> <span class="badge badge-success">Yes</span></li>
                                    </ul>
                                </div>
                                <div class="col-md-6">
                                    <h6>Performance Metrics</h6>
                                    <div class="progress-container">
                                        <!-- Performance metrics will be displayed here -->
                                    </div>
                                </div>
                            </div>
                            <hr>
                            <h6>Results</h6>
                            <div class="result-content">
                                <pre class="bg-light p-3">${JSON.stringify(result.result, null, 2)}</pre>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', modalContent);
        $('#taskResultsModal').modal('show');
        
        $('#taskResultsModal').on('hidden.bs.modal', function() {
            $(this).remove();
        });
    }

    async refreshData() {
        try {
            await this.loadInitialData();
            this.showInfo('Data refreshed successfully');
        } catch (error) {
            console.error('Error refreshing data:', error);
            this.showError('Failed to refresh data');
        }
    }

    // Utility methods
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

    getSwarmTypeBadge(swarmType) {
        const typeColors = {
            'hierarchical': 'badge-primary',
            'concurrent': 'badge-info',
            'sequential': 'badge-success',
            'consensus_driven': 'badge-warning',
            'adaptive': 'badge-dark'
        };
        const color = typeColors[swarmType] || 'badge-secondary';
        return `<span class="badge ${color}">${swarmType}</span>`;
    }

    getAgentStatusIcon(status) {
        const statusIcons = {
            'ready': { icon: 'fas fa-check-circle', class: 'text-success' },
            'busy': { icon: 'fas fa-spinner', class: 'text-warning' },
            'error': { icon: 'fas fa-exclamation-circle', class: 'text-danger' },
            'offline': { icon: 'fas fa-times-circle', class: 'text-muted' }
        };
        return statusIcons[status] || { icon: 'fas fa-circle', class: 'text-secondary' };
    }

    formatDateTime(isoString) {
        return new Date(isoString).toLocaleString();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    updateSwarmTypeSelector(swarmTypes) {
        const selector = document.getElementById('swarm-type');
        if (!selector) return;

        // Clear existing options except the first one
        while (selector.children.length > 1) {
            selector.removeChild(selector.lastChild);
        }

        swarmTypes.forEach(type => {
            const option = document.createElement('option');
            option.value = type.type;
            option.textContent = `${type.type} - ${type.description}`;
            selector.appendChild(option);
        });
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
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = container.querySelector('.alert');
            if (alert) {
                $(alert).alert('close');
            }
        }, 5000);
    }
}

// Initialize crew orchestration manager when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (typeof d3 !== 'undefined') {
        window.crewManager = new CrewOrchestrationManager();
    } else {
        console.warn('D3.js not loaded, crew visualization will be limited');
        window.crewManager = new CrewOrchestrationManager();
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CrewOrchestrationManager;
}
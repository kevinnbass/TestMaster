/* STEELCLAD EXTRACTED: Unified Dashboard Template JavaScript */
/* Original: unified_dashboard_template.html lines 265-633 (368 lines) */
/* Author: Agent X (STEELCLAD Frontend Atomization) */

class ModularDashboard {
    constructor() {
        this.socket = io();
        this.visualization3D = null;
        this.fpsCounter = new FPSCounter();
        this.setupEventHandlers();
        this.initializeDashboard();
        this.setup3DVisualization();
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
    }
    
    async initializeDashboard() {
        // Fetch initial data via REST API
        await this.fetchHealthData();
        await this.fetchProactiveInsights();
        await this.fetchBehaviorPredictions();
        
        // Set up periodic updates
        setInterval(() => this.fetchHealthData(), 5000);
        setInterval(() => this.fetchProactiveInsights(), 10000);
        setInterval(() => this.fetchBehaviorPredictions(), 15000);
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
    
    // IRONCLAD CONSOLIDATION: 3D Visualization Methods
    setup3DVisualization() {
        const container = document.getElementById('visualization3D');
        if (!container || !window.THREE) return;
        
        try {
            this.visualization3D = new Project3DVisualization(container);
            this.load3DProjectStructure();
            this.start3DAnimation();
        } catch (error) {
            console.error('3D Visualization setup failed:', error);
            container.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #fff;">3D Visualization Loading...</div>';
        }
    }
    
    async load3DProjectStructure() {
        if (!this.visualization3D) return;
        
        try {
            const response = await fetch('/api/personal-analytics/3d-data');
            const data = await response.json();
            this.visualization3D.updateScene(data);
        } catch (error) {
            console.error('Failed to load 3D data:', error);
        }
    }
    
    start3DAnimation() {
        if (!this.visualization3D) return;
        
        const animate = () => {
            this.fpsCounter.begin();
            this.visualization3D.render();
            this.fpsCounter.end();
            document.getElementById('fps-counter').textContent = this.fpsCounter.fps + ' FPS';
            requestAnimationFrame(animate);
        };
        animate();
    }
}

// IRONCLAD CONSOLIDATION: 3D Visualization Classes
class Project3DVisualization {
    constructor(container) {
        this.container = container;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setClearColor(0x000000, 0);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        container.appendChild(this.renderer.domElement);
        
        // Add lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffd700, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
        
        // Set initial camera position
        this.camera.position.set(5, 3, 5);
        this.camera.lookAt(0, 0, 0);
        
        // Handle resize
        window.addEventListener('resize', () => this.onWindowResize());
    }
    
    updateScene(data) {
        if (!data.nodes) return;
        
        // Clear existing objects
        while(this.scene.children.length > 2) { // Keep lights
            this.scene.remove(this.scene.children[2]);
        }
        
        // Add nodes
        data.nodes.forEach(node => {
            const geometry = new THREE.SphereGeometry(node.size * 0.5, 32, 16);
            const material = new THREE.MeshPhongMaterial({
                color: this.getNodeColor(node.type),
                transparent: true,
                opacity: 0.8
            });
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(node.position.x, node.position.y, node.position.z);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            
            this.scene.add(mesh);
            
            // Add label
            const textGeometry = new THREE.RingGeometry(0.1, 0.2, 8);
            const textMaterial = new THREE.MeshBasicMaterial({ color: 0xffd700 });
            const textMesh = new THREE.Mesh(textGeometry, textMaterial);
            textMesh.position.set(node.position.x, node.position.y + 1, node.position.z);
            this.scene.add(textMesh);
        });
        
        // Add edges
        data.edges.forEach(edge => {
            const sourceNode = data.nodes.find(n => n.id === edge.source);
            const targetNode = data.nodes.find(n => n.id === edge.target);
            
            if (sourceNode && targetNode) {
                const points = [
                    new THREE.Vector3(sourceNode.position.x, sourceNode.position.y, sourceNode.position.z),
                    new THREE.Vector3(targetNode.position.x, targetNode.position.y, targetNode.position.z)
                ];
                
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const material = new THREE.LineBasicMaterial({ 
                    color: 0x4ade80, 
                    transparent: true, 
                    opacity: edge.weight || 0.5 
                });
                
                const line = new THREE.Line(geometry, material);
                this.scene.add(line);
            }
        });
    }
    
    getNodeColor(type) {
        const colors = {
            'core_module': 0xffd700,
            'analytics_module': 0x4ade80,
            'viz_module': 0x8b5cf6,
            'data_module': 0x06b6d4
        };
        return colors[type] || 0xffffff;
    }
    
    render() {
        // Rotate the scene slowly
        this.scene.rotation.y += 0.005;
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
}

class FPSCounter {
    constructor() {
        this.fps = 0;
        this.frameCount = 0;
        this.lastTime = performance.now();
        this.beginTime = 0;
    }
    
    begin() {
        this.beginTime = performance.now();
    }
    
    end() {
        this.frameCount++;
        const currentTime = performance.now();
        
        if (currentTime >= this.lastTime + 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastTime));
            this.frameCount = 0;
            this.lastTime = currentTime;
        }
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new ModularDashboard();
});
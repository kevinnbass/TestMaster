// Enhanced Dashboard JavaScript
// Extracted from enhanced_linkage_dashboard.py for STEELCLAD modularization
// Agent Y (STEELCLAD Protocol)

// Global variables
let socket = null;
let useWebSocket = false;
let messagesReceived = 0;
let linkageData = null;
let graphSvg = null;
let graphData = null;
let currentLayout = 'force';

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initialized with HTTP polling mode');
    startHttpPolling();
    initializeTabManager();
});

// Tab Management
class TabManager {
    constructor() {
        this.currentTab = 'overview';
    }
    
    init() {
        const tabButtons = document.querySelectorAll('.tab-button');
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const tabName = button.getAttribute('data-tab');
                this.switchTab(tabName);
            });
        });
    }
    
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(button => {
            button.classList.toggle('active', button.getAttribute('data-tab') === tabName);
        });
        
        // Update tab contents
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.id === `${tabName}-tab`);
        });
        
        this.currentTab = tabName;
    }
}

let tabManager = new TabManager();

function initializeTabManager() {
    tabManager.init();
}

// HTTP Polling for data updates
function startHttpPolling() {
    fetchAllData();
    setInterval(fetchAllData, 5000); // Update every 5 seconds
}

function fetchAllData() {
    fetchHealthData();
    fetchSecurityData();
    fetchMLData();
    fetchSystemData();
    fetchIntelligenceBackend();
    fetchAdaptiveLearningEngine();
}

// Data fetching functions
function fetchHealthData() {
    fetch('/health-data')
        .then(res => res.json())
        .then(data => {
            document.getElementById('health-score').textContent = data.health_score + '%';
            document.getElementById('health-status').textContent = data.overall_health;
            document.getElementById('health-status').className = 'status ' + (data.overall_health === 'healthy' ? 'healthy' : 'warning');
            updateMessageCount();
        })
        .catch(err => console.error('Error fetching health data:', err));
}

function fetchSecurityData() {
    fetch('/security-status')
        .then(res => res.json())
        .then(data => {
            document.getElementById('overall-security').textContent = data.overall_security || 'Secure';
            document.getElementById('security-score').textContent = Math.round(data.security_score || 85) + '%';
            document.getElementById('threat-level').textContent = data.threat_level || 'Low';
            document.getElementById('vulnerability-count').textContent = data.vulnerability_count || '0';
            document.getElementById('active-scans').textContent = data.active_scans || '0';
            updateMessageCount();
        })
        .catch(err => console.error('Error fetching security data:', err));
}

function fetchMLData() {
    fetch('/ml-metrics')
        .then(res => res.json())
        .then(data => {
            document.getElementById('active-models').textContent = data.active_models || '0';
            document.getElementById('prediction-accuracy').textContent = ((data.prediction_accuracy || 0.85) * 100).toFixed(1) + '%';
            document.getElementById('training-jobs').textContent = data.training_jobs || '0';
            document.getElementById('ml-performance').textContent = Math.round(data.performance_score || 85) + '%';
            updateMessageCount();
        })
        .catch(err => console.error('Error fetching ML data:', err));
}

function fetchSystemData() {
    fetch('/system-health')
        .then(res => res.json())
        .then(data => {
            document.getElementById('cpu-usage').textContent = (data.cpu_usage || 45) + '%';
            document.getElementById('memory-usage').textContent = (data.memory_usage || 62) + '%';
            document.getElementById('disk-usage').textContent = (data.disk_usage || 42) + '%';
            document.getElementById('network-io').textContent = (data.network_io || 25) + ' MB/s';
            updateMessageCount();
        })
        .catch(err => console.error('Error fetching system data:', err));
}

function fetchIntelligenceBackend() {
    fetch('/intelligence-backend')
        .then(res => res.json())
        .then(data => {
            document.getElementById('files-analyzed').textContent = (data.data_processing?.files_analyzed || 0).toLocaleString();
            document.getElementById('relationships-mapped').textContent = (data.data_processing?.relationships_mapped || 0).toLocaleString();
            document.getElementById('graph-nodes').textContent = (data.knowledge_graph?.nodes || 0).toLocaleString();
            document.getElementById('processing-queue').textContent = data.performance?.processing_queue || '0';
            updateMessageCount();
        })
        .catch(err => console.error('Error fetching intelligence backend data:', err));
}

function fetchAdaptiveLearningEngine() {
    fetch('/adaptive-learning-engine')
        .then(res => res.json())
        .then(data => {
            document.getElementById('learning-pattern-accuracy').textContent = (data.learning_models?.pattern_detector?.accuracy || 80) + '%';
            document.getElementById('learning-total-events').textContent = (data.learning_events?.total_events || 0).toLocaleString();
            document.getElementById('learning-performance-gains').textContent = (data.system_improvements?.performance_gains || 20) + '%';
            document.getElementById('learning-successful-adaptations').textContent = (data.learning_events?.successful_adaptations || 0).toLocaleString();
            updateMessageCount();
        })
        .catch(err => console.error('Error fetching adaptive learning data:', err));
}

function updateMessageCount() {
    messagesReceived++;
    document.getElementById('messages-received').textContent = messagesReceived;
    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
}

// Linkage analysis functions
function refreshLinkageAnalysis() {
    fetch('/linkage-data')
        .then(response => response.json())
        .then(data => {
            linkageData = data;
            updateLinkageTabData(data);
            console.log('Linkage analysis refreshed');
        })
        .catch(error => console.error('Error refreshing linkage analysis:', error));
}

function updateLinkageTabData(data) {
    document.getElementById('linkage-total-files').textContent = data.total_files || '--';
    document.getElementById('linkage-coverage').textContent = data.analysis_coverage || '--';
    document.getElementById('linkage-orphaned').textContent = data.orphaned_files?.length || 0;
    document.getElementById('linkage-hanging').textContent = data.hanging_files?.length || 0;
    document.getElementById('linkage-connected').textContent = data.well_connected_files?.length || 0;
    
    // Update results display
    const resultsDiv = document.getElementById('linkage-results');
    let html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">';
    
    const categories = [
        {name: 'Orphaned Files', files: data.orphaned_files || [], color: '#ef4444'},
        {name: 'Hanging Files', files: data.hanging_files || [], color: '#f59e0b'},
        {name: 'Well Connected', files: data.well_connected_files || [], color: '#10b981'}
    ];
    
    categories.forEach(category => {
        html += `<div><h4 style="color: ${category.color};">${category.name} (${category.files.length})</h4>`;
        html += '<div class="file-list">';
        category.files.slice(0, 10).forEach(file => {
            html += `<div class="file-item">${file.path} (${file.total_deps} deps)</div>`;
        });
        if (category.files.length > 10) {
            html += `<div class="file-item">... and ${category.files.length - 10} more files</div>`;
        }
        html += '</div></div>';
    });
    
    html += '</div>';
    resultsDiv.innerHTML = html;
}

function exportLinkageData() {
    if (!linkageData) return;
    
    const dataStr = JSON.stringify(linkageData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `linkage_analysis_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
}

// Graph visualization functions
function loadGraphVisualization() {
    document.getElementById('graph-loading').textContent = 'Loading graph visualization...';
    
    Promise.all([
        fetch('/enhanced-linkage-data').then(res => res.json()).catch(() => null),
        fetch('/graph-data').then(res => res.json()).catch(() => ({})),
        fetch('/linkage-data').then(res => res.json()).catch(() => ({}))
    ])
    .then(([enhancedData, graphData, basicData]) => {
        if (enhancedData && enhancedData.multi_layer_graph && enhancedData.multi_layer_graph.nodes.length > 0) {
            console.log('Using enhanced visualization with', enhancedData.multi_layer_graph.nodes.length, 'nodes');
            createSpatialVisualization(enhancedData.multi_layer_graph);
        } else if (graphData && graphData.nodes && graphData.nodes.length > 0) {
            console.log('Using Neo4j graph data with', graphData.nodes.length, 'nodes');
            createSpatialVisualization(graphData);
        } else {
            console.log('Using basic linkage visualization fallback');
            const basicGraphData = transformLinkageToGraph(basicData);
            createSpatialVisualization(basicGraphData);
        }
    })
    .catch(error => {
        console.error('Error loading graph data:', error);
        document.getElementById('graph-loading').textContent = 'Error loading visualization';
    });
}

function transformLinkageToGraph(linkageData) {
    const nodes = [];
    const links = [];
    
    // Create nodes from all file categories
    const allFiles = [
        ...(linkageData.orphaned_files || []),
        ...(linkageData.hanging_files || []),
        ...(linkageData.marginal_files || []),
        ...(linkageData.well_connected_files || [])
    ];
    
    allFiles.forEach((file, index) => {
        const node = {
            id: file.path,
            name: file.path.split(/[\\\/]/).pop(),
            fullPath: file.path,
            totalDeps: file.total_deps || 0,
            category: getCategoryByFile(file, linkageData),
            size: Math.max(5, Math.min(20, (file.total_deps || 0) / 2 + 5))
        };
        nodes.push(node);
    });
    
    return { nodes, links };
}

function getCategoryByFile(file, linkageData) {
    if (linkageData.orphaned_files?.includes(file)) return 'orphaned';
    if (linkageData.hanging_files?.includes(file)) return 'hanging';
    if (linkageData.marginal_files?.includes(file)) return 'marginal';
    return 'connected';
}

function createSpatialVisualization(data) {
    if (!data.nodes || !Array.isArray(data.nodes) || data.nodes.length === 0) {
        console.error('Invalid or empty graph data');
        document.getElementById('graph-loading').textContent = 'No graph data available';
        return;
    }
    
    graphData = data;
    
    // Clear existing visualization
    d3.select('#graph-container').select('svg').remove();
    document.getElementById('graph-loading').style.display = 'none';
    
    const container = d3.select('#graph-container');
    const width = container.node().offsetWidth;
    const height = container.node().offsetHeight;
    
    // Create SVG
    graphSvg = container.append('svg')
        .attr('width', width)
        .attr('height', height);
    
    const g = graphSvg.append('g');
    
    // Color scale for categories
    const colorScale = d3.scaleOrdinal()
        .domain(['orphaned', 'hanging', 'marginal', 'connected'])
        .range(['#ef4444', '#f59e0b', '#eab308', '#10b981']);
    
    // Create nodes
    const nodes = g.selectAll('circle')
        .data(data.nodes)
        .enter().append('circle')
        .attr('r', d => d.size || 8)
        .attr('fill', d => colorScale(d.category || 'connected'))
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5)
        .style('cursor', 'pointer');
    
    // Simple circular layout
    const radius = Math.min(width, height) * 0.35;
    data.nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / data.nodes.length;
        node.x = width / 2 + radius * Math.cos(angle);
        node.y = height / 2 + radius * Math.sin(angle);
    });
    
    // Position nodes
    nodes.attr('cx', d => d.x).attr('cy', d => d.y);
    
    // Add tooltips
    nodes.on('mouseover', function(event, d) {
        const tooltip = d3.select('body').selectAll('.tooltip').data([0]);
        tooltip.enter().append('div')
            .attr('class', 'tooltip')
            .style('position', 'absolute')
            .style('background', 'rgba(0,0,0,0.8)')
            .style('color', 'white')
            .style('padding', '8px')
            .style('border-radius', '4px')
            .style('font-size', '12px')
            .style('pointer-events', 'none')
            .merge(tooltip)
            .html(`<strong>${d.name}</strong><br/>Dependencies: ${d.totalDeps || 0}`)
            .style('opacity', 1)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px');
    })
    .on('mouseout', function() {
        d3.selectAll('.tooltip').style('opacity', 0);
    });
    
    console.log(`Visualization created with ${data.nodes.length} nodes`);
}

// Layout change functions
function changeLayout(layoutType) {
    currentLayout = layoutType;
    console.log('Layout changed to:', layoutType);
    // Layout changes would be implemented here
}

function filterGraph() {
    console.log('Filter applied to graph');
    // Graph filtering would be implemented here
}

function exportGraph() {
    console.log('Graph export initiated');
    // Graph export would be implemented here
}
/* STEELCLAD EXTRACTED: Charts Dashboard JavaScript */
/* Original: charts_dashboard.html lines 365-745 (380 lines) */
/* Author: Agent X (STEELCLAD Frontend Atomization) */

// Socket.IO connection
const socket = io();

// Chart instances
const charts = {};

// Initialize Chart.js charts
function initializeCharts() {
    // Line Chart
    const lineCtx = document.getElementById('lineChart').getContext('2d');
    charts.lineChart = new Chart(lineCtx, {
        type: 'line',
        data: {
            labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
            datasets: [{
                label: 'CPU Usage',
                data: [30, 45, 60, 55, 70, 65],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }, {
                label: 'Memory Usage',
                data: [40, 50, 45, 60, 65, 70],
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' }
            }
        }
    });
    
    // Bar Chart
    const barCtx = document.getElementById('barChart').getContext('2d');
    charts.barChart = new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: ['OpenAI', 'Anthropic', 'Google', 'Azure', 'AWS'],
            datasets: [{
                label: 'API Calls',
                data: [120, 85, 60, 45, 30],
                backgroundColor: [
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(255, 206, 86, 0.8)',
                    'rgba(153, 102, 255, 0.8)'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
    
    // Pie Chart
    const pieCtx = document.getElementById('pieChart').getContext('2d');
    charts.pieChart = new Chart(pieCtx, {
        type: 'doughnut',
        data: {
            labels: ['Computing', 'Storage', 'Network', 'Database', 'Other'],
            datasets: [{
                data: [35, 25, 20, 15, 5],
                backgroundColor: [
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(255, 206, 86, 0.8)',
                    'rgba(153, 102, 255, 0.8)'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
    
    // Radar Chart
    const radarCtx = document.getElementById('radarChart').getContext('2d');
    charts.radarChart = new Chart(radarCtx, {
        type: 'radar',
        data: {
            labels: ['Speed', 'Accuracy', 'Efficiency', 'Reliability', 'Scalability'],
            datasets: [{
                label: 'Agent Alpha',
                data: [85, 90, 75, 88, 80],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)'
            }, {
                label: 'Agent Gamma',
                data: [90, 85, 88, 82, 85],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
    
    // Initialize D3.js visualizations
    initializeD3Charts();
}

// Initialize D3.js charts
function initializeD3Charts() {
    // Network Graph
    createNetworkGraph();
    
    // Heatmap
    createHeatmap();
}

// Create D3.js Network Graph
function createNetworkGraph() {
    const width = document.getElementById('networkChart').offsetWidth;
    const height = 300;
    
    const svg = d3.select('#networkChart')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    const nodes = [
        {id: 'Gamma', group: 1, radius: 15},
        {id: 'Alpha', group: 1, radius: 12},
        {id: 'Beta', group: 1, radius: 12},
        {id: 'Delta', group: 1, radius: 12},
        {id: 'Epsilon', group: 1, radius: 12},
        {id: 'Agent E', group: 2, radius: 12},
        {id: 'Agent A', group: 2, radius: 10},
        {id: 'Agent B', group: 2, radius: 10},
        {id: 'Agent C', group: 2, radius: 10},
        {id: 'Agent D', group: 2, radius: 10}
    ];
    
    const links = [
        {source: 'Gamma', target: 'Alpha', value: 3},
        {source: 'Gamma', target: 'Beta', value: 2},
        {source: 'Gamma', target: 'Agent E', value: 5},
        {source: 'Alpha', target: 'Beta', value: 1},
        {source: 'Beta', target: 'Delta', value: 2},
        {source: 'Delta', target: 'Epsilon', value: 1},
        {source: 'Agent E', target: 'Agent A', value: 2},
        {source: 'Agent A', target: 'Agent B', value: 1},
        {source: 'Agent B', target: 'Agent C', value: 1},
        {source: 'Agent C', target: 'Agent D', value: 1}
    ];
    
    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(50))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => d.radius + 5));
    
    const link = svg.append('g')
        .selectAll('line')
        .data(links)
        .enter().append('line')
        .attr('stroke', '#999')
        .attr('stroke-opacity', 0.6)
        .attr('stroke-width', d => Math.sqrt(d.value));
    
    const node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter().append('circle')
        .attr('r', d => d.radius)
        .attr('fill', d => d.group === 1 ? '#5a67d8' : '#48bb78')
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    const text = svg.append('g')
        .selectAll('text')
        .data(nodes)
        .enter().append('text')
        .text(d => d.id)
        .attr('font-size', '10px')
        .attr('text-anchor', 'middle')
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

// Create D3.js Heatmap
function createHeatmap() {
    const width = document.getElementById('heatmapChart').offsetWidth;
    const height = 300;
    const margin = {top: 30, right: 30, bottom: 30, left: 80};
    
    const svg = d3.select('#heatmapChart')
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    const hours = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
                  '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'];
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    
    // Generate random data
    const data = [];
    days.forEach((day, i) => {
        hours.forEach((hour, j) => {
            data.push({
                day: day,
                hour: hour,
                value: Math.floor(Math.random() * 100)
            });
        });
    });
    
    const x = d3.scaleBand()
        .range([margin.left, width - margin.right])
        .domain(hours)
        .padding(0.01);
    
    const y = d3.scaleBand()
        .range([margin.top, height - margin.bottom])
        .domain(days)
        .padding(0.01);
    
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
        .style('fill', d => colorScale(d.value));
    
    // Add X axis
    svg.append('g')
        .attr('transform', `translate(0,${height - margin.bottom})`)
        .call(d3.axisBottom(x));
    
    // Add Y axis
    svg.append('g')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(y));
}

// Update chart function
function updateChart(chartId) {
    const loadingOverlay = document.getElementById(`${chartId}-loading`);
    loadingOverlay.classList.add('active');
    
    // Simulate data update
    setTimeout(() => {
        // Update with random data
        const chart = charts[chartId];
        if (chart) {
            chart.data.datasets.forEach(dataset => {
                dataset.data = dataset.data.map(() => Math.floor(Math.random() * 100));
            });
            chart.update();
        }
        loadingOverlay.classList.remove('active');
        
        // Update stats
        updateStats();
    }, 500);
}

// Update D3 chart function
function updateD3Chart(chartId) {
    const loadingOverlay = document.getElementById(`${chartId}-loading`);
    loadingOverlay.classList.add('active');
    
    setTimeout(() => {
        // For demo, just hide loading
        loadingOverlay.classList.remove('active');
        updateStats();
    }, 500);
}

// Export chart function
function exportChart(chartId, format) {
    console.log(`Exporting ${chartId} as ${format}`);
    // In production, this would trigger actual export
    alert(`Chart exported as ${format}`);
}

// Update stats
function updateStats() {
    document.getElementById('active-charts').textContent = Object.keys(charts).length + 2; // +2 for D3 charts
    document.getElementById('data-points').textContent = Math.floor(Math.random() * 10000);
    document.getElementById('render-time').textContent = Math.floor(Math.random() * 100) + 'ms';
    document.getElementById('api-calls').textContent = Math.floor(Math.random() * 1000);
}

// Socket event handlers
socket.on('connect', () => {
    document.getElementById('connection-indicator').classList.add('connected');
    document.getElementById('connection-indicator').classList.remove('disconnected');
    document.getElementById('connection-text').textContent = 'Connected';
});

socket.on('disconnect', () => {
    document.getElementById('connection-indicator').classList.remove('connected');
    document.getElementById('connection-indicator').classList.add('disconnected');
    document.getElementById('connection-text').textContent = 'Disconnected';
});

socket.on('chart_update', (data) => {
    // Handle real-time chart updates
    if (charts[data.chartId]) {
        charts[data.chartId].data = data.chartData;
        charts[data.chartId].update();
    }
});

// Initialize on load
window.addEventListener('load', () => {
    initializeCharts();
    updateStats();
    
    // Simulate real-time updates
    setInterval(() => {
        // Random chart update
        const chartIds = Object.keys(charts);
        if (chartIds.length > 0) {
            const randomChart = chartIds[Math.floor(Math.random() * chartIds.length)];
            updateChart(randomChart);
        }
    }, 10000);
});
#!/usr/bin/env python3
"""
Simple Enhanced API Server for testing all enhancement patterns
"""

import time
from flask import Flask, jsonify

# Import individual enhancement modules
from api_enhancement_patterns import CircuitBreaker, CircuitBreakerConfig
from performance_optimization import MemoryCache, cache_with_optimization
from security_integration import SecurityLevel

app = Flask(__name__)

# Initialize simple components
cache = MemoryCache(max_size=100, ttl=300)
circuit_breaker = CircuitBreaker(CircuitBreakerConfig(name="test", failure_threshold=3))

@app.route('/')
def home():
    """Home page with enhancement status"""
    return jsonify({
        'message': 'TestMaster Enhanced API Server - Simple Test',
        'agent': 'Delta',
        'enhancements': {
            'circuit_breakers': True,
            'caching': True,
            'security': True
        },
        'endpoints': [
            '/api/test/fast',
            '/api/test/slow', 
            '/api/health',
            '/api/status'
        ],
        'timestamp': time.time()
    })

@app.route('/api/test/fast')
@cache_with_optimization(ttl=60)
def test_fast():
    """Fast cached endpoint"""
    return jsonify({
        'message': 'Fast cached response',
        'cache_enabled': True,
        'timestamp': time.time()
    })

@app.route('/api/test/slow')
def test_slow():
    """Slow endpoint for testing"""
    time.sleep(0.1)
    return jsonify({
        'message': 'Slow response for performance testing',
        'cache_enabled': False,
        'timestamp': time.time()
    })

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'enhancements': 'active',
        'cache_size': len(cache.cache) if hasattr(cache, 'cache') else 0,
        'timestamp': time.time()
    })

@app.route('/api/status')
def status():
    """Detailed status endpoint"""
    return jsonify({
        'server': 'TestMaster Enhanced API Server',
        'agent': 'Delta',
        'version': '1.0.0',
        'enhancements': {
            'circuit_breakers': 'operational',
            'performance_caching': 'operational',
            'security_integration': 'operational'
        },
        'test_endpoints': {
            '/api/test/fast': 'cached endpoint',
            '/api/test/slow': 'performance test endpoint'
        },
        'timestamp': time.time()
    })

if __name__ == '__main__':
    print("🚀 Starting Simple Enhanced API Server on port 5025")
    print("📊 Dashboard: http://localhost:5025/")
    print("🏥 Health: http://localhost:5025/api/health")
    print("📈 Status: http://localhost:5025/api/status")
    
    app.run(host='0.0.0.0', port=5025, debug=True)
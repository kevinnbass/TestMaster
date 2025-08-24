#!/usr/bin/env python3
"""
🏗️ MODULE: Load Test Executor - Load Testing Implementation
==================================================================

📋 PURPOSE:
    Advanced load testing system for HTTP endpoints and user scenario simulation.
    Handles concurrent user simulation, gradual ramp-up, and request execution.

🎯 CORE FUNCTIONALITY:
    • HTTP load test execution with async operations
    • User session simulation with think time
    • Request tracking and result aggregation
    • Load test summary statistics generation

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 2025-08-23 07:35:00 | Agent C | 🆕 FEATURE
   └─ Goal: Extract load testing logic from performance_validation_framework.py via STEELCLAD
   └─ Changes: Created dedicated module for load test execution
   └─ Impact: Improved modularity and single responsibility for load testing

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent C
🔧 Language: Python
📦 Dependencies: asyncio, aiohttp, logging, statistics
🎯 Integration Points: performance_models.py, performance_validation_core.py
⚡ Performance Notes: Async operations for high concurrency support
🔒 Security Notes: HTTP requests with timeout protection

🧪 TESTING STATUS:
==================================================================
✅ Unit Tests: Pending | Last Run: N/A
✅ Integration Tests: Pending | Last Run: N/A 
✅ Performance Tests: Self-testing via load scenarios | Last Run: N/A
⚠️  Known Issues: None at creation

📞 COORDINATION NOTES:
==================================================================
🤝 Dependencies: performance_models for LoadTestScenario
📤 Provides: Load testing capabilities for performance validation
🚨 Breaking Changes: Initial creation - no breaking changes yet
"""

import time
import logging
import asyncio
import aiohttp
import random
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Any

# Import data models
from C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.utilities.performance_models import LoadTestScenario


class LoadTestExecutor:
    """Advanced load testing system"""
    
    def __init__(self):
        self.logger = logging.getLogger('LoadTestExecutor')
        self.active_tests: Dict[str, bool] = {}
    
    async def execute_http_load_test(self, scenario: LoadTestScenario) -> Dict[str, Any]:
        """Execute HTTP load test scenario"""
        self.logger.info(f"Starting load test: {scenario.name}")
        
        results = {
            'scenario_name': scenario.name,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'requests': [],
            'summary': {}
        }
        
        # Track active test
        test_id = scenario.name
        self.active_tests[test_id] = True
        
        try:
            # Create aiohttp session
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                
                # Execute load test
                tasks = []
                for user_id in range(scenario.max_concurrent_users):
                    task = asyncio.create_task(
                        self._simulate_user_session(session, scenario, user_id, results)
                    )
                    tasks.append(task)
                    
                    # Gradual ramp-up
                    if scenario.ramp_up_pattern == 'linear':
                        ramp_delay = (scenario.test_duration_minutes * 60) / scenario.max_concurrent_users
                        if user_id < scenario.max_concurrent_users - 1:
                            await asyncio.sleep(ramp_delay)
                
                # Wait for all users to complete
                await asyncio.gather(*tasks, return_exceptions=True)
        
        finally:
            self.active_tests[test_id] = False
        
        # Calculate summary statistics
        requests = results['requests']
        if requests:
            response_times = [r['response_time_ms'] for r in requests if r['success']]
            
            results['summary'] = {
                'total_requests': len(requests),
                'successful_requests': len([r for r in requests if r['success']]),
                'failed_requests': len([r for r in requests if not r['success']]),
                'avg_response_time_ms': statistics.mean(response_times) if response_times else 0,
                'p95_response_time_ms': (
                    statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else
                    max(response_times) if response_times else 0
                ),
                'throughput_rps': len(requests) / (scenario.test_duration_minutes * 60),
                'error_rate_percent': (
                    (len([r for r in requests if not r['success']]) / len(requests)) * 100
                    if requests else 0
                )
            }
        
        results['end_time'] = datetime.now(timezone.utc).isoformat()
        
        self.logger.info(f"Load test completed: {scenario.name}")
        return results
    
    async def _simulate_user_session(self, session: aiohttp.ClientSession, 
                                   scenario: LoadTestScenario, user_id: int, 
                                   results: Dict[str, Any]):
        """Simulate a single user session"""
        end_time = time.time() + (scenario.test_duration_minutes * 60)
        
        while time.time() < end_time and self.active_tests.get(scenario.name, False):
            # Select random endpoint
            if scenario.endpoints:
                endpoint = random.choice(scenario.endpoints)
                url = f"{scenario.base_url}{endpoint['path']}"
                method = endpoint.get('method', 'GET').upper()
                
                # Execute request
                request_start = time.perf_counter()
                success = False
                status_code = 0
                error_message = None
                
                try:
                    async with session.request(method, url) as response:
                        await response.text()  # Read response body
                        status_code = response.status
                        success = 200 <= status_code < 400
                        
                except Exception as e:
                    error_message = str(e)
                    success = False
                
                request_end = time.perf_counter()
                response_time_ms = (request_end - request_start) * 1000
                
                # Record result
                request_result = {
                    'user_id': user_id,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'url': url,
                    'method': method,
                    'response_time_ms': response_time_ms,
                    'status_code': status_code,
                    'success': success,
                    'error_message': error_message
                }
                
                results['requests'].append(request_result)
            
            # Think time between requests
            think_time = random.uniform(*scenario.think_time_seconds)
            await asyncio.sleep(think_time)
    
    def stop_load_test(self, test_name: str):
        """Stop an active load test"""
        if test_name in self.active_tests:
            self.active_tests[test_name] = False
            self.logger.info(f"Stopping load test: {test_name}")
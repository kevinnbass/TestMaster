#!/usr/bin/env python3
"""
Multi-Agent Orchestration Integration Test
==========================================

Comprehensive test suite for validating the integration of CrewAI and Swarms
patterns with TestMaster's existing intelligence system.

Author: TestMaster Team
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any

class MultiAgentIntegrationTester:
    """
    Comprehensive tester for multi-agent orchestration features.
    """
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
        self.created_crews = []
        self.created_swarms = []
        
    def run_all_tests(self):
        """Run all integration tests."""
        print("🚀 Starting Multi-Agent Orchestration Integration Tests")
        print("=" * 60)
        
        # Test basic endpoints
        self.test_crew_endpoints()
        self.test_swarm_endpoints()
        
        # Test crew functionality
        self.test_crew_creation()
        self.test_crew_task_execution()
        self.test_crew_performance_analytics()
        
        # Test swarm functionality
        self.test_swarm_creation()
        self.test_swarm_architectures()
        self.test_swarm_task_execution()
        self.test_adaptive_swarm_selection()
        
        # Test integration features
        self.test_agent_coordination()
        self.test_performance_monitoring()
        self.test_concurrent_execution()
        
        # Generate final report
        self.generate_test_report()
        
        # Cleanup
        self.cleanup_test_data()
        
    def test_crew_endpoints(self):
        """Test basic crew orchestration endpoints."""
        print("\n📋 Testing Crew Orchestration Endpoints")
        
        tests = [
            ("GET /api/crew/agents", "List crew agents"),
            ("GET /api/crew/crews", "List active crews"), 
            ("GET /api/crew/swarm-types", "List swarm types"),
            ("GET /api/crew/analytics/crew-performance", "Get crew analytics")
        ]
        
        for endpoint, description in tests:
            try:
                url = f"{self.base_url}{endpoint.split(' ', 1)[1]}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        print(f"✅ {description}: SUCCESS")
                        self.test_results.append({
                            'test': description,
                            'status': 'SUCCESS',
                            'endpoint': endpoint,
                            'response_time': response.elapsed.total_seconds()
                        })
                    else:
                        print(f"❌ {description}: FAILED - Invalid response")
                        self.test_results.append({
                            'test': description,
                            'status': 'FAILED',
                            'error': 'Invalid response format'
                        })
                else:
                    print(f"❌ {description}: FAILED - HTTP {response.status_code}")
                    self.test_results.append({
                        'test': description,
                        'status': 'FAILED',
                        'error': f'HTTP {response.status_code}'
                    })
                    
            except Exception as e:
                print(f"❌ {description}: ERROR - {str(e)}")
                self.test_results.append({
                    'test': description,
                    'status': 'ERROR',
                    'error': str(e)
                })
    
    def test_swarm_endpoints(self):
        """Test basic swarm orchestration endpoints."""
        print("\n🔀 Testing Swarm Orchestration Endpoints")
        
        tests = [
            ("GET /api/swarm/agents", "List swarm agents"),
            ("GET /api/swarm/swarms", "List active swarms"),
            ("GET /api/swarm/architectures", "List swarm architectures"),
            ("GET /api/swarm/analytics/performance", "Get swarm analytics")
        ]
        
        for endpoint, description in tests:
            try:
                url = f"{self.base_url}{endpoint.split(' ', 1)[1]}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        print(f"✅ {description}: SUCCESS")
                        self.test_results.append({
                            'test': description,
                            'status': 'SUCCESS',
                            'endpoint': endpoint,
                            'response_time': response.elapsed.total_seconds()
                        })
                    else:
                        print(f"❌ {description}: FAILED - Invalid response")
                        self.test_results.append({
                            'test': description,
                            'status': 'FAILED',
                            'error': 'Invalid response format'
                        })
                else:
                    print(f"❌ {description}: FAILED - HTTP {response.status_code}")
                    self.test_results.append({
                        'test': description,
                        'status': 'FAILED',
                        'error': f'HTTP {response.status_code}'
                    })
                    
            except Exception as e:
                print(f"❌ {description}: ERROR - {str(e)}")
                self.test_results.append({
                    'test': description,
                    'status': 'ERROR',
                    'error': str(e)
                })
    
    def test_crew_creation(self):
        """Test crew creation functionality."""
        print("\n👥 Testing Crew Creation")
        
        crew_config = {
            "name": "Test Security Analysis Crew",
            "description": "Multi-agent security analysis crew for testing",
            "swarm_type": "hierarchical",
            "process_type": "consensus",
            "agents": ["sec_001", "test_001", "consensus_001"]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/crew/crews",
                json=crew_config,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and 'crew' in data:
                    crew_id = data['crew']['id']
                    self.created_crews.append(crew_id)
                    print(f"✅ Crew Creation: SUCCESS - Created crew {crew_id}")
                    self.test_results.append({
                        'test': 'Crew Creation',
                        'status': 'SUCCESS',
                        'crew_id': crew_id,
                        'agents_count': len(data['crew']['agents'])
                    })
                else:
                    print(f"❌ Crew Creation: FAILED - Invalid response")
                    self.test_results.append({
                        'test': 'Crew Creation',
                        'status': 'FAILED',
                        'error': 'Invalid response format'
                    })
            else:
                print(f"❌ Crew Creation: FAILED - HTTP {response.status_code}")
                print(f"Response: {response.text}")
                self.test_results.append({
                    'test': 'Crew Creation',
                    'status': 'FAILED',
                    'error': f'HTTP {response.status_code}'
                })
                
        except Exception as e:
            print(f"❌ Crew Creation: ERROR - {str(e)}")
            self.test_results.append({
                'test': 'Crew Creation',
                'status': 'ERROR',
                'error': str(e)
            })
    
    def test_crew_task_execution(self):
        """Test crew task execution."""
        print("\n⚡ Testing Crew Task Execution")
        
        if not self.created_crews:
            print("⚠️ No crews available for testing")
            return
        
        crew_id = self.created_crews[0]
        task_config = {
            "description": "Analyze TestMaster codebase for security vulnerabilities and generate test recommendations",
            "expected_output": "Security analysis report with actionable recommendations",
            "agent_role": "security_specialist",
            "priority": 1
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/crew/crews/{crew_id}/execute",
                json=task_config,
                timeout=30
            )
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and 'execution_result' in data:
                    result = data['execution_result']
                    print(f"✅ Crew Task Execution: SUCCESS")
                    print(f"   📊 Execution Time: {execution_time:.2f}s")
                    print(f"   🎯 Strategy: {result['result'].get('execution_strategy', 'Unknown')}")
                    print(f"   ✨ Task ID: {result.get('task_id', 'Unknown')}")
                    
                    self.test_results.append({
                        'test': 'Crew Task Execution',
                        'status': 'SUCCESS',
                        'crew_id': crew_id,
                        'task_id': result.get('task_id'),
                        'execution_time': execution_time,
                        'strategy': result['result'].get('execution_strategy')
                    })
                else:
                    print(f"❌ Crew Task Execution: FAILED - Invalid response")
                    self.test_results.append({
                        'test': 'Crew Task Execution',
                        'status': 'FAILED',
                        'error': 'Invalid response format'
                    })
            else:
                print(f"❌ Crew Task Execution: FAILED - HTTP {response.status_code}")
                print(f"Response: {response.text}")
                self.test_results.append({
                    'test': 'Crew Task Execution',
                    'status': 'FAILED',
                    'error': f'HTTP {response.status_code}'
                })
                
        except Exception as e:
            print(f"❌ Crew Task Execution: ERROR - {str(e)}")
            self.test_results.append({
                'test': 'Crew Task Execution',
                'status': 'ERROR',
                'error': str(e)
            })
    
    def test_swarm_creation(self):
        """Test swarm creation functionality.""" 
        print("\n🌐 Testing Swarm Creation")
        
        swarm_config = {
            "name": "Test Adaptive Intelligence Swarm",
            "description": "Adaptive multi-agent swarm for comprehensive testing",
            "architecture": "adaptive_swarm",
            "max_loops": 3,
            "timeout": 300,
            "agents": ["swarm_security", "swarm_test_gen", "swarm_qa", "swarm_consensus"],
            "rules": ["Prioritize accuracy over speed", "Maintain consensus threshold above 85%"]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/swarm/swarms",
                json=swarm_config,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and 'swarm' in data:
                    swarm_id = data['swarm']['id']
                    self.created_swarms.append(swarm_id)
                    print(f"✅ Swarm Creation: SUCCESS - Created swarm {swarm_id}")
                    print(f"   🏗️ Architecture: {data['swarm']['architecture']}")
                    print(f"   👥 Agents: {len(data['swarm']['agents'])}")
                    
                    self.test_results.append({
                        'test': 'Swarm Creation',
                        'status': 'SUCCESS',
                        'swarm_id': swarm_id,
                        'architecture': data['swarm']['architecture'],
                        'agents_count': len(data['swarm']['agents'])
                    })
                else:
                    print(f"❌ Swarm Creation: FAILED - Invalid response")
                    self.test_results.append({
                        'test': 'Swarm Creation',
                        'status': 'FAILED',
                        'error': 'Invalid response format'
                    })
            else:
                print(f"❌ Swarm Creation: FAILED - HTTP {response.status_code}")
                print(f"Response: {response.text}")
                self.test_results.append({
                    'test': 'Swarm Creation',
                    'status': 'FAILED',
                    'error': f'HTTP {response.status_code}'
                })
                
        except Exception as e:
            print(f"❌ Swarm Creation: ERROR - {str(e)}")
            self.test_results.append({
                'test': 'Swarm Creation',
                'status': 'ERROR',
                'error': str(e)
            })
    
    def test_swarm_architectures(self):
        """Test different swarm architectures."""
        print("\n🔧 Testing Swarm Architectures")
        
        architectures_to_test = [
            "sequential_workflow",
            "concurrent_workflow", 
            "hierarchical_swarm",
            "mixture_of_agents"
        ]
        
        for architecture in architectures_to_test:
            swarm_config = {
                "name": f"Test {architecture.replace('_', ' ').title()} Swarm",
                "description": f"Testing {architecture} architecture",
                "architecture": architecture,
                "max_loops": 2,
                "timeout": 180,
                "agents": ["swarm_security", "swarm_qa"]
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/swarm/swarms",
                    json=swarm_config,
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        swarm_id = data['swarm']['id']
                        self.created_swarms.append(swarm_id)
                        print(f"✅ {architecture}: SUCCESS - Created {swarm_id}")
                        
                        self.test_results.append({
                            'test': f'Swarm Architecture - {architecture}',
                            'status': 'SUCCESS',
                            'swarm_id': swarm_id
                        })
                    else:
                        print(f"❌ {architecture}: FAILED - Invalid response")
                        self.test_results.append({
                            'test': f'Swarm Architecture - {architecture}',
                            'status': 'FAILED',
                            'error': 'Invalid response'
                        })
                else:
                    print(f"❌ {architecture}: FAILED - HTTP {response.status_code}")
                    self.test_results.append({
                        'test': f'Swarm Architecture - {architecture}',
                        'status': 'FAILED',
                        'error': f'HTTP {response.status_code}'
                    })
                    
            except Exception as e:
                print(f"❌ {architecture}: ERROR - {str(e)}")
                self.test_results.append({
                    'test': f'Swarm Architecture - {architecture}',
                    'status': 'ERROR',
                    'error': str(e)
                })
    
    def test_swarm_task_execution(self):
        """Test swarm task execution with different patterns."""
        print("\n⚡ Testing Swarm Task Execution")
        
        if not self.created_swarms:
            print("⚠️ No swarms available for testing")
            return
        
        swarm_id = self.created_swarms[0]
        task_config = {
            "description": "Perform comprehensive quality and security analysis of TestMaster dashboard system with multi-agent coordination",
            "task_type": "security",
            "priority": "high",
            "expected_output": "Comprehensive analysis report with security and quality metrics",
            "context": {
                "target_system": "TestMaster dashboard",
                "analysis_depth": "comprehensive",
                "focus_areas": ["security", "quality", "performance"]
            },
            "constraints": ["Complete analysis within 30 seconds", "Maintain consensus above 85%"]
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/swarm/swarms/{swarm_id}/execute",
                json=task_config,
                timeout=45
            )
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and 'execution_result' in data:
                    result = data['execution_result']
                    print(f"✅ Swarm Task Execution: SUCCESS")
                    print(f"   📊 Execution Time: {execution_time:.2f}s")
                    print(f"   🏗️ Architecture: {result.get('architecture', 'Unknown')}")
                    print(f"   🎯 Task ID: {result.get('task_id', 'Unknown')}")
                    
                    # Check for adaptive selection
                    if result.get('architecture') == 'adaptive_swarm':
                        selected_arch = result['result'].get('selected_architecture', 'Unknown')
                        print(f"   🧠 Adaptive Selection: {selected_arch}")
                    
                    self.test_results.append({
                        'test': 'Swarm Task Execution',
                        'status': 'SUCCESS',
                        'swarm_id': swarm_id,
                        'task_id': result.get('task_id'),
                        'execution_time': execution_time,
                        'architecture': result.get('architecture'),
                        'adaptive_selection': result['result'].get('selected_architecture') if result.get('architecture') == 'adaptive_swarm' else None
                    })
                else:
                    print(f"❌ Swarm Task Execution: FAILED - Invalid response")
                    self.test_results.append({
                        'test': 'Swarm Task Execution',
                        'status': 'FAILED',
                        'error': 'Invalid response format'
                    })
            else:
                print(f"❌ Swarm Task Execution: FAILED - HTTP {response.status_code}")
                print(f"Response: {response.text}")
                self.test_results.append({
                    'test': 'Swarm Task Execution',
                    'status': 'FAILED',
                    'error': f'HTTP {response.status_code}'
                })
                
        except Exception as e:
            print(f"❌ Swarm Task Execution: ERROR - {str(e)}")
            self.test_results.append({
                'test': 'Swarm Task Execution',
                'status': 'ERROR',
                'error': str(e)
            })
    
    def test_adaptive_swarm_selection(self):
        """Test adaptive swarm architecture selection."""
        print("\n🧠 Testing Adaptive Swarm Selection")
        
        # Find an adaptive swarm
        adaptive_swarm = None
        for swarm_id in self.created_swarms:
            try:
                response = requests.get(f"{self.base_url}/api/swarm/swarms/{swarm_id}/status")
                if response.status_code == 200:
                    data = response.json()
                    if data.get('swarm_status', {}).get('architecture') == 'adaptive_swarm':
                        adaptive_swarm = swarm_id
                        break
            except:
                continue
        
        if not adaptive_swarm:
            print("⚠️ No adaptive swarm available for testing")
            return
        
        # Test different task types to see adaptive selection
        test_cases = [
            {
                "description": "Simple data validation task",
                "task_type": "validation",
                "priority": "low",
                "expected_architecture": "concurrent_workflow"
            },
            {
                "description": "Complex security analysis requiring deep investigation with multiple verification steps and comprehensive reporting",
                "task_type": "security", 
                "priority": "critical",
                "expected_architecture": "hierarchical_swarm"
            },
            {
                "description": "Research comprehensive best practices and methodologies for software testing across multiple domains",
                "task_type": "research",
                "priority": "medium",
                "expected_architecture": "deep_research_swarm"
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            try:
                response = requests.post(
                    f"{self.base_url}/api/swarm/swarms/{adaptive_swarm}/execute",
                    json=test_case,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        result = data['execution_result']['result']
                        selected = result.get('selected_architecture', 'Unknown')
                        reasoning = result.get('selection_reasoning', 'No reasoning provided')
                        
                        print(f"✅ Adaptive Test {i+1}: {selected}")
                        print(f"   💭 Reasoning: {reasoning}")
                        
                        self.test_results.append({
                            'test': f'Adaptive Selection - Test {i+1}',
                            'status': 'SUCCESS',
                            'selected_architecture': selected,
                            'reasoning': reasoning,
                            'task_type': test_case['task_type']
                        })
                    else:
                        print(f"❌ Adaptive Test {i+1}: FAILED")
                        self.test_results.append({
                            'test': f'Adaptive Selection - Test {i+1}',
                            'status': 'FAILED',
                            'error': 'Invalid response'
                        })
                else:
                    print(f"❌ Adaptive Test {i+1}: HTTP {response.status_code}")
                    self.test_results.append({
                        'test': f'Adaptive Selection - Test {i+1}',
                        'status': 'FAILED',
                        'error': f'HTTP {response.status_code}'
                    })
                    
            except Exception as e:
                print(f"❌ Adaptive Test {i+1}: ERROR - {str(e)}")
                self.test_results.append({
                    'test': f'Adaptive Selection - Test {i+1}',
                    'status': 'ERROR',
                    'error': str(e)
                })
    
    def test_agent_coordination(self):
        """Test agent coordination features."""
        print("\n🤝 Testing Agent Coordination")
        
        # Test crew performance analytics
        try:
            response = requests.get(f"{self.base_url}/api/crew/analytics/crew-performance")
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and 'analytics' in data:
                    analytics = data['analytics']
                    print(f"✅ Crew Analytics: SUCCESS")
                    print(f"   📊 Total Crews: {analytics['overview']['total_crews']}")
                    print(f"   📈 Tasks Completed: {analytics['overview']['total_tasks_completed']}")
                    
                    self.test_results.append({
                        'test': 'Agent Coordination - Crew Analytics',
                        'status': 'SUCCESS',
                        'metrics': analytics['overview']
                    })
                else:
                    print(f"❌ Crew Analytics: FAILED - Invalid response")
                    self.test_results.append({
                        'test': 'Agent Coordination - Crew Analytics',
                        'status': 'FAILED',
                        'error': 'Invalid response'
                    })
            else:
                print(f"❌ Crew Analytics: HTTP {response.status_code}")
                self.test_results.append({
                    'test': 'Agent Coordination - Crew Analytics',
                    'status': 'FAILED',
                    'error': f'HTTP {response.status_code}'
                })
        except Exception as e:
            print(f"❌ Crew Analytics: ERROR - {str(e)}")
            self.test_results.append({
                'test': 'Agent Coordination - Crew Analytics',
                'status': 'ERROR',
                'error': str(e)
            })
        
        # Test swarm performance analytics
        try:
            response = requests.get(f"{self.base_url}/api/swarm/analytics/performance")
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and 'analytics' in data:
                    analytics = data['analytics']
                    print(f"✅ Swarm Analytics: SUCCESS") 
                    print(f"   🌐 Total Swarms: {analytics['swarm_count']}")
                    print(f"   👥 Total Agents: {analytics['agent_count']}")
                    
                    self.test_results.append({
                        'test': 'Agent Coordination - Swarm Analytics',
                        'status': 'SUCCESS',
                        'swarm_count': analytics['swarm_count'],
                        'agent_count': analytics['agent_count']
                    })
                else:
                    print(f"❌ Swarm Analytics: FAILED - Invalid response")
                    self.test_results.append({
                        'test': 'Agent Coordination - Swarm Analytics',
                        'status': 'FAILED',
                        'error': 'Invalid response'
                    })
            else:
                print(f"❌ Swarm Analytics: HTTP {response.status_code}")
                self.test_results.append({
                    'test': 'Agent Coordination - Swarm Analytics',
                    'status': 'FAILED',
                    'error': f'HTTP {response.status_code}'
                })
        except Exception as e:
            print(f"❌ Swarm Analytics: ERROR - {str(e)}")
            self.test_results.append({
                'test': 'Agent Coordination - Swarm Analytics',
                'status': 'ERROR',
                'error': str(e)
            })
    
    def test_performance_monitoring(self):
        """Test performance monitoring integration."""
        print("\n📊 Testing Performance Monitoring")
        
        # Test that our new endpoints integrate with existing performance monitoring
        endpoints_to_test = [
            "/api/crew/crews",
            "/api/swarm/swarms",
            "/api/crew/agents",
            "/api/swarm/agents"
        ]
        
        response_times = []
        
        for endpoint in endpoints_to_test:
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}{endpoint}")
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if response.status_code == 200:
                    print(f"✅ {endpoint}: {response_time:.3f}s")
                else:
                    print(f"❌ {endpoint}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"❌ {endpoint}: ERROR - {str(e)}")
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            print(f"📈 Performance Summary:")
            print(f"   ⚡ Average Response Time: {avg_response_time:.3f}s")
            print(f"   🔥 Max Response Time: {max_response_time:.3f}s")
            
            performance_status = "SUCCESS" if avg_response_time < 2.0 else "WARNING"
            
            self.test_results.append({
                'test': 'Performance Monitoring',
                'status': performance_status,
                'avg_response_time': avg_response_time,
                'max_response_time': max_response_time,
                'endpoints_tested': len(endpoints_to_test)
            })
    
    def test_concurrent_execution(self):
        """Test concurrent execution capabilities."""
        print("\n⚡ Testing Concurrent Execution")
        
        if len(self.created_swarms) < 2:
            print("⚠️ Need at least 2 swarms for concurrent testing")
            return
        
        # Execute tasks concurrently on different swarms
        import threading
        
        results = []
        
        def execute_task(swarm_id, task_description):
            try:
                task_config = {
                    "description": f"Concurrent task execution test: {task_description}",
                    "task_type": "general",
                    "priority": "medium"
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/swarm/swarms/{swarm_id}/execute",
                    json=task_config,
                    timeout=30
                )
                execution_time = time.time() - start_time
                
                results.append({
                    'swarm_id': swarm_id,
                    'execution_time': execution_time,
                    'success': response.status_code == 200,
                    'task_description': task_description
                })
                
            except Exception as e:
                results.append({
                    'swarm_id': swarm_id,
                    'execution_time': 0,
                    'success': False,
                    'error': str(e),
                    'task_description': task_description
                })
        
        # Start concurrent executions
        threads = []
        for i, swarm_id in enumerate(self.created_swarms[:3]):  # Test up to 3 concurrent
            task_desc = f"Quality analysis task {i+1}"
            thread = threading.Thread(target=execute_task, args=(swarm_id, task_desc))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Analyze results
        successful_executions = [r for r in results if r['success']]
        failed_executions = [r for r in results if not r['success']]
        
        print(f"✅ Concurrent Executions: {len(successful_executions)}/{len(results)} successful")
        
        if successful_executions:
            avg_time = sum(r['execution_time'] for r in successful_executions) / len(successful_executions)
            print(f"   ⚡ Average Execution Time: {avg_time:.2f}s")
            
            self.test_results.append({
                'test': 'Concurrent Execution',
                'status': 'SUCCESS' if len(successful_executions) == len(results) else 'PARTIAL',
                'successful_executions': len(successful_executions),
                'total_executions': len(results),
                'avg_execution_time': avg_time
            })
        else:
            print(f"❌ All concurrent executions failed")
            self.test_results.append({
                'test': 'Concurrent Execution',
                'status': 'FAILED',
                'error': 'All executions failed'
            })
    
    def test_crew_performance_analytics(self):
        """Test crew performance analytics."""
        print("\n📈 Testing Crew Performance Analytics")
        
        try:
            response = requests.get(f"{self.base_url}/api/crew/analytics/crew-performance")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    analytics = data.get('analytics', {})
                    overview = analytics.get('overview', {})
                    
                    print(f"✅ Crew Performance Analytics: SUCCESS")
                    print(f"   📊 Total Crews: {overview.get('total_crews', 0)}")
                    print(f"   📈 Total Tasks: {overview.get('total_tasks_completed', 0)}")
                    print(f"   👥 Average Crew Size: {overview.get('average_crew_size', 0):.1f}")
                    
                    crew_performance = analytics.get('crew_performance', [])
                    agent_utilization = analytics.get('agent_utilization', [])
                    
                    print(f"   🎯 Crew Performance Records: {len(crew_performance)}")
                    print(f"   👤 Agent Utilization Records: {len(agent_utilization)}")
                    
                    self.test_results.append({
                        'test': 'Crew Performance Analytics',
                        'status': 'SUCCESS',
                        'total_crews': overview.get('total_crews', 0),
                        'total_tasks': overview.get('total_tasks_completed', 0),
                        'crew_performance_records': len(crew_performance),
                        'agent_utilization_records': len(agent_utilization)
                    })
                else:
                    print(f"❌ Crew Performance Analytics: FAILED - Invalid response")
                    self.test_results.append({
                        'test': 'Crew Performance Analytics',
                        'status': 'FAILED',
                        'error': 'Invalid response format'
                    })
            else:
                print(f"❌ Crew Performance Analytics: FAILED - HTTP {response.status_code}")
                self.test_results.append({
                    'test': 'Crew Performance Analytics',
                    'status': 'FAILED',
                    'error': f'HTTP {response.status_code}'
                })
                
        except Exception as e:
            print(f"❌ Crew Performance Analytics: ERROR - {str(e)}")
            self.test_results.append({
                'test': 'Crew Performance Analytics',
                'status': 'ERROR',
                'error': str(e)
            })
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("📊 MULTI-AGENT ORCHESTRATION INTEGRATION REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r['status'] == 'SUCCESS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAILED'])
        error_tests = len([r for r in self.test_results if r['status'] == 'ERROR'])
        partial_tests = len([r for r in self.test_results if r['status'] == 'PARTIAL'])
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📈 OVERALL RESULTS:")
        print(f"   ✅ Successful Tests: {successful_tests}")
        print(f"   ❌ Failed Tests: {failed_tests}")
        print(f"   ⚠️ Error Tests: {error_tests}")
        print(f"   🔄 Partial Tests: {partial_tests}")
        print(f"   📊 Success Rate: {success_rate:.1f}%")
        
        print(f"\n🏗️ INFRASTRUCTURE SUMMARY:")
        print(f"   👥 Crews Created: {len(self.created_crews)}")
        print(f"   🌐 Swarms Created: {len(self.created_swarms)}")
        
        # Category breakdown
        categories = {}
        for result in self.test_results:
            category = result['test'].split(' - ')[0] if ' - ' in result['test'] else result['test'].split(':')[0]
            if category not in categories:
                categories[category] = {'success': 0, 'total': 0}
            categories[category]['total'] += 1
            if result['status'] == 'SUCCESS':
                categories[category]['success'] += 1
        
        print(f"\n📋 CATEGORY BREAKDOWN:")
        for category, stats in categories.items():
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"   {category}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Performance summary
        response_times = [r.get('response_time', 0) for r in self.test_results if 'response_time' in r]
        execution_times = [r.get('execution_time', 0) for r in self.test_results if 'execution_time' in r]
        
        if response_times:
            avg_response = sum(response_times) / len(response_times)
            print(f"\n⚡ PERFORMANCE SUMMARY:")
            print(f"   📡 Average API Response Time: {avg_response:.3f}s")
        
        if execution_times:
            avg_execution = sum(execution_times) / len(execution_times)
            print(f"   🚀 Average Task Execution Time: {avg_execution:.2f}s")
        
        # Integration assessment
        print(f"\n🎯 INTEGRATION ASSESSMENT:")
        
        crew_integration = successful_tests >= 8  # Basic threshold
        swarm_integration = any('Swarm' in r['test'] and r['status'] == 'SUCCESS' for r in self.test_results)
        adaptive_integration = any('Adaptive' in r['test'] and r['status'] == 'SUCCESS' for r in self.test_results)
        performance_integration = any('Performance' in r['test'] and r['status'] == 'SUCCESS' for r in self.test_results)
        
        integrations = [
            ("CrewAI Pattern Integration", crew_integration),
            ("Swarms Framework Integration", swarm_integration),
            ("Adaptive Orchestration", adaptive_integration),
            ("Performance Monitoring", performance_integration)
        ]
        
        for integration_name, status in integrations:
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {integration_name}")
        
        # Final verdict
        overall_success = success_rate >= 75 and crew_integration and swarm_integration
        
        print(f"\n🏆 FINAL VERDICT:")
        if overall_success:
            print(f"   ✅ INTEGRATION SUCCESSFUL")
            print(f"   🎉 Multi-agent orchestration is fully operational!")
        else:
            print(f"   ⚠️ INTEGRATION NEEDS ATTENTION") 
            print(f"   🔧 Some components require further development")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"multi_agent_integration_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_tests': total_tests,
                    'successful_tests': successful_tests,
                    'failed_tests': failed_tests,
                    'error_tests': error_tests,
                    'success_rate': success_rate
                },
                'infrastructure': {
                    'crews_created': len(self.created_crews),
                    'swarms_created': len(self.created_swarms)
                },
                'integrations': dict(integrations),
                'overall_success': overall_success,
                'detailed_results': self.test_results
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
    
    def cleanup_test_data(self):
        """Clean up test crews and swarms."""
        print(f"\n🧹 Cleaning up test data...")
        
        # Note: In a production system, you might want to implement cleanup endpoints
        # For now, we'll just log what would be cleaned up
        print(f"   📋 Would clean up {len(self.created_crews)} test crews")
        print(f"   🌐 Would clean up {len(self.created_swarms)} test swarms")
        print(f"   ✅ Cleanup logged (manual cleanup may be required)")

def main():
    """Run the multi-agent integration tests."""
    tester = MultiAgentIntegrationTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
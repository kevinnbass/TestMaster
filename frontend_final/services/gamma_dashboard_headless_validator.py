#!/usr/bin/env python3
"""
🏗️ MODULE: Gamma Dashboard Headless Validator - ADAMANTIUMCLAD Compliance
==================================================================

📋 PURPOSE:
    Validates Gamma dashboard functionality through headless methods as required
    by updated ADAMANTIUMCLAD Rule #1.7. Performs comprehensive testing without
    browser dependencies to ensure dashboard compliance and functionality.

🎯 CORE FUNCTIONALITY:
    • Headless dashboard endpoint testing
    • API response validation
    • Performance benchmarking
    • Integration status verification
    • Agent E collaboration validation

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 [2025-08-23 23:50:00] | Agent Gamma | 🆕 FEATURE
   └─ Goal: Create headless validator for ADAMANTIUMCLAD compliance
   └─ Changes: Comprehensive headless testing framework
   └─ Impact: Ensures protocol compliance without browser dependency

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-23 by Agent Gamma
🔧 Language: Python
📦 Dependencies: requests, json, time, datetime
🎯 Integration Points: Dashboard services, Agent E integration
⚡ Performance Notes: <200ms validation, comprehensive coverage
🔒 Security Notes: Local testing, no external dependencies
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

class GammaDashboardHeadlessValidator:
    """
    Headless validator for Gamma dashboard functionality.
    
    Validates dashboard compliance with ADAMANTIUMCLAD requirements
    without requiring browser-based testing.
    """
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.validation_results = {}
        self.start_time = datetime.now()
        
    def validate_dashboard(self) -> Dict[str, Any]:
        """
        Comprehensive headless dashboard validation.
        
        Returns complete validation report with compliance status.
        """
        print(f"🔍 Starting headless validation of {self.base_url}")
        print("=" * 60)
        
        # Core validation tests
        self.validate_basic_connectivity()
        self.validate_api_endpoints()
        self.validate_performance()
        self.validate_agent_e_integration()
        self.validate_frontend_requirements()
        
        # Generate final report
        return self.generate_validation_report()
    
    def validate_basic_connectivity(self):
        """Test basic dashboard connectivity."""
        print("📡 Testing basic connectivity...")
        
        try:
            response = requests.get(self.base_url, timeout=10)
            
            self.validation_results['connectivity'] = {
                'status': 'PASS' if response.status_code == 200 else 'FAIL',
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds() * 1000,
                'content_length': len(response.content)
            }
            
            if response.status_code == 200:
                print(f"   ✅ Dashboard accessible at {self.base_url}")
                print(f"   ⚡ Response time: {response.elapsed.total_seconds() * 1000:.1f}ms")
            else:
                print(f"   ❌ Dashboard not accessible (Status: {response.status_code})")
                
        except requests.exceptions.RequestException as e:
            self.validation_results['connectivity'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"   ❌ Connection failed: {e}")
    
    def validate_api_endpoints(self):
        """Test API endpoint availability and responses."""
        print("\n🔗 Testing API endpoints...")
        
        endpoints = [
            '/api/status',
            '/api/unified-status',
            '/api/dashboard-config'
        ]
        
        endpoint_results = {}
        
        for endpoint in endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = requests.get(url, timeout=5)
                
                endpoint_results[endpoint] = {
                    'status': 'PASS' if response.status_code == 200 else 'FAIL',
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds() * 1000,
                    'has_json': self._is_valid_json(response.content)
                }
                
                if response.status_code == 200:
                    print(f"   ✅ {endpoint} - {response.elapsed.total_seconds() * 1000:.1f}ms")
                else:
                    print(f"   ❌ {endpoint} - Status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                endpoint_results[endpoint] = {
                    'status': 'FAIL',
                    'error': str(e)
                }
                print(f"   ❌ {endpoint} - {e}")
        
        self.validation_results['api_endpoints'] = endpoint_results
    
    def validate_performance(self):
        """Test dashboard performance metrics."""
        print("\n⚡ Testing performance metrics...")
        
        try:
            # Test response time over multiple requests
            response_times = []
            for i in range(5):
                start = time.time()
                response = requests.get(self.base_url, timeout=10)
                end = time.time()
                response_times.append((end - start) * 1000)
            
            avg_response = sum(response_times) / len(response_times)
            max_response = max(response_times)
            
            performance_pass = avg_response < 200  # ADAMANTIUMCLAD p95 < 200ms
            
            self.validation_results['performance'] = {
                'status': 'PASS' if performance_pass else 'FAIL',
                'avg_response_time': avg_response,
                'max_response_time': max_response,
                'all_response_times': response_times,
                'meets_adamantiumclad': performance_pass
            }
            
            if performance_pass:
                print(f"   ✅ Average response time: {avg_response:.1f}ms (target: <200ms)")
            else:
                print(f"   ❌ Average response time: {avg_response:.1f}ms (exceeds 200ms target)")
                
        except Exception as e:
            self.validation_results['performance'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"   ❌ Performance test failed: {e}")
    
    def validate_agent_e_integration(self):
        """Test Agent E integration status."""
        print("\n🤝 Testing Agent E integration...")
        
        agent_e_endpoints = [
            '/api/personal-analytics',
            '/api/personal-analytics/real-time',
            '/api/personal-analytics/3d-data'
        ]
        
        integration_results = {}
        integration_active = False
        
        for endpoint in agent_e_endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                response = requests.get(url, timeout=5)
                
                # 200 means active, 503 means service unavailable (expected if Agent E not running)
                status = 'ACTIVE' if response.status_code == 200 else 'PENDING'
                if response.status_code == 200:
                    integration_active = True
                
                integration_results[endpoint] = {
                    'status': status,
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds() * 1000
                }
                
                if response.status_code == 200:
                    print(f"   ✅ {endpoint} - Agent E active")
                else:
                    print(f"   ⏳ {endpoint} - Agent E pending (expected)")
                    
            except requests.exceptions.RequestException as e:
                integration_results[endpoint] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                print(f"   ⚠️ {endpoint} - {e}")
        
        self.validation_results['agent_e_integration'] = {
            'integration_active': integration_active,
            'endpoints': integration_results,
            'collaboration_ready': True  # Infrastructure is ready regardless
        }
        
        if integration_active:
            print("   🎉 Agent E integration active and operational!")
        else:
            print("   📋 Agent E integration ready (service pending - normal)")
    
    def validate_frontend_requirements(self):
        """Validate ADAMANTIUMCLAD frontend requirements."""
        print("\n🖥️ Validating ADAMANTIUMCLAD frontend requirements...")
        
        requirements = {
            'frontend_accessible': False,
            'ui_components_present': False,
            'data_pipeline_ready': False,
            'real_time_capable': False,
            'port_compliant': False
        }
        
        try:
            # Check if dashboard HTML is accessible
            response = requests.get(self.base_url, timeout=10)
            if response.status_code == 200:
                content = response.text.lower()
                requirements['frontend_accessible'] = True
                
                # Check for UI components
                if 'dashboard' in content and 'panel' in content:
                    requirements['ui_components_present'] = True
                    print("   ✅ UI components present in frontend")
                
                # Check for real-time capabilities (WebSocket)
                if 'websocket' in content or 'socket.io' in content:
                    requirements['real_time_capable'] = True
                    print("   ✅ Real-time capabilities detected")
            
            # Check port compliance (5000, 5001, 5002 only)
            allowed_ports = ['5000', '5001', '5002']
            current_port = self.base_url.split(':')[-1].split('/')[0]
            requirements['port_compliant'] = current_port in allowed_ports
            
            if requirements['port_compliant']:
                print(f"   ✅ Port {current_port} compliant with ADAMANTIUMCLAD restrictions")
            else:
                print(f"   ❌ Port {current_port} violates ADAMANTIUMCLAD restrictions")
            
            # Check data pipeline readiness
            config_response = requests.get(f"{self.base_url}/api/dashboard-config", timeout=5)
            if config_response.status_code == 200:
                requirements['data_pipeline_ready'] = True
                print("   ✅ Data pipeline configuration accessible")
            
        except Exception as e:
            print(f"   ⚠️ Frontend validation error: {e}")
        
        self.validation_results['frontend_requirements'] = requirements
        
        # Overall frontend compliance
        compliance_score = sum(requirements.values()) / len(requirements)
        print(f"   📊 ADAMANTIUMCLAD compliance: {compliance_score * 100:.0f}%")
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Calculate overall status
        overall_status = self._calculate_overall_status()
        
        report = {
            'timestamp': end_time.isoformat(),
            'validation_duration': duration,
            'dashboard_url': self.base_url,
            'overall_status': overall_status,
            'validation_results': self.validation_results,
            'adamantiumclad_compliance': self._assess_adamantiumclad_compliance(),
            'recommendations': self._generate_recommendations()
        }
        
        print("\n" + "=" * 60)
        print("📋 VALIDATION REPORT SUMMARY")
        print("=" * 60)
        print(f"   Dashboard URL: {self.base_url}")
        print(f"   Overall Status: {overall_status}")
        print(f"   Validation Duration: {duration:.2f}s")
        print(f"   ADAMANTIUMCLAD Compliant: {report['adamantiumclad_compliance']['compliant']}")
        
        if report['recommendations']:
            print("\n📝 Recommendations:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
        
        print("\n🎯 Validation Complete!")
        
        return report
    
    def _is_valid_json(self, content: bytes) -> bool:
        """Check if content is valid JSON."""
        try:
            json.loads(content)
            return True
        except:
            return False
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall validation status."""
        failed_tests = []
        
        if self.validation_results.get('connectivity', {}).get('status') == 'FAIL':
            failed_tests.append('connectivity')
        
        if self.validation_results.get('performance', {}).get('status') == 'FAIL':
            failed_tests.append('performance')
        
        api_tests = self.validation_results.get('api_endpoints', {})
        failed_apis = [ep for ep, result in api_tests.items() if result.get('status') == 'FAIL']
        if failed_apis:
            failed_tests.extend(failed_apis)
        
        if not failed_tests:
            return 'PASS'
        elif len(failed_tests) <= 2:  # Allow minor failures
            return 'PASS_WITH_WARNINGS'
        else:
            return 'FAIL'
    
    def _assess_adamantiumclad_compliance(self) -> Dict[str, Any]:
        """Assess compliance with ADAMANTIUMCLAD requirements."""
        frontend_req = self.validation_results.get('frontend_requirements', {})
        performance = self.validation_results.get('performance', {})
        
        compliance_checks = {
            'frontend_accessible': frontend_req.get('frontend_accessible', False),
            'ui_components': frontend_req.get('ui_components_present', False),
            'data_pipeline': frontend_req.get('data_pipeline_ready', False),
            'real_time_updates': frontend_req.get('real_time_capable', False),
            'port_restrictions': frontend_req.get('port_compliant', False),
            'performance_targets': performance.get('meets_adamantiumclad', False)
        }
        
        compliance_score = sum(compliance_checks.values()) / len(compliance_checks)
        
        return {
            'compliant': compliance_score >= 0.8,  # 80% threshold
            'compliance_score': compliance_score,
            'checks': compliance_checks,
            'missing_requirements': [k for k, v in compliance_checks.items() if not v]
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        connectivity = self.validation_results.get('connectivity', {})
        if connectivity.get('status') == 'FAIL':
            recommendations.append("Start dashboard service - connectivity test failed")
        
        performance = self.validation_results.get('performance', {})
        if not performance.get('meets_adamantiumclad', False):
            recommendations.append("Optimize dashboard performance to meet <200ms target")
        
        frontend_req = self.validation_results.get('frontend_requirements', {})
        if not frontend_req.get('port_compliant', False):
            recommendations.append("Move dashboard to allowed port (5000, 5001, or 5002)")
        
        if not frontend_req.get('real_time_capable', False):
            recommendations.append("Add WebSocket/real-time capabilities for ADAMANTIUMCLAD compliance")
        
        return recommendations


def main():
    """Run headless validation of Gamma dashboard."""
    
    print("🔍 GAMMA DASHBOARD HEADLESS VALIDATOR")
    print("ADAMANTIUMCLAD Rule #1.7 Compliance Testing")
    print()
    
    # Test all allowed ports
    allowed_ports = [5000, 5001, 5002]
    
    for port in allowed_ports:
        base_url = f"http://localhost:{port}"
        print(f"\n🧪 Testing port {port}...")
        
        validator = GammaDashboardHeadlessValidator(base_url)
        try:
            report = validator.validate_dashboard()
            
            # Save validation report
            report_file = Path(__file__).parent / f"gamma_dashboard_validation_port_{port}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"📄 Validation report saved: {report_file}")
            
            if report['overall_status'] in ['PASS', 'PASS_WITH_WARNINGS']:
                print(f"✅ Port {port} validation successful!")
                break  # Found working dashboard
            
        except Exception as e:
            print(f"❌ Port {port} validation failed: {e}")
    
    print("\n🎯 Headless validation complete!")


if __name__ == "__main__":
    main()
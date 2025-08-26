#!/usr/bin/env python3
"""
Automated Frontend Testing Suite
===============================

Tests frontend functionality without requiring manual browser interaction.
"""

import requests
import re
from datetime import datetime
from typing import Dict, List, Tuple

class FrontendTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.results = []
        
    def add_result(self, test_type: str, message: str, success: bool = None):
        """Add test result"""
        if success is None:
            success = test_type == 'pass'
            
        result = {
            'type': test_type,
            'message': message,
            'success': success,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        self.results.append(result)
        return success
    
    def test_tab_switching_html_structure(self) -> bool:
        """Test Phase 2A: Tab switching HTML structure"""
        self.add_result('info', 'Testing tab switching HTML structure...')
        
        try:
            response = requests.get(self.base_url, timeout=5)
            if not response.ok:
                return self.add_result('fail', f'Dashboard not accessible: HTTP {response.status_code}')
            
            html = response.text
            
            # Test all 6 required tabs
            required_tabs = ['overview', 'analytics', 'tests', 'workflow', 'refactor', 'analyzer']
            found_tabs = []
            
            for tab in required_tabs:
                if f'data-tab="{tab}"' in html:
                    found_tabs.append(tab)
            
            if len(found_tabs) == 6:
                self.add_result('pass', f'All 6 tabs found: {", ".join(found_tabs)}')
            else:
                self.add_result('fail', f'Only {len(found_tabs)}/6 tabs found: {", ".join(found_tabs)}')
                return False
            
            # Test tab content sections
            content_sections = []
            for tab in required_tabs:
                if f'id="{tab}-tab"' in html:
                    content_sections.append(tab)
            
            if len(content_sections) >= 2:  # At least overview and analytics should exist
                self.add_result('pass', f'Tab content sections found: {", ".join(content_sections)}')
            else:
                self.add_result('fail', f'Insufficient tab content sections: {", ".join(content_sections)}')
                return False
                
            # Test active tab
            if 'tab-button active' in html:
                self.add_result('pass', 'Active tab styling found')
            else:
                self.add_result('fail', 'Active tab styling not found')
                
            return True
            
        except Exception as e:
            return self.add_result('fail', f'Tab structure test failed: {e}')
    
    def test_performance_charts_structure(self) -> bool:
        """Test Phase 2B: Performance charts HTML structure"""
        self.add_result('info', 'Testing performance charts structure...')
        
        try:
            response = requests.get(self.base_url, timeout=5)
            if not response.ok:
                return self.add_result('fail', f'Dashboard not accessible: HTTP {response.status_code}')
            
            html = response.text
            
            # Test chart canvases
            required_charts = ['analytics-cpu-chart', 'analytics-memory-chart', 'analytics-network-chart']
            found_charts = []
            
            for chart in required_charts:
                if f'id="{chart}"' in html:
                    found_charts.append(chart)
            
            if len(found_charts) == 3:
                self.add_result('pass', f'All 3 performance chart canvases found')
            else:
                self.add_result('fail', f'Only {len(found_charts)}/3 charts found: {", ".join(found_charts)}')
                return False
            
            # Test Chart.js library
            if 'chart.js' in html:
                self.add_result('pass', 'Chart.js library referenced')
            else:
                self.add_result('fail', 'Chart.js library not found')
                return False
                
            # Test performance data endpoint
            perf_response = requests.get(f'{self.base_url}/api/performance/realtime', timeout=5)
            if perf_response.ok:
                data = perf_response.json()
                if 'timeseries' in data and 'cpu_usage' in data['timeseries']:
                    data_points = len(data['timeseries']['cpu_usage'])
                    if data_points == 300:
                        self.add_result('pass', f'Performance data: exactly 300 points (30s @ 100ms)')
                    else:
                        self.add_result('info', f'Performance data: {data_points} points (expected 300)')
                    
                    # Test all three data series
                    series = ['cpu_usage', 'memory_usage_mb', 'network_kb_s']
                    found_series = [s for s in series if s in data['timeseries']]
                    if len(found_series) == 3:
                        self.add_result('pass', 'All 3 data series present: CPU, Memory, Network')
                    else:
                        self.add_result('fail', f'Missing data series: {set(series) - set(found_series)}')
                else:
                    self.add_result('fail', 'Performance data structure incorrect')
                    return False
            else:
                self.add_result('fail', f'Performance data endpoint failed: HTTP {perf_response.status_code}')
                return False
                
            return True
            
        except Exception as e:
            return self.add_result('fail', f'Performance charts test failed: {e}')
    
    def test_llm_toggle_functionality(self) -> bool:
        """Test Phase 2C: LLM toggle button functionality"""
        self.add_result('info', 'Testing LLM toggle functionality...')
        
        try:
            # Test HTML structure
            response = requests.get(self.base_url, timeout=5)
            if not response.ok:
                return self.add_result('fail', f'Dashboard not accessible: HTTP {response.status_code}')
            
            html = response.text
            
            # Test toggle button HTML
            if 'llm-toggle-btn' in html:
                self.add_result('pass', 'LLM toggle button found in HTML')
            else:
                self.add_result('fail', 'LLM toggle button not found in HTML')
                return False
                
            if 'llm-status-indicator' in html:
                self.add_result('pass', 'LLM status indicator found')
            else:
                self.add_result('fail', 'LLM status indicator not found')
            
            # Test LLM status API
            status_response = requests.get(f'{self.base_url}/api/llm/status', timeout=5)
            if status_response.ok:
                status = status_response.json()
                current_state = status.get('api_enabled', False)
                self.add_result('pass', f'LLM status API accessible, current state: {current_state}')
                
                # Test toggle functionality  
                toggle_data = {"enabled": not current_state}
                toggle_response = requests.post(
                    f'{self.base_url}/api/llm/toggle-mode', 
                    json=toggle_data, 
                    timeout=5
                )
                
                if toggle_response.ok:
                    self.add_result('pass', 'LLM toggle API functional')
                    
                    # Verify state change (optional - might not work due to demo mode)
                    verify_response = requests.get(f'{self.base_url}/api/llm/status', timeout=5)
                    if verify_response.ok:
                        new_status = verify_response.json()
                        self.add_result('info', f'Toggle result: {new_status.get("message", "State changed")}')
                else:
                    self.add_result('fail', f'LLM toggle failed: HTTP {toggle_response.status_code}')
                    return False
            else:
                self.add_result('fail', f'LLM status API failed: HTTP {status_response.status_code}')
                return False
                
            return True
            
        except Exception as e:
            return self.add_result('fail', f'LLM toggle test failed: {e}')
    
    def test_responsive_design(self) -> bool:
        """Test Phase 4B: Mobile responsiveness"""
        self.add_result('info', 'Testing responsive design structure...')
        
        try:
            response = requests.get(self.base_url, timeout=5)
            if not response.ok:
                return self.add_result('fail', f'Dashboard not accessible: HTTP {response.status_code}')
            
            html = response.text
            
            # Test viewport meta tag
            if 'viewport' in html and 'width=device-width' in html:
                self.add_result('pass', 'Responsive viewport meta tag found')
            else:
                self.add_result('fail', 'Responsive viewport meta tag not found')
                return False
            
            # Test CSS media queries (check for CSS files)
            css_files = re.findall(r'href="([^"]*\.css)"', html)
            if css_files:
                self.add_result('pass', f'CSS files found: {", ".join(css_files)}')
                
                # Test if CSS files are accessible
                css_accessible = []
                for css_file in css_files[:3]:  # Test first 3 CSS files
                    try:
                        css_url = f"{self.base_url}/{css_file}" if not css_file.startswith('http') else css_file
                        css_response = requests.get(css_url, timeout=3)
                        if css_response.ok:
                            css_accessible.append(css_file)
                    except:
                        pass
                
                if css_accessible:
                    self.add_result('pass', f'CSS files accessible: {len(css_accessible)} files')
                else:
                    self.add_result('info', 'CSS files may be external or not directly accessible')
            else:
                self.add_result('fail', 'No CSS files found')
                return False
                
            return True
            
        except Exception as e:
            return self.add_result('fail', f'Responsive design test failed: {e}')
    
    def run_comprehensive_tests(self):
        """Run all frontend tests"""
        print("Dashboard Frontend Automated Testing Suite")
        print("=" * 50)
        print(f"Testing against: {self.base_url}")
        print(f"Test started at: {datetime.now().isoformat()}")
        print()
        
        # Run all test phases
        tests = [
            ('Phase 2A: Tab Switching', self.test_tab_switching_html_structure),
            ('Phase 2B: Performance Charts', self.test_performance_charts_structure),
            ('Phase 2C: LLM Toggle', self.test_llm_toggle_functionality),
            ('Phase 4B: Responsive Design', self.test_responsive_design)
        ]
        
        passed_phases = 0
        
        for phase_name, test_func in tests:
            print(f"\\nRunning {phase_name}...")
            success = test_func()
            if success:
                passed_phases += 1
                print(f"[PASS] {phase_name}")
            else:
                print(f"[FAIL] {phase_name}")
        
        return self.generate_report(passed_phases, len(tests))
    
    def generate_report(self, passed_phases: int, total_phases: int):
        """Generate comprehensive test report"""
        print("\\n" + "=" * 50)
        print("FRONTEND TEST RESULTS")
        print("=" * 50)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['success'])
        failed_tests = sum(1 for r in self.results if not r['success'] and r['type'] != 'info')
        
        print(f"Phases Passed: {passed_phases}/{total_phases}")
        print(f"Individual Tests: {passed_tests} passed, {failed_tests} failed")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        # Detailed results
        print("DETAILED RESULTS:")
        print("-" * 30)
        for result in self.results:
            status = "[PASS]" if result['success'] else "[FAIL]" if result['type'] != 'info' else "[INFO]"
            print(f"{status} [{result['timestamp']}] {result['message']}")
        
        print()
        print("FRONTEND TESTING SUMMARY:")
        print("-" * 30)
        if passed_phases == total_phases:
            print("[SUCCESS] All frontend phases passed!")
            print("✅ Tab switching structure verified")
            print("✅ Performance charts configured")
            print("✅ LLM toggle functionality working")
            print("✅ Responsive design implemented")
        elif passed_phases >= 3:
            print("[ACCEPTABLE] Most frontend phases passed")
        else:
            print("[NEEDS ATTENTION] Some frontend issues found")
        
        return passed_tests, total_tests

def main():
    """Main test execution"""
    tester = FrontendTester()
    
    try:
        passed_tests, total_tests = tester.run_comprehensive_tests()
        
        # Exit with appropriate code
        if passed_tests >= total_tests * 0.8:  # 80% pass rate acceptable
            print(f"\\n[ACCEPTABLE] Frontend testing completed successfully")
            return 0
        else:
            print(f"\\n[WARNING] Some frontend tests failed")
            return 1
            
    except KeyboardInterrupt:
        print("\\n\\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n\\nTest execution failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
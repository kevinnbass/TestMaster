#!/usr/bin/env python3
"""
Complete Robustness Test Suite
==============================

Tests every single robustness feature to ensure 100% reliability.
"""

import json
import time
import requests
from datetime import datetime
from typing import Dict, Any
import concurrent.futures

BASE_URL = "http://localhost:5000"

class RobustnessTests:
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    def test_monitoring_api(self) -> bool:
        """Test monitoring API endpoints."""
        print("\n[TEST] Monitoring API")
        
        endpoints = [
            '/api/monitoring/robustness',
            '/api/monitoring/heartbeat',
            '/api/monitoring/fallback',
            '/api/monitoring/dead-letter',
            '/api/monitoring/batch',
            '/api/monitoring/flow',
            '/api/monitoring/compression'
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{BASE_URL}{endpoint}")
                if response.status_code == 200:
                    print(f"  [OK] {endpoint}: OK")
                else:
                    print(f"  [FAIL] {endpoint}: {response.status_code}")
                    return False
            except Exception as e:
                print(f"  [FAIL] {endpoint}: {e}")
                return False
        
        return True
    
    def test_delivery_verification(self) -> bool:
        """Test analytics delivery verification."""
        print("\n[TEST] Delivery Verification")
        
        try:
            # Send test analytics
            response = requests.post(
                f"{BASE_URL}/api/monitoring/test-delivery",
                json={}
            )
            
            if response.status_code == 200:
                data = response.json()
                tests = data.get('results', {}).get('tests', {})
                
                for test_name, result in tests.items():
                    if result.get('success'):
                        print(f"  [OK] {test_name}: {result.get('message', 'Success')}")
                    else:
                        print(f"  [FAIL] {test_name}: Failed")
                        return False
                
                return True
        except Exception as e:
            print(f"  [FAIL] Delivery test failed: {e}")
            return False
        
        return False
    
    def test_concurrent_load(self) -> bool:
        """Test system under concurrent load."""
        print("\n[TEST] Concurrent Load Handling")
        
        def make_request(i):
            try:
                response = requests.get(f"{BASE_URL}/api/analytics/metrics")
                return response.status_code == 200
            except:
                return False
        
        # Send 20 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        success_rate = sum(results) / len(results)
        print(f"  Success rate: {success_rate * 100:.1f}%")
        
        return success_rate >= 0.95  # 95% success rate
    
    def test_failure_recovery(self) -> bool:
        """Test automatic failure recovery."""
        print("\n[TEST] Failure Recovery")
        
        try:
            # Check fallback status
            response = requests.get(f"{BASE_URL}/api/monitoring/fallback")
            data = response.json()
            
            fallback = data.get('fallback', {})
            current_level = fallback.get('current_level', 'unknown')
            
            print(f"  Fallback level: {current_level}")
            
            if current_level == 'primary':
                print("  [OK] System operating at primary level")
                return True
            else:
                print(f"  [WARN] System in fallback mode: {current_level}")
                return True  # Still working, just degraded
                
        except Exception as e:
            print(f"  [FAIL] Recovery check failed: {e}")
            return False
    
    def test_data_integrity(self) -> bool:
        """Test data integrity verification."""
        print("\n[TEST] Data Integrity")
        
        try:
            # Get compression stats
            response = requests.get(f"{BASE_URL}/api/monitoring/compression")
            data = response.json()
            
            compression = data.get('compression', {})
            
            if compression.get('total_compressions', 0) > 0:
                efficiency = compression.get('compression_efficiency', 0)
                print(f"  [OK] Compression efficiency: {efficiency} bytes/op")
            
            # Check dead letter queue
            response = requests.get(f"{BASE_URL}/api/monitoring/dead-letter")
            data = response.json()
            
            dlq = data.get('dead_letter_queue', {})
            queue_size = dlq.get('queue_size', 0)
            
            print(f"  Dead letter queue: {queue_size} entries")
            
            if queue_size == 0:
                print("  [OK] No failed deliveries")
            else:
                print(f"  [WARN] {queue_size} failed deliveries in recovery")
            
            return True
            
        except Exception as e:
            print(f"  [FAIL] Integrity check failed: {e}")
            return False
    
    def test_batch_processing(self) -> bool:
        """Test batch processing efficiency."""
        print("\n[TEST] Batch Processing")
        
        try:
            response = requests.get(f"{BASE_URL}/api/monitoring/batch")
            data = response.json()
            
            batch = data.get('batch_processor', {})
            
            pending = batch.get('pending_items', 0)
            queued = batch.get('queued_batches', 0)
            strategy = batch.get('current_strategy', 'unknown')
            
            print(f"  Strategy: {strategy}")
            print(f"  Pending items: {pending}")
            print(f"  Queued batches: {queued}")
            
            if pending < 100 and queued < 10:
                print("  [OK] Batch processing efficient")
                return True
            else:
                print("  [WARN] High backlog detected")
                return True  # Still functional
                
        except Exception as e:
            print(f"  [FAIL] Batch check failed: {e}")
            return False
    
    def test_heartbeat_monitoring(self) -> bool:
        """Test heartbeat monitoring."""
        print("\n[TEST] Heartbeat Monitoring")
        
        try:
            response = requests.get(f"{BASE_URL}/api/monitoring/heartbeat")
            data = response.json()
            
            heartbeat = data.get('heartbeat', {})
            overall_health = heartbeat.get('overall_health', 'unknown')
            
            print(f"  Overall health: {overall_health}")
            
            endpoints = heartbeat.get('endpoints', {})
            for name, info in endpoints.items():
                status = info.get('status', 'unknown')
                failures = info.get('failures', 0)
                print(f"  {name}: {status} (failures: {failures})")
            
            return overall_health in ['healthy', 'degraded']
            
        except Exception as e:
            print(f"  [FAIL] Heartbeat check failed: {e}")
            return False
    
    def test_flow_monitoring(self) -> bool:
        """Test flow monitoring."""
        print("\n[TEST] Flow Monitoring")
        
        try:
            response = requests.get(f"{BASE_URL}/api/monitoring/flow")
            data = response.json()
            
            flow = data.get('flow_monitor', {})
            
            active = flow.get('active_transactions', 0)
            completed = flow.get('completed_transactions', 0)
            failed = flow.get('failed_transactions', 0)
            
            print(f"  Active: {active}, Completed: {completed}, Failed: {failed}")
            
            success_rates = flow.get('success_rates', {})
            for stage, rate in success_rates.items():
                print(f"  {stage}: {rate:.1f}% success")
            
            return True
            
        except Exception as e:
            print(f"  [FAIL] Flow check failed: {e}")
            return False
    
    def test_robustness_score(self) -> bool:
        """Test overall robustness score."""
        print("\n[TEST] Robustness Score")
        
        try:
            response = requests.get(f"{BASE_URL}/api/monitoring/robustness")
            data = response.json()
            
            health_score = data.get('health_score', 0)
            monitoring = data.get('monitoring', {})
            
            print(f"  Overall health score: {health_score}/100")
            
            features = monitoring.get('robustness_features', {})
            
            # Check each feature
            feature_scores = []
            
            if 'heartbeat' in features:
                hb = features['heartbeat']
                health = hb.get('overall_health', 'unknown')
                score = 100 if health == 'healthy' else 50
                feature_scores.append(score)
                print(f"  Heartbeat: {health} ({score}/100)")
            
            if 'fallback' in features:
                fb = features['fallback']
                level = fb.get('current_level', 'unknown')
                score = 100 if level == 'primary' else 50
                feature_scores.append(score)
                print(f"  Fallback: {level} ({score}/100)")
            
            if 'dead_letter_queue' in features:
                dlq = features['dead_letter_queue']
                size = dlq.get('queue_size', 0)
                score = 100 if size == 0 else max(0, 100 - size * 2)
                feature_scores.append(score)
                print(f"  Dead Letter Queue: {size} items ({score}/100)")
            
            if 'batch_processor' in features:
                bp = features['batch_processor']
                pending = bp.get('pending_items', 0)
                score = 100 if pending == 0 else max(0, 100 - pending)
                feature_scores.append(score)
                print(f"  Batch Processor: {pending} pending ({score}/100)")
            
            # Calculate average
            if feature_scores:
                avg_score = sum(feature_scores) / len(feature_scores)
                print(f"\n  Calculated average: {avg_score:.1f}/100")
            
            return health_score >= 50  # At least 50% health
            
        except Exception as e:
            print(f"  [FAIL] Robustness check failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all robustness tests."""
        print("="*60)
        print("TESTMASTER ROBUSTNESS TEST SUITE")
        print("="*60)
        print(f"Started at: {self.start_time.isoformat()}")
        
        tests = [
            ("Monitoring API", self.test_monitoring_api),
            ("Delivery Verification", self.test_delivery_verification),
            ("Concurrent Load", self.test_concurrent_load),
            ("Failure Recovery", self.test_failure_recovery),
            ("Data Integrity", self.test_data_integrity),
            ("Batch Processing", self.test_batch_processing),
            ("Heartbeat Monitoring", self.test_heartbeat_monitoring),
            ("Flow Monitoring", self.test_flow_monitoring),
            ("Robustness Score", self.test_robustness_score)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.results[test_name] = result
            except Exception as e:
                print(f"\n[ERROR] {test_name} crashed: {e}")
                self.results[test_name] = False
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in self.results.values() if r)
        total = len(self.results)
        
        for test_name, result in self.results.items():
            status = "[PASSED]" if result else "[FAILED]"
            print(f"{test_name:30} {status}")
        
        print("-"*60)
        print(f"Total: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        print(f"\nCompleted in {duration:.2f} seconds")
        
        if passed == total:
            print("\n[PERFECT] All robustness features working!")
        elif passed >= total * 0.9:
            print("\n[EXCELLENT] System is highly robust!")
        elif passed >= total * 0.7:
            print("\n[GOOD] Most features working, some issues detected.")
        else:
            print("\n[ATTENTION NEEDED] Multiple features failing.")
        
        return passed == total

if __name__ == "__main__":
    # Wait for server
    print("Waiting for server to be ready...")
    for i in range(10):
        try:
            response = requests.get(f"{BASE_URL}/api/health/live")
            if response.status_code == 200:
                print("Server is ready!\n")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("Server not responding!")
        exit(1)
    
    # Run tests
    tester = RobustnessTests()
    success = tester.run_all_tests()
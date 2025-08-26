#!/usr/bin/env python3
"""
Debug integration tests to see why they're failing.
"""

import sys
import traceback

def test_cross_system_communication():
    """Debug cross-system communication test."""
    try:
        from integration.cross_system_communication import CrossSystemCommunication
        
        comm = CrossSystemCommunication()
        
        # Test message publishing/subscribing
        comm.subscribe("test_channel", lambda msg: print(f"Received: {msg}"))
        comm.publish("test_channel", {"test": "message"})
        
        # Test system registration
        comm.register_system("test_system", {"endpoint": "localhost:8080"})
        systems = comm.get_registered_systems()
        
        # Test health checks
        comm.send_health_check("test_system")
        
        # Test message routing
        comm.route_message("test_system", {"command": "status"})
        
        result = (hasattr(comm, 'publish') and 
                 hasattr(comm, 'subscribe') and
                 len(systems) >= 1)
        
        print(f"Cross-System Communication: {result}")
        print(f"  - Has publish: {hasattr(comm, 'publish')}")
        print(f"  - Has subscribe: {hasattr(comm, 'subscribe')}")
        print(f"  - Systems registered: {len(systems)}")
        return result
        
    except Exception as e:
        print(f"Cross-System Communication ERROR: {e}")
        traceback.print_exc()
        return False

def test_distributed_task_queue():
    """Debug distributed task queue test."""
    try:
        from integration.distributed_task_queue import DistributedTaskQueue
        
        queue = DistributedTaskQueue()
        
        # Test task submission
        task_id = queue.submit_task("test_task", {"param": "value"})
        
        # Test task status checking
        status = queue.get_task_status(task_id)
        
        # Test worker management
        queue.add_worker("worker_1", {"capacity": 10})
        workers = queue.get_active_workers()
        
        # Test task completion
        queue.complete_task(task_id, {"result": "success"})
        
        # Test queue statistics
        stats = queue.get_queue_statistics()
        
        result = (task_id is not None and
                 status is not None and
                 len(workers) >= 1 and
                 stats is not None)
        
        print(f"Distributed Task Queue: {result}")
        print(f"  - Task ID: {task_id}")
        print(f"  - Status: {status}")
        print(f"  - Workers: {len(workers)}")
        print(f"  - Stats: {stats is not None}")
        return result
        
    except Exception as e:
        print(f"Distributed Task Queue ERROR: {e}")
        traceback.print_exc()
        return False

def test_intelligent_caching_layer():
    """Debug intelligent caching test."""
    try:
        from integration.intelligent_caching_layer import IntelligentCachingLayer
        
        cache = IntelligentCachingLayer()
        
        # Test basic caching operations
        cache.set("test_key", "test_value", ttl=300)
        value = cache.get("test_key")
        
        # Test cache invalidation
        cache.invalidate("test_key")
        invalidated_value = cache.get("test_key")
        
        # Test cache statistics
        stats = cache.get_cache_statistics()
        
        # Test cache patterns
        cache.set_pattern("user:*", ttl=600)
        cache.set("user:123", {"name": "John"})
        
        # Test cache warming
        cache.warm_cache("frequent_queries", {"query1": "result1"})
        
        result = (value == "test_value" and
                 invalidated_value is None and
                 stats is not None)
        
        print(f"Intelligent Caching Layer: {result}")
        print(f"  - Value retrieved: {value == 'test_value'}")
        print(f"  - Invalidation worked: {invalidated_value is None}")
        print(f"  - Stats available: {stats is not None}")
        return result
        
    except Exception as e:
        print(f"Intelligent Caching Layer ERROR: {e}")
        traceback.print_exc()
        return False

def test_load_balancing_system():
    """Debug load balancing test."""
    try:
        from integration.load_balancing_system import LoadBalancingSystem
        
        lb = LoadBalancingSystem()
        
        # Test backend management
        lb.add_backend("backend1", {"host": "localhost", "port": 8001})
        lb.add_backend("backend2", {"host": "localhost", "port": 8002})
        backends = lb.get_active_backends()
        
        # Test request routing
        target = lb.route_request({"path": "/api/test"})
        
        # Test load statistics
        stats = lb.get_load_statistics()
        
        result = (len(backends) >= 2 and
                 target is not None and
                 stats is not None)
        
        print(f"Load Balancing System: {result}")
        print(f"  - Backends: {len(backends)}")
        print(f"  - Routing target: {target}")
        print(f"  - Stats available: {stats is not None}")
        return result
        
    except Exception as e:
        print(f"Load Balancing System ERROR: {e}")
        traceback.print_exc()
        return False

def main():
    """Debug all integration tests."""
    print("=" * 60)
    print("DEBUGGING INTEGRATION TESTS")
    print("=" * 60)
    
    test_cross_system_communication()
    print()
    test_distributed_task_queue()
    print()
    test_intelligent_caching_layer()
    print()
    test_load_balancing_system()
    
    print("=" * 60)

if __name__ == '__main__':
    main()
"""
Analytics Redundancy and Failover System
=========================================

Provides redundancy, failover mechanisms, and backup pathways to ensure
analytics data always reaches the dashboard even under failure conditions.

Author: TestMaster Team
"""

import logging
import time
import threading
import queue
import copy
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import json
import pickle
import os

logger = logging.getLogger(__name__)

class FailoverMode(Enum):
    ACTIVE_PASSIVE = "active_passive"
    ACTIVE_ACTIVE = "active_active"
    ROUND_ROBIN = "round_robin"
    PRIORITY_BASED = "priority_based"

class NodeStatus(Enum):
    ACTIVE = "active"
    PASSIVE = "passive"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"

@dataclass
class RedundancyNode:
    """Represents a redundant analytics processing node."""
    node_id: str
    priority: int
    status: NodeStatus
    last_heartbeat: datetime
    failure_count: int
    success_count: int
    processor_function: Optional[Callable] = None
    backup_data_path: Optional[str] = None
    
class AnalyticsRedundancyManager:
    """
    Manages redundancy and failover for analytics processing.
    """
    
    def __init__(self, failover_mode: FailoverMode = FailoverMode.ACTIVE_PASSIVE,
                 backup_directory: str = None):
        """
        Initialize redundancy manager.
        
        Args:
            failover_mode: Type of failover mechanism
            backup_directory: Directory for backup data storage
        """
        self.failover_mode = failover_mode
        self.backup_directory = backup_directory or os.path.join(
            os.path.dirname(__file__), '..', 'data', 'redundancy_backups'
        )
        
        # Ensure backup directory exists
        os.makedirs(self.backup_directory, exist_ok=True)
        
        # Redundancy nodes
        self.nodes = {}
        self.active_nodes = set()
        self.failed_nodes = set()
        
        # Data queuing and backup
        self.data_queue = queue.Queue(maxsize=10000)
        self.backup_queue = queue.Queue(maxsize=5000)
        self.processed_data_log = deque(maxlen=1000)
        
        # Failover configuration
        self.heartbeat_interval = 30  # seconds
        self.failure_threshold = 3
        self.recovery_timeout = 120  # seconds
        self.backup_interval = 60  # seconds
        
        # Threading
        self.redundancy_active = False
        self.heartbeat_thread = None
        self.failover_thread = None
        self.backup_thread = None
        
        # Statistics
        self.redundancy_stats = {
            'total_failovers': 0,
            'successful_recoveries': 0,
            'data_loss_events': 0,
            'backup_operations': 0,
            'queue_overflows': 0,
            'redundancy_violations': 0,
            'start_time': datetime.now()
        }
        
        # Callbacks
        self.failover_callbacks = []
        self.recovery_callbacks = []
        self.data_loss_callbacks = []
        
        logger.info(f"Analytics Redundancy Manager initialized: {failover_mode.value}")
    
    def register_node(self, node_id: str, processor_function: Callable,
                     priority: int = 1, backup_data_path: str = None) -> RedundancyNode:
        """
        Register a redundant analytics node.
        
        Args:
            node_id: Unique identifier for the node
            processor_function: Function that processes analytics data
            priority: Node priority (higher = more preferred)
            backup_data_path: Optional backup data path for this node
        
        Returns:
            Created redundancy node
        """
        node = RedundancyNode(
            node_id=node_id,
            priority=priority,
            status=NodeStatus.PASSIVE,
            last_heartbeat=datetime.now(),
            failure_count=0,
            success_count=0,
            processor_function=processor_function,
            backup_data_path=backup_data_path or os.path.join(
                self.backup_directory, f"{node_id}_backup.pkl"
            )
        )
        
        self.nodes[node_id] = node
        
        # Activate node based on failover mode
        if self.failover_mode == FailoverMode.ACTIVE_PASSIVE:
            if not self.active_nodes:
                self._activate_node(node_id)
        elif self.failover_mode == FailoverMode.ACTIVE_ACTIVE:
            self._activate_node(node_id)
        
        logger.info(f"Registered redundancy node: {node_id} (priority: {priority})")
        return node
    
    def start_redundancy(self):
        """Start redundancy monitoring and failover."""
        if self.redundancy_active:
            return
        
        self.redundancy_active = True
        
        # Start monitoring threads
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.failover_thread = threading.Thread(target=self._failover_loop, daemon=True)
        self.backup_thread = threading.Thread(target=self._backup_loop, daemon=True)
        
        self.heartbeat_thread.start()
        self.failover_thread.start()
        self.backup_thread.start()
        
        logger.info("Analytics redundancy monitoring started")
    
    def stop_redundancy(self):
        """Stop redundancy monitoring."""
        self.redundancy_active = False
        
        # Wait for threads to finish
        for thread in [self.heartbeat_thread, self.failover_thread, self.backup_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        logger.info("Analytics redundancy monitoring stopped")
    
    def process_analytics_data(self, data: Dict[str, Any], 
                             source: str = "unknown") -> Dict[str, Any]:
        """
        Process analytics data with redundancy and failover.
        
        Args:
            data: Analytics data to process
            source: Source identifier
        
        Returns:
            Processed analytics data
        """
        start_time = time.time()
        
        try:
            # Add to queue for backup
            self._queue_data_for_backup(data, source)
            
            # Process with active nodes
            result = self._process_with_redundancy(data, source)
            
            # Log successful processing
            self.processed_data_log.append({
                'timestamp': datetime.now(),
                'source': source,
                'processing_time': time.time() - start_time,
                'nodes_used': list(self.active_nodes),
                'success': True
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Analytics processing failed: {e}")
            
            # Log failure
            self.processed_data_log.append({
                'timestamp': datetime.now(),
                'source': source,
                'processing_time': time.time() - start_time,
                'nodes_used': list(self.active_nodes),
                'success': False,
                'error': str(e)
            })
            
            # Attempt recovery
            return self._attempt_data_recovery(data, source)
    
    def _process_with_redundancy(self, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Process data using redundant nodes."""
        if not self.active_nodes:
            raise Exception("No active redundancy nodes available")
        
        results = []
        processing_errors = []
        
        if self.failover_mode == FailoverMode.ACTIVE_PASSIVE:
            # Use highest priority active node
            primary_node_id = max(self.active_nodes, 
                                key=lambda nid: self.nodes[nid].priority)
            result = self._process_with_node(primary_node_id, data, source)
            if result is not None:
                return result
            else:
                # Primary failed, trigger failover
                self._trigger_failover(primary_node_id)
                raise Exception("Primary node failed and failover initiated")
        
        elif self.failover_mode == FailoverMode.ACTIVE_ACTIVE:
            # Process with all active nodes and aggregate results
            for node_id in self.active_nodes:
                try:
                    result = self._process_with_node(node_id, data, source)
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    processing_errors.append(f"Node {node_id}: {str(e)}")
                    self._record_node_failure(node_id)
            
            if results:
                # Aggregate results from multiple nodes
                return self._aggregate_results(results)
            else:
                raise Exception(f"All nodes failed: {processing_errors}")
        
        elif self.failover_mode == FailoverMode.ROUND_ROBIN:
            # Rotate through active nodes
            active_list = sorted(self.active_nodes)
            for i, node_id in enumerate(active_list):
                try:
                    result = self._process_with_node(node_id, data, source)
                    if result is not None:
                        return result
                except Exception as e:
                    processing_errors.append(f"Node {node_id}: {str(e)}")
                    self._record_node_failure(node_id)
            
            raise Exception(f"All nodes failed in round-robin: {processing_errors}")
        
        elif self.failover_mode == FailoverMode.PRIORITY_BASED:
            # Try nodes in priority order
            sorted_nodes = sorted(self.active_nodes, 
                                key=lambda nid: self.nodes[nid].priority, 
                                reverse=True)
            
            for node_id in sorted_nodes:
                try:
                    result = self._process_with_node(node_id, data, source)
                    if result is not None:
                        return result
                except Exception as e:
                    processing_errors.append(f"Node {node_id}: {str(e)}")
                    self._record_node_failure(node_id)
            
            raise Exception(f"All priority nodes failed: {processing_errors}")
    
    def _process_with_node(self, node_id: str, data: Dict[str, Any], 
                          source: str) -> Optional[Dict[str, Any]]:
        """Process data with a specific node."""
        node = self.nodes[node_id]
        
        try:
            # Update heartbeat
            node.last_heartbeat = datetime.now()
            
            # Process data
            if node.processor_function:
                result = node.processor_function(data, source)
                node.success_count += 1
                return result
            else:
                # No processor function, return data as-is
                return data
                
        except Exception as e:
            logger.warning(f"Node {node_id} processing failed: {e}")
            self._record_node_failure(node_id)
            return None
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple nodes."""
        if not results:
            return {}
        
        if len(results) == 1:
            return results[0]
        
        # Merge results - use majority voting for conflicts
        aggregated = {}
        
        # Collect all keys
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        
        for key in all_keys:
            values = [result.get(key) for result in results if key in result]
            
            if not values:
                continue
            
            # For numeric values, use average
            if all(isinstance(v, (int, float)) for v in values):
                aggregated[key] = sum(values) / len(values)
            # For strings, use majority vote
            elif all(isinstance(v, str) for v in values):
                from collections import Counter
                counter = Counter(values)
                aggregated[key] = counter.most_common(1)[0][0]
            # For other types, use first non-None value
            else:
                aggregated[key] = next((v for v in values if v is not None), values[0])
        
        # Add metadata about aggregation
        aggregated['_redundancy_metadata'] = {
            'aggregation_timestamp': datetime.now().isoformat(),
            'source_nodes': len(results),
            'aggregation_method': 'majority_voting'
        }
        
        return aggregated
    
    def _activate_node(self, node_id: str):
        """Activate a redundancy node."""
        if node_id in self.nodes:
            self.nodes[node_id].status = NodeStatus.ACTIVE
            self.active_nodes.add(node_id)
            self.failed_nodes.discard(node_id)
            logger.info(f"Activated redundancy node: {node_id}")
    
    def _deactivate_node(self, node_id: str):
        """Deactivate a redundancy node."""
        if node_id in self.nodes:
            self.nodes[node_id].status = NodeStatus.FAILED
            self.active_nodes.discard(node_id)
            self.failed_nodes.add(node_id)
            logger.warning(f"Deactivated redundancy node: {node_id}")
    
    def _record_node_failure(self, node_id: str):
        """Record a failure for a node."""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.failure_count += 1
            
            if node.failure_count >= self.failure_threshold:
                self._trigger_failover(node_id)
    
    def _trigger_failover(self, failed_node_id: str):
        """Trigger failover from a failed node."""
        logger.warning(f"Triggering failover for node: {failed_node_id}")
        
        # Deactivate failed node
        self._deactivate_node(failed_node_id)
        
        # Find replacement node
        available_nodes = [nid for nid, node in self.nodes.items() 
                          if nid not in self.active_nodes and 
                          node.status != NodeStatus.FAILED]
        
        if available_nodes:
            # Activate highest priority available node
            replacement_node = max(available_nodes, 
                                 key=lambda nid: self.nodes[nid].priority)
            self._activate_node(replacement_node)
            
            logger.info(f"Failover complete: {failed_node_id} -> {replacement_node}")
        else:
            logger.error("No available nodes for failover!")
            self.redundancy_stats['redundancy_violations'] += 1
        
        self.redundancy_stats['total_failovers'] += 1
        
        # Trigger callbacks
        for callback in self.failover_callbacks:
            try:
                callback(failed_node_id, available_nodes)
            except Exception as e:
                logger.error(f"Failover callback error: {e}")
    
    def _attempt_data_recovery(self, data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Attempt to recover data from backups."""
        try:
            # Try to load from most recent backup
            backup_files = [f for f in os.listdir(self.backup_directory) 
                           if f.endswith('_backup.pkl')]
            
            if backup_files:
                # Use most recent backup
                backup_files.sort(key=lambda f: os.path.getmtime(
                    os.path.join(self.backup_directory, f)
                ), reverse=True)
                
                backup_path = os.path.join(self.backup_directory, backup_files[0])
                
                with open(backup_path, 'rb') as f:
                    backup_data = SafePickleHandler.safe_load(f)
                
                logger.info(f"Recovered data from backup: {backup_files[0]}")
                
                # Merge with current data
                if isinstance(backup_data, dict):
                    merged_data = backup_data.copy()
                    merged_data.update(data)
                    merged_data['_recovery_metadata'] = {
                        'recovered_from_backup': True,
                        'backup_file': backup_files[0],
                        'recovery_timestamp': datetime.now().isoformat()
                    }
                    return merged_data
            
            # If no backup available, return original data with metadata
            data['_recovery_metadata'] = {
                'recovery_attempted': True,
                'backup_available': False,
                'fallback_data': True
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Data recovery failed: {e}")
            self.redundancy_stats['data_loss_events'] += 1
            
            # Trigger data loss callbacks
            for callback in self.data_loss_callbacks:
                try:
                    callback(data, source, str(e))
                except Exception as callback_error:
                    logger.error(f"Data loss callback error: {callback_error}")
            
            return data
    
    def _queue_data_for_backup(self, data: Dict[str, Any], source: str):
        """Queue data for backup storage."""
        try:
            backup_item = {
                'timestamp': datetime.now().isoformat(),
                'source': source,
                'data': copy.deepcopy(data)
            }
            
            self.backup_queue.put_nowait(backup_item)
            
        except queue.Full:
            self.redundancy_stats['queue_overflows'] += 1
            logger.warning("Backup queue full, discarding oldest item")
            try:
                self.backup_queue.get_nowait()  # Remove oldest
                self.backup_queue.put_nowait(backup_item)  # Add new
            except queue.Empty:
                pass
    
    def _heartbeat_loop(self):
        """Monitor node heartbeats."""
        while self.redundancy_active:
            try:
                time.sleep(self.heartbeat_interval)
                
                current_time = datetime.now()
                
                for node_id, node in self.nodes.items():
                    # Check if node has missed heartbeat
                    time_since_heartbeat = (current_time - node.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.heartbeat_interval * 2:
                        if node.status == NodeStatus.ACTIVE:
                            logger.warning(f"Node {node_id} missed heartbeat")
                            self._record_node_failure(node_id)
                
            except Exception as e:
                logger.error(f"Heartbeat monitoring error: {e}")
    
    def _failover_loop(self):
        """Monitor for recovery opportunities."""
        while self.redundancy_active:
            try:
                time.sleep(self.recovery_timeout)
                
                # Check if failed nodes can be recovered
                current_time = datetime.now()
                
                for node_id in list(self.failed_nodes):
                    node = self.nodes[node_id]
                    
                    # Attempt recovery if enough time has passed
                    time_since_failure = (current_time - node.last_heartbeat).total_seconds()
                    
                    if time_since_failure > self.recovery_timeout:
                        self._attempt_node_recovery(node_id)
                
            except Exception as e:
                logger.error(f"Failover monitoring error: {e}")
    
    def _attempt_node_recovery(self, node_id: str):
        """Attempt to recover a failed node."""
        try:
            node = self.nodes[node_id]
            
            # Test if node can process data
            test_data = {'test': True, 'timestamp': datetime.now().isoformat()}
            
            if node.processor_function:
                result = node.processor_function(test_data, 'recovery_test')
                if result is not None:
                    # Recovery successful
                    node.status = NodeStatus.ACTIVE
                    node.failure_count = 0
                    self.failed_nodes.discard(node_id)
                    
                    # Decide whether to activate based on failover mode
                    if (self.failover_mode in [FailoverMode.ACTIVE_ACTIVE, FailoverMode.ROUND_ROBIN] or
                        (self.failover_mode == FailoverMode.ACTIVE_PASSIVE and not self.active_nodes)):
                        self._activate_node(node_id)
                    
                    logger.info(f"Successfully recovered node: {node_id}")
                    self.redundancy_stats['successful_recoveries'] += 1
                    
                    # Trigger recovery callbacks
                    for callback in self.recovery_callbacks:
                        try:
                            callback(node_id)
                        except Exception as e:
                            logger.error(f"Recovery callback error: {e}")
                else:
                    logger.warning(f"Node {node_id} recovery test failed")
            
        except Exception as e:
            logger.warning(f"Node {node_id} recovery attempt failed: {e}")
    
    def _backup_loop(self):
        """Process backup queue."""
        while self.redundancy_active:
            try:
                time.sleep(self.backup_interval)
                
                # Process backup queue
                backup_items = []
                
                while not self.backup_queue.empty() and len(backup_items) < 100:
                    try:
                        item = self.backup_queue.get_nowait()
                        backup_items.append(item)
                    except queue.Empty:
                        break
                
                if backup_items:
                    self._create_backup(backup_items)
                
            except Exception as e:
                logger.error(f"Backup processing error: {e}")
    
    def _create_backup(self, backup_items: List[Dict[str, Any]]):
        """Create backup file from queued items."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"analytics_backup_{timestamp}.pkl"
            backup_path = os.path.join(self.backup_directory, backup_filename)
            
            with open(backup_path, 'wb') as f:
                pickle.dump(backup_items, f)
            
            self.redundancy_stats['backup_operations'] += 1
            logger.debug(f"Created backup: {backup_filename} ({len(backup_items)} items)")
            
            # Clean up old backups (keep last 10)
            self._cleanup_old_backups()
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
    
    def _cleanup_old_backups(self):
        """Clean up old backup files."""
        try:
            backup_files = [f for f in os.listdir(self.backup_directory) 
                           if f.startswith('analytics_backup_') and f.endswith('.pkl')]
            
            if len(backup_files) > 10:
                # Sort by modification time and remove oldest
                backup_files.sort(key=lambda f: os.path.getmtime(
                    os.path.join(self.backup_directory, f)
                ))
                
                for old_backup in backup_files[:-10]:
                    os.remove(os.path.join(self.backup_directory, old_backup))
                    logger.debug(f"Removed old backup: {old_backup}")
        
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    def add_failover_callback(self, callback: Callable[[str, List[str]], None]):
        """Add callback for failover events."""
        self.failover_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable[[str], None]):
        """Add callback for recovery events."""
        self.recovery_callbacks.append(callback)
    
    def add_data_loss_callback(self, callback: Callable[[Dict, str, str], None]):
        """Add callback for data loss events."""
        self.data_loss_callbacks.append(callback)
    
    def get_redundancy_status(self) -> Dict[str, Any]:
        """Get overall redundancy status."""
        uptime = (datetime.now() - self.redundancy_stats['start_time']).total_seconds()
        
        node_summary = {}
        for node_id, node in self.nodes.items():
            node_summary[node_id] = {
                'status': node.status.value,
                'priority': node.priority,
                'failure_count': node.failure_count,
                'success_count': node.success_count,
                'last_heartbeat': node.last_heartbeat.isoformat(),
                'is_active': node_id in self.active_nodes
            }
        
        return {
            'failover_mode': self.failover_mode.value,
            'redundancy_active': self.redundancy_active,
            'total_nodes': len(self.nodes),
            'active_nodes': len(self.active_nodes),
            'failed_nodes': len(self.failed_nodes),
            'queue_sizes': {
                'data_queue': self.data_queue.qsize(),
                'backup_queue': self.backup_queue.qsize()
            },
            'nodes': node_summary,
            'statistics': self.redundancy_stats.copy(),
            'uptime_seconds': uptime,
            'backup_directory': self.backup_directory
        }
    
    def shutdown(self):
        """Shutdown redundancy manager."""
        self.stop_redundancy()
        
        # Process remaining backup queue
        try:
            remaining_items = []
            while not self.backup_queue.empty():
                try:
                    item = self.backup_queue.get_nowait()
                    remaining_items.append(item)
                except queue.Empty:
                    break
            
            if remaining_items:
                self._create_backup(remaining_items)
        
        except Exception as e:
            logger.error(f"Error processing final backup: {e}")
        
        logger.info("Analytics Redundancy Manager shutdown")
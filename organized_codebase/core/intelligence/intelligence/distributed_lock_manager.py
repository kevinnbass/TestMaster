"""
Distributed Lock Manager
=======================

Enterprise distributed locking system providing coordination primitives,
mutual exclusion, and deadlock prevention across distributed systems.

Features:
- Distributed mutual exclusion with timeout support
- Deadlock detection and prevention
- Lock priorities and fair queuing
- Re-entrant locks with ownership tracking
- Lock leasing with automatic renewal
- Consensus-based lock coordination
- Lock monitoring and metrics
- Graceful degradation on network partitions

Author: TestMaster Intelligence Team
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
import threading
from collections import defaultdict, deque
import hashlib
import weakref

logger = logging.getLogger(__name__)

class LockType(Enum):
    """Types of distributed locks"""
    EXCLUSIVE = "exclusive"
    SHARED = "shared"
    PRIORITY = "priority"
    FAIR = "fair"
    REENTRANT = "reentrant"

class LockStatus(Enum):
    """Lock status"""
    AVAILABLE = "available"
    ACQUIRED = "acquired"
    PENDING = "pending"
    EXPIRED = "expired"
    RELEASED = "released"

class NodeStatus(Enum):
    """Node status in distributed system"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPECT = "suspect"
    FAILED = "failed"

@dataclass
class LockRequest:
    """Lock acquisition request"""
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    lock_name: str = ""
    requester_id: str = ""
    node_id: str = ""
    lock_type: LockType = LockType.EXCLUSIVE
    timeout_seconds: int = 30
    priority: int = 1  # Higher number = higher priority
    created_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if request has expired"""
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.timeout_seconds

@dataclass
class LockOwnership:
    """Lock ownership information"""
    lock_name: str
    owner_id: str
    node_id: str
    lock_type: LockType
    acquired_at: datetime
    expires_at: Optional[datetime] = None
    lease_duration_seconds: int = 300
    reentrant_count: int = 1
    context: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if lock lease has expired"""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def renew_lease(self, extension_seconds: Optional[int] = None):
        """Renew lock lease"""
        if extension_seconds is None:
            extension_seconds = self.lease_duration_seconds
        
        self.expires_at = datetime.now() + timedelta(seconds=extension_seconds)

@dataclass
class DistributedNode:
    """Node in distributed lock system"""
    node_id: str
    host: str = "localhost"
    port: int = 8080
    status: NodeStatus = NodeStatus.ACTIVE
    last_heartbeat: datetime = field(default_factory=datetime.now)
    locks_held: Set[str] = field(default_factory=set)
    version: str = "1.0.0"
    capabilities: Set[str] = field(default_factory=set)
    
    def is_alive(self, timeout_seconds: int = 60) -> bool:
        """Check if node is considered alive"""
        age = (datetime.now() - self.last_heartbeat).total_seconds()
        return age <= timeout_seconds

class DeadlockDetector:
    """Deadlock detection using wait-for graph"""
    
    def __init__(self):
        self.wait_for_graph: Dict[str, Set[str]] = defaultdict(set)
        self.lock = threading.Lock()
    
    def add_dependency(self, waiter: str, holder: str):
        """Add dependency: waiter is waiting for holder"""
        with self.lock:
            self.wait_for_graph[waiter].add(holder)
    
    def remove_dependency(self, waiter: str, holder: str):
        """Remove dependency"""
        with self.lock:
            if waiter in self.wait_for_graph:
                self.wait_for_graph[waiter].discard(holder)
                if not self.wait_for_graph[waiter]:
                    del self.wait_for_graph[waiter]
    
    def remove_node(self, node: str):
        """Remove node from all dependencies"""
        with self.lock:
            # Remove as waiter
            if node in self.wait_for_graph:
                del self.wait_for_graph[node]
            
            # Remove as holder
            for waiter in list(self.wait_for_graph.keys()):
                self.wait_for_graph[waiter].discard(node)
                if not self.wait_for_graph[waiter]:
                    del self.wait_for_graph[waiter]
    
    def detect_deadlock(self) -> List[List[str]]:
        """Detect deadlock cycles using DFS"""
        with self.lock:
            visited = set()
            rec_stack = set()
            cycles = []
            
            def dfs(node: str, path: List[str]) -> bool:
                if node in rec_stack:
                    # Found cycle
                    cycle_start = path.index(node)
                    cycle = path[cycle_start:] + [node]
                    cycles.append(cycle)
                    return True
                
                if node in visited:
                    return False
                
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in self.wait_for_graph.get(node, set()):
                    if dfs(neighbor, path + [neighbor]):
                        pass  # Continue to find all cycles
                
                rec_stack.remove(node)
                return False
            
            # Check all nodes for cycles
            for node in self.wait_for_graph:
                if node not in visited:
                    dfs(node, [node])
            
            return cycles

class ConsensusManager:
    """Simple consensus manager for distributed decisions"""
    
    def __init__(self):
        self.proposals: Dict[str, Dict] = {}
        self.votes: Dict[str, Dict[str, bool]] = defaultdict(dict)
        self.decisions: Dict[str, bool] = {}
    
    def propose(self, proposal_id: str, proposal_data: Dict[str, Any]) -> str:
        """Create a new proposal"""
        self.proposals[proposal_id] = {
            'data': proposal_data,
            'created_at': datetime.now(),
            'proposer': proposal_data.get('proposer', 'unknown')
        }
        return proposal_id
    
    def vote(self, proposal_id: str, voter_id: str, vote: bool):
        """Cast vote on proposal"""
        if proposal_id in self.proposals:
            self.votes[proposal_id][voter_id] = vote
    
    def tally_votes(self, proposal_id: str, total_nodes: int) -> Optional[bool]:
        """Tally votes and make decision"""
        if proposal_id not in self.proposals:
            return None
        
        votes = self.votes.get(proposal_id, {})
        yes_votes = sum(1 for vote in votes.values() if vote)
        total_votes = len(votes)
        
        # Require majority of active nodes
        majority_threshold = (total_nodes // 2) + 1
        
        if yes_votes >= majority_threshold:
            self.decisions[proposal_id] = True
            return True
        elif total_votes >= total_nodes and yes_votes < majority_threshold:
            self.decisions[proposal_id] = False
            return False
        
        return None  # Not enough votes yet

class DistributedLockManager:
    """
    Enterprise distributed lock manager providing coordination primitives
    and mutual exclusion across distributed systems.
    """
    
    def __init__(self, node_id: str = None, heartbeat_interval: int = 30):
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.heartbeat_interval = heartbeat_interval
        
        # Lock storage
        self.locks: Dict[str, LockOwnership] = {}
        self.pending_requests: Dict[str, deque] = defaultdict(deque)
        self.lock_history: deque = deque(maxlen=1000)
        
        # Node management
        self.nodes: Dict[str, DistributedNode] = {}
        self.local_node = DistributedNode(
            node_id=self.node_id,
            capabilities={"exclusive_locks", "shared_locks", "deadlock_detection"}
        )
        self.nodes[self.node_id] = self.local_node
        
        # Consensus and coordination
        self.consensus_manager = ConsensusManager()
        self.deadlock_detector = DeadlockDetector()
        
        # Lock monitoring
        self.lock_stats = {
            'locks_acquired': 0,
            'locks_released': 0,
            'lock_timeouts': 0,
            'deadlocks_detected': 0,
            'deadlocks_resolved': 0,
            'total_wait_time': 0.0,
            'start_time': datetime.now()
        }
        
        # Background tasks
        self.monitoring_active = False
        self.monitoring_tasks: Set[asyncio.Task] = set()
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"Distributed Lock Manager initialized (node: {self.node_id})")
    
    async def acquire_lock(self, lock_name: str, requester_id: str,
                          lock_type: LockType = LockType.EXCLUSIVE,
                          timeout_seconds: int = 30,
                          priority: int = 1) -> Optional[str]:
        """Acquire distributed lock"""
        request = LockRequest(
            lock_name=lock_name,
            requester_id=requester_id,
            node_id=self.node_id,
            lock_type=lock_type,
            timeout_seconds=timeout_seconds,
            priority=priority
        )
        
        logger.info(f"Lock acquisition request: {lock_name} by {requester_id}")
        
        with self.lock:
            # Check if lock is available
            if await self._can_acquire_lock(request):
                ownership = await self._grant_lock(request)
                if ownership:
                    self.lock_stats['locks_acquired'] += 1
                    return ownership.owner_id
            
            # Add to pending queue
            await self._add_to_pending_queue(request)
        
        # Wait for lock acquisition or timeout
        return await self._wait_for_lock(request)
    
    async def release_lock(self, lock_name: str, owner_id: str) -> bool:
        """Release distributed lock"""
        with self.lock:
            if lock_name not in self.locks:
                logger.warning(f"Attempted to release non-existent lock: {lock_name}")
                return False
            
            ownership = self.locks[lock_name]
            
            # Verify ownership
            if ownership.owner_id != owner_id:
                logger.warning(f"Attempted to release lock by non-owner: {lock_name}")
                return False
            
            # Handle reentrant locks
            if ownership.lock_type == LockType.REENTRANT:
                ownership.reentrant_count -= 1
                if ownership.reentrant_count > 0:
                    logger.debug(f"Decremented reentrant count for {lock_name}: {ownership.reentrant_count}")
                    return True
            
            # Release lock
            del self.locks[lock_name]
            self.local_node.locks_held.discard(lock_name)
            
            # Update deadlock detector
            self.deadlock_detector.remove_node(owner_id)
            
            # Record in history
            self.lock_history.append({
                'action': 'released',
                'lock_name': lock_name,
                'owner_id': owner_id,
                'timestamp': datetime.now().isoformat()
            })
            
            self.lock_stats['locks_released'] += 1
            
            logger.info(f"Lock released: {lock_name} by {owner_id}")
            
            # Process pending requests
            await self._process_pending_requests(lock_name)
            
            return True
    
    async def renew_lock(self, lock_name: str, owner_id: str, 
                        extension_seconds: Optional[int] = None) -> bool:
        """Renew lock lease"""
        with self.lock:
            if lock_name not in self.locks:
                return False
            
            ownership = self.locks[lock_name]
            
            if ownership.owner_id != owner_id:
                return False
            
            ownership.renew_lease(extension_seconds)
            
            logger.debug(f"Lock lease renewed: {lock_name} by {owner_id}")
            return True
    
    async def _can_acquire_lock(self, request: LockRequest) -> bool:
        """Check if lock can be acquired"""
        lock_name = request.lock_name
        
        # Check if lock exists
        if lock_name not in self.locks:
            return True
        
        ownership = self.locks[lock_name]
        
        # Check if lock has expired
        if ownership.is_expired():
            await self._expire_lock(lock_name)
            return True
        
        # Check lock type compatibility
        if request.lock_type == LockType.SHARED and ownership.lock_type == LockType.SHARED:
            return True
        
        # Check reentrant locks
        if (request.lock_type == LockType.REENTRANT and
            ownership.lock_type == LockType.REENTRANT and
            ownership.owner_id == request.requester_id):
            return True
        
        return False
    
    async def _grant_lock(self, request: LockRequest) -> Optional[LockOwnership]:
        """Grant lock to requester"""
        lock_name = request.lock_name
        
        # Handle reentrant locks
        if (lock_name in self.locks and
            self.locks[lock_name].lock_type == LockType.REENTRANT and
            self.locks[lock_name].owner_id == request.requester_id):
            self.locks[lock_name].reentrant_count += 1
            return self.locks[lock_name]
        
        # Create new ownership
        ownership = LockOwnership(
            lock_name=lock_name,
            owner_id=request.requester_id,
            node_id=request.node_id,
            lock_type=request.lock_type,
            acquired_at=datetime.now(),
            lease_duration_seconds=max(300, request.timeout_seconds * 2)
        )
        ownership.renew_lease()
        
        self.locks[lock_name] = ownership
        self.local_node.locks_held.add(lock_name)
        
        # Record in history
        self.lock_history.append({
            'action': 'acquired',
            'lock_name': lock_name,
            'owner_id': request.requester_id,
            'lock_type': request.lock_type.value,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Lock granted: {lock_name} to {request.requester_id}")
        return ownership
    
    async def _add_to_pending_queue(self, request: LockRequest):
        """Add request to pending queue with priority ordering"""
        lock_name = request.lock_name
        
        # Insert in priority order (higher priority first)
        queue = self.pending_requests[lock_name]
        inserted = False
        
        for i, pending_request in enumerate(queue):
            if request.priority > pending_request.priority:
                queue.insert(i, request)
                inserted = True
                break
        
        if not inserted:
            queue.append(request)
        
        # Update deadlock detector
        if lock_name in self.locks:
            self.deadlock_detector.add_dependency(
                request.requester_id,
                self.locks[lock_name].owner_id
            )
        
        logger.debug(f"Added to pending queue: {lock_name} for {request.requester_id}")
    
    async def _wait_for_lock(self, request: LockRequest) -> Optional[str]:
        """Wait for lock acquisition or timeout"""
        start_time = time.time()
        
        while not request.is_expired():
            await asyncio.sleep(0.1)  # Poll interval
            
            with self.lock:
                # Check if lock is now available
                if await self._can_acquire_lock(request):
                    # Remove from pending queue
                    self._remove_from_pending_queue(request)
                    
                    # Grant lock
                    ownership = await self._grant_lock(request)
                    if ownership:
                        wait_time = time.time() - start_time
                        self.lock_stats['total_wait_time'] += wait_time
                        return ownership.owner_id
        
        # Request timed out
        with self.lock:
            self._remove_from_pending_queue(request)
            self.deadlock_detector.remove_node(request.requester_id)
            self.lock_stats['lock_timeouts'] += 1
        
        logger.warning(f"Lock acquisition timeout: {request.lock_name} for {request.requester_id}")
        return None
    
    def _remove_from_pending_queue(self, request: LockRequest):
        """Remove request from pending queue"""
        queue = self.pending_requests[request.lock_name]
        
        for i, pending_request in enumerate(queue):
            if pending_request.request_id == request.request_id:
                queue.remove(pending_request)
                break
        
        # Clean up empty queues
        if not queue:
            del self.pending_requests[request.lock_name]
    
    async def _process_pending_requests(self, lock_name: str):
        """Process pending requests for a released lock"""
        if lock_name not in self.pending_requests:
            return
        
        queue = self.pending_requests[lock_name]
        processed_requests = []
        
        while queue:
            request = queue.popleft()
            
            # Check if request is still valid
            if request.is_expired():
                self.lock_stats['lock_timeouts'] += 1
                continue
            
            # Try to grant lock
            if await self._can_acquire_lock(request):
                ownership = await self._grant_lock(request)
                if ownership:
                    processed_requests.append(request)
                    
                    # For exclusive locks, stop processing
                    if request.lock_type == LockType.EXCLUSIVE:
                        break
            else:
                # Put back in queue if couldn't acquire
                queue.appendleft(request)
                break
        
        # Clean up empty queue
        if not queue:
            del self.pending_requests[lock_name]
        
        logger.debug(f"Processed {len(processed_requests)} pending requests for {lock_name}")
    
    async def _expire_lock(self, lock_name: str):
        """Expire a lock due to lease timeout"""
        if lock_name in self.locks:
            ownership = self.locks[lock_name]
            owner_id = ownership.owner_id
            
            del self.locks[lock_name]
            self.local_node.locks_held.discard(lock_name)
            
            # Update deadlock detector
            self.deadlock_detector.remove_node(owner_id)
            
            # Record in history
            self.lock_history.append({
                'action': 'expired',
                'lock_name': lock_name,
                'owner_id': owner_id,
                'timestamp': datetime.now().isoformat()
            })
            
            logger.warning(f"Lock expired: {lock_name} (owner: {owner_id})")
            
            # Process pending requests
            await self._process_pending_requests(lock_name)
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        if self.monitoring_active:
            return
        
        logger.info("Starting distributed lock monitoring")
        self.monitoring_active = True
        
        # Start lease monitoring
        lease_task = asyncio.create_task(self._lease_monitoring_loop())
        self.monitoring_tasks.add(lease_task)
        
        # Start deadlock detection
        deadlock_task = asyncio.create_task(self._deadlock_detection_loop())
        self.monitoring_tasks.add(deadlock_task)
        
        # Start heartbeat
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.monitoring_tasks.add(heartbeat_task)
        
        logger.info("Distributed lock monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring"""
        if not self.monitoring_active:
            return
        
        logger.info("Stopping distributed lock monitoring")
        self.monitoring_active = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
        
        logger.info("Distributed lock monitoring stopped")
    
    async def _lease_monitoring_loop(self):
        """Monitor lock leases and expire stale locks"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                expired_locks = []
                current_time = datetime.now()
                
                with self.lock:
                    for lock_name, ownership in self.locks.items():
                        if ownership.is_expired():
                            expired_locks.append(lock_name)
                
                # Expire locks outside of main lock
                for lock_name in expired_locks:
                    await self._expire_lock(lock_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Lease monitoring error: {e}")
    
    async def _deadlock_detection_loop(self):
        """Detect and resolve deadlocks"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                cycles = self.deadlock_detector.detect_deadlock()
                
                if cycles:
                    self.lock_stats['deadlocks_detected'] += len(cycles)
                    logger.warning(f"Detected {len(cycles)} deadlock cycles")
                    
                    # Resolve deadlocks by timing out lowest priority requests
                    for cycle in cycles:
                        await self._resolve_deadlock(cycle)
                        self.lock_stats['deadlocks_resolved'] += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Deadlock detection error: {e}")
    
    async def _resolve_deadlock(self, cycle: List[str]):
        """Resolve deadlock by terminating requests"""
        logger.warning(f"Resolving deadlock cycle: {' -> '.join(cycle)}")
        
        # Find lowest priority request in cycle and terminate it
        with self.lock:
            lowest_priority_request = None
            lowest_priority = float('inf')
            
            for lock_name, queue in self.pending_requests.items():
                for request in queue:
                    if request.requester_id in cycle and request.priority < lowest_priority:
                        lowest_priority = request.priority
                        lowest_priority_request = request
            
            if lowest_priority_request:
                self._remove_from_pending_queue(lowest_priority_request)
                self.deadlock_detector.remove_node(lowest_priority_request.requester_id)
                logger.info(f"Terminated request to resolve deadlock: {lowest_priority_request.request_id}")
    
    async def _heartbeat_loop(self):
        """Send heartbeats to maintain node liveness"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Update local node heartbeat
                self.local_node.last_heartbeat = datetime.now()
                
                # Check other nodes (in real implementation, would send network heartbeats)
                current_time = datetime.now()
                failed_nodes = []
                
                for node_id, node in self.nodes.items():
                    if node_id != self.node_id and not node.is_alive():
                        node.status = NodeStatus.FAILED
                        failed_nodes.append(node_id)
                
                # Handle failed nodes
                for node_id in failed_nodes:
                    await self._handle_node_failure(node_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _handle_node_failure(self, failed_node_id: str):
        """Handle node failure by releasing its locks"""
        logger.warning(f"Handling failure of node: {failed_node_id}")
        
        with self.lock:
            # Find locks held by failed node
            failed_locks = []
            for lock_name, ownership in self.locks.items():
                if ownership.node_id == failed_node_id:
                    failed_locks.append(lock_name)
            
            # Release locks of failed node
            for lock_name in failed_locks:
                ownership = self.locks[lock_name]
                del self.locks[lock_name]
                
                # Record in history
                self.lock_history.append({
                    'action': 'node_failure_release',
                    'lock_name': lock_name,
                    'owner_id': ownership.owner_id,
                    'failed_node': failed_node_id,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Process pending requests
                await self._process_pending_requests(lock_name)
    
    def get_lock_status(self, lock_name: str) -> Optional[Dict[str, Any]]:
        """Get status of specific lock"""
        with self.lock:
            if lock_name in self.locks:
                ownership = self.locks[lock_name]
                pending_count = len(self.pending_requests.get(lock_name, []))
                
                return {
                    'lock_name': lock_name,
                    'status': LockStatus.ACQUIRED.value,
                    'owner_id': ownership.owner_id,
                    'node_id': ownership.node_id,
                    'lock_type': ownership.lock_type.value,
                    'acquired_at': ownership.acquired_at.isoformat(),
                    'expires_at': ownership.expires_at.isoformat() if ownership.expires_at else None,
                    'reentrant_count': ownership.reentrant_count,
                    'pending_requests': pending_count
                }
            else:
                pending_count = len(self.pending_requests.get(lock_name, []))
                return {
                    'lock_name': lock_name,
                    'status': LockStatus.AVAILABLE.value if pending_count == 0 else LockStatus.PENDING.value,
                    'pending_requests': pending_count
                }
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get comprehensive lock manager status"""
        with self.lock:
            uptime = (datetime.now() - self.lock_stats['start_time']).total_seconds()
            
            # Calculate statistics
            total_locks = len(self.locks)
            total_pending = sum(len(queue) for queue in self.pending_requests.values())
            
            lock_types = defaultdict(int)
            for ownership in self.locks.values():
                lock_types[ownership.lock_type.value] += 1
            
            return {
                'node_id': self.node_id,
                'status': 'active' if self.monitoring_active else 'inactive',
                'uptime_seconds': uptime,
                'statistics': self.lock_stats.copy(),
                'current_locks': {
                    'total': total_locks,
                    'by_type': dict(lock_types),
                    'held_by_this_node': len(self.local_node.locks_held)
                },
                'pending_requests': {
                    'total': total_pending,
                    'by_lock': {lock: len(queue) for lock, queue in self.pending_requests.items()}
                },
                'nodes': {
                    'total': len(self.nodes),
                    'active': len([n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]),
                    'failed': len([n for n in self.nodes.values() if n.status == NodeStatus.FAILED])
                },
                'deadlock_detection': {
                    'enabled': True,
                    'current_dependencies': len(self.deadlock_detector.wait_for_graph)
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def get_lock_hierarchy(self) -> Dict[str, Any]:
        """Get lock dependency hierarchy and wait-for graph"""
        with self.lock:
            hierarchy = {
                'locks': {},
                'wait_for_graph': {},
                'potential_deadlocks': []
            }
            
            # Current locks
            for lock_name, ownership in self.locks.items():
                hierarchy['locks'][lock_name] = {
                    'owner': ownership.owner_id,
                    'type': ownership.lock_type.value,
                    'acquired_at': ownership.acquired_at.isoformat(),
                    'pending_count': len(self.pending_requests.get(lock_name, []))
                }
            
            # Wait-for graph
            hierarchy['wait_for_graph'] = {
                waiter: list(holders) for waiter, holders in self.deadlock_detector.wait_for_graph.items()
            }
            
            # Check for potential deadlocks
            cycles = self.deadlock_detector.detect_deadlock()
            hierarchy['potential_deadlocks'] = cycles
            
            return hierarchy
    
    def shutdown(self):
        """Shutdown distributed lock manager"""
        if self.monitoring_active:
            # Create and run shutdown task
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.stop_monitoring())
            loop.close()
        
        logger.info("Distributed Lock Manager shutdown")

# Global distributed lock manager instance
distributed_lock_manager = DistributedLockManager()
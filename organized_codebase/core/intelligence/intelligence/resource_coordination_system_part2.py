"""
        
        # Sort preemptable by priority (lowest first)
        preemptable_allocations.sort(key=lambda a: a.priority)
        
        resolved = []
        additional_capacity = 0
        
        for allocation in preemptable_allocations:
            pool.deallocate(allocation.allocation_id)
            additional_capacity += allocation.amount
            
            # Check if we can now satisfy high priority requests
            remaining_capacity = conflict.available_capacity + additional_capacity
            for request in high_priority_requests:
                if request.amount <= remaining_capacity:
                    resolved.append(request)
                    remaining_capacity -= request.amount
            
            if len(resolved) == len(high_priority_requests):
                break
        
        return resolved
    
    def _resolve_alternative(self, conflict: ResourceConflict,
                           requests: List[ResourceRequest],
                           pools: Dict[str, ResourcePool]) -> List[ResourceRequest]:
        """Resolve by finding alternative resources"""
        # For now, return empty list (would need resource substitution logic)
        return []
    
    def _resolve_degraded(self, conflict: ResourceConflict,
                         requests: List[ResourceRequest],
                         pools: Dict[str, ResourcePool]) -> List[ResourceRequest]:
        """Resolve by offering degraded service"""
        conflicting_requests = [r for r in requests if r.request_id in conflict.conflicting_requests]
        
        # Reduce all requests to fit capacity
        total_demand = sum(r.amount for r in conflicting_requests)
        reduction_factor = conflict.available_capacity / total_demand
        
        resolved = []
        for request in conflicting_requests:
            request.amount *= reduction_factor
            request.context['degraded_service'] = True
            request.context['original_amount'] = request.amount / reduction_factor
            resolved.append(request)
        
        return resolved

class ResourceCoordinationSystem:
    """
    Enterprise resource coordination system providing distributed resource allocation,
    conflict resolution, and intelligent resource management.
    """
    
    def __init__(self):
        # Resource management
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.tenant_quotas: Dict[str, Dict[ResourceType, ResourceQuota]] = defaultdict(dict)
        self.pending_requests: deque = deque()
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        
        # Scheduling and conflict resolution
        self.scheduler = AllocationScheduler()
        self.conflict_resolver = ConflictResolver()
        self.detected_conflicts: List[ResourceConflict] = []
        
        # Monitoring and analytics
        self.coordination_stats = {
            'total_requests': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'preemptions_performed': 0,
            'start_time': datetime.now()
        }
        
        # Background processing
        self.coordination_active = False
        self.coordination_tasks: Set[asyncio.Task] = set()
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("Resource Coordination System initialized")
    
    def register_resource(self, definition: ResourceDefinition) -> bool:
        """Register a new resource"""
        with self.lock:
            if definition.resource_id in self.resource_pools:
                logger.warning(f"Resource already registered: {definition.resource_id}")
                return False
            
            pool = ResourcePool(definition)
            self.resource_pools[definition.resource_id] = pool
            
            logger.info(f"Registered resource: {definition.resource_id} ({definition.resource_type.value})")
            return True
    
    def set_tenant_quota(self, tenant_id: str, resource_type: ResourceType,
                        max_allocation: float, priority: int = 1) -> bool:
        """Set resource quota for tenant"""
        with self.lock:
            quota = ResourceQuota(
                tenant_id=tenant_id,
                resource_type=resource_type,
                max_allocation=max_allocation,
                priority=priority
            )
            
            self.tenant_quotas[tenant_id][resource_type] = quota
            
            logger.info(f"Set quota for {tenant_id}: {resource_type.value} = {max_allocation}")
            return True
    
    async def request_resource(self, request: ResourceRequest) -> Optional[str]:
        """Request resource allocation"""
        with self.lock:
            self.coordination_stats['total_requests'] += 1
            
            # Check tenant quota
            if not self._check_tenant_quota(request):
                logger.warning(f"Quota exceeded for request: {request.request_id}")
                self.coordination_stats['failed_allocations'] += 1
                return None
            
            # Add to pending requests
            self.pending_requests.append(request)
            
            logger.info(f"Resource request queued: {request.request_id}")
        
        # Process request
        return await self._process_resource_requests()
    
    async def release_resource(self, allocation_id: str) -> bool:
        """Release resource allocation"""
        with self.lock:
            if allocation_id not in self.active_allocations:
                logger.warning(f"Attempted to release non-existent allocation: {allocation_id}")
                return False
            
            allocation = self.active_allocations[allocation_id]
            
            # Find and update pool
            pool = None
            for p in self.resource_pools.values():
                if allocation_id in p.allocations:
                    pool = p
                    break
            
            if pool:
                pool.deallocate(allocation_id)
                
                # Update tenant quota
                quota = self.tenant_quotas.get(allocation.tenant_id, {}).get(
                    pool.definition.resource_type
                )
                if quota:
                    quota.current_allocation -= allocation.amount
            
            del self.active_allocations[allocation_id]
            
            logger.info(f"Resource released: {allocation_id}")
            
            # Process pending requests
            await self._process_resource_requests()
            
            return True
    
    def _check_tenant_quota(self, request: ResourceRequest) -> bool:
        """Check if request respects tenant quota"""
        quota = self.tenant_quotas.get(request.tenant_id, {}).get(request.resource_type)
        
        if not quota:
            # No quota set, allow request
            return True
        
        return quota.available_quota() >= request.amount
    
    async def _process_resource_requests(self) -> Optional[str]:
        """Process pending resource requests"""
        with self.lock:
            if not self.pending_requests:
                return None
            
            # Get current requests
            current_requests = list(self.pending_requests)
            self.pending_requests.clear()
            
            # Detect conflicts
            conflicts = self._detect_conflicts(current_requests)
            
            if conflicts:
                # Resolve conflicts
                resolved_requests = []
                for conflict in conflicts:
                    self.coordination_stats['conflicts_detected'] += 1
                    resolved = self.conflict_resolver.resolve_conflict(
                        conflict, current_requests, self.resource_pools
                    )
                    resolved_requests.extend(resolved)
                    self.coordination_stats['conflicts_resolved'] += 1
                
                current_requests = resolved_requests
            
            # Schedule requests
            scheduled = self.scheduler.schedule(
                current_requests,
                self.resource_pools,
                AllocationStrategy.PRIORITY_BASED
            )
            
            # Allocate resources
            allocated_id = None
            for request, pool_id in scheduled:
                allocation_id = await self._allocate_resource(request, pool_id)
                if allocation_id and not allocated_id:
                    allocated_id = allocation_id
            
            # Re-queue unscheduled requests
            scheduled_request_ids = {req.request_id for req, _ in scheduled}
            for request in current_requests:
                if request.request_id not in scheduled_request_ids:
                    if not request.is_expired():
                        self.pending_requests.append(request)
            
            return allocated_id
    
    def _detect_conflicts(self, requests: List[ResourceRequest]) -> List[ResourceConflict]:
        """Detect resource allocation conflicts"""
        conflicts = []
        resource_demands = defaultdict(lambda: {'requests': [], 'total_demand': 0.0})
        
        # Group requests by resource type
        for request in requests:
            # Find available pools of this type
            available_pools = [
                pool for pool in self.resource_pools.values()
                if pool.definition.resource_type == request.resource_type
            ]
            
            if available_pools:
                total_capacity = sum(pool.available_capacity for pool in available_pools)
                key = request.resource_type.value
                resource_demands[key]['requests'].append(request)
                resource_demands[key]['total_demand'] += request.amount
                resource_demands[key]['available_capacity'] = total_capacity
        
        # Check for conflicts
        for resource_type, demand_info in resource_demands.items():
            if demand_info['total_demand'] > demand_info['available_capacity']:
                conflict = ResourceConflict(
                    resource_id=resource_type,
                    conflicting_requests=[req.request_id for req in demand_info['requests']],
                    total_demand=demand_info['total_demand'],
                    available_capacity=demand_info['available_capacity'],
                    resolution_strategy=ConflictResolutionStrategy.PRIORITY_BASED
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def _allocate_resource(self, request: ResourceRequest, pool_id: str) -> Optional[str]:
        """Allocate resource from specific pool"""
        pool = self.resource_pools.get(pool_id)
        if not pool:
            return None
        
        # Create allocation
        allocation = ResourceAllocation(
            resource_id=pool.definition.resource_id,
            tenant_id=request.tenant_id,
            requester_id=request.requester_id,
            amount=request.amount,
            priority=request.priority,
            context=request.context.copy()
        )
        
        # Set expiration if duration specified
        if request.duration_seconds:
            allocation.expires_at = datetime.now() + timedelta(seconds=request.duration_seconds)
        
        # Allocate in pool
        if pool.allocate(allocation):
            self.active_allocations[allocation.allocation_id] = allocation
            
            # Update tenant quota
            quota = self.tenant_quotas.get(request.tenant_id, {}).get(request.resource_type)
            if quota:
                quota.current_allocation += request.amount
            
            self.coordination_stats['successful_allocations'] += 1
            
            logger.info(f"Resource allocated: {allocation.allocation_id} in pool {pool_id}")
            return allocation.allocation_id
        else:
            self.coordination_stats['failed_allocations'] += 1
            return None
    
    async def start_coordination(self):
        """Start resource coordination background tasks"""
        if self.coordination_active:
            return
        
        logger.info("Starting resource coordination")
        self.coordination_active = True
        
        # Start lease monitoring
        lease_task = asyncio.create_task(self._lease_monitoring_loop())
        self.coordination_tasks.add(lease_task)
        
        # Start resource optimization
        optimization_task = asyncio.create_task(self._optimization_loop())
        self.coordination_tasks.add(optimization_task)
        
        # Start request processing
        processing_task = asyncio.create_task(self._request_processing_loop())
        self.coordination_tasks.add(processing_task)
        
        logger.info("Resource coordination started")
    
    async def stop_coordination(self):
        """Stop resource coordination"""
        if not self.coordination_active:
            return
        
        logger.info("Stopping resource coordination")
        self.coordination_active = False
        
        # Cancel all coordination tasks
        for task in self.coordination_tasks:
            task.cancel()
        
        await asyncio.gather(*self.coordination_tasks, return_exceptions=True)
        self.coordination_tasks.clear()
        
        logger.info("Resource coordination stopped")
    
    async def _lease_monitoring_loop(self):
        """Monitor resource leases and handle expirations"""
        while self.coordination_active:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                expired_allocations = []
                current_time = datetime.now()
                
                with self.lock:
                    for allocation_id, allocation in self.active_allocations.items():
                        if allocation.is_expired():
                            expired_allocations.append(allocation_id)
                
                # Release expired allocations
                for allocation_id in expired_allocations:
                    await self.release_resource(allocation_id)
                    logger.info(f"Released expired allocation: {allocation_id}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Lease monitoring error: {e}")
    
    async def _optimization_loop(self):
        """Optimize resource allocation and detect inefficiencies"""
        while self.coordination_active:
            try:
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
                with self.lock:
                    # Calculate utilization metrics
                    for pool in self.resource_pools.values():
                        utilization = pool.get_utilization()
                        
                        # Update average utilization
                        current_avg = pool.usage_stats.get('average_utilization', 0.0)
                        pool.usage_stats['average_utilization'] = (current_avg + utilization) / 2
                
                # Log optimization opportunities
                self._identify_optimization_opportunities()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
    
    async def _request_processing_loop(self):
        """Continuously process pending requests"""
        while self.coordination_active:
            try:
                await asyncio.sleep(1)  # Process every second
                
                if self.pending_requests:
                    await self._process_resource_requests()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Request processing error: {e}")
    
    def _identify_optimization_opportunities(self):
        """Identify resource optimization opportunities"""
        with self.lock:
            for pool_id, pool in self.resource_pools.items():
                utilization = pool.get_utilization()
                
                if utilization > 90:
                    logger.warning(f"High utilization in pool {pool_id}: {utilization:.1f}%")
                elif utilization < 10:
                    logger.info(f"Low utilization in pool {pool_id}: {utilization:.1f}%")
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination system status"""
        with self.lock:
            uptime = (datetime.now() - self.coordination_stats['start_time']).total_seconds()
            
            # Pool statistics
            pool_stats = {}
            for pool_id, pool in self.resource_pools.items():
                pool_stats[pool_id] = pool.get_status()
            
            # Tenant statistics
            tenant_stats = {}
            for tenant_id, quotas in self.tenant_quotas.items():
                tenant_utilization = {}
                for resource_type, quota in quotas.items():
                    tenant_utilization[resource_type.value] = quota.utilization_percentage()
                tenant_stats[tenant_id] = tenant_utilization
            
            return {
                'status': 'active' if self.coordination_active else 'inactive',
                'uptime_seconds': uptime,
                'statistics': self.coordination_stats.copy(),
                'resource_pools': {
                    'total': len(self.resource_pools),
                    'details': pool_stats
                },
                'allocations': {
                    'active': len(self.active_allocations),
                    'pending_requests': len(self.pending_requests)
                },
                'tenants': {
                    'total': len(self.tenant_quotas),
                    'quota_utilization': tenant_stats
                },
                'conflicts': {
                    'detected': len(self.detected_conflicts),
                    'resolved': self.coordination_stats['conflicts_resolved']
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def get_resource_topology(self) -> Dict[str, Any]:
        """Get resource topology and allocation map"""
        with self.lock:
            topology = {
                'resource_pools': {},
                'allocations': {},
                'tenant_quotas': {},
                'utilization_map': {}
            }
            
            # Resource pools
            for pool_id, pool in self.resource_pools.items():
                topology['resource_pools'][pool_id] = {
                    'type': pool.definition.resource_type.value,
                    'capacity': pool.definition.total_capacity,
                    'available': pool.available_capacity,
                    'utilization': pool.get_utilization(),
                    'allocations': len(pool.allocations)
                }
            
            # Active allocations
            for alloc_id, allocation in self.active_allocations.items():
                topology['allocations'][alloc_id] = {
                    'tenant_id': allocation.tenant_id,
                    'resource_id': allocation.resource_id,
                    'amount': allocation.amount,
                    'allocated_at': allocation.allocated_at.isoformat(),
                    'expires_at': allocation.expires_at.isoformat() if allocation.expires_at else None
                }
            
            # Tenant quotas
            for tenant_id, quotas in self.tenant_quotas.items():
                tenant_quota_info = {}
                for resource_type, quota in quotas.items():
                    tenant_quota_info[resource_type.value] = {
                        'max_allocation': quota.max_allocation,
                        'current_allocation': quota.current_allocation,
                        'utilization': quota.utilization_percentage()
                    }
                topology['tenant_quotas'][tenant_id] = tenant_quota_info
            
            return topology
    
    def shutdown(self):
        """Shutdown resource coordination system"""
        if self.coordination_active:
            # Create and run shutdown task
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.stop_coordination())
            loop.close()
        
        logger.info("Resource Coordination System shutdown")

# Global resource coordination system instance
resource_coordination_system = ResourceCoordinationSystem()
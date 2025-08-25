"""
        execution_thread.start()
        
        self.logger.info(f"Started workflow execution: {workflow.name} ({execution_id})")
        
        return True
    
    def _execute_workflow_steps(self, workflow: MLWorkflowInstance, execution: WorkflowExecution):
        """Execute all steps in a workflow with dependency management"""
        
        try:
            completed_steps = set()
            failed_steps = set()
            max_iterations = len(workflow.steps) * 3  # Prevent infinite loops
            iteration = 0
            
            while (len(completed_steps) + len(failed_steps) < len(workflow.steps) and 
                   iteration < max_iterations):
                
                iteration += 1
                
                # Find steps ready to execute
                ready_steps = []
                for step in workflow.steps:
                    if (step.status == "pending" and 
                        step.step_id not in failed_steps and
                        all(dep in completed_steps for dep in step.dependencies)):
                        ready_steps.append(step)
                
                if not ready_steps:
                    # Check for deadlock or completion
                    pending_steps = [s for s in workflow.steps if s.status == "pending"]
                    if pending_steps:
                        self.logger.error(f"Workflow deadlock detected: {workflow.workflow_id}")
                        workflow.status = "failed"
                        break
                    else:
                        break  # All steps completed or failed
                
                # Execute ready steps based on execution mode
                if workflow.execution_mode == ExecutionMode.SEQUENTIAL:
                    # Execute one step at a time
                    step = ready_steps[0]
                    success = self._execute_single_step(step, workflow, execution)
                    if success:
                        completed_steps.add(step.step_id)
                    else:
                        failed_steps.add(step.step_id)
                        
                elif workflow.execution_mode == ExecutionMode.PARALLEL:
                    # Execute all ready steps in parallel
                    max_workers = min(len(ready_steps), 
                                    self.orchestration_config["execution_engine"]["max_parallel_steps"])
                    
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_step = {
                            executor.submit(self._execute_single_step, step, workflow, execution): step
                            for step in ready_steps
                        }
                        
                        for future in as_completed(future_to_step):
                            step = future_to_step[future]
                            try:
                                success = future.result()
                                if success:
                                    completed_steps.add(step.step_id)
                                else:
                                    failed_steps.add(step.step_id)
                            except Exception as e:
                                self.logger.error(f"Step execution error: {step.step_id} - {e}")
                                step.status = "failed"
                                step.error_details = {"exception": str(e)}
                                failed_steps.add(step.step_id)
                
                elif workflow.execution_mode in [ExecutionMode.HYBRID, ExecutionMode.ADAPTIVE]:
                    # Intelligent mixed execution
                    parallel_steps = []
                    sequential_steps = []
                    
                    for step in ready_steps:
                        # Determine if step can run in parallel based on resource requirements
                        if (step.resource_requirements.cpu_cores <= 4 and 
                            step.resource_requirements.memory_gb <= 8 and
                            step.resource_requirements.gpu_count == 0):
                            parallel_steps.append(step)
                        else:
                            sequential_steps.append(step)
                    
                    # Execute parallel steps first
                    if parallel_steps:
                        max_workers = min(len(parallel_steps), 3)
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            future_to_step = {
                                executor.submit(self._execute_single_step, step, workflow, execution): step
                                for step in parallel_steps
                            }
                            
                            for future in as_completed(future_to_step):
                                step = future_to_step[future]
                                try:
                                    success = future.result()
                                    if success:
                                        completed_steps.add(step.step_id)
                                    else:
                                        failed_steps.add(step.step_id)
                                except Exception as e:
                                    self.logger.error(f"Step execution error: {step.step_id} - {e}")
                                    failed_steps.add(step.step_id)
                    
                    # Execute sequential steps one by one
                    for step in sequential_steps:
                        success = self._execute_single_step(step, workflow, execution)
                        if success:
                            completed_steps.add(step.step_id)
                        else:
                            failed_steps.add(step.step_id)
                
                # Update workflow progress
                workflow.completed_steps = len(completed_steps)
                workflow.failed_steps = len(failed_steps)
                workflow.progress_percentage = (len(completed_steps) / len(workflow.steps)) * 100
                
                # Update execution progress
                execution.progress_details = {
                    "completed_steps": len(completed_steps),
                    "failed_steps": len(failed_steps),
                    "total_steps": len(workflow.steps),
                    "progress_percentage": workflow.progress_percentage
                }
                
                # Brief pause between iterations
                time.sleep(1)
            
            # Finalize workflow execution
            if failed_steps:
                workflow.status = "failed"
                execution.status = "failed"
            else:
                workflow.status = "completed"
                execution.status = "completed"
            
            workflow.completion_time = datetime.now()
            workflow.estimated_completion = None
            
            # Calculate performance metrics
            if workflow.start_time and workflow.completion_time:
                total_duration = (workflow.completion_time - workflow.start_time).total_seconds()
                workflow.performance_metrics = {
                    "total_duration_seconds": total_duration,
                    "average_step_duration": total_duration / len(workflow.steps),
                    "resource_efficiency": self._calculate_resource_efficiency(workflow),
                    "success_rate": len(completed_steps) / len(workflow.steps)
                }
            
            # Release resources
            self._release_workflow_resources(workflow)
            
            # Move to history
            self.workflow_history.append(workflow)
            if workflow.workflow_id in self.active_workflows:
                del self.active_workflows[workflow.workflow_id]
            
            # Remove from running executions
            if execution.execution_id in self.running_executions:
                del self.running_executions[execution.execution_id]
            
            self.logger.info(f"Workflow execution completed: {workflow.name} - Status: {workflow.status}")
            
        except Exception as e:
            workflow.status = "failed"
            execution.status = "failed"
            self.logger.error(f"Workflow execution error: {workflow.workflow_id} - {e}")
    
    def _execute_single_step(self, step: WorkflowStep, workflow: MLWorkflowInstance, 
                           execution: WorkflowExecution) -> bool:
        """Execute a single workflow step"""
        
        try:
            step.status = "running"
            step.start_time = datetime.now()
            
            self.logger.info(f"Executing step: {step.name} ({step.step_id})")
            
            # Prepare step execution context
            execution_context = {
                "workflow_id": workflow.workflow_id,
                "step_id": step.step_id,
                "parameters": step.parameters,
                "global_context": workflow.global_context,
                "workflow_parameters": workflow.parameters,
                "inputs": step.inputs,
                "resource_allocation": self._get_step_resource_allocation(step)
            }
            
            # Execute step based on agent type and operation
            success = self._dispatch_step_execution(step, execution_context)
            
            step.completion_time = datetime.now()
            step.execution_duration = (step.completion_time - step.start_time).total_seconds()
            
            if success:
                step.status = "completed"
                self.logger.info(f"Step completed successfully: {step.name}")
            else:
                step.status = "failed"
                self.logger.error(f"Step failed: {step.name}")
            
            # Update execution current step
            execution.current_step = step.step_id
            
            return success
            
        except Exception as e:
            step.status = "failed"
            step.error_details = {"exception": str(e), "timestamp": datetime.now().isoformat()}
            step.completion_time = datetime.now()
            
            self.logger.error(f"Step execution error: {step.step_id} - {e}")
            return False
    
    def _dispatch_step_execution(self, step: WorkflowStep, context: Dict[str, Any]) -> bool:
        """Dispatch step execution to appropriate agent"""
        
        # Simulate step execution - in production, this would make actual calls to agents
        execution_time = step.resource_requirements.estimated_duration_minutes * 60 / 10  # Scaled down for demo
        
        # Simulate varying execution times and success rates
        import random
        actual_time = execution_time * random.uniform(0.8, 1.5)
        success_rate = 0.95  # 95% success rate
        
        # Simulate execution delay
        time.sleep(min(actual_time, 5))  # Cap at 5 seconds for demo
        
        # Simulate execution result
        success = random.random() < success_rate
        
        if success:
            step.result_data = {
                "execution_time": actual_time,
                "resource_usage": {
                    "cpu_percent": random.uniform(20, 80),
                    "memory_percent": random.uniform(30, 70),
                    "gpu_percent": random.uniform(0, 90) if step.resource_requirements.gpu_count > 0 else 0
                },
                "output_data": f"Step {step.name} completed successfully",
                "metrics": {
                    "throughput": random.uniform(100, 1000),
                    "latency_ms": random.uniform(10, 500),
                    "accuracy": random.uniform(0.85, 0.99)
                }
            }
        else:
            step.error_details = {
                "error_code": f"ERR_{random.randint(1000, 9999)}",
                "error_message": f"Simulated failure in step: {step.name}",
                "error_type": random.choice(["timeout", "resource_exhaustion", "data_error", "network_error"])
            }
        
        return success
    
    def _workflow_execution_loop(self):
        """Main workflow execution management loop"""
        while self.orchestration_active:
            try:
                # Process execution queue
                if self.execution_queue:
                    workflow_id = self.execution_queue.popleft()
                    
                    if workflow_id in self.active_workflows:
                        workflow = self.active_workflows[workflow_id]
                        
                        # Check if it's time to execute
                        current_time = datetime.now()
                        if (workflow.scheduled_time is None or 
                            current_time >= workflow.scheduled_time):
                            
                            # Check resource availability and concurrent workflow limit
                            max_concurrent = self.orchestration_config["execution_engine"]["max_concurrent_workflows"]
                            if (len(self.running_executions) < max_concurrent and
                                self._check_resource_availability(workflow)):
                                
                                self.execute_workflow(workflow_id)
                            else:
                                # Put back in queue for later
                                self.execution_queue.append(workflow_id)
                        else:
                            # Put back in queue - not time yet
                            self.execution_queue.append(workflow_id)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in workflow execution loop: {e}")
                time.sleep(10)
    
    def _resource_monitoring_loop(self):
        """Monitor and manage resource allocation"""
        while self.orchestration_active:
            try:
                self._update_resource_usage()
                self._optimize_resource_allocation()
                
                interval = self.orchestration_config["resource_management"]["resource_monitoring_interval"]
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(60)
    
    def _performance_optimization_loop(self):
        """Optimize workflow performance and learn from execution patterns"""
        while self.orchestration_active:
            try:
                if self.orchestration_config["optimization"]["performance_learning"]:
                    self._analyze_workflow_performance()
                    self._optimize_workflow_templates()
                
                time.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in performance optimization: {e}")
                time.sleep(600)
    
    def _checkpoint_loop(self):
        """Create periodic checkpoints of workflow state"""
        while self.orchestration_active:
            try:
                self._create_checkpoint()
                
                interval = self.orchestration_config["persistence"]["checkpoint_interval"]
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in checkpoint loop: {e}")
                time.sleep(600)
    
    def _workflow_monitoring_loop(self):
        """Monitor workflow health and performance"""
        while self.orchestration_active:
            try:
                self._monitor_workflow_health()
                self._update_performance_predictions()
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in workflow monitoring: {e}")
                time.sleep(60)
    
    # Helper methods for resource management, persistence, optimization, etc.
    def _calculate_template_resource_profile(self, steps: List[Dict[str, Any]]) -> WorkflowResource:
        """Calculate total resource profile for a template"""
        
        total_cpu = sum(step.get("cpu_cores", 2) for step in steps)
        total_memory = sum(step.get("memory_gb", 4.0) for step in steps)
        max_gpu = max(step.get("gpu_count", 0) for step in steps)
        total_storage = sum(step.get("storage_gb", 10.0) for step in steps)
        max_bandwidth = max(step.get("network_bandwidth", 100) for step in steps)
        total_duration = sum(step.get("estimated_duration", 30) for step in steps)
        
        return WorkflowResource(
            cpu_cores=total_cpu,
            memory_gb=total_memory,
            gpu_count=max_gpu,
            storage_gb=total_storage,
            network_bandwidth_mbps=max_bandwidth,
            estimated_duration_minutes=total_duration
        )
    
    def _check_resource_availability(self, workflow: MLWorkflowInstance) -> bool:
        """Check if resources are available for workflow execution"""
        
        # Calculate peak resource requirements
        peak_cpu = 0
        peak_memory = 0.0
        peak_gpu = 0
        peak_storage = 0.0
        peak_bandwidth = 0
        
        for step in workflow.steps:
            peak_cpu = max(peak_cpu, step.resource_requirements.cpu_cores)
            peak_memory = max(peak_memory, step.resource_requirements.memory_gb)
            peak_gpu = max(peak_gpu, step.resource_requirements.gpu_count)
            peak_storage = max(peak_storage, step.resource_requirements.storage_gb)
            peak_bandwidth = max(peak_bandwidth, step.resource_requirements.network_bandwidth_mbps)
        
        # Check availability
        available_cpu = self.available_resources["cpu_cores"] - self.allocated_resources["cpu_cores"]
        available_memory = self.available_resources["memory_gb"] - self.allocated_resources["memory_gb"]
        available_gpu = self.available_resources["gpu_count"] - self.allocated_resources["gpu_count"]
        available_storage = self.available_resources["storage_gb"] - self.allocated_resources["storage_gb"]
        available_bandwidth = self.available_resources["network_bandwidth_mbps"] - self.allocated_resources["network_bandwidth_mbps"]
        
        return (available_cpu >= peak_cpu and
                available_memory >= peak_memory and
                available_gpu >= peak_gpu and
                available_storage >= peak_storage and
                available_bandwidth >= peak_bandwidth)
    
    def _reserve_workflow_resources(self, workflow: MLWorkflowInstance):
        """Reserve resources for workflow execution"""
        
        for step in workflow.steps:
            self.allocated_resources["cpu_cores"] += step.resource_requirements.cpu_cores
            self.allocated_resources["memory_gb"] += step.resource_requirements.memory_gb
            self.allocated_resources["gpu_count"] += step.resource_requirements.gpu_count
            self.allocated_resources["storage_gb"] += step.resource_requirements.storage_gb
            self.allocated_resources["network_bandwidth_mbps"] += step.resource_requirements.network_bandwidth_mbps
    
    def _release_workflow_resources(self, workflow: MLWorkflowInstance):
        """Release resources after workflow completion"""
        
        for step in workflow.steps:
            self.allocated_resources["cpu_cores"] -= step.resource_requirements.cpu_cores
            self.allocated_resources["memory_gb"] -= step.resource_requirements.memory_gb
            self.allocated_resources["gpu_count"] -= step.resource_requirements.gpu_count
            self.allocated_resources["storage_gb"] -= step.resource_requirements.storage_gb
            self.allocated_resources["network_bandwidth_mbps"] -= step.resource_requirements.network_bandwidth_mbps
        
        # Ensure no negative values
        for resource in self.allocated_resources:
            self.allocated_resources[resource] = max(0, self.allocated_resources[resource])
    
    def _get_step_resource_allocation(self, step: WorkflowStep) -> Dict[str, Any]:
        """Get current resource allocation for a step"""
        return {
            "cpu_cores": step.resource_requirements.cpu_cores,
            "memory_gb": step.resource_requirements.memory_gb,
            "gpu_count": step.resource_requirements.gpu_count,
            "storage_gb": step.resource_requirements.storage_gb,
            "network_bandwidth_mbps": step.resource_requirements.network_bandwidth_mbps
        }
    
    def _calculate_resource_efficiency(self, workflow: MLWorkflowInstance) -> float:
        """Calculate resource efficiency for a workflow"""
        
        if not workflow.start_time or not workflow.completion_time:
            return 0.0
        
        total_duration = (workflow.completion_time - workflow.start_time).total_seconds()
        estimated_duration = sum(step.resource_requirements.estimated_duration_minutes * 60 for step in workflow.steps)
        
        if estimated_duration == 0:
            return 1.0
        
        return min(1.0, estimated_duration / total_duration)
    
    def _update_resource_usage(self):
        """Update current resource usage metrics"""
        # Implementation for real resource monitoring
        pass
    
    def _optimize_resource_allocation(self):
        """Optimize resource allocation across workflows"""
        # Implementation for resource optimization
        pass
    
    def _analyze_workflow_performance(self):
        """Analyze performance of completed workflows"""
        # Implementation for performance analysis
        pass
    
    def _optimize_workflow_templates(self):
        """Optimize workflow templates based on performance data"""
        # Implementation for template optimization
        pass
    
    def _create_checkpoint(self):
        """Create checkpoint of current state"""
        # Implementation for state persistence
        pass
    
    def _monitor_workflow_health(self):
        """Monitor health of running workflows"""
        # Implementation for health monitoring
        pass
    
    def _update_performance_predictions(self):
        """Update performance predictions for workflows"""
        # Implementation for prediction updates
        pass
    
    def _persist_template(self, template: WorkflowTemplate):
        """Persist template to database"""
        
        with self.db_lock:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO workflow_templates 
                (template_id, name, description, version, workflow_type, execution_mode, template_data, created_time, usage_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                template.template_id,
                template.name,
                template.description,
                template.version,
                template.workflow_type.value,
                template.execution_mode.value,
                json.dumps(asdict(template)),
                template.created_time,
                template.usage_count
            ))
            self.db_connection.commit()
    
    def _persist_workflow(self, workflow: MLWorkflowInstance):
        """Persist workflow to database"""
        
        with self.db_lock:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO workflow_instances 
                (workflow_id, template_id, name, status, created_time, start_time, completion_time, progress_percentage, workflow_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                workflow.workflow_id,
                workflow.template_id,
                workflow.name,
                workflow.status,
                workflow.created_time,
                workflow.start_time,
                workflow.completion_time,
                workflow.progress_percentage,
                json.dumps(asdict(workflow))
            ))
            self.db_connection.commit()
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration status"""
        
        active_count = len(self.active_workflows)
        running_count = len(self.running_executions)
        queued_count = len(self.execution_queue)
        template_count = len(self.workflow_templates)
        
        # Calculate resource utilization
        resource_utilization = {}
        for resource, total in self.available_resources.items():
            allocated = self.allocated_resources[resource]
            utilization = (allocated / total * 100) if total > 0 else 0
            resource_utilization[resource] = {
                "total": total,
                "allocated": allocated,
                "available": total - allocated,
                "utilization_percentage": utilization
            }
        
        return {
            "orchestrator_overview": {
                "orchestrator_id": self.orchestrator_id,
                "status": "active" if self.orchestration_active else "inactive",
                "workflow_templates": template_count,
                "active_workflows": active_count,
                "running_workflows": running_count,
                "queued_workflows": queued_count
            },
            "resource_utilization": resource_utilization,
            "recent_workflows": [
                {
                    "workflow_id": w.workflow_id,
                    "name": w.name,
                    "status": w.status,
                    "progress": w.progress_percentage,
                    "created": w.created_time.isoformat()
                }
                for w in list(self.workflow_history)[-10:]
            ],
            "running_executions": [
                {
                    "execution_id": exec.execution_id,
                    "workflow_id": exec.workflow_id,
                    "status": exec.status,
                    "current_step": exec.current_step,
                    "start_time": exec.start_time.isoformat()
                }
                for exec in self.running_executions.values()
            ],
            "performance_metrics": {
                "average_execution_time": self._calculate_average_execution_time(),
                "success_rate": self._calculate_success_rate(),
                "resource_efficiency": self._calculate_overall_resource_efficiency(),
                "throughput_per_hour": self._calculate_throughput()
            },
            "configuration": {
                "max_concurrent_workflows": self.orchestration_config["execution_engine"]["max_concurrent_workflows"],
                "max_parallel_steps": self.orchestration_config["execution_engine"]["max_parallel_steps"],
                "optimization_enabled": self.orchestration_config["optimization"]["workflow_optimization"],
                "persistence_enabled": True
            }
        }
    
    def _calculate_average_execution_time(self) -> float:
        """Calculate average workflow execution time"""
        completed_workflows = [w for w in self.workflow_history 
                             if w.start_time and w.completion_time]
        if not completed_workflows:
            return 0.0
        
        total_time = sum((w.completion_time - w.start_time).total_seconds() 
                        for w in completed_workflows)
        return total_time / len(completed_workflows)
    
    def _calculate_success_rate(self) -> float:
        """Calculate workflow success rate"""
        completed_workflows = [w for w in self.workflow_history if w.status in ["completed", "failed"]]
        if not completed_workflows:
            return 1.0
        
        successful = len([w for w in completed_workflows if w.status == "completed"])
        return successful / len(completed_workflows)
    
    def _calculate_overall_resource_efficiency(self) -> float:
        """Calculate overall resource efficiency"""
        workflows_with_metrics = [w for w in self.workflow_history 
                                if w.performance_metrics.get("resource_efficiency")]
        if not workflows_with_metrics:
            return 1.0
        
        total_efficiency = sum(w.performance_metrics["resource_efficiency"] 
                             for w in workflows_with_metrics)
        return total_efficiency / len(workflows_with_metrics)
    
    def _calculate_throughput(self) -> float:
        """Calculate workflow throughput per hour"""
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_workflows = [w for w in self.workflow_history 
                          if w.completion_time and w.completion_time > one_hour_ago]
        return len(recent_workflows)
    
    def stop_orchestration(self):
        """Stop workflow orchestration"""
        self.orchestration_active = False
        
        # Close database connection
        if hasattr(self, 'db_connection'):
            self.db_connection.close()
        
        self.logger.info("Unified workflow orchestration stopped")

def main():
    """Main function for standalone execution"""
    orchestrator = UnifiedWorkflowOrchestrator()
    
    try:
        # Create a sample workflow
        workflow_id = orchestrator.create_workflow_from_template(
            list(orchestrator.workflow_templates.keys())[0],  # Use first template
            name="Sample ML Training Workflow",
            parameters={"dataset_path": "/data/sample", "model_type": "classification"}
        )
        
        # Schedule and execute the workflow
        orchestrator.schedule_workflow(workflow_id)
        
        # Monitor status
        while True:
            status = orchestrator.get_orchestration_status()
            print(f"\n{'='*80}")
            print("UNIFIED WORKFLOW ORCHESTRATOR STATUS")
            print(f"{'='*80}")
            print(f"Templates: {status['orchestrator_overview']['workflow_templates']}")
            print(f"Active Workflows: {status['orchestrator_overview']['active_workflows']}")
            print(f"Running Workflows: {status['orchestrator_overview']['running_workflows']}")
            print(f"Success Rate: {status['performance_metrics']['success_rate']:.1%}")
            print(f"Avg Execution Time: {status['performance_metrics']['average_execution_time']:.1f}s")
            print(f"{'='*80}")
            
            time.sleep(30)  # Status update every 30 seconds
            
    except KeyboardInterrupt:
        orchestrator.stop_orchestration()
        print("\nWorkflow orchestration stopped.")

if __name__ == "__main__":
    main()
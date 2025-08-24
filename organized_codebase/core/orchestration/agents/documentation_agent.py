"""
Documentation Intelligence Agent

This agent acts as an intelligent coordinator between the classical analysis system
and the documentation generation system, providing context-aware documentation
that leverages deep code insights.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

from ...classical_analysis.analysis_orchestrator import AnalysisOrchestrator
from ...documentation.core.doc_generator import DocumentationEngine
from ...documentation.core.context_builder import AnalysisContextBuilder
from ...documentation.quality.doc_validator import DocumentationValidator


@dataclass
class DocumentationRequest:
    """Request for documentation generation."""
    request_id: str
    project_path: str
    document_types: List[str]  # ['docstrings', 'readme', 'api', 'architecture', 'tutorials']
    priority: int = 5  # 1-10, higher is more important
    context_depth: str = "standard"  # 'minimal', 'standard', 'comprehensive'
    style_preferences: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    requester: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "project_path": self.project_path,
            "document_types": self.document_types,
            "priority": self.priority,
            "context_depth": self.context_depth,
            "style_preferences": self.style_preferences,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "requester": self.requester
        }


@dataclass
class DocumentationResult:
    """Result of documentation generation."""
    request_id: str
    success: bool
    documents: Dict[str, str]  # document_type -> content
    analysis_insights: Dict[str, Any]
    quality_metrics: Dict[str, float]
    generation_time: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "success": self.success,
            "documents": self.documents,
            "analysis_insights": self.analysis_insights,
            "quality_metrics": self.quality_metrics,
            "generation_time": self.generation_time,
            "errors": self.errors,
            "warnings": self.warnings
        }


class DocumentationIntelligenceAgent:
    """
    Intelligent agent that bridges classical analysis and documentation systems.
    
    Key Responsibilities:
    - Queue and prioritize documentation requests
    - Coordinate classical analysis for documentation context
    - Generate context-aware documentation using analysis insights
    - Monitor documentation quality and suggest improvements
    - Learn from user feedback to improve future documentation
    """
    
    def __init__(self, 
                 max_concurrent_requests: int = 3,
                 analysis_timeout: int = 300,
                 documentation_timeout: int = 600):
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.analysis_orchestrator = AnalysisOrchestrator()
        self.doc_engine = DocumentationEngine()
        self.context_builder = AnalysisContextBuilder()
        self.doc_validator = DocumentationValidator()
        
        # Agent configuration
        self.max_concurrent_requests = max_concurrent_requests
        self.analysis_timeout = analysis_timeout
        self.documentation_timeout = documentation_timeout
        
        # State management
        self.request_queue: List[DocumentationRequest] = []
        self.active_requests: Dict[str, DocumentationRequest] = {}
        self.completed_requests: Dict[str, DocumentationResult] = {}
        self.failed_requests: Dict[str, DocumentationResult] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_generation_time": 0.0,
            "average_quality_score": 0.0
        }
        
        # Learning system
        self.feedback_history: List[Dict[str, Any]] = []
        self.learned_patterns: Dict[str, Any] = {}
        
        # Agent status
        self.is_running = False
        self.agent_task: Optional[asyncio.Task] = None
    
    async def start_agent(self):
        """Start the documentation intelligence agent."""
        if self.is_running:
            self.logger.warning("Agent is already running")
            return
        
        self.is_running = True
        self.agent_task = asyncio.create_task(self._agent_main_loop())
        self.logger.info("Documentation Intelligence Agent started")
    
    async def stop_agent(self):
        """Stop the documentation intelligence agent."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.agent_task:
            self.agent_task.cancel()
            try:
                await self.agent_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Documentation Intelligence Agent stopped")
    
    async def _agent_main_loop(self):
        """Main agent processing loop."""
        while self.is_running:
            try:
                # Process pending requests
                await self._process_request_queue()
                
                # Check on active requests
                await self._monitor_active_requests()
                
                # Cleanup completed requests (keep recent ones)
                self._cleanup_old_requests()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Brief pause to prevent CPU spinning
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in agent main loop: {e}")
                await asyncio.sleep(5.0)  # Longer pause on error
    
    async def submit_documentation_request(self, request: DocumentationRequest) -> str:
        """Submit a new documentation request."""
        # Validate request
        if not Path(request.project_path).exists():
            raise ValueError(f"Project path does not exist: {request.project_path}")
        
        # Add to queue
        self.request_queue.append(request)
        self.request_queue.sort(key=lambda r: (-r.priority, r.deadline or datetime.max))
        
        self.logger.info(f"Documentation request queued: {request.request_id}")
        return request.request_id
    
    async def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get the status of a documentation request."""
        # Check if in queue
        for req in self.request_queue:
            if req.request_id == request_id:
                position = self.request_queue.index(req) + 1
                return {
                    "status": "queued",
                    "position": position,
                    "estimated_start_time": self._estimate_start_time(position)
                }
        
        # Check if active
        if request_id in self.active_requests:
            return {
                "status": "processing",
                "stage": "analysis_and_documentation",
                "started_at": self.active_requests[request_id].deadline  # Reusing field for start time
            }
        
        # Check if completed
        if request_id in self.completed_requests:
            result = self.completed_requests[request_id]
            return {
                "status": "completed",
                "result": result.to_dict()
            }
        
        # Check if failed
        if request_id in self.failed_requests:
            result = self.failed_requests[request_id]
            return {
                "status": "failed",
                "result": result.to_dict()
            }
        
        return {"status": "not_found"}
    
    async def _process_request_queue(self):
        """Process requests from the queue."""
        # Don't process if at capacity
        if len(self.active_requests) >= self.max_concurrent_requests:
            return
        
        # Get next request
        if not self.request_queue:
            return
        
        request = self.request_queue.pop(0)
        
        # Move to active requests
        self.active_requests[request.request_id] = request
        
        # Process asynchronously
        asyncio.create_task(self._process_documentation_request(request))
    
    async def _process_documentation_request(self, request: DocumentationRequest):
        """Process a single documentation request."""
        start_time = datetime.now()
        request_id = request.request_id
        
        try:
            self.logger.info(f"Processing documentation request: {request_id}")
            
            # Step 1: Run classical analysis for context
            analysis_context = await self._gather_analysis_context(request)
            
            # Step 2: Generate documentation using analysis insights
            documents = await self._generate_documentation_with_context(request, analysis_context)
            
            # Step 3: Validate generated documentation
            quality_metrics = await self._validate_documentation_quality(request, documents)
            
            # Step 4: Create result
            generation_time = (datetime.now() - start_time).total_seconds()
            
            result = DocumentationResult(
                request_id=request_id,
                success=True,
                documents=documents,
                analysis_insights=analysis_context,
                quality_metrics=quality_metrics,
                generation_time=generation_time
            )
            
            # Move to completed
            self.completed_requests[request_id] = result
            self.logger.info(f"Documentation request completed: {request_id} in {generation_time:.1f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to process documentation request {request_id}: {e}")
            
            generation_time = (datetime.now() - start_time).total_seconds()
            result = DocumentationResult(
                request_id=request_id,
                success=False,
                documents={},
                analysis_insights={},
                quality_metrics={},
                generation_time=generation_time,
                errors=[str(e)]
            )
            
            # Move to failed
            self.failed_requests[request_id] = result
        
        finally:
            # Remove from active requests
            if request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def _gather_analysis_context(self, request: DocumentationRequest) -> Dict[str, Any]:
        """Gather classical analysis context for documentation generation."""
        try:
            # Configure analysis depth based on request
            analysis_config = self._get_analysis_config_for_context_depth(request.context_depth)
            
            # Run classical analysis
            analysis_results = await asyncio.wait_for(
                self.analysis_orchestrator.run_comprehensive_analysis(
                    request.project_path,
                    **analysis_config
                ),
                timeout=self.analysis_timeout
            )
            
            # Extract relevant insights for documentation
            context = {
                "project_structure": analysis_results.get("structure_analysis", {}),
                "complexity_metrics": analysis_results.get("complexity_analysis", {}),
                "security_insights": analysis_results.get("security_analysis", {}),
                "performance_patterns": analysis_results.get("performance_analysis", {}),
                "api_patterns": analysis_results.get("api_analysis", {}),
                "test_coverage": analysis_results.get("test_analysis", {}),
                "business_logic": analysis_results.get("business_analysis", {}),
                "dependencies": analysis_results.get("dependency_analysis", {}),
                "code_quality": analysis_results.get("quality_metrics", {})
            }
            
            return context
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Analysis timeout for request {request.request_id}")
            return {"error": "analysis_timeout", "limited_context": True}
        except Exception as e:
            self.logger.error(f"Analysis failed for request {request.request_id}: {e}")
            return {"error": str(e), "limited_context": True}
    
    async def _generate_documentation_with_context(self, 
                                                    request: DocumentationRequest, 
                                                    analysis_context: Dict[str, Any]) -> Dict[str, str]:
        """Generate documentation using analysis context."""
        documents = {}
        
        try:
            # Configure documentation engine with analysis context
            self.doc_engine.set_analysis_context(analysis_context)
            
            # Generate each requested document type
            for doc_type in request.document_types:
                if doc_type == "docstrings":
                    documents["docstrings"] = await self._generate_intelligent_docstrings(
                        request, analysis_context
                    )
                elif doc_type == "readme":
                    documents["readme"] = await self._generate_intelligent_readme(
                        request, analysis_context
                    )
                elif doc_type == "api":
                    documents["api"] = await self._generate_intelligent_api_docs(
                        request, analysis_context
                    )
                elif doc_type == "architecture":
                    documents["architecture"] = await self._generate_architecture_docs(
                        request, analysis_context
                    )
                elif doc_type == "tutorials":
                    documents["tutorials"] = await self._generate_tutorials(
                        request, analysis_context
                    )
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}")
            raise
    
    async def _generate_intelligent_docstrings(self, 
                                               request: DocumentationRequest, 
                                               context: Dict[str, Any]) -> str:
        """Generate intelligent docstrings using analysis context."""
        # Use complexity and API analysis to prioritize functions for documentation
        complex_functions = context.get("complexity_metrics", {}).get("high_complexity_functions", [])
        api_endpoints = context.get("api_patterns", {}).get("endpoints", [])
        
        # Generate docstrings with enhanced context
        docstring_config = {
            "style": request.style_preferences.get("docstring_style", "google"),
            "include_examples": True,
            "include_type_hints": True,
            "focus_areas": complex_functions + api_endpoints,
            "security_notes": context.get("security_insights", {}).get("recommendations", [])
        }
        
        return await self.doc_engine.generate_intelligent_docstrings(
            request.project_path, 
            config=docstring_config
        )
    
    async def _generate_intelligent_readme(self, 
                                           request: DocumentationRequest, 
                                           context: Dict[str, Any]) -> str:
        """Generate intelligent README using analysis context."""
        readme_config = {
            "project_type": context.get("project_structure", {}).get("type", "library"),
            "complexity_level": context.get("complexity_metrics", {}).get("overall_complexity", "medium"),
            "security_features": context.get("security_insights", {}).get("features", []),
            "performance_notes": context.get("performance_patterns", {}).get("optimizations", []),
            "api_overview": context.get("api_patterns", {}).get("overview", {}),
            "test_coverage_stats": context.get("test_coverage", {})
        }
        
        return await self.doc_engine.generate_intelligent_readme(
            request.project_path,
            config=readme_config
        )
    
    async def _generate_intelligent_api_docs(self, 
                                             request: DocumentationRequest, 
                                             context: Dict[str, Any]) -> str:
        """Generate intelligent API documentation using analysis context."""
        api_config = {
            "endpoints": context.get("api_patterns", {}).get("endpoints", []),
            "security_patterns": context.get("security_insights", {}).get("api_security", {}),
            "performance_considerations": context.get("performance_patterns", {}).get("api_performance", {}),
            "error_handling": context.get("api_patterns", {}).get("error_patterns", []),
            "authentication": context.get("security_insights", {}).get("auth_patterns", {})
        }
        
        return await self.doc_engine.generate_intelligent_api_docs(
            request.project_path,
            config=api_config
        )
    
    async def _generate_architecture_docs(self, 
                                          request: DocumentationRequest, 
                                          context: Dict[str, Any]) -> str:
        """Generate architecture documentation using analysis context."""
        arch_config = {
            "structure": context.get("project_structure", {}),
            "dependencies": context.get("dependencies", {}),
            "complexity_analysis": context.get("complexity_metrics", {}),
            "security_architecture": context.get("security_insights", {}),
            "performance_patterns": context.get("performance_patterns", {})
        }
        
        return await self.doc_engine.generate_architecture_docs(
            request.project_path,
            config=arch_config
        )
    
    async def _generate_tutorials(self, 
                                  request: DocumentationRequest, 
                                  context: Dict[str, Any]) -> str:
        """Generate tutorials using analysis context."""
        tutorial_config = {
            "complexity_level": context.get("complexity_metrics", {}).get("overall_complexity", "medium"),
            "api_examples": context.get("api_patterns", {}).get("examples", []),
            "business_flows": context.get("business_logic", {}).get("workflows", []),
            "common_patterns": context.get("code_quality", {}).get("patterns", [])
        }
        
        return await self.doc_engine.generate_tutorials(
            request.project_path,
            config=tutorial_config
        )
    
    async def _validate_documentation_quality(self, 
                                               request: DocumentationRequest, 
                                               documents: Dict[str, str]) -> Dict[str, float]:
        """Validate the quality of generated documentation."""
        quality_metrics = {}
        
        for doc_type, content in documents.items():
            try:
                # Basic validation
                validation_result = await self.doc_validator.validate_content(content, doc_type)
                
                # Calculate quality score (0-1)
                quality_score = self._calculate_quality_score(validation_result, doc_type)
                quality_metrics[doc_type] = quality_score
                
            except Exception as e:
                self.logger.error(f"Quality validation failed for {doc_type}: {e}")
                quality_metrics[doc_type] = 0.0
        
        return quality_metrics
    
    def _calculate_quality_score(self, validation_result: Any, doc_type: str) -> float:
        """Calculate a quality score for documentation."""
        # Simplified quality scoring - can be enhanced
        if hasattr(validation_result, 'score'):
            return validation_result.score
        elif hasattr(validation_result, 'is_valid'):
            return 1.0 if validation_result.is_valid else 0.5
        else:
            # Default scoring based on content length and structure
            return 0.8  # Placeholder
    
    def _get_analysis_config_for_context_depth(self, depth: str) -> Dict[str, Any]:
        """Get analysis configuration based on requested context depth."""
        if depth == "minimal":
            return {
                "enable_security_analysis": False,
                "enable_performance_analysis": False,
                "enable_business_analysis": False,
                "analysis_depth": "basic"
            }
        elif depth == "comprehensive":
            return {
                "enable_security_analysis": True,
                "enable_performance_analysis": True,
                "enable_business_analysis": True,
                "enable_ml_analysis": True,
                "analysis_depth": "deep"
            }
        else:  # standard
            return {
                "enable_security_analysis": True,
                "enable_performance_analysis": True,
                "enable_business_analysis": False,
                "analysis_depth": "standard"
            }
    
    def _estimate_start_time(self, queue_position: int) -> datetime:
        """Estimate when a queued request will start processing."""
        # Estimate based on current active requests and average processing time
        avg_time = self.performance_metrics.get("average_generation_time", 300)  # 5 minutes default
        concurrent_slots = self.max_concurrent_requests
        
        # Account for currently active requests
        active_count = len(self.active_requests)
        remaining_slots = max(0, concurrent_slots - active_count)
        
        if queue_position <= remaining_slots:
            # Can start immediately
            return datetime.now()
        else:
            # Need to wait for slots to open
            batches_to_wait = (queue_position - remaining_slots - 1) // concurrent_slots + 1
            estimated_wait = batches_to_wait * avg_time
            return datetime.now() + timedelta(seconds=estimated_wait)
    
    async def _monitor_active_requests(self):
        """Monitor active requests for timeouts or issues."""
        current_time = datetime.now()
        
        for request_id, request in list(self.active_requests.items()):
            # Check for deadline timeout
            if request.deadline and current_time > request.deadline:
                self.logger.warning(f"Request {request_id} exceeded deadline")
                # Could implement deadline handling here
            
            # Check for processing timeout (would need start time tracking)
            # This is simplified - in real implementation, track start times
    
    def _cleanup_old_requests(self):
        """Clean up old completed/failed requests to prevent memory leaks."""
        cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours of history
        
        # This is simplified - would need actual timestamp tracking
        if len(self.completed_requests) > 100:  # Simple limit
            # Remove oldest requests (simplified approach)
            oldest_keys = list(self.completed_requests.keys())[:50]
            for key in oldest_keys:
                del self.completed_requests[key]
        
        if len(self.failed_requests) > 50:
            oldest_keys = list(self.failed_requests.keys())[:25]
            for key in oldest_keys:
                del self.failed_requests[key]
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        total_completed = len(self.completed_requests)
        total_failed = len(self.failed_requests)
        
        self.performance_metrics.update({
            "total_requests": total_completed + total_failed,
            "successful_requests": total_completed,
            "failed_requests": total_failed
        })
        
        if total_completed > 0:
            # Calculate averages (simplified)
            avg_time = sum(r.generation_time for r in self.completed_requests.values()) / total_completed
            avg_quality = sum(
                sum(r.quality_metrics.values()) / len(r.quality_metrics) 
                for r in self.completed_requests.values() 
                if r.quality_metrics
            ) / total_completed if total_completed > 0 else 0.0
            
            self.performance_metrics.update({
                "average_generation_time": avg_time,
                "average_quality_score": avg_quality
            })
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        return {
            "is_running": self.is_running,
            "queue_length": len(self.request_queue),
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.completed_requests),
            "failed_requests": len(self.failed_requests),
            "performance_metrics": self.performance_metrics.copy(),
            "configuration": {
                "max_concurrent_requests": self.max_concurrent_requests,
                "analysis_timeout": self.analysis_timeout,
                "documentation_timeout": self.documentation_timeout
            }
        }
    
    async def add_feedback(self, request_id: str, feedback: Dict[str, Any]):
        """Add user feedback for learning and improvement."""
        if request_id in self.completed_requests:
            feedback_entry = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "feedback": feedback,
                "result": self.completed_requests[request_id].to_dict()
            }
            self.feedback_history.append(feedback_entry)
            
            # Simple learning - track patterns
            self._update_learned_patterns(feedback_entry)
    
    def _update_learned_patterns(self, feedback_entry: Dict[str, Any]):
        """Update learned patterns based on feedback."""
        # Simplified learning system - can be enhanced with ML
        feedback = feedback_entry["feedback"]
        
        if "style_preference" in feedback:
            style = feedback["style_preference"]
            if "preferred_styles" not in self.learned_patterns:
                self.learned_patterns["preferred_styles"] = {}
            
            self.learned_patterns["preferred_styles"][style] = \
                self.learned_patterns["preferred_styles"].get(style, 0) + 1
        
        if "quality_rating" in feedback:
            rating = feedback["quality_rating"]
            if "quality_feedback" not in self.learned_patterns:
                self.learned_patterns["quality_feedback"] = []
            
            self.learned_patterns["quality_feedback"].append({
                "rating": rating,
                "patterns": feedback_entry["result"]["analysis_insights"]
            })


# Factory function for easy instantiation
def create_documentation_intelligence_agent(**config) -> DocumentationIntelligenceAgent:
    """Create a documentation intelligence agent with configuration."""
    return DocumentationIntelligenceAgent(**config)
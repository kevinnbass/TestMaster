"""
Intelligence Integration Framework - Agent B Implementation
Unified framework for integrating all AI/ML intelligence components

Key Features:
- Cross-component data flow orchestration
- Unified intelligence API
- Real-time analysis coordination
- Performance optimization
- Intelligent caching and data sharing
- Component health monitoring

Enhanced Features (Hours 10-20):
- Orchestration integration with OrchestratorBase
- Analytics integration with BaseAnalytics
- Converter integration with ConverterStrategy
- Unified processing protocols
- Enterprise-grade consolidation patterns
"""

import json
import logging
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time
import hashlib
import queue

# Enhanced orchestration imports (Hours 10-20 enhancements)
try:
    from ...orchestration.foundations.abstractions.orchestrator_base import (
        OrchestratorBase, OrchestratorType, OrchestratorCapabilities, ExecutionStrategy
    )
    from ....analytics.core.base_analytics import (
        BaseAnalytics, AnalyticsConfig, ProcessingStrategy
    )
    from ...orchestration.converters.converter_base import (
        ConverterStrategy, ConversionConfig
    )
    ORCHESTRATION_AVAILABLE = True
except ImportError:
    ORCHESTRATION_AVAILABLE = False
    logger.warning("Enhanced orchestration components not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IntelligenceRequest:
    """Represents a request for intelligence analysis"""
    request_id: str = field(default_factory=lambda: f"req_{int(datetime.now().timestamp() * 1000000)}")
    request_type: str = ""
    target_path: str = ""
    analysis_types: List[str] = field(default_factory=list)
    priority: str = "medium"  # low, medium, high, critical
    context: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'request_id': self.request_id,
            'request_type': self.request_type,
            'target_path': self.target_path,
            'analysis_types': self.analysis_types,
            'priority': self.priority,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class IntelligenceResult:
    """Represents the result of intelligence analysis"""
    result_id: str = field(default_factory=lambda: f"result_{int(datetime.now().timestamp() * 1000000)}")
    request_id: str = ""
    component_type: str = ""
    analysis_data: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    status: str = "completed"  # completed, failed, partial
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'result_id': self.result_id,
            'request_id': self.request_id,
            'component_type': self.component_type,
            'analysis_data': self.analysis_data,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'status': self.status,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ComponentHealth:
    """Represents the health status of an intelligence component"""
    component_name: str = ""
    status: str = "unknown"  # healthy, degraded, failed, unknown
    last_check: datetime = field(default_factory=datetime.now)
    response_time: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    total_requests: int = 0
    memory_usage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'component_name': self.component_name,
            'status': self.status,
            'last_check': self.last_check.isoformat(),
            'response_time': self.response_time,
            'success_rate': self.success_rate,
            'error_count': self.error_count,
            'total_requests': self.total_requests,
            'memory_usage': self.memory_usage
        }


class IntelligenceCache:
    """
    Intelligent caching system for analysis results
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self._lock = threading.RLock()
    
    def _generate_key(self, request: IntelligenceRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            'target_path': request.target_path,
            'analysis_types': sorted(request.analysis_types),
            'context_hash': hashlib.md5(str(sorted(request.context.items())).encode()).hexdigest()
        }
        return hashlib.md5(str(key_data).encode()).hexdigest()
    
    def get(self, request: IntelligenceRequest) -> Optional[IntelligenceResult]:
        """Get cached result for request"""
        with self._lock:
            key = self._generate_key(request)
            
            if key in self.cache:
                # Check TTL
                if datetime.now() - self.creation_times[key] < timedelta(seconds=self.ttl_seconds):
                    self.access_times[key] = datetime.now()
                    return self.cache[key]
                else:
                    # Remove expired entry
                    del self.cache[key]
                    del self.access_times[key]
                    del self.creation_times[key]
            
            return None
    
    def put(self, request: IntelligenceRequest, result: IntelligenceResult):
        """Store result in cache"""
        with self._lock:
            key = self._generate_key(request)
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            self.cache[key] = result
            self.access_times[key] = datetime.now()
            self.creation_times[key] = datetime.now()
    
    def _evict_oldest(self):
        """Evict oldest accessed item"""
        if self.access_times:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            del self.creation_times[oldest_key]
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1)
            }


class IntelligenceComponentManager:
    """
    Manages individual intelligence components
    """
    
    def __init__(self):
        self.components = {}
        self.component_health = {}
        self.component_configs = {}
        self._health_check_interval = 60  # seconds
        self._monitoring_thread = None
        self._monitoring_active = False
    
    def register_component(self, name: str, component: Any, config: Dict[str, Any] = None):
        """Register an intelligence component"""
        self.components[name] = component
        self.component_configs[name] = config or {}
        self.component_health[name] = ComponentHealth(component_name=name)
        
        logger.info(f"Registered intelligence component: {name}")
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get a registered component"""
        return self.components.get(name)
    
    def start_monitoring(self):
        """Start component health monitoring"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(target=self._monitor_components, daemon=True)
            self._monitoring_thread.start()
            logger.info("Started component health monitoring")
    
    def stop_monitoring(self):
        """Stop component health monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Stopped component health monitoring")
    
    def _monitor_components(self):
        """Monitor component health in background thread"""
        while self._monitoring_active:
            for name, component in self.components.items():
                try:
                    start_time = time.time()
                    
                    # Perform health check
                    if hasattr(component, 'health_check'):
                        health_status = component.health_check()
                    else:
                        health_status = 'healthy'  # Assume healthy if no health check
                    
                    response_time = time.time() - start_time
                    
                    # Update health status
                    health = self.component_health[name]
                    health.status = health_status
                    health.last_check = datetime.now()
                    health.response_time = response_time
                    
                except Exception as e:
                    logger.error(f"Health check failed for component {name}: {e}")
                    health = self.component_health[name]
                    health.status = 'failed'
                    health.last_check = datetime.now()
                    health.error_count += 1
            
            time.sleep(self._health_check_interval)
    
    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all components"""
        return {name: health.to_dict() for name, health in self.component_health.items()}
    
    def update_component_stats(self, component_name: str, success: bool, processing_time: float):
        """Update component performance statistics"""
        if component_name in self.component_health:
            health = self.component_health[component_name]
            health.total_requests += 1
            health.response_time = (health.response_time + processing_time) / 2  # Moving average
            
            if not success:
                health.error_count += 1
            
            health.success_rate = (health.total_requests - health.error_count) / health.total_requests


class IntelligenceOrchestrator:
    """
    Orchestrates intelligence analysis across multiple components
    """
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.request_queue = queue.PriorityQueue()
        self.result_handlers = {}
        self._processing_active = False
        self._processing_thread = None
    
    def start_processing(self):
        """Start request processing"""
        if not self._processing_active:
            self._processing_active = True
            self._processing_thread = threading.Thread(target=self._process_requests, daemon=True)
            self._processing_thread.start()
            logger.info("Started intelligence request processing")
    
    def stop_processing(self):
        """Stop request processing"""
        self._processing_active = False
        if self._processing_thread:
            self._processing_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("Stopped intelligence request processing")
    
    def submit_request(self, request: IntelligenceRequest) -> str:
        """Submit intelligence analysis request"""
        # Priority mapping: critical=0, high=1, medium=2, low=3
        priority_map = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        priority = priority_map.get(request.priority, 2)
        
        self.request_queue.put((priority, request))
        logger.debug(f"Submitted request {request.request_id} with priority {request.priority}")
        
        return request.request_id
    
    def register_result_handler(self, component_type: str, handler: Callable[[IntelligenceResult], None]):
        """Register result handler for component type"""
        self.result_handlers[component_type] = handler
    
    def _process_requests(self):
        """Process requests from queue"""
        while self._processing_active:
            try:
                if not self.request_queue.empty():
                    priority, request = self.request_queue.get(timeout=1)
                    
                    # Submit to executor for processing
                    future = self.executor.submit(self._execute_request, request)
                    
                    # Handle result when complete
                    future.add_done_callback(lambda f: self._handle_result(f.result()))
                else:
                    time.sleep(0.1)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in request processing: {e}")
    
    def _execute_request(self, request: IntelligenceRequest) -> IntelligenceResult:
        """Execute intelligence analysis request"""
        # This would integrate with actual component execution
        # For now, return a placeholder result
        result = IntelligenceResult(
            request_id=request.request_id,
            component_type="orchestrator",
            analysis_data={"message": "Request processed by orchestrator"},
            confidence_score=0.8,
            processing_time=1.0,
            status="completed"
        )
        
        return result
    
    def _handle_result(self, result: IntelligenceResult):
        """Handle completed analysis result"""
        component_type = result.component_type
        
        if component_type in self.result_handlers:
            try:
                self.result_handlers[component_type](result)
            except Exception as e:
                logger.error(f"Error in result handler for {component_type}: {e}")
        
        logger.debug(f"Processed result {result.result_id} from {component_type}")


class IntelligenceIntegrationFramework:
    """
    Main Intelligence Integration Framework - Agent B Implementation
    
    Coordinates all intelligence components with unified API,
    intelligent caching, and performance optimization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core components
        self.cache = IntelligenceCache(
            max_size=self.config.get('cache_size', 1000),
            ttl_seconds=self.config.get('cache_ttl', 3600)
        )
        
        self.component_manager = IntelligenceComponentManager()
        self.orchestrator = IntelligenceOrchestrator(
            max_workers=self.config.get('max_workers', 10)
        )
        
        # Enhanced orchestration integration (Hours 10-20)
        self.orchestration_enabled = ORCHESTRATION_AVAILABLE and self.config.get('enable_orchestration', True)
        self.master_orchestrator = None
        self.analytics_processors = {}
        self.converters = {}
        
        if self.orchestration_enabled:
            self._initialize_enhanced_orchestration()
        
        # Analysis pipeline
        self.analysis_pipeline = []
        self.pipeline_configs = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_response_time': 0.0,
            'error_rate': 0.0
        }
        
        # Initialize intelligence components
        self._initialize_components()
        
        # Start background services
        self.component_manager.start_monitoring()
        self.orchestrator.start_processing()
    
    def _initialize_enhanced_orchestration(self):
        """Initialize enhanced orchestration integration (Hours 10-20)"""
        try:
            # Create master orchestrator for intelligence coordination
            class IntelligenceOrchestrator(OrchestratorBase):
                """Intelligence-specialized orchestrator"""
                
                def __init__(self, framework_instance):
                    super().__init__(
                        orchestrator_type=OrchestratorType.INTELLIGENCE,
                        name="IntelligenceOrchestrator"
                    )
                    self.framework = framework_instance
                    
                    # Set enhanced capabilities
                    self.capabilities.supports_workflow_design = True
                    self.capabilities.supports_workflow_optimization = True
                    self.capabilities.supports_semantic_learning = True
                    self.capabilities.supports_cross_system_coordination = True
                    self.capabilities.supports_adaptive_execution = True
                    self.capabilities.supports_intelligent_routing = True
                    self.capabilities.workflow_patterns.update({
                        'intelligence_analysis', 'semantic_processing', 'cross_system_coordination'
                    })
                    self.capabilities.semantic_capabilities.update({
                        'concept_extraction', 'relationship_discovery', 'behavior_detection'
                    })
                
                async def execute_task(self, task: Any) -> Any:
                    """Execute intelligence task with orchestration"""
                    if isinstance(task, dict) and task.get('type') == 'intelligence_analysis':
                        return await self.framework._orchestrated_intelligence_analysis(task)
                    elif isinstance(task, dict) and task.get('type') == 'conversion':
                        return await self.framework._orchestrated_conversion(task)
                    else:
                        # Default task execution
                        return {"status": "completed", "result": task}
                
                async def execute_batch(self, tasks: List[Any]) -> Dict[str, Any]:
                    """Execute batch of intelligence tasks"""
                    results = []
                    for task in tasks:
                        result = await self.execute_task(task)
                        results.append(result)
                    return {"batch_results": results, "total": len(tasks)}
                
                async def start(self) -> bool:
                    """Start the intelligence orchestrator"""
                    return await super().initialize()
                
                async def stop(self) -> bool:
                    """Stop the intelligence orchestrator"""
                    return await super().shutdown()
                
                def get_supported_capabilities(self) -> OrchestratorCapabilities:
                    """Get intelligence orchestrator capabilities"""
                    return self.capabilities
            
            # Initialize the master orchestrator
            self.master_orchestrator = IntelligenceOrchestrator(self)
            
            # Initialize analytics processors with enhanced capabilities
            analytics_config = AnalyticsConfig(
                component_name="IntelligenceAnalytics",
                processing_strategy=ProcessingStrategy.INTELLIGENT,
                enable_workflow_analytics=True,
                enable_semantic_analytics=True,
                enable_coordination_analytics=True,
                consolidation_enabled=True,
                cross_system_integration=True,
                intelligent_routing=True,
                pattern_recognition=True
            )
            
            # Create enhanced analytics processor
            class IntelligenceAnalyticsProcessor(BaseAnalytics):
                """Intelligence-specialized analytics processor"""
                
                def __init__(self, config, framework_instance):
                    super().__init__(config)
                    self.framework = framework_instance
                
                async def process(self, data: Any) -> Any:
                    """Process intelligence analytics data"""
                    return self.create_result(
                        success=True,
                        message="Intelligence analytics processed",
                        data={"processed": True, "input_type": type(data).__name__}
                    )
                
                def get_status(self) -> Dict[str, Any]:
                    """Get analytics processor status"""
                    base_status = self.get_base_status()
                    base_status.update({
                        'intelligence_integration': True,
                        'orchestration_enabled': True
                    })
                    return base_status
            
            self.analytics_processors['intelligence'] = IntelligenceAnalyticsProcessor(
                analytics_config, self
            )
            
            logger.info("Enhanced orchestration integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced orchestration: {e}")
            self.orchestration_enabled = False
    
    async def _orchestrated_intelligence_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform orchestrated intelligence analysis using enhanced protocols"""
        try:
            # Use enhanced analytics processor
            if 'intelligence' in self.analytics_processors:
                processor = self.analytics_processors['intelligence']
                result = await processor.process_with_strategy(
                    task, ProcessingStrategy.INTELLIGENT
                )
                return {
                    "status": "completed",
                    "orchestrated": True,
                    "result": result.__dict__ if hasattr(result, '__dict__') else str(result)
                }
            else:
                # Fall back to standard processing
                return {"status": "completed", "orchestrated": False, "fallback": True}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _orchestrated_conversion(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform orchestrated conversion using enhanced protocols"""
        try:
            # This would integrate with converter strategies if available
            return {
                "status": "completed",
                "orchestrated": True,
                "conversion_result": "placeholder"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive status including orchestration enhancements"""
        base_status = {
            'framework_name': 'IntelligenceIntegrationFramework',
            'orchestration_enabled': self.orchestration_enabled,
            'components_registered': len(self.component_manager.components),
            'cache_entries': len(self.cache.cache),
            'performance_metrics': self.performance_metrics.copy()
        }
        
        if self.orchestration_enabled and self.master_orchestrator:
            orchestrator_status = self.master_orchestrator.get_status_summary()
            base_status['orchestrator'] = orchestrator_status
            
            # Add analytics processor status
            analytics_status = {}
            for name, processor in self.analytics_processors.items():
                analytics_status[name] = processor.get_status()
            base_status['analytics_processors'] = analytics_status
        
        return base_status
    
    async def process_with_unified_protocols(self, request: 'IntelligenceRequest') -> 'IntelligenceResult':
        """Process request using unified orchestration and analytics protocols"""
        if not self.orchestration_enabled:
            # Fall back to standard processing
            return await self.process_request(request)
        
        try:
            # Create orchestrated task
            orchestrated_task = {
                'type': 'intelligence_analysis',
                'request_id': request.request_id,
                'target_path': request.target_path,
                'analysis_types': request.analysis_types,
                'context': request.context
            }
            
            # Execute through master orchestrator
            result = await self.master_orchestrator.execute_task(orchestrated_task)
            
            # Convert to IntelligenceResult format
            return IntelligenceResult(
                request_id=request.request_id,
                success=result.get('status') == 'completed',
                results=result,
                metadata={
                    'orchestrated': True,
                    'processing_time': 0.0,  # Would be calculated in production
                    'orchestrator_used': self.master_orchestrator.name
                }
            )
            
        except Exception as e:
            logger.error(f"Unified protocol processing failed: {e}")
            # Fall back to standard processing
            return await self.process_request(request)
    
    def _initialize_components(self):
        """Initialize all intelligence components"""
        try:
            # Import and register components
            from ..analysis.advanced_code_analyzer import create_advanced_code_analyzer
            from ..analysis.business_logic_analyzer import create_business_logic_analyzer
            from ..analysis.technical_debt_analyzer import TechnicalDebtAnalyzer
            from ..analysis.ai_code_understanding_engine import create_ai_code_understanding_engine
            
            # Register components
            self.component_manager.register_component(
                'advanced_code_analyzer',
                create_advanced_code_analyzer(),
                {'priority': 'high', 'parallel': True}
            )
            
            self.component_manager.register_component(
                'business_logic_analyzer',
                create_business_logic_analyzer(),
                {'priority': 'medium', 'parallel': True}
            )
            
            self.component_manager.register_component(
                'technical_debt_analyzer',
                TechnicalDebtAnalyzer(),
                {'priority': 'medium', 'parallel': True}
            )
            
            self.component_manager.register_component(
                'ai_code_understanding',
                create_ai_code_understanding_engine(),
                {'priority': 'low', 'parallel': False}  # AI analysis is more resource intensive
            )
            
            logger.info("Initialized all intelligence components")
            
        except ImportError as e:
            logger.warning(f"Could not import some intelligence components: {e}")
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def analyze_code(self, target_path: str, analysis_types: List[str] = None, priority: str = "medium", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform comprehensive code analysis using multiple intelligence components
        """
        start_time = time.time()
        
        try:
            # Create request
            request = IntelligenceRequest(
                request_type="code_analysis",
                target_path=target_path,
                analysis_types=analysis_types or ['all'],
                priority=priority,
                context=context or {}
            )
            
            # Check cache first
            cached_result = self.cache.get(request)
            if cached_result:
                self.performance_metrics['cache_hits'] += 1
                logger.debug(f"Cache hit for request {request.request_id}")
                return cached_result.analysis_data
            
            self.performance_metrics['cache_misses'] += 1
            
            # Determine which components to run
            components_to_run = self._determine_components(analysis_types or ['all'])
            
            # Execute analysis
            results = self._execute_parallel_analysis(request, components_to_run)
            
            # Aggregate results
            aggregated_result = self._aggregate_results(results)
            
            # Create final result
            final_result = IntelligenceResult(
                request_id=request.request_id,
                component_type="integration_framework",
                analysis_data=aggregated_result,
                confidence_score=self._calculate_overall_confidence(results),
                processing_time=time.time() - start_time,
                status="completed"
            )
            
            # Cache result
            self.cache.put(request, final_result)
            
            # Update metrics
            self._update_performance_metrics(final_result)
            
            return aggregated_result
            
        except Exception as e:
            logger.error(f"Error in code analysis: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - start_time
            }
    
    def _determine_components(self, analysis_types: List[str]) -> List[str]:
        """Determine which components to run based on analysis types"""
        component_mapping = {
            'code_analysis': ['advanced_code_analyzer'],
            'business_logic': ['business_logic_analyzer'],
            'technical_debt': ['technical_debt_analyzer'],
            'ai_understanding': ['ai_code_understanding'],
            'all': list(self.component_manager.components.keys())
        }
        
        components = set()
        for analysis_type in analysis_types:
            if analysis_type in component_mapping:
                components.update(component_mapping[analysis_type])
        
        return list(components)
    
    def _execute_parallel_analysis(self, request: IntelligenceRequest, components: List[str]) -> List[IntelligenceResult]:
        """Execute analysis across multiple components in parallel"""
        results = []
        futures = []
        
        with ThreadPoolExecutor(max_workers=len(components)) as executor:
            for component_name in components:
                future = executor.submit(self._execute_component_analysis, component_name, request)
                futures.append((component_name, future))
            
            for component_name, future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout per component
                    results.append(result)
                    
                    # Update component stats
                    self.component_manager.update_component_stats(
                        component_name, 
                        result.status == "completed",
                        result.processing_time
                    )
                    
                except Exception as e:
                    logger.error(f"Component {component_name} failed: {e}")
                    
                    # Create error result
                    error_result = IntelligenceResult(
                        request_id=request.request_id,
                        component_type=component_name,
                        status="failed",
                        error_message=str(e)
                    )
                    results.append(error_result)
                    
                    # Update component stats
                    self.component_manager.update_component_stats(component_name, False, 0.0)
        
        return results
    
    def _execute_component_analysis(self, component_name: str, request: IntelligenceRequest) -> IntelligenceResult:
        """Execute analysis for a single component"""
        start_time = time.time()
        
        try:
            component = self.component_manager.get_component(component_name)
            if not component:
                raise ValueError(f"Component {component_name} not found")
            
            # Execute component-specific analysis
            if component_name == 'advanced_code_analyzer':
                analysis_data = component.analyze_codebase(request.target_path)
            elif component_name == 'business_logic_analyzer':
                analysis_data = component.analyze_business_logic(request.target_path)
            elif component_name == 'technical_debt_analyzer':
                analysis_data = component.analyze_project(request.target_path)
            elif component_name == 'ai_code_understanding':
                # For AI understanding, we need to read the code first
                if Path(request.target_path).is_file():
                    with open(request.target_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    understanding = component.understand_code(code, request.context)
                    analysis_data = understanding.to_dict()
                else:
                    analysis_data = {'error': 'AI understanding requires a single file'}
            else:
                analysis_data = {'error': f'Unknown component: {component_name}'}
            
            return IntelligenceResult(
                request_id=request.request_id,
                component_type=component_name,
                analysis_data=analysis_data,
                confidence_score=self._extract_confidence_score(analysis_data),
                processing_time=time.time() - start_time,
                status="completed"
            )
            
        except Exception as e:
            return IntelligenceResult(
                request_id=request.request_id,
                component_type=component_name,
                status="failed",
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _extract_confidence_score(self, analysis_data: Dict[str, Any]) -> float:
        """Extract confidence score from analysis data"""
        if 'confidence_score' in analysis_data:
            return analysis_data['confidence_score']
        elif 'error' in analysis_data:
            return 0.0
        else:
            return 0.8  # Default confidence for successful analysis
    
    def _aggregate_results(self, results: List[IntelligenceResult]) -> Dict[str, Any]:
        """Aggregate results from multiple components"""
        aggregated = {
            'integration_metadata': {
                'total_components': len(results),
                'successful_components': len([r for r in results if r.status == "completed"]),
                'failed_components': len([r for r in results if r.status == "failed"]),
                'processing_times': {r.component_type: r.processing_time for r in results}
            },
            'component_results': {}
        }
        
        for result in results:
            aggregated['component_results'][result.component_type] = {
                'status': result.status,
                'confidence_score': result.confidence_score,
                'processing_time': result.processing_time,
                'data': result.analysis_data,
                'error_message': result.error_message
            }
        
        # Generate unified insights
        aggregated['unified_insights'] = self._generate_unified_insights(results)
        
        return aggregated
    
    def _generate_unified_insights(self, results: List[IntelligenceResult]) -> Dict[str, Any]:
        """Generate unified insights across all component results"""
        insights = {
            'summary': [],
            'recommendations': [],
            'quality_score': 0.0,
            'risk_assessment': 'low'
        }
        
        successful_results = [r for r in results if r.status == "completed"]
        
        if not successful_results:
            insights['summary'].append("No successful analysis results available")
            return insights
        
        # Aggregate recommendations
        all_recommendations = []
        for result in successful_results:
            data = result.analysis_data
            if isinstance(data, dict):
                recommendations = data.get('recommendations', [])
                if isinstance(recommendations, list):
                    all_recommendations.extend(recommendations)
        
        # Deduplicate and prioritize recommendations
        unique_recommendations = list(set(all_recommendations))
        insights['recommendations'] = unique_recommendations[:10]  # Top 10 recommendations
        
        # Calculate quality score
        confidence_scores = [r.confidence_score for r in successful_results]
        insights['quality_score'] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Risk assessment
        if insights['quality_score'] < 0.4:
            insights['risk_assessment'] = 'high'
        elif insights['quality_score'] < 0.7:
            insights['risk_assessment'] = 'medium'
        else:
            insights['risk_assessment'] = 'low'
        
        # Generate summary
        insights['summary'] = [
            f"Analyzed with {len(successful_results)} intelligence components",
            f"Overall quality score: {insights['quality_score']:.2f}",
            f"Risk assessment: {insights['risk_assessment']}",
            f"Generated {len(insights['recommendations'])} recommendations"
        ]
        
        return insights
    
    def _calculate_overall_confidence(self, results: List[IntelligenceResult]) -> float:
        """Calculate overall confidence score"""
        successful_results = [r for r in results if r.status == "completed"]
        
        if not successful_results:
            return 0.0
        
        confidence_scores = [r.confidence_score for r in successful_results]
        return sum(confidence_scores) / len(confidence_scores)
    
    def _update_performance_metrics(self, result: IntelligenceResult):
        """Update framework performance metrics"""
        self.performance_metrics['total_requests'] += 1
        
        # Update average response time
        current_avg = self.performance_metrics['average_response_time']
        total_requests = self.performance_metrics['total_requests']
        new_avg = (current_avg * (total_requests - 1) + result.processing_time) / total_requests
        self.performance_metrics['average_response_time'] = new_avg
        
        # Update error rate
        if result.status == "failed":
            errors = self.performance_metrics.get('errors', 0) + 1
            self.performance_metrics['errors'] = errors
            self.performance_metrics['error_rate'] = errors / total_requests
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get comprehensive framework status"""
        return {
            'component_health': self.component_manager.get_health_status(),
            'cache_stats': self.cache.get_stats(),
            'performance_metrics': self.performance_metrics,
            'configuration': {
                'max_workers': self.config.get('max_workers', 10),
                'cache_size': self.config.get('cache_size', 1000),
                'cache_ttl': self.config.get('cache_ttl', 3600)
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Optimize framework performance based on current metrics"""
        optimizations = []
        
        # Check cache hit rate
        cache_stats = self.cache.get_stats()
        if cache_stats['hit_rate'] < 0.3:
            # Increase cache size
            self.cache.max_size = min(self.cache.max_size * 2, 5000)
            optimizations.append("Increased cache size due to low hit rate")
        
        # Check component health
        health_status = self.component_manager.get_health_status()
        for component_name, health in health_status.items():
            if health['success_rate'] < 0.8:
                optimizations.append(f"Component {component_name} has low success rate: {health['success_rate']:.2f}")
        
        # Check response times
        if self.performance_metrics['average_response_time'] > 10.0:
            optimizations.append("High average response time detected - consider scaling")
        
        return {
            'optimizations_applied': optimizations,
            'timestamp': datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Gracefully shutdown the framework"""
        logger.info("Shutting down Intelligence Integration Framework")
        
        self.orchestrator.stop_processing()
        self.component_manager.stop_monitoring()
        self.cache.clear()
        
        logger.info("Intelligence Integration Framework shutdown complete")


# Export classes
__all__ = [
    'IntelligenceIntegrationFramework', 'IntelligenceRequest', 'IntelligenceResult',
    'ComponentHealth', 'IntelligenceCache', 'IntelligenceComponentManager', 'IntelligenceOrchestrator'
]


# Factory function for easy instantiation
def create_intelligence_integration_framework(config: Dict[str, Any] = None) -> IntelligenceIntegrationFramework:
    """Factory function to create a configured Intelligence Integration Framework"""
    return IntelligenceIntegrationFramework(config)


if __name__ == "__main__":
    # Example usage
    framework = create_intelligence_integration_framework()
    
    try:
        # Example analysis
        import os
        current_dir = os.getcwd()
        
        print("Starting comprehensive intelligence analysis...")
        
        result = framework.analyze_code(
            target_path=current_dir,
            analysis_types=['code_analysis', 'business_logic', 'technical_debt'],
            priority='high'
        )
        
        print(f"Analysis completed!")
        print(f"Successful components: {result['integration_metadata']['successful_components']}")
        print(f"Quality score: {result['unified_insights']['quality_score']:.2f}")
        print(f"Risk assessment: {result['unified_insights']['risk_assessment']}")
        
        # Get framework status
        status = framework.get_framework_status()
        print(f"\\nFramework Status:")
        print(f"Cache hit rate: {status['cache_stats']['hit_rate']:.2f}")
        print(f"Average response time: {status['performance_metrics']['average_response_time']:.2f}s")
        
        # Save results
        with open("intelligence_analysis_results.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print("\\nResults saved to intelligence_analysis_results.json")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
    
    finally:
        framework.shutdown()
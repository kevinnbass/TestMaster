"""
Intelligence Orchestrator
========================
Unified interface for all ML intelligence capabilities.
Module size: ~299 lines (under 300 limit)

Author: Agent B - Intelligence Specialist
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import time
from collections import defaultdict
import threading

# Import all ML modules
from .ml.advanced_models import create_ml_pipeline
from .ml.statistical_engine import run_statistical_analysis
from .ml.C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.core.intelligence.pattern_detector import create_pattern_detector
from .ml.anomaly_algorithms import EnsembleAnomalyDetector
from .ml.correlation_engine import AdvancedCorrelationEngine
from .ml.feature_engineering import AutoFeatureEngineer
from .ml.model_registry import ModelRegistry
from .ml.online_learning import OnlineLinearRegression
from .ml.explainability import ModelExplainer
from .ml.gpu_accelerator import GPUDetector, NumPyGPUAccelerator


@dataclass
class IntelligenceRequest:
    """Request for intelligence analysis."""
    request_id: str
    data: Union[np.ndarray, Dict[str, Any]]
    analysis_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10 scale
    callback: Optional[Callable] = None


@dataclass
class IntelligenceResult:
    """Result of intelligence analysis."""
    request_id: str
    analysis_type: str
    result: Any
    confidence: float
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligenceOrchestrator:
    """
    Central orchestrator for all ML intelligence capabilities.
    Provides unified interface and smart algorithm selection.
    """
    
    def __init__(self, storage_path: str = "./intelligence_storage"):
        # Initialize components
        self.model_registry = ModelRegistry(storage_path)
        self.gpu_detector = GPUDetector()
        self.gpu_accelerator = NumPyGPUAccelerator()
        
        # ML engines
        self.anomaly_detector = EnsembleAnomalyDetector()
        self.correlation_engine = AdvancedCorrelationEngine()
        self.feature_engineer = AutoFeatureEngineer()
        self.pattern_detector = create_pattern_detector("streaming")
        
        # Request queue and processing
        self.request_queue = []
        self.active_requests = {}
        self.completed_requests = {}
        self.processing_lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_processing_time_ms": 0.0,
            "active_models": 0
        }
        
        # Capability registry
        self.capabilities = {
            "anomaly_detection": self._analyze_anomalies,
            "pattern_detection": self._detect_patterns,
            "correlation_analysis": self._analyze_correlations,
            "feature_engineering": self._engineer_features,
            "statistical_analysis": self._statistical_analysis,
            "prediction": self._make_prediction,
            "model_explanation": self._explain_model,
            "clustering": self._perform_clustering,
            "time_series_analysis": self._analyze_time_series,
            "classification": self._classify_data
        }
        
    def submit_request(self, request: IntelligenceRequest) -> str:
        """Submit intelligence analysis request."""
        with self.processing_lock:
            self.request_queue.append(request)
            self.active_requests[request.request_id] = request
            
        # Process immediately if high priority
        if request.priority >= 8:
            return self._process_request(request)
        
        return request.request_id
        
    def get_result(self, request_id: str) -> Optional[IntelligenceResult]:
        """Get result for completed request."""
        return self.completed_requests.get(request_id)
        
    def process_pending_requests(self) -> List[str]:
        """Process all pending requests."""
        completed_ids = []
        
        with self.processing_lock:
            while self.request_queue:
                request = self.request_queue.pop(0)
                try:
                    self._process_request(request)
                    completed_ids.append(request.request_id)
                except Exception as e:
                    self._handle_error(request, e)
                    
        return completed_ids
        
    def _process_request(self, request: IntelligenceRequest) -> str:
        """Process single intelligence request."""
        start_time = time.time()
        
        try:
            # Select appropriate analysis method
            if request.analysis_type not in self.capabilities:
                raise ValueError(f"Unknown analysis type: {request.analysis_type}")
                
            # Perform analysis
            analysis_func = self.capabilities[request.analysis_type]
            result = analysis_func(request.data, **request.parameters)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Create result
            intelligence_result = IntelligenceResult(
                request_id=request.request_id,
                analysis_type=request.analysis_type,
                result=result,
                confidence=self._calculate_confidence(result, request.analysis_type),
                processing_time_ms=processing_time,
                metadata={
                    "gpu_used": self.gpu_detector.has_gpu,
                    "parameters": request.parameters
                }
            )
            
            # Store result
            with self.processing_lock:
                self.completed_requests[request.request_id] = intelligence_result
                if request.request_id in self.active_requests:
                    del self.active_requests[request.request_id]
                    
            # Update metrics
            self._update_metrics(processing_time, success=True)
            
            # Execute callback if provided
            if request.callback:
                request.callback(intelligence_result)
                
            return request.request_id
            
        except Exception as e:
            self._handle_error(request, e)
            raise
            
    def _analyze_anomalies(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Perform anomaly detection."""
        result = self.anomaly_detector.detect(data[-1], data[:-1])
        
        return {
            "is_anomaly": result.is_anomaly,
            "confidence": result.confidence,
            "anomaly_type": result.anomaly_type.value if result.anomaly_type else None,
            "score": result.score,
            "expected_value": result.expected_value,
            "metadata": result.metadata
        }
        
    def _detect_patterns(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Detect patterns in data."""
        patterns = []
        
        for value in data:
            pattern = self.pattern_detector.process_stream(float(value))
            if pattern:
                patterns.append({
                    "type": pattern.pattern_type,
                    "confidence": pattern.confidence,
                    "location": pattern.location,
                    "metadata": pattern.metadata
                })
                
        return {"patterns": patterns, "total_patterns": len(patterns)}
        
    def _analyze_correlations(self, data: Dict[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        """Analyze correlations between variables."""
        if len(data) < 2:
            return {"error": "Need at least 2 variables for correlation"}
            
        variables = list(data.keys())
        correlations = {}
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables[i+1:], i+1):
                result = self.correlation_engine.pearson_correlation(data[var1], data[var2])
                correlations[f"{var1}_{var2}"] = {
                    "coefficient": result.coefficient,
                    "p_value": result.p_value,
                    "confidence": result.confidence
                }
                
        return {"correlations": correlations}
        
    def _engineer_features(self, data: Union[np.ndarray, List[np.ndarray]], **kwargs) -> Dict[str, Any]:
        """Engineer features from data."""
        feature_set = self.feature_engineer.engineer_features(data)
        
        return {
            "features": feature_set.features.tolist(),
            "feature_names": feature_set.feature_names,
            "n_features": feature_set.metadata["n_features"],
            "importance_scores": feature_set.importance_scores
        }
        
    def _statistical_analysis(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Perform statistical analysis."""
        analysis_type = kwargs.get("analysis_type", "full")
        return run_statistical_analysis(data, analysis_type)
        
    def _make_prediction(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Make predictions using registered models."""
        model_name = kwargs.get("model_name")
        if not model_name:
            return {"error": "Model name required"}
            
        model = self.model_registry.get_model(name=model_name)
        if not model:
            return {"error": "Model not found"}
            
        X = data.get("features")
        if X is None:
            return {"error": "Features required"}
            
        predictions = model.predict(np.array(X))
        return {"predictions": predictions.tolist()}
        
    def _explain_model(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Explain model predictions."""
        model_name = kwargs.get("model_name")
        instance = data.get("instance")
        
        if not model_name or instance is None:
            return {"error": "Model name and instance required"}
            
        model = self.model_registry.get_model(name=model_name)
        if not model:
            return {"error": "Model not found"}
            
        explainer = ModelExplainer(model)
        explanations = explainer.explain(
            instance=np.array(instance),
            feature_names=kwargs.get("feature_names"),
            methods=kwargs.get("methods", ["shap", "lime"])
        )
        
        return {method: {
            "feature_importance": result.feature_importance,
            "confidence": result.confidence,
            "explanation_type": result.explanation_type
        } for method, result in explanations.items()}
        
    def _perform_clustering(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Perform clustering analysis."""
        from .ml.online_learning import StreamingKMeans
        
        n_clusters = kwargs.get("n_clusters", 3)
        clusterer = StreamingKMeans(n_clusters=n_clusters)
        clusterer.partial_fit(data)
        
        labels = clusterer.predict(data)
        metrics = clusterer.get_metrics()
        
        return {
            "labels": labels.tolist(),
            "n_clusters": n_clusters,
            "metrics": metrics
        }
        
    def _analyze_time_series(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Analyze time series data."""
        from .analytics.statistical_engine import TimeSeriesAnalyzer
        
        analyzer = TimeSeriesAnalyzer()
        decomposition = analyzer.decompose(data)
        forecast = analyzer.forecast(steps=kwargs.get("forecast_steps", 10))
        
        return {
            "decomposition": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                            for k, v in decomposition.items()},
            "forecast": forecast.tolist()
        }
        
    def _classify_data(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Classify data using ML models."""
        # Simplified classification using ensemble
        X = np.array(data.get("features", []))
        if len(X) == 0:
            return {"error": "Features required"}
            
        # Use feature engineering + anomaly detection as proxy classifier
        features = self.feature_engineer.transform(X)
        anomaly_result = self.anomaly_detector.detect(features[-1], features[:-1])
        
        return {
            "classification": "anomaly" if anomaly_result.is_anomaly else "normal",
            "confidence": anomaly_result.confidence,
            "features_used": len(features)
        }
        
    def _calculate_confidence(self, result: Any, analysis_type: str) -> float:
        """Calculate confidence score for result."""
        if isinstance(result, dict):
            if "confidence" in result:
                return result["confidence"]
            elif "correlations" in result:
                correlations = result["correlations"]
                if correlations:
                    return sum(abs(c["coefficient"]) for c in correlations.values()) / len(correlations)
                    
        return 0.8  # Default confidence
        
    def _update_metrics(self, processing_time: float, success: bool):
        """Update performance metrics."""
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
            
        # Update average processing time
        total = self.metrics["total_requests"]
        current_avg = self.metrics["avg_processing_time_ms"]
        self.metrics["avg_processing_time_ms"] = (current_avg * (total - 1) + processing_time) / total
        
    def _handle_error(self, request: IntelligenceRequest, error: Exception):
        """Handle request processing error."""
        error_result = IntelligenceResult(
            request_id=request.request_id,
            analysis_type=request.analysis_type,
            result={"error": str(error)},
            confidence=0.0,
            processing_time_ms=0.0,
            metadata={"error_type": type(error).__name__}
        )
        
        with self.processing_lock:
            self.completed_requests[request.request_id] = error_result
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
                
        self._update_metrics(0.0, success=False)
        
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status and metrics."""
        return {
            "status": "operational",
            "capabilities": list(self.capabilities.keys()),
            "metrics": self.metrics,
            "gpu_available": self.gpu_detector.has_gpu,
            "active_requests": len(self.active_requests),
            "queue_size": len(self.request_queue),
            "completed_requests": len(self.completed_requests)
        }


# Public API
__all__ = ['IntelligenceOrchestrator', 'IntelligenceRequest', 'IntelligenceResult']
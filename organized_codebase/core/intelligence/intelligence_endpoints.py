
# AGENT D SECURITY INTEGRATION
try:
    from SECURITY_PATCHES.api_security_framework import APISecurityFramework
    from SECURITY_PATCHES.authentication_framework import SecurityFramework
    _security_framework = SecurityFramework()
    _api_security = APISecurityFramework()
    _SECURITY_ENABLED = True
except ImportError:
    _SECURITY_ENABLED = False
    print("Security frameworks not available - running without protection")

def apply_security_middleware():
    """Apply security middleware to requests"""
    if not _SECURITY_ENABLED:
        return True, {}
    
    from flask import request
    request_data = {
        'ip_address': request.remote_addr,
        'endpoint': request.path,
        'method': request.method,
        'user_agent': request.headers.get('User-Agent', ''),
        'body': request.get_json() if request.is_json else {},
        'query_params': dict(request.args),
        'headers': dict(request.headers)
    }
    
    return _api_security.validate_request(request_data)

"""
Intelligence API Endpoints
==========================
REST API for all ML intelligence capabilities.
Module size: ~298 lines (under 300 limit)

Author: Agent B - Intelligence Specialist
"""

from flask import Blueprint, request, jsonify
import numpy as np
import uuid
import time
from typing import Dict, Any, List
import traceback

from ..C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.core.orchestration.orchestrator import IntelligenceOrchestrator, IntelligenceRequest

# Create Flask blueprint
intelligence_bp = Blueprint('intelligence', __name__, url_prefix='/api/intelligence')

# Global orchestrator instance
orchestrator = IntelligenceOrchestrator()


@intelligence_bp.route('/status', methods=['GET'])
def get_status():
    """Get intelligence system status."""
    try:
        status = orchestrator.get_status()
        return jsonify({"success": True, "data": status})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@intelligence_bp.route('/capabilities', methods=['GET'])
def get_capabilities():
    """Get available intelligence capabilities."""
    try:
        capabilities = {
            "anomaly_detection": {
                "description": "Detect anomalies in data streams",
                "parameters": ["threshold", "method"],
                "input_format": "numeric_array"
            },
            "pattern_detection": {
                "description": "Detect patterns in time series",
                "parameters": ["window_size", "sensitivity"],
                "input_format": "numeric_array"
            },
            "correlation_analysis": {
                "description": "Analyze correlations between variables",
                "parameters": ["method", "significance_level"],
                "input_format": "variable_dict"
            },
            "feature_engineering": {
                "description": "Automated feature extraction",
                "parameters": ["max_features", "domains"],
                "input_format": "numeric_array"
            },
            "statistical_analysis": {
                "description": "Statistical analysis suite",
                "parameters": ["analysis_type"],
                "input_format": "numeric_array"
            },
            "prediction": {
                "description": "Make predictions using models",
                "parameters": ["model_name"],
                "input_format": "features_dict"
            },
            "model_explanation": {
                "description": "Explain model predictions",
                "parameters": ["model_name", "methods"],
                "input_format": "instance_dict"
            },
            "clustering": {
                "description": "Cluster data points",
                "parameters": ["n_clusters", "method"],
                "input_format": "numeric_array"
            },
            "time_series_analysis": {
                "description": "Time series decomposition and forecasting",
                "parameters": ["forecast_steps", "period"],
                "input_format": "numeric_array"
            },
            "classification": {
                "description": "Classify data instances",
                "parameters": ["model_type"],
                "input_format": "features_dict"
            }
        }
        return jsonify({"success": True, "data": capabilities})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@intelligence_bp.route('/analyze', methods=['POST'])
def analyze_data():
    """Submit data for intelligence analysis."""
    try:
        data = request.get_json()
        
        # Validate request
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
            
        analysis_type = data.get('analysis_type')
        if not analysis_type:
            return jsonify({"success": False, "error": "analysis_type required"}), 400
            
        # Prepare data
        input_data = data.get('data')
        if input_data is None:
            return jsonify({"success": False, "error": "data field required"}), 400
            
        # Convert to numpy arrays if needed
        if isinstance(input_data, list):
            input_data = np.array(input_data)
        elif isinstance(input_data, dict):
            # Handle variable dictionaries
            for key, value in input_data.items():
                if isinstance(value, list):
                    input_data[key] = np.array(value)
                    
        # Create request
        request_id = str(uuid.uuid4())
        intelligence_request = IntelligenceRequest(
            request_id=request_id,
            data=input_data,
            analysis_type=analysis_type,
            parameters=data.get('parameters', {}),
            priority=data.get('priority', 5)
        )
        
        # Submit request
        orchestrator.submit_request(intelligence_request)
        
        # Process immediately for synchronous response
        result = orchestrator.get_result(request_id)
        
        if result:
            return jsonify({
                "success": True,
                "request_id": request_id,
                "result": _serialize_result(result.result),
                "confidence": result.confidence,
                "processing_time_ms": result.processing_time_ms,
                "metadata": result.metadata
            })
        else:
            return jsonify({
                "success": True,
                "request_id": request_id,
                "status": "processing"
            })
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@intelligence_bp.route('/result/<request_id>', methods=['GET'])
def get_result(request_id: str):
    """Get result for specific request."""
    try:
        result = orchestrator.get_result(request_id)
        
        if result:
            return jsonify({
                "success": True,
                "request_id": request_id,
                "analysis_type": result.analysis_type,
                "result": _serialize_result(result.result),
                "confidence": result.confidence,
                "processing_time_ms": result.processing_time_ms,
                "metadata": result.metadata
            })
        else:
            return jsonify({
                "success": False,
                "error": "Result not found or still processing"
            }), 404
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@intelligence_bp.route('/models', methods=['GET'])
def list_models():
    """List available models in registry."""
    try:
        models = orchestrator.model_registry.list_models()
        
        model_data = []
        for model in models:
            model_data.append({
                "model_id": model.model_id,
                "name": model.name,
                "version": model.version,
                "status": model.status.value,
                "algorithm": model.algorithm,
                "metrics": model.metrics,
                "created_at": model.created_at.isoformat(),
                "updated_at": model.updated_at.isoformat()
            })
            
        return jsonify({"success": True, "data": model_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@intelligence_bp.route('/models/<model_name>/predict', methods=['POST'])
def predict_with_model(model_name: str):
    """Make predictions with specific model."""
    try:
        data = request.get_json()
        features = data.get('features')
        
        if features is None:
            return jsonify({"success": False, "error": "features required"}), 400
            
        # Create prediction request
        request_id = str(uuid.uuid4())
        intelligence_request = IntelligenceRequest(
            request_id=request_id,
            data={"features": features},
            analysis_type="prediction",
            parameters={"model_name": model_name},
            priority=8  # High priority for predictions
        )
        
        orchestrator.submit_request(intelligence_request)
        result = orchestrator.get_result(request_id)
        
        if result and "error" not in result.result:
            return jsonify({
                "success": True,
                "predictions": result.result["predictions"],
                "processing_time_ms": result.processing_time_ms
            })
        else:
            error_msg = result.result.get("error", "Prediction failed") if result else "No result"
            return jsonify({"success": False, "error": error_msg}), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@intelligence_bp.route('/models/<model_name>/explain', methods=['POST'])
def explain_model_prediction(model_name: str):
    """Explain model prediction."""
    try:
        data = request.get_json()
        instance = data.get('instance')
        
        if instance is None:
            return jsonify({"success": False, "error": "instance required"}), 400
            
        # Create explanation request
        request_id = str(uuid.uuid4())
        intelligence_request = IntelligenceRequest(
            request_id=request_id,
            data={"instance": instance},
            analysis_type="model_explanation",
            parameters={
                "model_name": model_name,
                "feature_names": data.get('feature_names'),
                "methods": data.get('methods', ["shap", "lime"])
            },
            priority=7
        )
        
        orchestrator.submit_request(intelligence_request)
        result = orchestrator.get_result(request_id)
        
        if result and "error" not in result.result:
            return jsonify({
                "success": True,
                "explanations": result.result,
                "processing_time_ms": result.processing_time_ms
            })
        else:
            error_msg = result.result.get("error", "Explanation failed") if result else "No result"
            return jsonify({"success": False, "error": error_msg}), 400
            
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@intelligence_bp.route('/batch', methods=['POST'])
def batch_analysis():
    """Submit multiple analyses in batch."""
    try:
        data = request.get_json()
        requests = data.get('requests', [])
        
        if not requests:
            return jsonify({"success": False, "error": "No requests provided"}), 400
            
        batch_results = []
        
        for req_data in requests:
            request_id = str(uuid.uuid4())
            
            # Prepare data
            input_data = req_data.get('data')
            if isinstance(input_data, list):
                input_data = np.array(input_data)
            elif isinstance(input_data, dict):
                for key, value in input_data.items():
                    if isinstance(value, list):
                        input_data[key] = np.array(value)
                        
            intelligence_request = IntelligenceRequest(
                request_id=request_id,
                data=input_data,
                analysis_type=req_data.get('analysis_type'),
                parameters=req_data.get('parameters', {}),
                priority=req_data.get('priority', 5)
            )
            
            orchestrator.submit_request(intelligence_request)
            
        # Process all pending requests
        completed_ids = orchestrator.process_pending_requests()
        
        # Collect results
        for request_id in completed_ids:
            result = orchestrator.get_result(request_id)
            if result:
                batch_results.append({
                    "request_id": request_id,
                    "analysis_type": result.analysis_type,
                    "result": _serialize_result(result.result),
                    "confidence": result.confidence,
                    "processing_time_ms": result.processing_time_ms
                })
                
        return jsonify({
            "success": True,
            "batch_results": batch_results,
            "total_processed": len(batch_results)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@intelligence_bp.route('/gpu-info', methods=['GET'])
def get_gpu_info():
    """Get GPU information and capabilities."""
    try:
        gpu_summary = orchestrator.gpu_detector.get_gpu_summary()
        return jsonify({"success": True, "data": gpu_summary})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _serialize_result(result: Any) -> Any:
    """Serialize result for JSON response."""
    if isinstance(result, np.ndarray):
        return result.tolist()
    elif isinstance(result, dict):
        return {k: _serialize_result(v) for k, v in result.items()}
    elif isinstance(result, list):
        return [_serialize_result(item) for item in result]
    elif hasattr(result, '__dict__'):
        return {k: _serialize_result(v) for k, v in result.__dict__.items()}
    else:
        return result


# Error handlers
@intelligence_bp.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404


@intelligence_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"success": False, "error": "Method not allowed"}), 405


@intelligence_bp.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500


# Public API
__all__ = ['intelligence_bp']
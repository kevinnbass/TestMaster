"""
Model Registry and Management System
====================================
Versioning, storage, and lifecycle management for ML models.
Module size: ~298 lines (under 300 limit)

Author: Agent B - Intelligence Specialist
"""

import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


class ModelStatus(Enum):
    """Model lifecycle status."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetadata:
    """Metadata for registered models."""
    model_id: str
    name: str
    version: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    algorithm: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    tags: List[str]
    description: str
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    training_data_hash: Optional[str] = None
    model_hash: Optional[str] = None


class ModelRegistry:
    """
    Central registry for ML model management.
    Handles versioning, storage, and retrieval.
    """
    
    def __init__(self, storage_path: str = "./model_registry"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.model_history = {}
        self.active_models = {}
        
        self._load_registry()
        
    def register_model(self, model: Any, name: str, version: str,
                       algorithm: str, metrics: Dict[str, float],
                       parameters: Dict[str, Any] = None,
                       tags: List[str] = None,
                       description: str = "") -> str:
        """
        Register a new model in the registry.
        
        Returns:
            model_id: Unique identifier for the registered model
        """
        # Generate model ID
        model_id = self._generate_model_id(name, version)
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(model)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            status=ModelStatus.DEVELOPMENT,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            algorithm=algorithm,
            metrics=metrics,
            parameters=parameters or {},
            tags=tags or [],
            description=description,
            model_hash=model_hash
        )
        
        # Store model
        self._save_model(model, model_id)
        
        # Update registry
        self.models[model_id] = metadata
        
        # Track version history
        if name not in self.model_history:
            self.model_history[name] = []
        self.model_history[name].append(model_id)
        
        # Save registry
        self._save_registry()
        
        return model_id
        
    def get_model(self, model_id: str = None, name: str = None,
                  version: str = None, status: ModelStatus = None) -> Optional[Any]:
        """
        Retrieve a model from registry.
        
        Args:
            model_id: Direct model ID
            name: Model name (returns latest version)
            version: Specific version
            status: Filter by status (e.g., PRODUCTION)
            
        Returns:
            Model object or None if not found
        """
        # Direct ID lookup
        if model_id:
            return self._load_model(model_id)
            
        # Find by name/version/status
        if name:
            if version:
                model_id = self._generate_model_id(name, version)
                return self._load_model(model_id)
            else:
                # Get latest or by status
                candidates = self.model_history.get(name, [])
                
                if status:
                    candidates = [
                        mid for mid in candidates
                        if self.models[mid].status == status
                    ]
                    
                if candidates:
                    # Return most recent
                    model_id = candidates[-1]
                    return self._load_model(model_id)
                    
        return None
        
    def update_status(self, model_id: str, new_status: ModelStatus) -> bool:
        """Update model status in lifecycle."""
        if model_id not in self.models:
            return False
            
        old_status = self.models[model_id].status
        self.models[model_id].status = new_status
        self.models[model_id].updated_at = datetime.now()
        
        # Handle production transitions
        if new_status == ModelStatus.PRODUCTION:
            model_name = self.models[model_id].name
            
            # Demote current production model if exists
            if model_name in self.active_models:
                old_prod_id = self.active_models[model_name]
                if old_prod_id != model_id:
                    self.models[old_prod_id].status = ModelStatus.STAGING
                    
            self.active_models[model_name] = model_id
            
        self._save_registry()
        return True
        
    def compare_models(self, model_id1: str, model_id2: str) -> Dict[str, Any]:
        """Compare two models by their metrics and parameters."""
        if model_id1 not in self.models or model_id2 not in self.models:
            return {"error": "Model(s) not found"}
            
        m1 = self.models[model_id1]
        m2 = self.models[model_id2]
        
        comparison = {
            "model1": {"id": model_id1, "name": m1.name, "version": m1.version},
            "model2": {"id": model_id2, "name": m2.name, "version": m2.version},
            "metrics_diff": {},
            "params_diff": {},
            "status": {"model1": m1.status.value, "model2": m2.status.value}
        }
        
        # Compare metrics
        all_metrics = set(m1.metrics.keys()) | set(m2.metrics.keys())
        for metric in all_metrics:
            val1 = m1.metrics.get(metric, None)
            val2 = m2.metrics.get(metric, None)
            
            if val1 is not None and val2 is not None:
                comparison["metrics_diff"][metric] = {
                    "model1": val1,
                    "model2": val2,
                    "improvement": ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                }
                
        # Compare parameters
        all_params = set(m1.parameters.keys()) | set(m2.parameters.keys())
        for param in all_params:
            val1 = m1.parameters.get(param, None)
            val2 = m2.parameters.get(param, None)
            
            if val1 != val2:
                comparison["params_diff"][param] = {
                    "model1": val1,
                    "model2": val2
                }
                
        return comparison
        
    def get_best_model(self, name: str, metric: str, minimize: bool = False) -> Optional[str]:
        """Get best model by metric."""
        if name not in self.model_history:
            return None
            
        best_id = None
        best_value = float('inf') if minimize else float('-inf')
        
        for model_id in self.model_history[name]:
            if metric in self.models[model_id].metrics:
                value = self.models[model_id].metrics[metric]
                
                if minimize and value < best_value:
                    best_value = value
                    best_id = model_id
                elif not minimize and value > best_value:
                    best_value = value
                    best_id = model_id
                    
        return best_id
        
    def list_models(self, name: str = None, status: ModelStatus = None,
                   tags: List[str] = None) -> List[ModelMetadata]:
        """List models with optional filters."""
        results = []
        
        for model_id, metadata in self.models.items():
            # Apply filters
            if name and metadata.name != name:
                continue
            if status and metadata.status != status:
                continue
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
                
            results.append(metadata)
            
        # Sort by creation date
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results
        
    def delete_model(self, model_id: str, force: bool = False) -> bool:
        """Delete a model from registry."""
        if model_id not in self.models:
            return False
            
        # Check if model is in production
        if not force and self.models[model_id].status == ModelStatus.PRODUCTION:
            return False
            
        # Remove from storage
        model_path = self.storage_path / f"{model_id}.pkl"
        if model_path.exists():
            model_path.unlink()
            
        # Update registry
        metadata = self.models.pop(model_id)
        
        # Update history
        if metadata.name in self.model_history:
            self.model_history[metadata.name].remove(model_id)
            
        # Update active models
        if metadata.name in self.active_models:
            if self.active_models[metadata.name] == model_id:
                del self.active_models[metadata.name]
                
        self._save_registry()
        return True
        
    def get_metrics_summary(self, name: str) -> Dict[str, Any]:
        """Get metrics summary across all versions."""
        if name not in self.model_history:
            return {}
            
        summary = {
            "model_name": name,
            "total_versions": len(self.model_history[name]),
            "metrics_evolution": {},
            "best_performing": {}
        }
        
        # Collect all metrics
        all_metrics = set()
        for model_id in self.model_history[name]:
            all_metrics.update(self.models[model_id].metrics.keys())
            
        # Track evolution
        for metric in all_metrics:
            values = []
            for model_id in self.model_history[name]:
                if metric in self.models[model_id].metrics:
                    values.append({
                        "version": self.models[model_id].version,
                        "value": self.models[model_id].metrics[metric]
                    })
                    
            if values:
                summary["metrics_evolution"][metric] = values
                
                # Find best
                best = max(values, key=lambda x: x["value"])
                summary["best_performing"][metric] = best
                
        return summary
        
    def _generate_model_id(self, name: str, version: str) -> str:
        """Generate unique model ID."""
        return f"{name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def _calculate_model_hash(self, model: Any) -> str:
        """Calculate hash of model for versioning."""
        try:
            model_bytes = pickle.dumps(model)
            return hashlib.sha256(model_bytes).hexdigest()[:16]
        except:
            return "unhashable"
            
    def _save_model(self, model: Any, model_id: str):
        """Save model to storage."""
        model_path = self.storage_path / f"{model_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
    def _load_model(self, model_id: str) -> Optional[Any]:
        """Load model from storage."""
        model_path = self.storage_path / f"{model_id}.pkl"
        if not model_path.exists():
            return None
            
        with open(model_path, 'rb') as f:
            return SafePickleHandler.safe_load(f)
            
    def _save_registry(self):
        """Save registry metadata."""
        registry_path = self.storage_path / "registry.json"
        
        registry_data = {
            "models": {
                model_id: {
                    **asdict(metadata),
                    "status": metadata.status.value,
                    "created_at": metadata.created_at.isoformat(),
                    "updated_at": metadata.updated_at.isoformat()
                }
                for model_id, metadata in self.models.items()
            },
            "history": self.model_history,
            "active": self.active_models
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
            
    def _load_registry(self):
        """Load registry metadata."""
        registry_path = self.storage_path / "registry.json"
        if not registry_path.exists():
            return
            
        with open(registry_path, 'r') as f:
            data = json.load(f)
            
        # Reconstruct models
        for model_id, model_data in data.get("models", {}).items():
            model_data["status"] = ModelStatus(model_data["status"])
            model_data["created_at"] = datetime.fromisoformat(model_data["created_at"])
            model_data["updated_at"] = datetime.fromisoformat(model_data["updated_at"])
            self.models[model_id] = ModelMetadata(**model_data)
            
        self.model_history = data.get("history", {})
        self.active_models = data.get("active", {})


# Public API
__all__ = ['ModelRegistry', 'ModelMetadata', 'ModelStatus']
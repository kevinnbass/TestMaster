"""
Machine Learning Configuration Module
=====================================

ML/AI-related configuration settings including model management, training, and inference.
Modularized from testmaster_config.py and enhanced_unified_config.py.

Author: Agent E - Infrastructure Consolidation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum

from .data_models import ConfigBase


class ModelFramework(Enum):
    """ML frameworks."""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SCIKIT_LEARN = "scikit-learn"
    XGBOOST = "xgboost"
    TRANSFORMERS = "transformers"
    LANGCHAIN = "langchain"
    CUSTOM = "custom"


class ModelType(Enum):
    """Model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    REINFORCEMENT = "reinforcement"
    GENERATIVE = "generative"


@dataclass
class ModelConfig(ConfigBase):
    """Model configuration."""
    
    # Model Selection
    framework: ModelFramework = ModelFramework.PYTORCH
    model_type: ModelType = ModelType.CLASSIFICATION
    model_name: str = "default_model"
    model_version: str = "1.0.0"
    
    # Model Paths
    model_path: Path = Path("models")
    checkpoint_path: Path = Path("checkpoints")
    weights_path: Optional[Path] = None
    config_path: Optional[Path] = None
    
    # Model Parameters
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    activation: str = "relu"
    dropout_rate: float = 0.2
    
    # Optimization
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adam"
    loss_function: str = "categorical_crossentropy"
    
    # Hardware
    use_gpu: bool = True
    gpu_memory_limit: Optional[int] = None
    distributed_training: bool = False
    num_gpus: int = 1
    
    def validate(self) -> List[str]:
        """Validate model configuration."""
        errors = []
        
        if self.dropout_rate < 0 or self.dropout_rate > 1:
            errors.append("Dropout rate must be between 0 and 1")
        
        if self.learning_rate <= 0:
            errors.append("Learning rate must be positive")
        
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
        
        if self.epochs <= 0:
            errors.append("Epochs must be positive")
        
        if self.num_gpus < 0:
            errors.append("Number of GPUs cannot be negative")
        
        return errors


@dataclass
class TrainingConfig(ConfigBase):
    """Training configuration."""
    
    # Data Settings
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    shuffle_data: bool = True
    random_seed: int = 42
    
    # Training Settings
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    restore_best_weights: bool = True
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5
    keep_best_only: bool = True
    max_checkpoints: int = 5
    
    # Data Augmentation
    augmentation_enabled: bool = False
    augmentation_factor: float = 2.0
    augmentation_techniques: List[str] = field(default_factory=lambda: [
        "rotation", "flip", "zoom", "shift"
    ])
    
    # Regularization
    l1_regularization: float = 0.0
    l2_regularization: float = 0.01
    batch_normalization: bool = True
    gradient_clipping: float = 1.0
    
    # Monitoring
    tensorboard_enabled: bool = True
    log_frequency: int = 10
    profile_training: bool = False
    
    def validate(self) -> List[str]:
        """Validate training configuration."""
        errors = []
        
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 0.001:
            errors.append(f"Train/val/test splits must sum to 1.0, got {total_split}")
        
        if self.patience <= 0:
            errors.append("Patience must be positive")
        
        if self.checkpoint_frequency <= 0:
            errors.append("Checkpoint frequency must be positive")
        
        if self.l1_regularization < 0 or self.l2_regularization < 0:
            errors.append("Regularization values cannot be negative")
        
        return errors


@dataclass
class InferenceConfig(ConfigBase):
    """Inference configuration."""
    
    # Inference Settings
    batch_inference: bool = True
    inference_batch_size: int = 64
    max_sequence_length: int = 512
    
    # Performance
    use_onnx: bool = False
    use_tensorrt: bool = False
    quantization_enabled: bool = False
    quantization_bits: int = 8
    
    # Caching
    cache_predictions: bool = True
    cache_size_mb: int = 1024
    cache_ttl_seconds: int = 3600
    
    # Serving
    model_server_enabled: bool = False
    server_port: int = 8501
    server_workers: int = 4
    max_concurrent_requests: int = 100
    
    # Post-processing
    confidence_threshold: float = 0.5
    top_k_predictions: int = 5
    apply_softmax: bool = True
    
    def validate(self) -> List[str]:
        """Validate inference configuration."""
        errors = []
        
        if self.inference_batch_size <= 0:
            errors.append("Inference batch size must be positive")
        
        if self.cache_size_mb <= 0:
            errors.append("Cache size must be positive")
        
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        if self.server_port <= 0 or self.server_port > 65535:
            errors.append("Server port must be valid")
        
        if self.quantization_bits not in [4, 8, 16]:
            errors.append("Quantization bits must be 4, 8, or 16")
        
        return errors


@dataclass
class AutoMLConfig(ConfigBase):
    """AutoML configuration."""
    
    # AutoML Settings
    enabled: bool = False
    time_budget_hours: float = 2.0
    metric_to_optimize: str = "accuracy"
    
    # Search Space
    search_algorithms: List[str] = field(default_factory=lambda: [
        "random_search", "grid_search", "bayesian_optimization"
    ])
    max_trials: int = 100
    parallel_trials: int = 4
    
    # Hyperparameter Ranges
    hyperparameter_space: Dict[str, Any] = field(default_factory=lambda: {
        "learning_rate": [0.0001, 0.1],
        "batch_size": [16, 128],
        "dropout_rate": [0.1, 0.5],
        "hidden_units": [[32], [64], [128], [256]]
    })
    
    # Feature Engineering
    auto_feature_engineering: bool = True
    feature_selection_method: str = "mutual_info"
    max_features: Optional[int] = None
    
    def validate(self) -> List[str]:
        """Validate AutoML configuration."""
        errors = []
        
        if self.time_budget_hours <= 0:
            errors.append("Time budget must be positive")
        
        if self.max_trials <= 0:
            errors.append("Max trials must be positive")
        
        if self.parallel_trials <= 0:
            errors.append("Parallel trials must be positive")
        
        return errors


@dataclass
class MLConfig(ConfigBase):
    """Combined ML configuration."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    automl: AutoMLConfig = field(default_factory=AutoMLConfig)
    
    # Experiment Tracking
    experiment_tracking_enabled: bool = True
    experiment_name: str = "default_experiment"
    mlflow_tracking_uri: Optional[str] = None
    wandb_project: Optional[str] = None
    
    # Data Pipeline
    data_pipeline_enabled: bool = True
    data_validation_enabled: bool = True
    data_versioning_enabled: bool = True
    
    # Model Registry
    model_registry_enabled: bool = True
    registry_uri: Optional[str] = None
    auto_register_models: bool = True
    
    def validate(self) -> List[str]:
        """Validate all ML configurations."""
        errors = []
        errors.extend(self.model.validate())
        errors.extend(self.training.validate())
        errors.extend(self.inference.validate())
        errors.extend(self.automl.validate())
        
        if self.experiment_tracking_enabled:
            if not self.mlflow_tracking_uri and not self.wandb_project:
                errors.append("Experiment tracking enabled but no tracking URI configured")
        
        return errors


__all__ = [
    'ModelFramework',
    'ModelType',
    'ModelConfig',
    'TrainingConfig',
    'InferenceConfig',
    'AutoMLConfig',
    'MLConfig'
]
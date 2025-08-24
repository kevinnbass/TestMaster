"""
GPU Acceleration Support
========================
GPU detection, memory management, and acceleration utilities.
Module size: ~297 lines (under 300 limit)

Author: Agent B - Intelligence Specialist
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import platform
import subprocess
import warnings
from dataclasses import dataclass


@dataclass
class GPUInfo:
    """GPU information."""
    device_id: int
    name: str
    memory_total: int
    memory_free: int
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None


class GPUDetector:
    """
    Detects and reports available GPU resources.
    Works without requiring CUDA/GPU libraries.
    """
    
    def __init__(self):
        self.gpus = []
        self.has_gpu = False
        self._detect_gpus()
        
    def _detect_gpus(self):
        """Detect available GPUs."""
        # Try NVIDIA GPUs first
        self._detect_nvidia_gpus()
        
        # Try other GPU vendors if no NVIDIA found
        if not self.gpus:
            self._detect_other_gpus()
            
        self.has_gpu = len(self.gpus) > 0
        
    def _detect_nvidia_gpus(self):
        """Detect NVIDIA GPUs using nvidia-smi."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,driver_version',
                 '--format=csv,nounits,noheader'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            gpu_info = GPUInfo(
                                device_id=int(parts[0]),
                                name=parts[1],
                                memory_total=int(parts[2]) * 1024 * 1024,  # Convert MB to bytes
                                memory_free=int(parts[3]) * 1024 * 1024,
                                driver_version=parts[4] if len(parts) > 4 else None
                            )
                            self.gpus.append(gpu_info)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass
            
    def _detect_other_gpus(self):
        """Detect other GPU types."""
        # Check for Intel GPUs on Windows
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                    capture_output=True, text=True, timeout=10
                )
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.strip() and 'Name' not in line and ('Intel' in line or 'AMD' in line):
                            gpu_info = GPUInfo(
                                device_id=len(self.gpus),
                                name=line.strip(),
                                memory_total=0,  # Unknown
                                memory_free=0
                            )
                            self.gpus.append(gpu_info)
            except:
                pass
                
    def get_best_gpu(self) -> Optional[GPUInfo]:
        """Get GPU with most free memory."""
        if not self.gpus:
            return None
            
        return max(self.gpus, key=lambda gpu: gpu.memory_free)
        
    def get_gpu_summary(self) -> Dict[str, Any]:
        """Get summary of available GPUs."""
        return {
            'has_gpu': self.has_gpu,
            'num_gpus': len(self.gpus),
            'gpus': [
                {
                    'id': gpu.device_id,
                    'name': gpu.name,
                    'memory_gb': gpu.memory_total / (1024**3) if gpu.memory_total > 0 else 0,
                    'free_gb': gpu.memory_free / (1024**3) if gpu.memory_free > 0 else 0
                }
                for gpu in self.gpus
            ]
        }


class NumPyGPUAccelerator:
    """
    GPU acceleration for NumPy operations using available backends.
    Falls back gracefully to CPU if GPU unavailable.
    """
    
    def __init__(self):
        self.gpu_detector = GPUDetector()
        self.backend = self._detect_backend()
        self.device_id = 0
        
    def _detect_backend(self) -> str:
        """Detect available GPU backend."""
        # Try CuPy
        try:
            import cupy
            return "cupy"
        except ImportError:
            pass
            
        # Try TensorFlow
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                return "tensorflow"
        except ImportError:
            pass
            
        # Try PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                return "pytorch"
        except ImportError:
            pass
            
        return "numpy"
        
    def to_gpu(self, array: np.ndarray) -> Any:
        """Move array to GPU."""
        if self.backend == "cupy":
            import cupy as cp
            return cp.asarray(array)
        elif self.backend == "tensorflow":
            import tensorflow as tf
            return tf.constant(array)
        elif self.backend == "pytorch":
            import torch
            return torch.tensor(array).cuda()
        else:
            return array
            
    def to_cpu(self, gpu_array: Any) -> np.ndarray:
        """Move array back to CPU."""
        if self.backend == "cupy":
            return gpu_array.get()
        elif self.backend == "tensorflow":
            return gpu_array.numpy()
        elif self.backend == "pytorch":
            return gpu_array.cpu().numpy()
        else:
            return gpu_array
            
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """GPU-accelerated matrix multiplication."""
        if self.backend == "numpy":
            return np.matmul(a, b)
            
        try:
            gpu_a = self.to_gpu(a)
            gpu_b = self.to_gpu(b)
            
            if self.backend == "cupy":
                import cupy as cp
                result = cp.matmul(gpu_a, gpu_b)
            elif self.backend == "tensorflow":
                import tensorflow as tf
                result = tf.matmul(gpu_a, gpu_b)
            elif self.backend == "pytorch":
                import torch
                result = torch.matmul(gpu_a, gpu_b)
            else:
                result = gpu_a @ gpu_b
                
            return self.to_cpu(result)
        except:
            # Fallback to CPU
            return np.matmul(a, b)
            
    def norm(self, array: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        """GPU-accelerated norm calculation."""
        if self.backend == "numpy":
            return np.linalg.norm(array, axis=axis)
            
        try:
            gpu_array = self.to_gpu(array)
            
            if self.backend == "cupy":
                import cupy as cp
                result = cp.linalg.norm(gpu_array, axis=axis)
            elif self.backend == "tensorflow":
                import tensorflow as tf
                result = tf.norm(gpu_array, axis=axis)
            elif self.backend == "pytorch":
                import torch
                result = torch.norm(gpu_array, dim=axis)
            else:
                result = np.linalg.norm(gpu_array, axis=axis)
                
            return self.to_cpu(result)
        except:
            return np.linalg.norm(array, axis=axis)
            
    def einsum(self, subscripts: str, *operands: np.ndarray) -> np.ndarray:
        """GPU-accelerated einsum operation."""
        if self.backend == "numpy":
            return np.einsum(subscripts, *operands)
            
        try:
            gpu_operands = [self.to_gpu(op) for op in operands]
            
            if self.backend == "cupy":
                import cupy as cp
                result = cp.einsum(subscripts, *gpu_operands)
            elif self.backend == "tensorflow":
                import tensorflow as tf
                result = tf.einsum(subscripts, *gpu_operands)
            elif self.backend == "pytorch":
                import torch
                result = torch.einsum(subscripts, *gpu_operands)
            else:
                result = np.einsum(subscripts, *gpu_operands)
                
            return self.to_cpu(result)
        except:
            return np.einsum(subscripts, *operands)


class BatchProcessor:
    """
    Processes large arrays in GPU-friendly batches.
    Automatically determines optimal batch size based on GPU memory.
    """
    
    def __init__(self, max_memory_gb: float = 2.0):
        self.max_memory_gb = max_memory_gb
        self.gpu_accelerator = NumPyGPUAccelerator()
        
    def process_batches(self, data: np.ndarray, process_func: callable,
                       axis: int = 0, **kwargs) -> np.ndarray:
        """
        Process data in batches to avoid memory issues.
        
        Args:
            data: Input data array
            process_func: Function to apply to each batch
            axis: Axis along which to batch
            **kwargs: Additional arguments for process_func
            
        Returns:
            Processed data
        """
        batch_size = self._calculate_batch_size(data, axis)
        
        if batch_size >= data.shape[axis]:
            # Process all at once
            return process_func(data, **kwargs)
            
        # Process in batches
        results = []
        num_batches = (data.shape[axis] + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, data.shape[axis])
            
            # Extract batch
            batch_slice = [slice(None)] * data.ndim
            batch_slice[axis] = slice(start_idx, end_idx)
            batch = data[tuple(batch_slice)]
            
            # Process batch
            batch_result = process_func(batch, **kwargs)
            results.append(batch_result)
            
        # Concatenate results
        return np.concatenate(results, axis=axis)
        
    def _calculate_batch_size(self, data: np.ndarray, axis: int) -> int:
        """Calculate optimal batch size based on memory constraints."""
        # Estimate memory per sample
        sample_shape = list(data.shape)
        sample_shape[axis] = 1
        bytes_per_sample = np.prod(sample_shape) * data.itemsize
        
        # Calculate batch size
        max_bytes = self.max_memory_gb * 1024**3
        batch_size = max(1, int(max_bytes / bytes_per_sample))
        
        return min(batch_size, data.shape[axis])


class MemoryManager:
    """
    Manages GPU memory allocation and cleanup.
    Provides memory monitoring and optimization.
    """
    
    def __init__(self):
        self.gpu_detector = GPUDetector()
        self.allocated_arrays = []
        
    def allocate_gpu_memory(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> Optional[Any]:
        """Allocate GPU memory for array."""
        try:
            if self.gpu_detector.has_gpu:
                # Try to allocate on GPU
                backend = NumPyGPUAccelerator()._detect_backend()
                
                if backend == "cupy":
                    import cupy as cp
                    array = cp.zeros(shape, dtype=dtype)
                    self.allocated_arrays.append(array)
                    return array
                elif backend == "pytorch":
                    import torch
                    array = torch.zeros(shape, dtype=torch.float32).cuda()
                    self.allocated_arrays.append(array)
                    return array
        except:
            pass
            
        # Fallback to CPU
        return np.zeros(shape, dtype=dtype)
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        usage = {"cpu_allocated_gb": 0.0, "gpu_allocated_gb": 0.0}
        
        try:
            if self.gpu_detector.has_gpu:
                # Try to get actual GPU memory usage
                for gpu in self.gpu_detector.gpus:
                    if gpu.memory_total > 0:
                        used_memory = gpu.memory_total - gpu.memory_free
                        usage[f"gpu_{gpu.device_id}_used_gb"] = used_memory / (1024**3)
                        usage[f"gpu_{gpu.device_id}_total_gb"] = gpu.memory_total / (1024**3)
        except:
            pass
            
        return usage
        
    def cleanup(self):
        """Clean up allocated GPU memory."""
        try:
            if hasattr(self, 'allocated_arrays'):
                backend = NumPyGPUAccelerator()._detect_backend()
                
                if backend == "cupy":
                    import cupy as cp
                    cp.get_default_memory_pool().free_all_blocks()
                elif backend == "pytorch":
                    import torch
                    torch.cuda.empty_cache()
                    
                self.allocated_arrays.clear()
        except:
            pass


# Public API
__all__ = [
    'GPUDetector',
    'NumPyGPUAccelerator', 
    'BatchProcessor',
    'MemoryManager',
    'GPUInfo'
]
"""
Analytics Data Compression System
==================================

Intelligent compression for large analytics payloads with multiple
algorithms, adaptive selection, and streaming support.

Author: TestMaster Team
"""

import logging
import time
import json
import zlib
import gzip
import bz2
import lzma
import base64
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import io
import pickle

logger = logging.getLogger(__name__)

class CompressionAlgorithm(Enum):
    """Available compression algorithms."""
    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"
    ADAPTIVE = "adaptive"

@dataclass
class CompressionResult:
    """Result of compression operation."""
    algorithm: CompressionAlgorithm
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time_ms: float
    data: bytes
    metadata: Dict[str, Any]

class AnalyticsCompressor:
    """
    Intelligent compression system for analytics data.
    """
    
    def __init__(self,
                 threshold_bytes: int = 10240,  # 10KB
                 adaptive: bool = True,
                 compression_level: int = 6):
        """
        Initialize compressor.
        
        Args:
            threshold_bytes: Minimum size for compression
            adaptive: Enable adaptive algorithm selection
            compression_level: Compression level (1-9)
        """
        self.threshold_bytes = threshold_bytes
        self.adaptive = adaptive
        self.compression_level = compression_level
        
        # Algorithm configurations
        self.algorithms = {
            CompressionAlgorithm.ZLIB: {
                'compress': lambda d: zlib.compress(d, self.compression_level),
                'decompress': zlib.decompress,
                'speed': 8,  # Speed rating 1-10
                'ratio': 7   # Compression ratio rating 1-10
            },
            CompressionAlgorithm.GZIP: {
                'compress': lambda d: gzip.compress(d, compresslevel=self.compression_level),
                'decompress': gzip.decompress,
                'speed': 7,
                'ratio': 7
            },
            CompressionAlgorithm.BZIP2: {
                'compress': lambda d: bz2.compress(d, compresslevel=self.compression_level),
                'decompress': bz2.decompress,
                'speed': 4,
                'ratio': 9
            },
            CompressionAlgorithm.LZMA: {
                'compress': lambda d: lzma.compress(d, preset=self.compression_level),
                'decompress': lzma.decompress,
                'speed': 3,
                'ratio': 10
            }
        }
        
        # Performance statistics
        self.stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'bytes_saved': 0,
            'total_time_ms': 0,
            'algorithm_usage': {alg.value: 0 for alg in CompressionAlgorithm}
        }
        
        # Algorithm performance history for adaptive selection
        self.performance_history = {alg: [] for alg in CompressionAlgorithm}
        
        logger.info("Analytics Compressor initialized")
    
    def compress(self,
                data: Any,
                algorithm: Optional[CompressionAlgorithm] = None,
                metadata: Optional[Dict[str, Any]] = None) -> CompressionResult:
        """
        Compress analytics data.
        
        Args:
            data: Data to compress
            algorithm: Optional algorithm override
            metadata: Optional metadata
            
        Returns:
            Compression result
        """
        start_time = time.time()
        
        # Serialize data
        if isinstance(data, bytes):
            data_bytes = data
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            # Use JSON for dicts/lists, pickle for others
            if isinstance(data, (dict, list)):
                data_bytes = json.dumps(data, separators=(',', ':')).encode('utf-8')
            else:
                data_bytes = pickle.dumps(data)
        
        original_size = len(data_bytes)
        
        # Check if compression is needed
        if original_size < self.threshold_bytes:
            return CompressionResult(
                algorithm=CompressionAlgorithm.NONE,
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,
                compression_time_ms=0,
                data=data_bytes,
                metadata=metadata or {}
            )
        
        # Select algorithm
        if algorithm is None:
            if self.adaptive:
                algorithm = self._select_adaptive_algorithm(data_bytes)
            else:
                algorithm = CompressionAlgorithm.ZLIB
        
        # Compress
        try:
            if algorithm == CompressionAlgorithm.ADAPTIVE:
                result = self._compress_adaptive(data_bytes)
            elif algorithm in self.algorithms:
                compressed = self.algorithms[algorithm]['compress'](data_bytes)
                result = CompressionResult(
                    algorithm=algorithm,
                    original_size=original_size,
                    compressed_size=len(compressed),
                    compression_ratio=original_size / len(compressed) if compressed else 1.0,
                    compression_time_ms=(time.time() - start_time) * 1000,
                    data=compressed,
                    metadata=metadata or {}
                )
            else:
                # No compression
                result = CompressionResult(
                    algorithm=CompressionAlgorithm.NONE,
                    original_size=original_size,
                    compressed_size=original_size,
                    compression_ratio=1.0,
                    compression_time_ms=(time.time() - start_time) * 1000,
                    data=data_bytes,
                    metadata=metadata or {}
                )
            
            # Update statistics
            self.stats['total_compressions'] += 1
            self.stats['bytes_saved'] += original_size - result.compressed_size
            self.stats['total_time_ms'] += result.compression_time_ms
            self.stats['algorithm_usage'][result.algorithm.value] += 1
            
            # Update performance history
            if result.algorithm in self.performance_history:
                self.performance_history[result.algorithm].append({
                    'ratio': result.compression_ratio,
                    'time_ms': result.compression_time_ms,
                    'size': original_size
                })
                
                # Keep only last 100 entries
                if len(self.performance_history[result.algorithm]) > 100:
                    self.performance_history[result.algorithm].pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            # Return uncompressed on failure
            return CompressionResult(
                algorithm=CompressionAlgorithm.NONE,
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,
                compression_time_ms=(time.time() - start_time) * 1000,
                data=data_bytes,
                metadata=metadata or {}
            )
    
    def decompress(self, compressed_result: CompressionResult) -> Any:
        """
        Decompress data.
        
        Args:
            compressed_result: Compression result to decompress
            
        Returns:
            Original data
        """
        start_time = time.time()
        
        try:
            # No decompression needed
            if compressed_result.algorithm == CompressionAlgorithm.NONE:
                data_bytes = compressed_result.data
            elif compressed_result.algorithm in self.algorithms:
                data_bytes = self.algorithms[compressed_result.algorithm]['decompress'](
                    compressed_result.data
                )
            else:
                raise ValueError(f"Unknown algorithm: {compressed_result.algorithm}")
            
            # Deserialize based on metadata hints
            if compressed_result.metadata.get('format') == 'json':
                result = json.loads(data_bytes.decode('utf-8'))
            elif compressed_result.metadata.get('format') == 'pickle':
                result = SafePickleHandler.safe_load(data_bytes)
            else:
                # Try to auto-detect
                try:
                    result = json.loads(data_bytes.decode('utf-8'))
                except:
                    try:
                        result = SafePickleHandler.safe_load(data_bytes)
                    except:
                        result = data_bytes
            
            # Update statistics
            self.stats['total_decompressions'] += 1
            self.stats['total_time_ms'] += (time.time() - start_time) * 1000
            
            return result
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise
    
    def _compress_adaptive(self, data: bytes) -> CompressionResult:
        """Adaptive compression with best algorithm selection."""
        best_result = None
        best_score = 0
        
        # Test each algorithm
        for algorithm, config in self.algorithms.items():
            try:
                start = time.time()
                compressed = config['compress'](data)
                compression_time = (time.time() - start) * 1000
                
                ratio = len(data) / len(compressed) if compressed else 1.0
                
                # Calculate score (balance speed and ratio)
                speed_weight = 0.3
                ratio_weight = 0.7
                
                # Normalize scores
                speed_score = (10 - compression_time / 100) if compression_time < 1000 else 0
                ratio_score = min(ratio * 2, 10)  # Cap at 10
                
                score = (speed_score * speed_weight) + (ratio_score * ratio_weight)
                
                if score > best_score:
                    best_score = score
                    best_result = CompressionResult(
                        algorithm=algorithm,
                        original_size=len(data),
                        compressed_size=len(compressed),
                        compression_ratio=ratio,
                        compression_time_ms=compression_time,
                        data=compressed,
                        metadata={}
                    )
            except Exception as e:
                logger.debug(f"Algorithm {algorithm} failed: {e}")
                continue
        
        # Fallback to uncompressed if all fail
        if best_result is None:
            best_result = CompressionResult(
                algorithm=CompressionAlgorithm.NONE,
                original_size=len(data),
                compressed_size=len(data),
                compression_ratio=1.0,
                compression_time_ms=0,
                data=data,
                metadata={}
            )
        
        return best_result
    
    def _select_adaptive_algorithm(self, data: bytes) -> CompressionAlgorithm:
        """Select best algorithm based on data characteristics and history."""
        data_size = len(data)
        
        # Quick heuristics based on size
        if data_size < 1024:  # < 1KB
            return CompressionAlgorithm.ZLIB  # Fast for small data
        elif data_size < 10240:  # < 10KB
            return CompressionAlgorithm.GZIP  # Good balance
        elif data_size < 102400:  # < 100KB
            # Check data entropy for compressibility
            if self._estimate_entropy(data) > 0.9:
                return CompressionAlgorithm.LZMA  # Best for high entropy
            else:
                return CompressionAlgorithm.BZIP2  # Good for medium data
        else:
            # Large data - use history to decide
            return self._select_from_history(data_size)
    
    def _estimate_entropy(self, data: bytes) -> float:
        """Estimate data entropy (0-1, higher = more random)."""
        if not data:
            return 0
        
        # Simple byte frequency analysis
        frequency = {}
        for byte in data:
            frequency[byte] = frequency.get(byte, 0) + 1
        
        entropy = 0
        data_len = len(data)
        
        for count in frequency.values():
            if count > 0:
                probability = count / data_len
                entropy -= probability * (probability if probability > 0 else 0)
        
        # Normalize to 0-1
        return min(1.0, entropy / 0.693)  # ln(2) = 0.693
    
    def _select_from_history(self, data_size: int) -> CompressionAlgorithm:
        """Select algorithm based on historical performance."""
        best_algorithm = CompressionAlgorithm.ZLIB
        best_score = 0
        
        for algorithm, history in self.performance_history.items():
            if algorithm == CompressionAlgorithm.NONE:
                continue
            
            if history:
                # Calculate average performance for similar sizes
                similar = [h for h in history 
                          if abs(h['size'] - data_size) / data_size < 0.5]
                
                if similar:
                    avg_ratio = sum(h['ratio'] for h in similar) / len(similar)
                    avg_time = sum(h['time_ms'] for h in similar) / len(similar)
                    
                    # Score based on ratio and speed
                    score = (avg_ratio * 10) / (1 + avg_time / 100)
                    
                    if score > best_score:
                        best_score = score
                        best_algorithm = algorithm
        
        return best_algorithm
    
    def compress_stream(self, 
                       data_stream,
                       chunk_size: int = 8192) -> List[CompressionResult]:
        """
        Compress data stream in chunks.
        
        Args:
            data_stream: Input stream
            chunk_size: Chunk size in bytes
            
        Returns:
            List of compression results
        """
        results = []
        
        while True:
            chunk = data_stream.read(chunk_size)
            if not chunk:
                break
            
            result = self.compress(chunk)
            results.append(result)
        
        return results
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            'total_compressions': self.stats['total_compressions'],
            'total_decompressions': self.stats['total_decompressions'],
            'bytes_saved': self.stats['bytes_saved'],
            'average_time_ms': (self.stats['total_time_ms'] / 
                              max(1, self.stats['total_compressions'])),
            'algorithm_usage': self.stats['algorithm_usage'],
            'compression_efficiency': (self.stats['bytes_saved'] / 
                                     max(1, self.stats['total_compressions'])),
            'adaptive_enabled': self.adaptive
        }
    
    def encode_for_transport(self, compressed_result: CompressionResult) -> str:
        """
        Encode compressed data for network transport.
        
        Args:
            compressed_result: Compression result
            
        Returns:
            Base64 encoded string
        """
        # Create transport package
        package = {
            'algorithm': compressed_result.algorithm.value,
            'original_size': compressed_result.original_size,
            'compressed_size': compressed_result.compressed_size,
            'data': base64.b64encode(compressed_result.data).decode('ascii'),
            'metadata': compressed_result.metadata
        }
        
        return json.dumps(package)
    
    def decode_from_transport(self, encoded: str) -> CompressionResult:
        """
        Decode transported data.
        
        Args:
            encoded: Base64 encoded transport string
            
        Returns:
            Compression result
        """
        package = json.loads(encoded)
        
        return CompressionResult(
            algorithm=CompressionAlgorithm(package['algorithm']),
            original_size=package['original_size'],
            compressed_size=package['compressed_size'],
            compression_ratio=package['original_size'] / package['compressed_size'],
            compression_time_ms=0,
            data=base64.b64decode(package['data']),
            metadata=package.get('metadata', {})
        )

# Global compressor instance
analytics_compressor = AnalyticsCompressor()
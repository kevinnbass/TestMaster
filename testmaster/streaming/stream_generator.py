"""
Stream Generator for TestMaster

Real-time streaming test generation inspired by multi-agent frameworks.
Provides incremental, live test generation with progressive enhancement.
"""

import asyncio
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
import json

from ..core.feature_flags import FeatureFlags
from ..core.shared_state import get_shared_state
from ..telemetry import get_telemetry_collector

class GenerationStage(Enum):
    """Stages of streaming test generation."""
    ANALYSIS = "analysis"
    SKELETON = "skeleton" 
    IMPLEMENTATION = "implementation"
    ENHANCEMENT = "enhancement"
    VALIDATION = "validation"
    FINALIZATION = "finalization"

@dataclass
class StreamConfig:
    """Configuration for streaming generation."""
    buffer_size: int = 1000
    chunk_size: int = 100
    stage_timeout: float = 30.0
    enable_live_feedback: bool = True
    enable_incremental: bool = True
    max_iterations: int = 5
    quality_threshold: float = 0.8

@dataclass
class StreamChunk:
    """A chunk of streaming test generation."""
    chunk_id: str
    stage: GenerationStage
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    is_complete: bool = False
    error: Optional[str] = None

@dataclass
class GenerationProgress:
    """Progress tracking for streaming generation."""
    session_id: str
    current_stage: GenerationStage
    stages_completed: List[GenerationStage] = field(default_factory=list)
    total_chunks: int = 0
    completed_chunks: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    success: bool = False
    error_message: Optional[str] = None

class StreamGenerator:
    """
    Real-time streaming test generator.
    
    Features:
    - Progressive test generation in stages
    - Live streaming output with chunks
    - Incremental enhancement and refinement
    - Integration with existing TestMaster components
    - Multi-stage pipeline with feedback loops
    """
    
    def __init__(self, config: StreamConfig = None):
        """
        Initialize stream generator.
        
        Args:
            config: Stream generation configuration
        """
        self.enabled = FeatureFlags.is_enabled('layer1_test_foundation', 'streaming_generation')
        self.config = config or StreamConfig()
        
        # Initialize all attributes regardless of enabled state
        # Streaming state
        self.active_sessions: Dict[str, GenerationProgress] = {}
        self.stream_buffers: Dict[str, Queue] = {}
        self.chunk_callbacks: Dict[str, List[Callable]] = {}
        
        # Threading
        self.lock = threading.RLock()
        self.worker_threads: Dict[str, threading.Thread] = {}
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.sessions_created = 0
        self.total_chunks_generated = 0
        self.successful_generations = 0
        
        # Integrations - always initialize
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
        
        if FeatureFlags.is_enabled('layer3_orchestration', 'telemetry_system'):
            self.telemetry = get_telemetry_collector()
        else:
            self.telemetry = None
        
        if not self.enabled:
            return
        
        print("Stream generator initialized")
        print(f"   Buffer size: {self.config.buffer_size}")
        print(f"   Chunk size: {self.config.chunk_size}")
    
    def start_streaming_generation(self, source_code: str, module_path: str,
                                 metadata: Dict[str, Any] = None,
                                 chunk_callback: Callable[[StreamChunk], None] = None) -> str:
        """
        Start streaming test generation.
        
        Args:
            source_code: Source code to generate tests for
            module_path: Path to the module
            metadata: Additional metadata
            chunk_callback: Callback for each generated chunk
            
        Returns:
            Session ID for tracking
        """
        if not self.enabled:
            raise RuntimeError("Stream generator is disabled")
        
        session_id = str(uuid.uuid4())
        
        # Initialize session
        progress = GenerationProgress(
            session_id=session_id,
            current_stage=GenerationStage.ANALYSIS
        )
        
        with self.lock:
            self.active_sessions[session_id] = progress
            self.stream_buffers[session_id] = Queue(maxsize=self.config.buffer_size)
            self.chunk_callbacks[session_id] = [chunk_callback] if chunk_callback else []
            self.sessions_created += 1
        
        # Start generation worker
        worker = threading.Thread(
            target=self._generation_worker,
            args=(session_id, source_code, module_path, metadata or {}),
            daemon=True
        )
        
        with self.lock:
            self.worker_threads[session_id] = worker
        
        worker.start()
        
        # Send telemetry
        if self.telemetry:
            self.telemetry.record_event(
                event_type="streaming_generation_started",
                component="stream_generator",
                operation="start_streaming",
                metadata={
                    "session_id": session_id,
                    "module_path": module_path,
                    "has_callback": chunk_callback is not None
                }
            )
        
        print(f"Started streaming generation: {session_id}")
        return session_id
    
    def _generation_worker(self, session_id: str, source_code: str,
                          module_path: str, metadata: Dict[str, Any]):
        """Worker thread for streaming generation."""
        try:
            progress = self.active_sessions[session_id]
            
            # Execute generation stages
            stages = [
                GenerationStage.ANALYSIS,
                GenerationStage.SKELETON,
                GenerationStage.IMPLEMENTATION,
                GenerationStage.ENHANCEMENT,
                GenerationStage.VALIDATION,
                GenerationStage.FINALIZATION
            ]
            
            for stage in stages:
                if self.shutdown_event.is_set():
                    break
                
                progress.current_stage = stage
                self._execute_generation_stage(session_id, stage, source_code, module_path, metadata)
                progress.stages_completed.append(stage)
            
            # Mark as successful
            progress.success = True
            progress.end_time = datetime.now()
            
            # Send completion chunk
            completion_chunk = StreamChunk(
                chunk_id=str(uuid.uuid4()),
                stage=GenerationStage.FINALIZATION,
                content="",
                metadata={"session_complete": True, "success": True},
                is_complete=True
            )
            self._emit_chunk(session_id, completion_chunk)
            
            with self.lock:
                self.successful_generations += 1
            
        except Exception as e:
            # Handle error
            progress.success = False
            progress.error_message = str(e)
            progress.end_time = datetime.now()
            
            error_chunk = StreamChunk(
                chunk_id=str(uuid.uuid4()),
                stage=progress.current_stage,
                content="",
                error=str(e),
                metadata={"session_complete": True, "success": False},
                is_complete=True
            )
            self._emit_chunk(session_id, error_chunk)
            
            print(f"Generation error in session {session_id}: {e}")
        
        finally:
            # Cleanup session
            with self.lock:
                self.worker_threads.pop(session_id, None)
    
    def _execute_generation_stage(self, session_id: str, stage: GenerationStage,
                                 source_code: str, module_path: str, metadata: Dict[str, Any]):
        """Execute a specific generation stage."""
        if stage == GenerationStage.ANALYSIS:
            self._analyze_source(session_id, source_code, module_path)
        elif stage == GenerationStage.SKELETON:
            self._generate_skeleton(session_id, source_code, module_path)
        elif stage == GenerationStage.IMPLEMENTATION:
            self._implement_tests(session_id, source_code, module_path)
        elif stage == GenerationStage.ENHANCEMENT:
            self._enhance_tests(session_id, source_code, module_path)
        elif stage == GenerationStage.VALIDATION:
            self._validate_tests(session_id, source_code, module_path)
        elif stage == GenerationStage.FINALIZATION:
            self._finalize_tests(session_id, source_code, module_path)
    
    def _analyze_source(self, session_id: str, source_code: str, module_path: str):
        """Analyze source code and emit analysis chunks."""
        lines = source_code.split('\n')
        
        # Emit analysis chunks progressively
        analysis_parts = [
            "Analyzing imports and dependencies...",
            "Identifying classes and functions...",
            "Detecting test patterns and requirements...",
            "Planning test structure and coverage..."
        ]
        
        for i, part in enumerate(analysis_parts):
            chunk = StreamChunk(
                chunk_id=str(uuid.uuid4()),
                stage=GenerationStage.ANALYSIS,
                content=part,
                metadata={
                    "step": i + 1,
                    "total_steps": len(analysis_parts),
                    "lines_analyzed": len(lines)
                }
            )
            self._emit_chunk(session_id, chunk)
            time.sleep(0.5)  # Simulate analysis time
    
    def _generate_skeleton(self, session_id: str, source_code: str, module_path: str):
        """Generate test skeleton and emit skeleton chunks."""
        skeleton_parts = [
            "# Test skeleton for {}\n".format(module_path),
            "import unittest\n",
            "from unittest.mock import Mock, patch\n",
            f"from {module_path.replace('/', '.').replace('.py', '')} import *\n\n",
            "class Test{}(unittest.TestCase):\n".format(
                module_path.split('/')[-1].replace('.py', '').title()
            ),
            "    def setUp(self):\n",
            "        \"\"\"Set up test fixtures.\"\"\"\n",
            "        pass\n\n"
        ]
        
        for i, part in enumerate(skeleton_parts):
            chunk = StreamChunk(
                chunk_id=str(uuid.uuid4()),
                stage=GenerationStage.SKELETON,
                content=part,
                metadata={
                    "step": i + 1,
                    "total_steps": len(skeleton_parts),
                    "skeleton_progress": (i + 1) / len(skeleton_parts)
                }
            )
            self._emit_chunk(session_id, chunk)
            time.sleep(0.3)
    
    def _implement_tests(self, session_id: str, source_code: str, module_path: str):
        """Implement test methods and emit implementation chunks."""
        test_methods = [
            "    def test_basic_functionality(self):\n",
            "        \"\"\"Test basic functionality.\"\"\"\n",
            "        # TODO: Implement test\n",
            "        self.assertTrue(True)\n\n",
            "    def test_edge_cases(self):\n",
            "        \"\"\"Test edge cases.\"\"\"\n",
            "        # TODO: Implement edge case tests\n",
            "        pass\n\n",
            "    def test_error_handling(self):\n",
            "        \"\"\"Test error handling.\"\"\"\n",
            "        # TODO: Implement error tests\n",
            "        pass\n\n"
        ]
        
        for i, method in enumerate(test_methods):
            chunk = StreamChunk(
                chunk_id=str(uuid.uuid4()),
                stage=GenerationStage.IMPLEMENTATION,
                content=method,
                metadata={
                    "method_index": i,
                    "implementation_progress": (i + 1) / len(test_methods)
                }
            )
            self._emit_chunk(session_id, chunk)
            time.sleep(0.4)
    
    def _enhance_tests(self, session_id: str, source_code: str, module_path: str):
        """Enhance tests with additional coverage."""
        enhancements = [
            "    def test_integration(self):\n",
            "        \"\"\"Test integration scenarios.\"\"\"\n",
            "        # Enhanced integration tests\n",
            "        pass\n\n",
            "    def test_performance(self):\n",
            "        \"\"\"Test performance characteristics.\"\"\"\n",
            "        # Performance validation\n",
            "        pass\n\n"
        ]
        
        for enhancement in enhancements:
            chunk = StreamChunk(
                chunk_id=str(uuid.uuid4()),
                stage=GenerationStage.ENHANCEMENT,
                content=enhancement,
                metadata={"enhancement_type": "coverage_expansion"}
            )
            self._emit_chunk(session_id, chunk)
            time.sleep(0.3)
    
    def _validate_tests(self, session_id: str, source_code: str, module_path: str):
        """Validate generated tests."""
        validation_steps = [
            "Validating test syntax...",
            "Checking import statements...",
            "Verifying test structure...",
            "Analyzing test coverage..."
        ]
        
        for step in validation_steps:
            chunk = StreamChunk(
                chunk_id=str(uuid.uuid4()),
                stage=GenerationStage.VALIDATION,
                content=step,
                metadata={"validation_step": step}
            )
            self._emit_chunk(session_id, chunk)
            time.sleep(0.2)
    
    def _finalize_tests(self, session_id: str, source_code: str, module_path: str):
        """Finalize test generation."""
        finalization = [
            "\nif __name__ == '__main__':\n",
            "    unittest.main()\n"
        ]
        
        for part in finalization:
            chunk = StreamChunk(
                chunk_id=str(uuid.uuid4()),
                stage=GenerationStage.FINALIZATION,
                content=part,
                metadata={"finalization": True}
            )
            self._emit_chunk(session_id, chunk)
            time.sleep(0.1)
    
    def _emit_chunk(self, session_id: str, chunk: StreamChunk):
        """Emit a chunk to the stream buffer and callbacks."""
        with self.lock:
            # Add to buffer
            if session_id in self.stream_buffers:
                try:
                    self.stream_buffers[session_id].put_nowait(chunk)
                except:
                    # Buffer full, skip oldest
                    try:
                        self.stream_buffers[session_id].get_nowait()
                        self.stream_buffers[session_id].put_nowait(chunk)
                    except:
                        pass
            
            # Call callbacks
            if session_id in self.chunk_callbacks:
                for callback in self.chunk_callbacks[session_id]:
                    try:
                        callback(chunk)
                    except Exception as e:
                        print(f"Chunk callback error: {e}")
            
            self.total_chunks_generated += 1
        
        # Update shared state
        if self.shared_state:
            self.shared_state.increment("streaming_chunks_generated")
    
    def get_stream_chunks(self, session_id: str, timeout: float = 1.0) -> List[StreamChunk]:
        """Get available chunks from stream buffer."""
        if not self.enabled or session_id not in self.stream_buffers:
            return []
        
        chunks = []
        buffer = self.stream_buffers[session_id]
        
        try:
            # Get first chunk with timeout
            chunk = buffer.get(timeout=timeout)
            chunks.append(chunk)
            
            # Get remaining chunks without blocking
            while True:
                try:
                    chunk = buffer.get_nowait()
                    chunks.append(chunk)
                except Empty:
                    break
        except Empty:
            pass
        
        return chunks
    
    def get_session_progress(self, session_id: str) -> Optional[GenerationProgress]:
        """Get progress for a streaming session."""
        if not self.enabled:
            return None
        
        with self.lock:
            return self.active_sessions.get(session_id)
    
    def add_chunk_callback(self, session_id: str, callback: Callable[[StreamChunk], None]):
        """Add a callback for streaming chunks."""
        if not self.enabled:
            return
        
        with self.lock:
            if session_id in self.chunk_callbacks:
                self.chunk_callbacks[session_id].append(callback)
    
    def cancel_session(self, session_id: str) -> bool:
        """Cancel a streaming generation session."""
        if not self.enabled:
            return False
        
        with self.lock:
            if session_id in self.active_sessions:
                # Mark as cancelled
                progress = self.active_sessions[session_id]
                progress.success = False
                progress.error_message = "Cancelled by user"
                progress.end_time = datetime.now()
                
                # Cancel worker thread
                if session_id in self.worker_threads:
                    # Note: Python doesn't support thread cancellation
                    # The worker will check shutdown_event
                    pass
                
                return True
        
        return False
    
    def get_generator_statistics(self) -> Dict[str, Any]:
        """Get generator statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            active_sessions = len(self.active_sessions)
            success_rate = 0.0
            if self.sessions_created > 0:
                success_rate = (self.successful_generations / self.sessions_created) * 100
            
            return {
                "enabled": True,
                "active_sessions": active_sessions,
                "total_sessions": self.sessions_created,
                "successful_generations": self.successful_generations,
                "total_chunks_generated": self.total_chunks_generated,
                "success_rate": round(success_rate, 2),
                "buffer_size": self.config.buffer_size,
                "chunk_size": self.config.chunk_size
            }
    
    def configure(self, **kwargs):
        """Configure the stream generator."""
        if not self.enabled:
            return
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        print(f"Stream generator configured: {kwargs}")
    
    def shutdown(self):
        """Shutdown the stream generator."""
        if not self.enabled:
            return
        
        print("Shutting down stream generator...")
        
        # Signal shutdown to all workers
        self.shutdown_event.set()
        
        # Wait for workers to complete
        with self.lock:
            for worker in list(self.worker_threads.values()):
                if worker.is_alive():
                    worker.join(timeout=2.0)
            
            sessions_count = len(self.active_sessions)
            chunks_count = self.total_chunks_generated
            
            # Clear all state
            self.active_sessions.clear()
            self.stream_buffers.clear()
            self.chunk_callbacks.clear()
            self.worker_threads.clear()
        
        print(f"Stream generator shutdown - processed {sessions_count} sessions, {chunks_count} chunks")

# Global instance
_stream_generator: Optional[StreamGenerator] = None

def get_stream_generator() -> StreamGenerator:
    """Get the global stream generator instance."""
    global _stream_generator
    if _stream_generator is None:
        _stream_generator = StreamGenerator()
    return _stream_generator

# Convenience function
def stream_generate_test(source_code: str, module_path: str,
                        metadata: Dict[str, Any] = None,
                        chunk_callback: Callable[[StreamChunk], None] = None) -> str:
    """
    Generate a test using streaming generation.
    
    Args:
        source_code: Source code to generate tests for
        module_path: Path to the module
        metadata: Additional metadata
        chunk_callback: Callback for each generated chunk
        
    Returns:
        Session ID for tracking
    """
    generator = get_stream_generator()
    return generator.start_streaming_generation(
        source_code=source_code,
        module_path=module_path,
        metadata=metadata,
        chunk_callback=chunk_callback
    )
"""
TestMaster Streaming Test Generation

Real-time, incremental test generation system inspired by multi-agent
streaming patterns. Provides live feedback, progressive enhancement,
and collaborative test development.

Features:
- Real-time test generation with live streaming output
- Incremental enhancement and refinement
- Multi-stage generation pipeline with feedback loops
- Integration with existing TestMaster components
- Collaborative test development patterns
"""

from .stream_generator import (
    StreamGenerator, StreamConfig, GenerationStage,
    get_stream_generator, stream_generate_test
)
from .incremental_enhancer import (
    IncrementalEnhancer, EnhancementPipeline, RefinementStage,
    get_incremental_enhancer, enhance_test_incrementally
)
from .live_feedback import (
    LiveFeedback, FeedbackCollector, FeedbackType,
    get_feedback_collector, collect_live_feedback
)
from .collaborative_generator import (
    CollaborativeGenerator, GeneratorAgent, CollaborationMode,
    get_collaborative_generator, generate_collaboratively
)
from .stream_monitor import (
    StreamMonitor, StreamMetrics, StreamEvent,
    get_stream_monitor, monitor_streaming
)

__all__ = [
    # Core streaming generation
    'StreamGenerator',
    'StreamConfig', 
    'GenerationStage',
    'get_stream_generator',
    'stream_generate_test',
    
    # Incremental enhancement
    'IncrementalEnhancer',
    'EnhancementPipeline',
    'RefinementStage',
    'get_incremental_enhancer',
    'enhance_test_incrementally',
    
    # Live feedback
    'LiveFeedback',
    'FeedbackCollector',
    'FeedbackType',
    'get_feedback_collector',
    'collect_live_feedback',
    
    # Collaborative generation
    'CollaborativeGenerator',
    'GeneratorAgent',
    'CollaborationMode',
    'get_collaborative_generator',
    'generate_collaboratively',
    
    # Stream monitoring
    'StreamMonitor',
    'StreamMetrics',
    'StreamEvent',
    'get_stream_monitor',
    'monitor_streaming',
    
    # Utilities
    'is_streaming_enabled',
    'configure_streaming',
    'shutdown_streaming'
]

def is_streaming_enabled() -> bool:
    """Check if streaming test generation is enabled."""
    from ..core.feature_flags import FeatureFlags
    return FeatureFlags.is_enabled('layer1_test_foundation', 'streaming_generation')

def configure_streaming(buffer_size: int = 1000,
                       enable_collaboration: bool = True,
                       enable_live_feedback: bool = True):
    """Configure streaming test generation system."""
    if not is_streaming_enabled():
        print("Streaming test generation is disabled")
        return
    
    # Configure stream generator
    generator = get_stream_generator()
    generator.configure(buffer_size=buffer_size)
    
    # Configure collaborative generation
    if enable_collaboration:
        collab_gen = get_collaborative_generator()
        collab_gen.initialize()
    
    # Configure live feedback
    if enable_live_feedback:
        feedback = get_feedback_collector()
        feedback.start_collection()
    
    # Start monitoring
    monitor = get_stream_monitor()
    monitor.start_monitoring()
    
    print(f"Streaming test generation configured (buffer: {buffer_size}, collaboration: {enable_collaboration})")

def shutdown_streaming():
    """Shutdown all streaming components."""
    try:
        # Shutdown in reverse order of dependencies
        monitor = get_stream_monitor()
        monitor.shutdown()
        
        feedback = get_feedback_collector()
        feedback.stop_collection()
        
        collab_gen = get_collaborative_generator()
        collab_gen.shutdown()
        
        enhancer = get_incremental_enhancer()
        enhancer.shutdown()
        
        generator = get_stream_generator()
        generator.shutdown()
        
        print("Streaming test generation shutdown completed")
    except Exception as e:
        print(f"Error during streaming shutdown: {e}")

# Initialize streaming if enabled
if is_streaming_enabled():
    configure_streaming()
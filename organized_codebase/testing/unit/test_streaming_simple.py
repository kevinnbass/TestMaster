"""
Simple test for streaming generation components.
Tests the components directly without feature flag dependencies.
"""

import asyncio
import time
from testmaster.streaming.stream_generator import StreamGenerator, StreamConfig, GenerationStage
from testmaster.streaming.incremental_enhancer import IncrementalEnhancer, EnhancementPipeline
from testmaster.streaming.live_feedback import FeedbackCollector, FeedbackType

async def test_stream_generator():
    """Test stream generator directly."""
    print("\\n[*] Testing StreamGenerator directly...")
    
    # Create generator with manual enabling
    config = StreamConfig(buffer_size=100, chunk_size=50)
    generator = StreamGenerator(config)
    
    # Manually enable for testing
    generator.enabled = True
    
    chunks_received = []
    
    def chunk_callback(chunk):
        chunks_received.append(chunk)
        print(f"   [>] {chunk.stage.value}: {chunk.content[:30]}...")
    
    test_code = '''
def add(a, b):
    return a + b
'''
    
    session_id = generator.start_streaming_generation(
        source_code=test_code,
        module_path="test.py",
        chunk_callback=chunk_callback
    )
    
    print(f"   [+] Session started: {session_id}")
    
    # Wait for completion
    await asyncio.sleep(4.0)
    
    # Check results
    progress = generator.get_session_progress(session_id)
    if progress:
        print(f"   [i] Stages completed: {len(progress.stages_completed)}")
        print(f"   [i] Success: {progress.success}")
    
    stats = generator.get_generator_statistics()
    print(f"   [i] Total chunks: {stats['total_chunks_generated']}")
    
    generator.shutdown()
    return len(chunks_received) > 0

async def test_incremental_enhancer():
    """Test incremental enhancer directly."""
    print("\\n[*] Testing IncrementalEnhancer directly...")
    
    pipeline = EnhancementPipeline(max_iterations=2, quality_threshold=0.7)
    enhancer = IncrementalEnhancer(pipeline)
    
    # Manually enable for testing
    enhancer.enabled = True
    
    basic_test = '''
import unittest
class TestAdd(unittest.TestCase):
    def test_basic(self):
        pass
'''
    
    enhancement_id = enhancer.enhance_test_incrementally(
        test_content=basic_test,
        module_path="test.py"
    )
    
    print(f"   [+] Enhancement started: {enhancement_id}")
    
    # Wait for completion
    await asyncio.sleep(2.0)
    
    result = enhancer.get_enhancement_result(enhancement_id)
    if result:
        print(f"   [i] Success: {result.success}")
        print(f"   [i] Quality score: {result.quality_score:.2f}")
        print(f"   [i] Enhanced length: {len(result.enhanced_test)}")
    
    enhancer.shutdown()
    return result and result.success

async def test_feedback_collector():
    """Test feedback collector directly."""
    print("\\n[*] Testing FeedbackCollector directly...")
    
    collector = FeedbackCollector(collection_interval=1.0)
    
    # Manually enable for testing
    collector.enabled = True
    collector.start_collection()
    
    session_id = collector.start_feedback_session("generation", "test_target")
    print(f"   [+] Feedback session: {session_id}")
    
    # Submit feedback
    feedback_id = collector.submit_feedback(
        session_id=session_id,
        feedback_type=FeedbackType.QUALITY_RATING,
        content="Good test quality",
        rating=0.85
    )
    
    print(f"   [+] Feedback submitted: {feedback_id}")
    
    # Wait for processing
    await asyncio.sleep(1.5)
    
    # Check results
    feedback_list = collector.get_session_feedback(session_id)
    summary = collector.get_feedback_summary(session_id)
    
    print(f"   [i] Feedback count: {len(feedback_list)}")
    if summary:
        print(f"   [i] Average rating: {summary.average_rating:.2f}")
    
    collector.stop_collection()
    return len(feedback_list) > 0

async def main():
    """Run simple streaming tests."""
    print("=" * 50)
    print("Simple Streaming Generation Test")
    print("=" * 50)
    
    results = {}
    
    try:
        results['stream_generator'] = await test_stream_generator()
        results['incremental_enhancer'] = await test_incremental_enhancer()
        results['feedback_collector'] = await test_feedback_collector()
        
        print("\\n" + "=" * 50)
        print("Results Summary")
        print("=" * 50)
        
        for component, success in results.items():
            status = "[PASS]" if success else "[FAIL]"
            print(f"{component}: {status}")
        
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        print(f"\\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("All streaming tests PASSED!")
        else:
            print("Some tests failed")
            
    except Exception as e:
        print(f"Test error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
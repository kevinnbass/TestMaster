"""
Test script for TestMaster Streaming Test Generation System

Comprehensive testing of streaming generation components:
- StreamGenerator: Real-time test generation with progressive stages
- IncrementalEnhancer: Progressive test improvement and refinement
- LiveFeedback: Real-time feedback collection and processing
- CollaborativeGenerator: Multi-agent collaborative generation
- StreamMonitor: Performance and usage monitoring
"""

import asyncio
import time
import threading
from datetime import datetime
from testmaster.core.feature_flags import FeatureFlags
from testmaster.streaming import (
    # Core components
    StreamGenerator, IncrementalEnhancer, FeedbackCollector,
    CollaborativeGenerator, StreamMonitor,
    
    # Convenience functions
    stream_generate_test, enhance_test_incrementally, collect_live_feedback,
    generate_collaboratively, monitor_streaming,
    
    # Enums and configs
    GenerationStage, RefinementStage, FeedbackType, CollaborationMode,
    StreamConfig, EnhancementPipeline,
    
    # Global instances
    get_stream_generator, get_incremental_enhancer, get_feedback_collector,
    get_collaborative_generator, get_stream_monitor,
    
    # Utilities
    is_streaming_enabled, configure_streaming, shutdown_streaming
)

class StreamingGenerationTest:
    """Comprehensive test suite for streaming generation."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        self.test_source = '''
def calculate_fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def is_prime(num):
    """Check if a number is prime."""
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True
'''
        
    async def run_all_tests(self):
        """Run all streaming generation tests."""
        print("=" * 60)
        print("TestMaster Streaming Test Generation System Test")
        print("=" * 60)
        
        # Initialize feature flags
        FeatureFlags.initialize("testmaster_config.yaml")
        
        # Debug feature flags
        print(f"[i] Feature flags initialized")
        print(f"[i] Layer1 enabled: {FeatureFlags.is_enabled('layer1_test_foundation', 'test_generation')}")
        print(f"[i] Streaming enabled: {FeatureFlags.is_enabled('layer1_test_foundation', 'streaming_generation')}")
        
        # Check if streaming is enabled
        if not is_streaming_enabled():
            print("[!] Streaming test generation is disabled")
            
            # Force enable for testing
            print("[i] Force enabling streaming for test...")
            # We'll test the components directly even if feature flag is off
            
        else:
            print("[+] Streaming test generation is enabled")
        
        # Configure streaming
        configure_streaming(
            buffer_size=500,
            enable_collaboration=True,
            enable_live_feedback=True
        )
        
        # Test individual components
        await self.test_stream_generator()
        await self.test_incremental_enhancer()
        await self.test_live_feedback()
        await self.test_collaborative_generator()
        await self.test_stream_monitor()
        await self.test_integration()
        
        # Display results
        self.display_results()
    
    async def test_stream_generator(self):
        """Test StreamGenerator functionality."""
        print("\\n[*] Testing StreamGenerator...")
        
        try:
            generator = get_stream_generator()
            
            # Test streaming generation
            chunks_received = []
            
            def chunk_callback(chunk):
                chunks_received.append(chunk)
                print(f"   [>] Stage {chunk.stage.value}: {chunk.content[:50]}...")
            
            session_id = generator.start_streaming_generation(
                source_code=self.test_source,
                module_path="test_module.py",
                metadata={"test": "stream_generator"},
                chunk_callback=chunk_callback
            )
            
            print(f"   [+] Started streaming session: {session_id}")
            
            # Wait for generation to complete
            await asyncio.sleep(3.0)
            
            # Check session progress
            progress = generator.get_session_progress(session_id)
            if progress:
                print(f"   [i] Progress: {len(progress.stages_completed)} stages completed")
                print(f"   [i] Success: {progress.success}")
            
            # Get final chunks
            final_chunks = generator.get_stream_chunks(session_id, timeout=0.1)
            print(f"   [i] Final chunks received: {len(final_chunks)}")
            
            # Check statistics
            stats = generator.get_generator_statistics()
            print(f"   [i] Generator stats: {stats['total_sessions']} sessions, {stats['total_chunks_generated']} chunks")
            
            self.test_results['stream_generator'] = len(chunks_received) > 0
            
        except Exception as e:
            print(f"   [!] StreamGenerator test failed: {e}")
            self.test_results['stream_generator'] = False
    
    async def test_incremental_enhancer(self):
        """Test IncrementalEnhancer functionality."""
        print("\\n[*] Testing IncrementalEnhancer...")
        
        try:
            enhancer = get_incremental_enhancer()
            
            # Test incremental enhancement
            basic_test = '''
import unittest

class TestCalculator(unittest.TestCase):
    def test_something(self):
        self.assertTrue(True)
'''
            
            enhancement_updates = []
            
            def enhancement_callback(result):
                enhancement_updates.append(result)
                print(f"   [>] Enhancement iteration {result.iteration_count}, quality: {result.quality_score:.2f}")
            
            enhancement_id = enhancer.enhance_test_incrementally(
                test_content=basic_test,
                module_path="test_module.py",
                metadata={"test": "incremental_enhancer"},
                callback=enhancement_callback
            )
            
            print(f"   [+] Started enhancement: {enhancement_id}")
            
            # Wait for enhancement to complete
            await asyncio.sleep(2.0)
            
            # Check enhancement result
            result = enhancer.get_enhancement_result(enhancement_id)
            if result:
                print(f"   [i] Enhancement success: {result.success}")
                print(f"   [i] Final quality score: {result.quality_score:.2f}")
                print(f"   [i] Improvements made: {len(result.improvements)}")
            
            # Check refinement history
            history = enhancer.get_refinement_history(enhancement_id)
            print(f"   [i] Refinement steps: {len(history)}")
            
            # Check statistics
            stats = enhancer.get_enhancer_statistics()
            print(f"   [i] Enhancer stats: {stats['total_enhancements']} processed, {stats['success_rate']}% success")
            
            self.test_results['incremental_enhancer'] = len(enhancement_updates) > 0
            
        except Exception as e:
            print(f"   [!] IncrementalEnhancer test failed: {e}")
            self.test_results['incremental_enhancer'] = False
    
    async def test_live_feedback(self):
        """Test FeedbackCollector functionality."""
        print("\\n[*] Testing FeedbackCollector...")
        
        try:
            collector = get_feedback_collector()
            collector.start_collection()
            
            # Start feedback session
            session_id = collector.start_feedback_session(
                target_type="generation",
                target_id="test_target_123"
            )
            
            print(f"   [+] Started feedback session: {session_id}")
            
            # Submit various types of feedback
            feedback_types = [
                (FeedbackType.QUALITY_RATING, "Good test structure", 0.8),
                (FeedbackType.SUGGESTION, "Add more edge cases", None),
                (FeedbackType.APPROVAL, "Looks good!", None),
                (FeedbackType.ISSUE_REPORT, "Missing import statement", None)
            ]
            
            for feedback_type, content, rating in feedback_types:
                feedback_id = collector.submit_feedback(
                    session_id=session_id,
                    feedback_type=feedback_type,
                    content=content,
                    rating=rating,
                    metadata={"test": "live_feedback"}
                )
                print(f"   [+] Submitted feedback: {feedback_type.value}")
            
            # Wait for processing
            await asyncio.sleep(1.0)
            
            # Check session feedback
            session_feedback = collector.get_session_feedback(session_id)
            print(f"   [i] Session feedback count: {len(session_feedback)}")
            
            # Get feedback summary
            summary = collector.get_feedback_summary(session_id)
            if summary:
                print(f"   [i] Average rating: {summary.average_rating:.2f}")
                print(f"   [i] Approval rate: {summary.approval_rate:.2f}")
                print(f"   [i] Feedback types: {len(summary.feedback_by_type)}")
            
            # Check statistics
            stats = collector.get_collector_statistics()
            print(f"   [i] Collector stats: {stats['total_feedback']} feedback, {stats['active_sessions']} sessions")
            
            # Close session
            collector.close_session(session_id)
            
            self.test_results['live_feedback'] = len(session_feedback) > 0
            
        except Exception as e:
            print(f"   [!] FeedbackCollector test failed: {e}")
            self.test_results['live_feedback'] = False
    
    async def test_collaborative_generator(self):
        """Test CollaborativeGenerator functionality."""
        print("\\n[*] Testing CollaborativeGenerator...")
        
        try:
            generator = get_collaborative_generator()
            generator.initialize()
            
            # Test collaborative generation
            result = generator.generate_collaboratively(
                source_code=self.test_source,
                module_path="test_module.py",
                mode=CollaborationMode.SEQUENTIAL
            )
            
            print(f"   [+] Generated collaborative test")
            print(f"   [i] Result length: {len(result)} characters")
            
            # Check if result contains expected content
            has_collaboration = "Collaborative" in result
            has_contributions = "contribution" in result
            
            print(f"   [i] Has collaboration marker: {has_collaboration}")
            print(f"   [i] Has agent contributions: {has_contributions}")
            
            self.test_results['collaborative_generator'] = has_collaboration and has_contributions
            
        except Exception as e:
            print(f"   [!] CollaborativeGenerator test failed: {e}")
            self.test_results['collaborative_generator'] = False
    
    async def test_stream_monitor(self):
        """Test StreamMonitor functionality."""
        print("\\n[*] Testing StreamMonitor...")
        
        try:
            monitor = get_stream_monitor()
            monitor.start_monitoring()
            
            # Test monitoring
            session_id = monitor_streaming("test_session_123")
            
            print(f"   [+] Started monitoring session: {session_id}")
            print(f"   [i] Monitor is running: {monitor.is_monitoring}")
            
            # Check metrics
            print(f"   [i] Current metrics: {monitor.metrics.total_sessions} sessions")
            
            monitor.shutdown()
            print(f"   [i] Monitor shutdown: {not monitor.is_monitoring}")
            
            self.test_results['stream_monitor'] = True
            
        except Exception as e:
            print(f"   [!] StreamMonitor test failed: {e}")
            self.test_results['stream_monitor'] = False
    
    async def test_integration(self):
        """Test integrated streaming functionality."""
        print("\\n[*] Testing Integration...")
        
        try:
            # Test end-to-end streaming workflow
            print("   [>] Starting integrated streaming workflow...")
            
            # 1. Start streaming generation
            session_id = stream_generate_test(
                source_code=self.test_source,
                module_path="integration_test.py",
                metadata={"test": "integration"}
            )
            
            # 2. Start feedback collection
            feedback_session = collect_live_feedback("generation", session_id)
            
            # 3. Submit some feedback
            collector = get_feedback_collector()
            collector.submit_feedback(
                session_id=feedback_session,
                feedback_type=FeedbackType.QUALITY_RATING,
                content="Integration test feedback",
                rating=0.9
            )
            
            # 4. Generate collaboratively
            collab_result = generate_collaboratively(
                self.test_source,
                "integration_test.py"
            )
            
            # 5. Enhance incrementally
            enhancement_id = enhance_test_incrementally(
                test_content="# Basic test\\npass",
                module_path="integration_test.py"
            )
            
            # Wait for all processes
            await asyncio.sleep(2.0)
            
            # Check results
            generator = get_stream_generator()
            enhancer = get_incremental_enhancer()
            
            gen_stats = generator.get_generator_statistics()
            enh_stats = enhancer.get_enhancer_statistics()
            fb_stats = collector.get_collector_statistics()
            
            print(f"   [i] Integration stats:")
            print(f"      - Generator: {gen_stats['total_sessions']} sessions")
            print(f"      - Enhancer: {enh_stats['total_enhancements']} enhancements")
            print(f"      - Feedback: {fb_stats['total_feedback']} feedback")
            print(f"      - Collaborative result: {len(collab_result)} chars")
            
            # Check overall integration success
            integration_success = (
                gen_stats['total_sessions'] > 0 and
                fb_stats['total_feedback'] > 0 and
                len(collab_result) > 0
            )
            
            self.test_results['integration'] = integration_success
            
        except Exception as e:
            print(f"   [!] Integration test failed: {e}")
            self.test_results['integration'] = False
    
    def display_results(self):
        """Display test results summary."""
        print("\\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        for component, result in self.test_results.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"{component.replace('_', ' ').title()}: {status}")
        
        print(f"\\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("All streaming generation tests PASSED!")
        else:
            print("Some tests failed - check implementation")
        
        execution_time = time.time() - self.start_time
        print(f"Total execution time: {execution_time:.2f} seconds")

async def main():
    """Main test execution."""
    try:
        # Run tests
        test_suite = StreamingGenerationTest()
        await test_suite.run_all_tests()
        
    finally:
        # Cleanup
        print("\\nCleaning up streaming generation...")
        shutdown_streaming()
        print("Cleanup completed")

if __name__ == "__main__":
    asyncio.run(main())
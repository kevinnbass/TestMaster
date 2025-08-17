"""
Incremental Enhancer for TestMaster

Progressive test enhancement system inspired by iterative agent patterns.
Continuously refines and improves generated tests through multiple passes.
"""

import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import re

from ..core.feature_flags import FeatureFlags
from ..core.shared_state import get_shared_state
from ..telemetry import get_telemetry_collector

class RefinementStage(Enum):
    """Stages of incremental refinement."""
    SYNTAX_REVIEW = "syntax_review"
    LOGIC_ENHANCEMENT = "logic_enhancement"
    COVERAGE_EXPANSION = "coverage_expansion"
    ASSERTION_IMPROVEMENT = "assertion_improvement"
    DOCUMENTATION_ENHANCEMENT = "documentation_enhancement"
    OPTIMIZATION = "optimization"

@dataclass
class EnhancementPipeline:
    """Configuration for enhancement pipeline."""
    max_iterations: int = 3
    quality_threshold: float = 0.8
    enable_parallel: bool = True
    refinement_stages: List[RefinementStage] = field(default_factory=lambda: [
        RefinementStage.SYNTAX_REVIEW,
        RefinementStage.LOGIC_ENHANCEMENT,
        RefinementStage.COVERAGE_EXPANSION,
        RefinementStage.ASSERTION_IMPROVEMENT,
        RefinementStage.DOCUMENTATION_ENHANCEMENT,
        RefinementStage.OPTIMIZATION
    ])

@dataclass
class EnhancementResult:
    """Result of incremental enhancement."""
    enhancement_id: str
    original_test: str
    enhanced_test: str
    improvements: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    iteration_count: int = 0
    stages_completed: List[RefinementStage] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RefinementStep:
    """Individual refinement step."""
    step_id: str
    stage: RefinementStage
    before_content: str
    after_content: str
    improvement_description: str
    quality_impact: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class IncrementalEnhancer:
    """
    Incremental test enhancer for progressive improvement.
    
    Features:
    - Multi-stage refinement pipeline
    - Iterative quality improvement
    - Parallel enhancement processing
    - Quality scoring and threshold checking
    - Integration with streaming generation
    """
    
    def __init__(self, pipeline: EnhancementPipeline = None):
        """
        Initialize incremental enhancer.
        
        Args:
            pipeline: Enhancement pipeline configuration
        """
        self.enabled = FeatureFlags.is_enabled('layer1_test_foundation', 'streaming_generation')
        self.pipeline = pipeline or EnhancementPipeline()
        
        # Initialize all attributes regardless of enabled state
        # Enhancement state
        self.active_enhancements: Dict[str, EnhancementResult] = {}
        self.refinement_history: Dict[str, List[RefinementStep]] = {}
        
        # Threading
        self.lock = threading.RLock()
        self.worker_threads: Dict[str, threading.Thread] = {}
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.enhancements_processed = 0
        self.successful_enhancements = 0
        self.total_refinement_steps = 0
        
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
        
        print("Incremental enhancer initialized")
        print(f"   Max iterations: {self.pipeline.max_iterations}")
        print(f"   Quality threshold: {self.pipeline.quality_threshold}")
    
    def enhance_test_incrementally(self, test_content: str, module_path: str,
                                 metadata: Dict[str, Any] = None,
                                 callback: Callable[[EnhancementResult], None] = None) -> str:
        """
        Start incremental enhancement of a test.
        
        Args:
            test_content: Test content to enhance
            module_path: Path to the module
            metadata: Additional metadata
            callback: Callback for enhancement updates
            
        Returns:
            Enhancement ID for tracking
        """
        if not self.enabled:
            raise RuntimeError("Incremental enhancer is disabled")
        
        enhancement_id = str(uuid.uuid4())
        
        # Initialize enhancement
        result = EnhancementResult(
            enhancement_id=enhancement_id,
            original_test=test_content,
            enhanced_test=test_content,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.active_enhancements[enhancement_id] = result
            self.refinement_history[enhancement_id] = []
            self.enhancements_processed += 1
        
        # Start enhancement worker
        worker = threading.Thread(
            target=self._enhancement_worker,
            args=(enhancement_id, test_content, module_path, callback),
            daemon=True
        )
        
        with self.lock:
            self.worker_threads[enhancement_id] = worker
        
        worker.start()
        
        # Send telemetry
        if self.telemetry:
            self.telemetry.record_event(
                event_type="incremental_enhancement_started",
                component="incremental_enhancer",
                operation="enhance_test",
                metadata={
                    "enhancement_id": enhancement_id,
                    "module_path": module_path,
                    "original_length": len(test_content)
                }
            )
        
        print(f"Started incremental enhancement: {enhancement_id}")
        return enhancement_id
    
    def _enhancement_worker(self, enhancement_id: str, test_content: str,
                          module_path: str, callback: Optional[Callable]):
        """Worker thread for incremental enhancement."""
        try:
            result = self.active_enhancements[enhancement_id]
            current_content = test_content
            
            # Perform iterative enhancement
            for iteration in range(self.pipeline.max_iterations):
                if self.shutdown_event.is_set():
                    break
                
                result.iteration_count = iteration + 1
                iteration_improved = False
                
                # Apply refinement stages
                for stage in self.pipeline.refinement_stages:
                    if self.shutdown_event.is_set():
                        break
                    
                    enhanced_content = self._apply_refinement_stage(
                        enhancement_id, stage, current_content, module_path
                    )
                    
                    if enhanced_content != current_content:
                        # Record refinement step
                        step = RefinementStep(
                            step_id=str(uuid.uuid4()),
                            stage=stage,
                            before_content=current_content,
                            after_content=enhanced_content,
                            improvement_description=f"Applied {stage.value} refinement"
                        )
                        
                        with self.lock:
                            self.refinement_history[enhancement_id].append(step)
                            self.total_refinement_steps += 1
                        
                        current_content = enhanced_content
                        iteration_improved = True
                        result.stages_completed.append(stage)
                
                # Update result
                result.enhanced_test = current_content
                result.quality_score = self._calculate_quality_score(current_content)
                
                # Call callback if provided
                if callback:
                    try:
                        callback(result)
                    except Exception as e:
                        print(f"Enhancement callback error: {e}")
                
                # Check if quality threshold reached
                if result.quality_score >= self.pipeline.quality_threshold:
                    break
                
                # If no improvements in this iteration, stop
                if not iteration_improved:
                    break
            
            # Mark as successful
            result.success = True
            result.end_time = datetime.now()
            
            with self.lock:
                self.successful_enhancements += 1
            
        except Exception as e:
            # Handle error
            result = self.active_enhancements[enhancement_id]
            result.success = False
            result.end_time = datetime.now()
            result.metadata["error"] = str(e)
            
            print(f"Enhancement error in {enhancement_id}: {e}")
        
        finally:
            # Cleanup worker
            with self.lock:
                self.worker_threads.pop(enhancement_id, None)
    
    def _apply_refinement_stage(self, enhancement_id: str, stage: RefinementStage,
                              content: str, module_path: str) -> str:
        """Apply a specific refinement stage."""
        if stage == RefinementStage.SYNTAX_REVIEW:
            return self._refine_syntax(content)
        elif stage == RefinementStage.LOGIC_ENHANCEMENT:
            return self._enhance_logic(content)
        elif stage == RefinementStage.COVERAGE_EXPANSION:
            return self._expand_coverage(content, module_path)
        elif stage == RefinementStage.ASSERTION_IMPROVEMENT:
            return self._improve_assertions(content)
        elif stage == RefinementStage.DOCUMENTATION_ENHANCEMENT:
            return self._enhance_documentation(content)
        elif stage == RefinementStage.OPTIMIZATION:
            return self._optimize_test(content)
        
        return content
    
    def _refine_syntax(self, content: str) -> str:
        """Refine test syntax and structure."""
        lines = content.split('\n')
        refined_lines = []
        
        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            
            # Fix common syntax issues
            if line.strip().startswith('def test_') and not line.endswith(':'):
                line += ':'
            
            # Ensure proper indentation for test methods
            if line.strip().startswith('def test_') and not line.startswith('    '):
                line = '    ' + line.strip()
            
            refined_lines.append(line)
        
        return '\n'.join(refined_lines)
    
    def _enhance_logic(self, content: str) -> str:
        """Enhance test logic and flow."""
        # Add setup/teardown if missing
        if 'def setUp(self):' not in content and 'class Test' in content:
            # Find class definition and add setUp
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('class Test') and line.endswith(':'):
                    setup_method = [
                        '',
                        '    def setUp(self):',
                        '        """Set up test fixtures."""',
                        '        pass',
                        ''
                    ]
                    lines[i+1:i+1] = setup_method
                    break
            content = '\n'.join(lines)
        
        # Improve placeholder tests
        content = content.replace(
            'self.assertTrue(True)',
            'self.assertIsNotNone(None)  # TODO: Implement meaningful assertion'
        )
        
        return content
    
    def _expand_coverage(self, content: str, module_path: str) -> str:
        """Expand test coverage."""
        # Add edge case tests if missing
        if 'test_edge_cases' not in content:
            edge_case_test = '''
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # TODO: Test empty inputs
        # TODO: Test None values
        # TODO: Test boundary values
        pass
'''
            content += edge_case_test
        
        # Add error handling tests if missing
        if 'test_error_handling' not in content and 'test_exceptions' not in content:
            error_test = '''
    def test_error_handling(self):
        """Test error handling and exceptions."""
        # TODO: Test invalid inputs
        # TODO: Test exception scenarios
        with self.assertRaises(Exception):
            pass
'''
            content += error_test
        
        return content
    
    def _improve_assertions(self, content: str) -> str:
        """Improve test assertions."""
        # Replace weak assertions with stronger ones
        improvements = [
            ('self.assertTrue(True)', 'self.assertEqual(actual, expected)'),
            ('assert True', 'self.assertTrue(condition)'),
            ('pass  # TODO', 'self.fail("Test not implemented")'),
        ]
        
        for old, new in improvements:
            content = content.replace(old, new)
        
        return content
    
    def _enhance_documentation(self, content: str) -> str:
        """Enhance test documentation."""
        lines = content.split('\n')
        enhanced_lines = []
        
        for line in lines:
            enhanced_lines.append(line)
            
            # Add docstrings to test methods without them
            if (line.strip().startswith('def test_') and 
                line.endswith(':') and 
                len(enhanced_lines) > 1):
                
                # Check if next line is a docstring
                next_line_idx = len(enhanced_lines)
                if (next_line_idx < len(lines) and 
                    not lines[next_line_idx].strip().startswith('"""')):
                    
                    method_name = line.strip().replace('def ', '').replace('(self):', '').replace('_', ' ').title()
                    docstring = f'        """Test {method_name.replace("Test ", "").lower()}."""'
                    enhanced_lines.append(docstring)
        
        return '\n'.join(enhanced_lines)
    
    def _optimize_test(self, content: str) -> str:
        """Optimize test performance and structure."""
        # Remove duplicate imports
        lines = content.split('\n')
        imports = set()
        optimized_lines = []
        
        for line in lines:
            if line.startswith('import ') or line.startswith('from '):
                if line not in imports:
                    imports.add(line)
                    optimized_lines.append(line)
            else:
                optimized_lines.append(line)
        
        # Remove excessive empty lines
        final_lines = []
        empty_count = 0
        
        for line in optimized_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:  # Max 2 consecutive empty lines
                    final_lines.append(line)
            else:
                empty_count = 0
                final_lines.append(line)
        
        return '\n'.join(final_lines)
    
    def _calculate_quality_score(self, content: str) -> float:
        """Calculate quality score for test content."""
        score = 0.0
        max_score = 10.0
        
        # Basic structure (2 points)
        if 'class Test' in content:
            score += 1.0
        if 'def test_' in content:
            score += 1.0
        
        # Assertions (2 points)
        assertion_count = len(re.findall(r'self\.assert\w+', content))
        score += min(2.0, assertion_count * 0.5)
        
        # Documentation (2 points)
        docstring_count = content.count('"""')
        score += min(2.0, docstring_count * 0.5)
        
        # Coverage (2 points)
        if 'test_edge_cases' in content or 'edge' in content.lower():
            score += 1.0
        if 'test_error' in content or 'exception' in content.lower():
            score += 1.0
        
        # Setup/teardown (1 point)
        if 'def setUp' in content or 'def tearDown' in content:
            score += 1.0
        
        # Import statements (1 point)
        if 'import unittest' in content:
            score += 1.0
        
        return score / max_score
    
    def get_enhancement_result(self, enhancement_id: str) -> Optional[EnhancementResult]:
        """Get enhancement result."""
        if not self.enabled:
            return None
        
        with self.lock:
            return self.active_enhancements.get(enhancement_id)
    
    def get_refinement_history(self, enhancement_id: str) -> List[RefinementStep]:
        """Get refinement history for an enhancement."""
        if not self.enabled:
            return []
        
        with self.lock:
            return self.refinement_history.get(enhancement_id, [])
    
    def get_enhancer_statistics(self) -> Dict[str, Any]:
        """Get enhancer statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            active_enhancements = len(self.active_enhancements)
            success_rate = 0.0
            if self.enhancements_processed > 0:
                success_rate = (self.successful_enhancements / self.enhancements_processed) * 100
            
            return {
                "enabled": True,
                "active_enhancements": active_enhancements,
                "total_enhancements": self.enhancements_processed,
                "successful_enhancements": self.successful_enhancements,
                "total_refinement_steps": self.total_refinement_steps,
                "success_rate": round(success_rate, 2),
                "max_iterations": self.pipeline.max_iterations,
                "quality_threshold": self.pipeline.quality_threshold
            }
    
    def shutdown(self):
        """Shutdown the incremental enhancer."""
        if not self.enabled:
            return
        
        print("Shutting down incremental enhancer...")
        
        # Signal shutdown to all workers
        self.shutdown_event.set()
        
        # Wait for workers to complete
        with self.lock:
            for worker in list(self.worker_threads.values()):
                if worker.is_alive():
                    worker.join(timeout=2.0)
            
            enhancements_count = len(self.active_enhancements)
            steps_count = self.total_refinement_steps
            
            # Clear all state
            self.active_enhancements.clear()
            self.refinement_history.clear()
            self.worker_threads.clear()
        
        print(f"Incremental enhancer shutdown - processed {enhancements_count} enhancements, {steps_count} steps")

# Global instance
_incremental_enhancer: Optional[IncrementalEnhancer] = None

def get_incremental_enhancer() -> IncrementalEnhancer:
    """Get the global incremental enhancer instance."""
    global _incremental_enhancer
    if _incremental_enhancer is None:
        _incremental_enhancer = IncrementalEnhancer()
    return _incremental_enhancer

# Convenience function
def enhance_test_incrementally(test_content: str, module_path: str,
                              metadata: Dict[str, Any] = None,
                              callback: Callable[[EnhancementResult], None] = None) -> str:
    """
    Enhance a test incrementally.
    
    Args:
        test_content: Test content to enhance
        module_path: Path to the module
        metadata: Additional metadata
        callback: Callback for enhancement updates
        
    Returns:
        Enhancement ID for tracking
    """
    enhancer = get_incremental_enhancer()
    return enhancer.enhance_test_incrementally(
        test_content=test_content,
        module_path=module_path,
        metadata=metadata,
        callback=callback
    )
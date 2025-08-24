"""
Universal AI Test Generator for TestMaster
Coordinates multiple AI providers for optimal test generation
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from .claude_test_generator import ClaudeTestGenerator, TestGenerationStrategy, GeneratedTest
from .gemini_test_generator import GeminiTestGenerator, TestGenerationConfig, TestSuite

class AIProvider(Enum):
    """Available AI providers"""
    CLAUDE = "claude"
    GEMINI = "gemini"
    AUTO = "auto"

class GenerationMode(Enum):
    """Test generation modes"""
    FAST = "fast"
    COMPREHENSIVE = "comprehensive"
    MULTI_STRATEGY = "multi_strategy"
    BEST_QUALITY = "best_quality"

@dataclass
class UniversalTestConfig:
    """Universal configuration for test generation"""
    provider: AIProvider = AIProvider.AUTO
    mode: GenerationMode = GenerationMode.COMPREHENSIVE
    target_coverage: float = 0.95
    max_generation_time: int = 120  # seconds
    fallback_enabled: bool = True
    quality_threshold: float = 0.7
    parallel_generation: bool = True

@dataclass
class GenerationResult:
    """Result from universal test generation"""
    test_code: str
    provider_used: AIProvider
    generation_time: float
    quality_score: float
    coverage_estimate: float
    metadata: Dict[str, Any]
    fallback_used: bool = False

class UniversalAIGenerator:
    """Universal AI test generator coordinating multiple providers"""
    
    def __init__(self, claude_api_key: Optional[str] = None, 
                 gemini_api_key: Optional[str] = None):
        self.claude_generator = None
        self.gemini_generator = None
        
        # Initialize available generators
        try:
            if claude_api_key:
                self.claude_generator = ClaudeTestGenerator(claude_api_key)
        except Exception:
            pass
        
        try:
            if gemini_api_key:
                self.gemini_generator = GeminiTestGenerator(gemini_api_key)
        except Exception:
            pass
        
        self.generation_history: List[GenerationResult] = []
        self.provider_performance: Dict[AIProvider, Dict[str, float]] = {
            AIProvider.CLAUDE: {'avg_quality': 0.8, 'avg_time': 15.0, 'success_rate': 0.9},
            AIProvider.GEMINI: {'avg_quality': 0.8, 'avg_time': 12.0, 'success_rate': 0.9}
        }
    
    async def generate_tests(self, file_path: str, 
                           config: UniversalTestConfig) -> GenerationResult:
        """Generate tests using optimal strategy"""
        start_time = time.time()
        
        try:
            if config.mode == GenerationMode.FAST:
                return await self._generate_fast(file_path, config)
            elif config.mode == GenerationMode.COMPREHENSIVE:
                return await self._generate_comprehensive(file_path, config)
            elif config.mode == GenerationMode.MULTI_STRATEGY:
                return await self._generate_multi_strategy(file_path, config)
            elif config.mode == GenerationMode.BEST_QUALITY:
                return await self._generate_best_quality(file_path, config)
            else:
                return await self._generate_comprehensive(file_path, config)
                
        except Exception as e:
            if config.fallback_enabled:
                return self._generate_fallback(file_path, str(e))
            raise e
    
    async def _generate_fast(self, file_path: str, 
                           config: UniversalTestConfig) -> GenerationResult:
        """Fast generation using best performing provider"""
        provider = self._select_fastest_provider(config)
        
        if provider == AIProvider.GEMINI and self.gemini_generator:
            insights = await self.gemini_generator.analyze_code_with_gemini(file_path)
            test_config = TestGenerationConfig(coverage_target=config.target_coverage)
            result = await self.gemini_generator.generate_intelligent_test_suite(insights, test_config)
            
            return GenerationResult(
                test_code=result.test_code,
                provider_used=AIProvider.GEMINI,
                generation_time=result.generation_metrics['generation_time'],
                quality_score=result.quality_score,
                coverage_estimate=result.estimated_coverage,
                metadata={'insights': insights.__dict__, 'config': test_config.__dict__}
            )
        
        elif provider == AIProvider.CLAUDE and self.claude_generator:
            analysis = await self.claude_generator.analyze_code_intelligence(file_path)
            result = await self.claude_generator.generate_comprehensive_tests(analysis)
            
            return GenerationResult(
                test_code=result.test_code,
                provider_used=AIProvider.CLAUDE,
                generation_time=result.generation_time,
                quality_score=result.confidence_score,
                coverage_estimate=result.estimated_effectiveness,
                metadata={'analysis': analysis.__dict__, 'strategy': result.strategy.value}
            )
        
        else:
            return self._generate_fallback(file_path, "No providers available")
    
    async def _generate_comprehensive(self, file_path: str, 
                                    config: UniversalTestConfig) -> GenerationResult:
        """Comprehensive generation with best available provider"""
        provider = self._select_best_provider(config)
        
        if provider == AIProvider.CLAUDE and self.claude_generator:
            return await self._generate_with_claude(file_path, config)
        elif provider == AIProvider.GEMINI and self.gemini_generator:
            return await self._generate_with_gemini(file_path, config)
        else:
            return self._generate_fallback(file_path, "No providers available")
    
    async def _generate_multi_strategy(self, file_path: str, 
                                     config: UniversalTestConfig) -> GenerationResult:
        """Generate tests using multiple strategies and combine best"""
        if not (self.claude_generator and self.gemini_generator):
            return await self._generate_comprehensive(file_path, config)
        
        # Generate with both providers in parallel
        tasks = [
            self._generate_with_claude(file_path, config),
            self._generate_with_gemini(file_path, config)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Select best result
        valid_results = [r for r in results if isinstance(r, GenerationResult)]
        if not valid_results:
            return self._generate_fallback(file_path, "All providers failed")
        
        best_result = max(valid_results, key=lambda r: r.quality_score)
        
        # Optionally combine results
        if len(valid_results) > 1:
            combined_code = self._combine_test_codes(valid_results)
            best_result.test_code = combined_code
            best_result.metadata['combined_from'] = [r.provider_used.value for r in valid_results]
        
        return best_result
    
    async def _generate_best_quality(self, file_path: str, 
                                   config: UniversalTestConfig) -> GenerationResult:
        """Generate highest quality tests regardless of time"""
        if not self.claude_generator:
            return await self._generate_comprehensive(file_path, config)
        
        # Use Claude with advanced strategy for highest quality
        analysis = await self.claude_generator.analyze_code_intelligence(file_path)
        
        # Try multiple strategies and pick best
        strategies = [
            TestGenerationStrategy.COMPREHENSIVE,
            TestGenerationStrategy.SECURITY,
            TestGenerationStrategy.BUSINESS_LOGIC
        ]
        
        best_result = None
        best_score = 0.0
        
        for strategy in strategies:
            try:
                result = await self.claude_generator.generate_comprehensive_tests(analysis, strategy)
                if result.confidence_score > best_score:
                    best_score = result.confidence_score
                    best_result = result
            except Exception:
                continue
        
        if best_result:
            return GenerationResult(
                test_code=best_result.test_code,
                provider_used=AIProvider.CLAUDE,
                generation_time=best_result.generation_time,
                quality_score=best_result.confidence_score,
                coverage_estimate=best_result.estimated_effectiveness,
                metadata={'analysis': analysis.__dict__, 'strategy': best_result.strategy.value}
            )
        
        return self._generate_fallback(file_path, "Quality generation failed")
    
    async def _generate_with_claude(self, file_path: str, 
                                  config: UniversalTestConfig) -> GenerationResult:
        """Generate tests using Claude"""
        analysis = await self.claude_generator.analyze_code_intelligence(file_path)
        
        # Select strategy based on analysis
        strategy = TestGenerationStrategy.COMPREHENSIVE
        if analysis.security_concerns:
            strategy = TestGenerationStrategy.SECURITY
        elif analysis.business_logic_patterns:
            strategy = TestGenerationStrategy.BUSINESS_LOGIC
        
        result = await self.claude_generator.generate_comprehensive_tests(analysis, strategy)
        
        return GenerationResult(
            test_code=result.test_code,
            provider_used=AIProvider.CLAUDE,
            generation_time=result.generation_time,
            quality_score=result.confidence_score,
            coverage_estimate=result.estimated_effectiveness,
            metadata={
                'analysis': analysis.__dict__,
                'strategy': result.strategy.value,
                'coverage_targets': result.coverage_targets
            }
        )
    
    async def _generate_with_gemini(self, file_path: str, 
                                  config: UniversalTestConfig) -> GenerationResult:
        """Generate tests using Gemini"""
        insights = await self.gemini_generator.analyze_code_with_gemini(file_path)
        
        # Configure based on insights
        test_config = TestGenerationConfig(
            coverage_target=config.target_coverage,
            include_edge_cases=True,
            include_error_handling=True,
            include_performance_tests=bool(insights.performance_concerns)
        )
        
        result = await self.gemini_generator.generate_intelligent_test_suite(insights, test_config)
        
        return GenerationResult(
            test_code=result.test_code,
            provider_used=AIProvider.GEMINI,
            generation_time=result.generation_metrics['generation_time'],
            quality_score=result.quality_score,
            coverage_estimate=result.estimated_coverage,
            metadata={
                'insights': insights.__dict__,
                'config': test_config.__dict__,
                'focus_areas': [f.value for f in result.focus_areas]
            }
        )
    
    def _select_best_provider(self, config: UniversalTestConfig) -> AIProvider:
        """Select best provider based on configuration and performance"""
        if config.provider != AIProvider.AUTO:
            return config.provider
        
        # Select based on performance metrics
        claude_score = self._calculate_provider_score(AIProvider.CLAUDE, config)
        gemini_score = self._calculate_provider_score(AIProvider.GEMINI, config)
        
        if claude_score > gemini_score and self.claude_generator:
            return AIProvider.CLAUDE
        elif self.gemini_generator:
            return AIProvider.GEMINI
        elif self.claude_generator:
            return AIProvider.CLAUDE
        else:
            return AIProvider.AUTO  # Will trigger fallback
    
    def _select_fastest_provider(self, config: UniversalTestConfig) -> AIProvider:
        """Select fastest provider"""
        if config.provider != AIProvider.AUTO:
            return config.provider
        
        claude_time = self.provider_performance[AIProvider.CLAUDE]['avg_time']
        gemini_time = self.provider_performance[AIProvider.GEMINI]['avg_time']
        
        if gemini_time < claude_time and self.gemini_generator:
            return AIProvider.GEMINI
        elif self.claude_generator:
            return AIProvider.CLAUDE
        elif self.gemini_generator:
            return AIProvider.GEMINI
        else:
            return AIProvider.AUTO
    
    def _calculate_provider_score(self, provider: AIProvider, 
                                config: UniversalTestConfig) -> float:
        """Calculate provider score based on requirements"""
        if provider not in self.provider_performance:
            return 0.0
        
        metrics = self.provider_performance[provider]
        
        # Weight factors based on mode
        if config.mode == GenerationMode.FAST:
            time_weight = 0.6
            quality_weight = 0.3
            reliability_weight = 0.1
        else:
            time_weight = 0.2
            quality_weight = 0.6
            reliability_weight = 0.2
        
        # Normalize scores (lower time is better)
        time_score = max(0, 1 - (metrics['avg_time'] / 30.0))
        quality_score = metrics['avg_quality']
        reliability_score = metrics['success_rate']
        
        return (time_score * time_weight + 
                quality_score * quality_weight + 
                reliability_score * reliability_weight)
    
    def _combine_test_codes(self, results: List[GenerationResult]) -> str:
        """Intelligently combine test codes from multiple providers"""
        combined_imports = set()
        combined_fixtures = []
        combined_tests = []
        
        for result in results:
            lines = result.test_code.split('\n')
            
            # Extract imports
            for line in lines:
                if line.strip().startswith(('import ', 'from ')):
                    combined_imports.add(line.strip())
            
            # Extract test functions (simple approach)
            in_test = False
            current_test = []
            
            for line in lines:
                if line.strip().startswith('def test_'):
                    if current_test:
                        combined_tests.append('\n'.join(current_test))
                    current_test = [line]
                    in_test = True
                elif in_test:
                    if line.strip().startswith('def ') and not line.strip().startswith('def test_'):
                        if current_test:
                            combined_tests.append('\n'.join(current_test))
                        current_test = []
                        in_test = False
                    else:
                        current_test.append(line)
            
            if current_test:
                combined_tests.append('\n'.join(current_test))
        
        # Build combined test file
        combined_code = '\n'.join(sorted(combined_imports)) + '\n\n'
        combined_code += '\n\n'.join(combined_fixtures) + '\n\n'
        combined_code += '\n\n'.join(combined_tests)
        
        return combined_code
    
    def _generate_fallback(self, file_path: str, error: str) -> GenerationResult:
        """Generate fallback test when all providers fail"""
        fallback_code = f'''import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestFallback:
    """Fallback tests when AI generation fails"""
    
    def test_module_import(self):
        """Test that the module can be imported"""
        # TODO: Replace with actual module import
        # from module_name import function_name
        assert True, "Module import test - replace with actual import"
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        # TODO: Add specific tests based on module
        assert True, "Basic functionality test - implement based on {Path(file_path).name}"
    
    @pytest.mark.parametrize("input_val,expected", [
        (None, None),
        ("", ""),
        (0, 0),
    ])
    def test_edge_cases(self, input_val, expected):
        """Test edge cases"""
        # TODO: Implement edge case testing
        assert input_val == expected
    
    def test_error_handling(self):
        """Test error handling scenarios"""
        # TODO: Add error handling tests
        with pytest.raises(Exception):
            # Add code that should raise exception
            pass

# Generation failed: {error}
# File: {file_path}
'''
        
        return GenerationResult(
            test_code=fallback_code,
            provider_used=AIProvider.AUTO,
            generation_time=0.1,
            quality_score=0.3,
            coverage_estimate=0.2,
            metadata={'error': error, 'fallback_reason': 'provider_failure'},
            fallback_used=True
        )
    
    def update_provider_performance(self, result: GenerationResult):
        """Update provider performance metrics"""
        provider = result.provider_used
        if provider in self.provider_performance:
            metrics = self.provider_performance[provider]
            
            # Update with exponential moving average
            alpha = 0.1
            metrics['avg_quality'] = (1 - alpha) * metrics['avg_quality'] + alpha * result.quality_score
            metrics['avg_time'] = (1 - alpha) * metrics['avg_time'] + alpha * result.generation_time
            
            if not result.fallback_used:
                metrics['success_rate'] = (1 - alpha) * metrics['success_rate'] + alpha * 1.0
            else:
                metrics['success_rate'] = (1 - alpha) * metrics['success_rate'] + alpha * 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'provider_performance': self.provider_performance,
            'total_generations': len(self.generation_history),
            'average_quality': sum(r.quality_score for r in self.generation_history) / len(self.generation_history) if self.generation_history else 0,
            'providers_available': {
                'claude': bool(self.claude_generator),
                'gemini': bool(self.gemini_generator)
            }
        }
    
    async def batch_generate(self, file_paths: List[str], 
                           config: UniversalTestConfig) -> List[GenerationResult]:
        """Generate tests for multiple files"""
        if config.parallel_generation:
            tasks = [self.generate_tests(fp, config) for fp in file_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if isinstance(r, GenerationResult)]
        else:
            results = []
            for file_path in file_paths:
                result = await self.generate_tests(file_path, config)
                results.append(result)
            return results
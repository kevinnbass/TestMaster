#!/usr/bin/env python3
"""
LLM Analysis Monitor - Enhanced Intelligence Integration

Real-time monitoring of LLM API calls, token usage, and live module analysis 
using the Gemini SDK with comprehensive intelligence tracking.
"""

import sys
import os
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import asyncio
from collections import defaultdict, deque

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google.generativeai not available - install with: pip install google-generativeai")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class LLMProvider(Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class AnalysisType(Enum):
    """Types of LLM analysis."""
    MODULE_ANALYSIS = "module_analysis"
    TEST_GENERATION = "test_generation"
    CODE_REVIEW = "code_review"
    SECURITY_SCAN = "security_scan"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"

@dataclass
class LLMAPICall:
    """Track individual LLM API calls."""
    timestamp: datetime
    provider: LLMProvider
    model: str
    analysis_type: AnalysisType
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_estimate: float
    duration_ms: int
    success: bool
    error_message: Optional[str] = None
    module_path: Optional[str] = None

@dataclass
class LLMMetrics:
    """Aggregate LLM usage metrics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_estimate: float = 0.0
    avg_response_time_ms: float = 0.0
    calls_per_minute: float = 0.0
    active_analyses: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ModuleAnalysis:
    """Live module analysis results."""
    module_path: str
    analysis_timestamp: datetime
    complexity_score: float
    test_coverage_estimate: float
    security_issues: List[str]
    optimization_suggestions: List[str]
    analysis_summary: str
    quality_score: float
    llm_model_used: str

class LLMAnalysisMonitor:
    """
    Enhanced LLM Analysis Monitor
    
    Features:
    - Real-time API call tracking with Gemini SDK
    - Token usage and cost monitoring
    - Live module analysis with intelligence
    - Multi-provider LLM support
    - Performance and quality metrics
    - Integration with TestMaster monitoring
    """
    
    def __init__(self, api_key: Optional[str] = None, demo_mode: bool = False):
        """Initialize LLM Analysis Monitor."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.demo_mode = demo_mode
        
        # Initialize Gemini if available
        self.gemini_available = False
        if GENAI_AVAILABLE and self.api_key and not demo_mode:
            try:
                genai.configure(api_key=self.api_key)
                # Try gemini-2.5-pro first, fallback to 1.5-pro if not available
                try:
                    self.gemini_model = genai.GenerativeModel('models/gemini-2.5-pro')
                    self.model_name = 'gemini-2.5-pro'
                    print("[PASS] Gemini SDK initialized with gemini-2.5-pro")
                except Exception:
                    self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
                    self.model_name = 'gemini-1.5-pro'
                    print("[PASS] Gemini SDK initialized with gemini-1.5-pro (fallback)")
                self.gemini_available = True
            except Exception as e:
                print(f"[FAIL] Gemini SDK initialization failed: {e}")
        
        # Metrics tracking
        self.api_calls: List[LLMAPICall] = []
        self.current_metrics = LLMMetrics()
        self.analysis_results: Dict[str, ModuleAnalysis] = {}
        
        # Demo mode setup
        if demo_mode or not self.gemini_available:
            print("[DEMO] Demo mode enabled - generating simulated LLM data")
            self.demo_mode = True
            self._setup_demo_data()
        
        # Real-time tracking
        self.active_analyses: Dict[str, datetime] = {}
        self.analysis_queue = deque()
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Configuration
        self.max_history = 1000  # Keep last 1000 API calls
        self.analysis_interval = 30 if not demo_mode else 10  # Faster in demo mode
        
        print("LLM Analysis Monitor initialized")
        print(f"Gemini available: {self.gemini_available}")
        print(f"Demo mode: {self.demo_mode}")
        print(f"API calls history limit: {self.max_history}")
    
    def _setup_demo_data(self):
        """Set up demo data for demonstration purposes."""
        import random
        
        # Generate some historical API calls
        demo_modules = [
            "real_time_monitor.py",
            "web_monitor.py", 
            "llm_analysis_monitor.py",
            "intelligent_test_builder.py",
            "orchestrator.py"
        ]
        
        # Generate demo API calls over the last hour
        now = datetime.now()
        for i in range(25):  # 25 demo calls
            call_time = now - timedelta(minutes=random.randint(1, 60))
            
            self.api_calls.append(LLMAPICall(
                timestamp=call_time,
                provider=LLMProvider.GEMINI,
                model="gemini-1.5-pro",
                analysis_type=random.choice(list(AnalysisType)),
                input_tokens=random.randint(800, 2500),
                output_tokens=random.randint(200, 800),
                total_tokens=random.randint(1000, 3300),
                cost_estimate=random.uniform(0.001, 0.008),
                duration_ms=random.randint(1200, 4500),
                success=random.choice([True, True, True, True, False]),  # 80% success rate
                module_path=random.choice(demo_modules)
            ))
        
        # Generate some demo module analyses
        for module in demo_modules[:3]:
            self.analysis_results[module] = ModuleAnalysis(
                module_path=module,
                analysis_timestamp=now - timedelta(minutes=random.randint(5, 30)),
                complexity_score=random.uniform(3.0, 8.5),
                test_coverage_estimate=random.uniform(45.0, 85.0),
                security_issues=[
                    "Input validation needed for user data",
                    "Consider rate limiting for API endpoints"
                ] if random.choice([True, False]) else [],
                optimization_suggestions=[
                    "Consider caching frequently accessed data",
                    "Implement async processing for long operations",
                    "Add connection pooling for database access"
                ][:random.randint(1, 3)],
                analysis_summary=f"Well-structured module with good separation of concerns. Quality score reflects clean code patterns.",
                quality_score=random.uniform(68.0, 92.0),
                llm_model_used="gemini-1.5-pro-demo"
            )
    
    def start_monitoring(self):
        """Start LLM analysis monitoring."""
        if self.monitoring_active:
            print("LLM monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        print("LLM Analysis Monitor started")
    
    def stop_monitoring(self):
        """Stop LLM analysis monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        print("LLM Analysis Monitor stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_analysis_time = 0
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Update metrics
                self._update_metrics()
                
                # DISABLED: Automatic periodic analysis (too costly)
                # if current_time - last_analysis_time >= self.analysis_interval:
                #     self._run_periodic_analysis()
                #     last_analysis_time = current_time
                
                # Process analysis queue (only manual requests)
                self._process_analysis_queue()
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                print(f"LLM monitoring error: {e}")
                time.sleep(5)
    
    def track_api_call(self, provider: LLMProvider, model: str, analysis_type: AnalysisType,
                      input_tokens: int, output_tokens: int, duration_ms: int,
                      success: bool = True, error_message: Optional[str] = None,
                      module_path: Optional[str] = None) -> None:
        """Track an LLM API call."""
        
        # Estimate cost (rough estimates)
        cost_per_1k_input = {
            LLMProvider.GEMINI: 0.00025,  # Gemini 1.5 Pro
            LLMProvider.OPENAI: 0.005,    # GPT-4
            LLMProvider.ANTHROPIC: 0.008  # Claude-3
        }.get(provider, 0.001)
        
        cost_per_1k_output = cost_per_1k_input * 3  # Output typically 3x more expensive
        
        total_tokens = input_tokens + output_tokens
        cost_estimate = (input_tokens / 1000 * cost_per_1k_input + 
                        output_tokens / 1000 * cost_per_1k_output)
        
        api_call = LLMAPICall(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            analysis_type=analysis_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_estimate=cost_estimate,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            module_path=module_path
        )
        
        self.api_calls.append(api_call)
        
        # Keep only recent calls
        if len(self.api_calls) > self.max_history:
            self.api_calls = self.api_calls[-self.max_history:]
        
        print(f"API call tracked: {provider.value} - {analysis_type.value} - "
              f"{total_tokens} tokens - ${cost_estimate:.4f}")
    
    def analyze_module_with_gemini(self, module_path: str) -> Optional[ModuleAnalysis]:
        """Analyze a module using Gemini SDK."""
        if not self.gemini_available:
            print("Gemini not available for module analysis")
            return None
        
        try:
            start_time = time.time()
            
            # Read module content
            with open(module_path, 'r', encoding='utf-8') as f:
                module_content = f.read()
            
            # Create analysis prompt
            prompt = f"""
            Please analyze this Python module and provide a comprehensive assessment:
            
            Module Path: {module_path}
            
            Module Content:
            ```python
            {module_content[:5000]}  # Limit to first 5000 chars
            ```
            
            Provide analysis in JSON format with these fields:
            - complexity_score: Float 0-10 (0=simple, 10=very complex)
            - test_coverage_estimate: Float 0-100 (estimated test coverage percentage)
            - security_issues: Array of security concerns found
            - optimization_suggestions: Array of performance optimization suggestions
            - analysis_summary: String summary of the module's purpose and quality
            - quality_score: Float 0-100 (overall code quality score)
            
            Focus on practical, actionable insights.
            """
            
            # Make API call
            response = self.gemini_model.generate_content(prompt)
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Estimate tokens (rough approximation)
            input_tokens = len(prompt) // 4  # ~4 chars per token
            output_tokens = len(response.text) // 4
            
            # Track the API call
            self.track_api_call(
                provider=LLMProvider.GEMINI,
                model=self.model_name if hasattr(self, 'model_name') else 'gemini-1.5-pro',
                analysis_type=AnalysisType.MODULE_ANALYSIS,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
                success=True,
                module_path=module_path
            )
            
            # Parse response
            try:
                # Try to extract JSON from response
                response_text = response.text
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    json_text = response_text[json_start:json_end].strip()
                else:
                    json_text = response_text
                
                analysis_data = json.loads(json_text)
                
                analysis = ModuleAnalysis(
                    module_path=module_path,
                    analysis_timestamp=datetime.now(),
                    complexity_score=float(analysis_data.get('complexity_score', 5.0)),
                    test_coverage_estimate=float(analysis_data.get('test_coverage_estimate', 50.0)),
                    security_issues=analysis_data.get('security_issues', []),
                    optimization_suggestions=analysis_data.get('optimization_suggestions', []),
                    analysis_summary=analysis_data.get('analysis_summary', 'Analysis completed'),
                    quality_score=float(analysis_data.get('quality_score', 70.0)),
                    llm_model_used=self.model_name if hasattr(self, 'model_name') else 'gemini-1.5-pro'
                )
                
                self.analysis_results[module_path] = analysis
                print(f"[PASS] Module analyzed: {module_path} - Quality: {analysis.quality_score:.1f}/100")
                return analysis
                
            except json.JSONDecodeError:
                # Fallback: create basic analysis from text
                analysis = ModuleAnalysis(
                    module_path=module_path,
                    analysis_timestamp=datetime.now(),
                    complexity_score=5.0,
                    test_coverage_estimate=50.0,
                    security_issues=[],
                    optimization_suggestions=[],
                    analysis_summary=response.text[:200] + "...",
                    quality_score=70.0,
                    llm_model_used=self.model_name if hasattr(self, 'model_name') else 'gemini-1.5-pro'
                )
                
                self.analysis_results[module_path] = analysis
                return analysis
                
        except Exception as e:
            # Track failed API call
            self.track_api_call(
                provider=LLMProvider.GEMINI,
                model="gemini-1.5-pro",
                analysis_type=AnalysisType.MODULE_ANALYSIS,
                input_tokens=0,
                output_tokens=0,
                duration_ms=0,
                success=False,
                error_message=str(e),
                module_path=module_path
            )
            print(f"[FAIL] Module analysis failed: {module_path} - {e}")
            return None
    
    def queue_module_analysis(self, module_path: str):
        """Queue a module for analysis."""
        if module_path not in self.analysis_queue:
            self.analysis_queue.append(module_path)
            print(f"Queued for analysis: {module_path}")
    
    def _process_analysis_queue(self):
        """Process queued module analyses."""
        if not self.analysis_queue:
            return
        
        # In demo mode, simulate analysis without real API calls
        if self.demo_mode:
            self._process_demo_analysis_queue()
            return
        
        # Real mode - only process if Gemini is available
        if not self.gemini_available:
            return
        
        # Limit concurrent analyses
        if len(self.active_analyses) >= 2:  # Max 2 concurrent analyses
            return
        
        module_path = self.analysis_queue.popleft()
        
        # Check if already analyzed recently
        if module_path in self.analysis_results:
            last_analysis = self.analysis_results[module_path].analysis_timestamp
            if (datetime.now() - last_analysis).total_seconds() < 300:  # 5 minutes
                return
        
        # Start analysis in background
        self.active_analyses[module_path] = datetime.now()
        
        def analyze_worker():
            try:
                self.analyze_module_with_gemini(module_path)
            finally:
                self.active_analyses.pop(module_path, None)
        
        thread = threading.Thread(target=analyze_worker, daemon=True)
        thread.start()
    
    def _process_demo_analysis_queue(self):
        """Process analysis queue in demo mode."""
        import random
        
        if not self.analysis_queue or len(self.active_analyses) >= 2:
            return
        
        module_path = self.analysis_queue.popleft()
        self.active_analyses[module_path] = datetime.now()
        
        def demo_analyze_worker():
            try:
                # Simulate analysis time
                time.sleep(random.uniform(2, 5))
                
                # Generate demo API call
                self.track_api_call(
                    provider=LLMProvider.GEMINI,
                    model="gemini-1.5-pro-demo",
                    analysis_type=AnalysisType.MODULE_ANALYSIS,
                    input_tokens=random.randint(1200, 2800),
                    output_tokens=random.randint(300, 900),
                    duration_ms=random.randint(2000, 5000),
                    success=True,
                    module_path=module_path
                )
                
                # Generate demo analysis result
                self.analysis_results[module_path] = ModuleAnalysis(
                    module_path=module_path,
                    analysis_timestamp=datetime.now(),
                    complexity_score=random.uniform(3.0, 8.5),
                    test_coverage_estimate=random.uniform(45.0, 85.0),
                    security_issues=[
                        "Consider input validation",
                        "Add error handling for edge cases"
                    ] if random.choice([True, False]) else [],
                    optimization_suggestions=[
                        "Cache computed results",
                        "Use async operations where possible",
                        "Consider connection pooling"
                    ][:random.randint(1, 3)],
                    analysis_summary=f"Analysis of {module_path} completed. Code structure appears well-organized with room for optimization.",
                    quality_score=random.uniform(65.0, 90.0),
                    llm_model_used="gemini-1.5-pro-demo"
                )
                
                print(f"[DEMO] Demo analysis completed: {module_path}")
                
            finally:
                self.active_analyses.pop(module_path, None)
        
        thread = threading.Thread(target=demo_analyze_worker, daemon=True)
        thread.start()
    
    def _run_periodic_analysis(self):
        """Run periodic analysis of project modules."""
        # Find Python modules in the project
        project_root = Path.cwd()
        python_files = list(project_root.glob("**/*.py"))
        
        # Filter to important modules (not tests, not __pycache__)
        important_modules = [
            str(f) for f in python_files
            if not any(part.startswith('.') or part == '__pycache__' or 'test' in part.lower()
                      for part in f.parts)
        ]
        
        # Queue a few modules for analysis
        for module_path in important_modules[:3]:  # Analyze 3 modules per interval
            self.queue_module_analysis(module_path)
    
    def _update_metrics(self):
        """Update aggregate metrics."""
        recent_calls = [call for call in self.api_calls 
                       if (datetime.now() - call.timestamp).total_seconds() < 3600]  # Last hour
        
        # Always update metrics even if no recent calls
        self.current_metrics.total_calls = len(self.api_calls)
        self.current_metrics.successful_calls = len([c for c in self.api_calls if c.success])
        self.current_metrics.failed_calls = len([c for c in self.api_calls if not c.success])
        self.current_metrics.total_input_tokens = sum(c.input_tokens for c in self.api_calls)
        self.current_metrics.total_output_tokens = sum(c.output_tokens for c in self.api_calls)
        self.current_metrics.total_cost_estimate = sum(c.cost_estimate for c in self.api_calls)
        
        if recent_calls:
            self.current_metrics.avg_response_time_ms = sum(c.duration_ms for c in recent_calls) / len(recent_calls)
            
            # Calls per minute (last 5 minutes)
            recent_5min = [c for c in recent_calls 
                          if (datetime.now() - c.timestamp).total_seconds() < 300]
            self.current_metrics.calls_per_minute = len(recent_5min) / 5.0
        
        self.current_metrics.active_analyses = len(self.active_analyses)
        self.current_metrics.timestamp = datetime.now()
    
    def get_llm_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive LLM metrics summary."""
        return {
            'timestamp': self.current_metrics.timestamp.isoformat(),
            'api_calls': {
                'total_calls': self.current_metrics.total_calls,
                'successful_calls': self.current_metrics.successful_calls,
                'failed_calls': self.current_metrics.failed_calls,
                'success_rate': (self.current_metrics.successful_calls / max(1, self.current_metrics.total_calls)) * 100,
                'calls_per_minute': self.current_metrics.calls_per_minute,
                'avg_response_time_ms': self.current_metrics.avg_response_time_ms
            },
            'token_usage': {
                'total_input_tokens': self.current_metrics.total_input_tokens,
                'total_output_tokens': self.current_metrics.total_output_tokens,
                'total_tokens': self.current_metrics.total_input_tokens + self.current_metrics.total_output_tokens,
                'input_output_ratio': (self.current_metrics.total_output_tokens / 
                                     max(1, self.current_metrics.total_input_tokens))
            },
            'cost_tracking': {
                'total_cost_estimate': self.current_metrics.total_cost_estimate,
                'avg_cost_per_call': (self.current_metrics.total_cost_estimate / 
                                     max(1, self.current_metrics.total_calls)),
                'hourly_cost_estimate': self.current_metrics.calls_per_minute * 60 * 
                                       (self.current_metrics.total_cost_estimate / max(1, self.current_metrics.total_calls))
            },
            'analysis_status': {
                'active_analyses': self.current_metrics.active_analyses,
                'completed_analyses': len(self.analysis_results),
                'queue_size': len(self.analysis_queue),
                'gemini_available': self.gemini_available
            },
            'recent_analyses': [
                {
                    'module_path': analysis.module_path,
                    'quality_score': analysis.quality_score,
                    'complexity_score': analysis.complexity_score,
                    'test_coverage_estimate': analysis.test_coverage_estimate,
                    'timestamp': analysis.analysis_timestamp.isoformat()
                }
                for analysis in sorted(
                    self.analysis_results.values(),
                    key=lambda x: x.analysis_timestamp,
                    reverse=True
                )[:5]  # Last 5 analyses
            ]
        }
    
    def get_module_analysis(self, module_path: str) -> Optional[Dict[str, Any]]:
        """Get analysis results for a specific module."""
        if module_path in self.analysis_results:
            analysis = self.analysis_results[module_path]
            return {
                'module_path': analysis.module_path,
                'analysis_timestamp': analysis.analysis_timestamp.isoformat(),
                'complexity_score': analysis.complexity_score,
                'test_coverage_estimate': analysis.test_coverage_estimate,
                'security_issues': analysis.security_issues,
                'optimization_suggestions': analysis.optimization_suggestions,
                'analysis_summary': analysis.analysis_summary,
                'quality_score': analysis.quality_score,
                'llm_model_used': analysis.llm_model_used
            }
        return None

# Global LLM monitor instance
_llm_monitor = None

def get_llm_monitor(demo_mode: bool = None) -> LLMAnalysisMonitor:
    """Get global LLM monitor instance."""
    global _llm_monitor
    if _llm_monitor is None:
        # Enable demo mode by default if no API key is available
        if demo_mode is None:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            demo_mode = not api_key
        _llm_monitor = LLMAnalysisMonitor(demo_mode=demo_mode)
    return _llm_monitor

def start_llm_monitoring():
    """Start LLM monitoring."""
    monitor = get_llm_monitor()
    monitor.start_monitoring()
    return monitor

def main():
    """Test LLM Analysis Monitor."""
    print("Testing LLM Analysis Monitor...")
    
    monitor = LLMAnalysisMonitor()
    
    # Test module analysis if Gemini is available
    if monitor.gemini_available:
        test_module = __file__  # Analyze this file
        print(f"Analyzing module: {test_module}")
        
        analysis = monitor.analyze_module_with_gemini(test_module)
        if analysis:
            print(f"Analysis completed - Quality Score: {analysis.quality_score}")
            print(f"Complexity Score: {analysis.complexity_score}")
            print(f"Summary: {analysis.analysis_summary[:100]}...")
    
    # Show metrics
    metrics = monitor.get_llm_metrics_summary()
    print("\nLLM Metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
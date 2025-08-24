#!/usr/bin/env python3
"""
AI Analysis Wrapper - Mandatory Cost Control
============================================

MANDATORY wrapper for ALL AI-powered analysis tools.
Ensures cost tracking and budget compliance before any LLM calls.

CRITICAL: All AI analysis must go through this wrapper!
"""

import logging
import time
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime
import warnings

# Import the API tracker
from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.monitoring.api_usage_tracker import (
    track_api_call, pre_check_cost, get_usage_stats,
    APICallType, CostWarningLevel
)


class AIAnalysisError(Exception):
    """Exception raised when AI analysis fails budget checks"""
    pass


class CostBudgetExceededError(AIAnalysisError):
    """Exception raised when API call would exceed budget"""
    pass


def require_cost_approval(func):
    """
    Decorator that requires cost approval for AI analysis functions
    MANDATORY for all functions that make LLM API calls
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract analysis parameters
        component = kwargs.get('component', func.__module__)
        purpose = kwargs.get('purpose', func.__name__)
        model = kwargs.get('model', 'gpt-3.5-turbo')
        
        # Estimate token usage
        estimated_input = kwargs.get('estimated_input_tokens', 1000)
        estimated_output = kwargs.get('estimated_output_tokens', 500)
        
        # Pre-check cost
        allowed, message, estimated_cost = pre_check_cost(
            model, estimated_input, estimated_output
        )
        
        if not allowed:
            error_msg = f"AI Analysis BLOCKED: {message} (Estimated: ${estimated_cost:.4f})"
            logging.error(error_msg)
            raise CostBudgetExceededError(error_msg)
        
        # Log the analysis attempt
        logging.warning(f"AI Analysis Starting: {purpose} using {model} (Est: ${estimated_cost:.4f})")
        
        # Execute the function with tracking
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            
            # Track successful call
            actual_input = getattr(result, 'input_tokens', estimated_input)
            actual_output = getattr(result, 'output_tokens', estimated_output)
            
            track_api_call(
                model=model,
                call_type=APICallType.ANALYSIS,
                purpose=purpose,
                component=component,
                input_tokens=actual_input,
                output_tokens=actual_output,
                response_time=time.time() - start_time,
                success=True
            )
            
            logging.info(f"AI Analysis Complete: {purpose} (${estimated_cost:.4f})")
            return result
            
        except Exception as e:
            # Track failed call
            track_api_call(
                model=model,
                call_type=APICallType.ANALYSIS,
                purpose=purpose,
                component=component,
                input_tokens=estimated_input,
                output_tokens=0,
                response_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            
            logging.error(f"AI Analysis Failed: {purpose} - {e}")
            raise
    
    return wrapper


class AIAnalysisWrapper:
    """
    Wrapper class for AI analysis tools with mandatory cost control
    """
    
    def __init__(self, component_name: str, default_model: str = "gemini-2.5-pro"):
        self.component_name = component_name
        self.default_model = default_model
        self.logger = logging.getLogger(f"{__name__}.{component_name}")
        
        # Register component for tracking
        self.logger.info(f"AI Analysis Wrapper initialized for {component_name}")
    
    def analyze_with_budget_check(self,
                                 analysis_function: Callable,
                                 purpose: str,
                                 estimated_input_tokens: int,
                                 estimated_output_tokens: int,
                                 model: Optional[str] = None,
                                 **kwargs) -> Any:
        """
        Execute AI analysis with mandatory budget checking
        
        Args:
            analysis_function: The actual AI analysis function to call
            purpose: Description of what the analysis is for
            estimated_input_tokens: Estimated input token count
            estimated_output_tokens: Estimated output token count
            model: Model to use (defaults to component default)
            **kwargs: Additional arguments for the analysis function
        
        Returns:
            Result of the analysis function
            
        Raises:
            CostBudgetExceededError: If the analysis would exceed budget
        """
        model = model or self.default_model
        
        # Pre-check budget
        allowed, message, estimated_cost = pre_check_cost(
            model, estimated_input_tokens, estimated_output_tokens
        )
        
        if not allowed:
            error_msg = f"AI Analysis BLOCKED for {self.component_name}: {message}"
            self.logger.error(error_msg)
            raise CostBudgetExceededError(error_msg)
        
        # Warn if expensive
        if estimated_cost > 0.1:
            self.logger.warning(f"EXPENSIVE AI Analysis: {purpose} - ${estimated_cost:.4f}")
        
        # Execute with tracking
        start_time = time.time()
        try:
            self.logger.info(f"Starting AI Analysis: {purpose} using {model}")
            
            result = analysis_function(**kwargs)
            
            # Track successful call
            track_api_call(
                model=model,
                call_type=APICallType.ANALYSIS,
                purpose=purpose,
                component=self.component_name,
                input_tokens=estimated_input_tokens,
                output_tokens=estimated_output_tokens,
                response_time=time.time() - start_time,
                success=True
            )
            
            self.logger.info(f"AI Analysis Complete: {purpose}")
            return result
            
        except Exception as e:
            # Track failed call
            track_api_call(
                model=model,
                call_type=APICallType.ANALYSIS,
                purpose=purpose,
                component=self.component_name,
                input_tokens=estimated_input_tokens,
                output_tokens=0,
                response_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
            
            self.logger.error(f"AI Analysis Failed: {purpose} - {e}")
            raise
    
    def batch_analyze_with_budget(self,
                                 analysis_items: List[Dict[str, Any]],
                                 batch_size: int = 5) -> List[Any]:
        """
        Execute batch AI analysis with budget management
        
        Args:
            analysis_items: List of analysis configurations
            batch_size: Number of items to process in each batch
            
        Returns:
            List of analysis results
        """
        results = []
        total_estimated_cost = 0.0
        
        # Calculate total estimated cost
        for item in analysis_items:
            _, _, cost = pre_check_cost(
                item.get('model', self.default_model),
                item.get('estimated_input_tokens', 1000),
                item.get('estimated_output_tokens', 500)
            )
            total_estimated_cost += cost
        
        # Check total budget
        if total_estimated_cost > 5.0:  # $5 batch limit
            raise CostBudgetExceededError(
                f"Batch analysis would cost ${total_estimated_cost:.2f} - exceeds $5 batch limit"
            )
        
        self.logger.warning(f"Batch Analysis: {len(analysis_items)} items, ${total_estimated_cost:.2f}")
        
        # Process in batches
        for i in range(0, len(analysis_items), batch_size):
            batch = analysis_items[i:i + batch_size]
            
            for item in batch:
                try:
                    result = self.analyze_with_budget_check(**item)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch item failed: {e}")
                    results.append(None)
            
            # Brief pause between batches to avoid rate limits
            time.sleep(1)
        
        return results
    
    def get_cost_estimate(self,
                         analysis_type: str,
                         input_size: int,
                         model: Optional[str] = None) -> Dict[str, float]:
        """
        Get cost estimate for an analysis without executing it
        
        Args:
            analysis_type: Type of analysis to estimate
            input_size: Size of input (characters, lines, etc.)
            model: Model to use for estimation
            
        Returns:
            Dictionary with cost estimates
        """
        model = model or self.default_model
        
        # Estimate tokens based on input size and analysis type
        token_multipliers = {
            'code_analysis': {'input': 4, 'output': 2},  # 4 chars per input token, 2:1 output ratio
            'architecture_validation': {'input': 3, 'output': 3},
            'pattern_detection': {'input': 5, 'output': 1.5},
            'documentation_generation': {'input': 4, 'output': 4},
            'test_generation': {'input': 3, 'output': 5},
            'refactoring_suggestions': {'input': 4, 'output': 3}
        }
        
        multiplier = token_multipliers.get(analysis_type, {'input': 4, 'output': 2})
        
        estimated_input_tokens = int(input_size / multiplier['input'])
        estimated_output_tokens = int(estimated_input_tokens * multiplier['output'])
        
        _, _, cost = pre_check_cost(model, estimated_input_tokens, estimated_output_tokens)
        
        return {
            'estimated_input_tokens': estimated_input_tokens,
            'estimated_output_tokens': estimated_output_tokens,
            'estimated_cost': cost,
            'model': model,
            'analysis_type': analysis_type
        }


def create_ai_wrapper(component_name: str, default_model: str = "gemini-2.5-pro") -> AIAnalysisWrapper:
    """Create an AI analysis wrapper for a component"""
    return AIAnalysisWrapper(component_name, default_model)


def mandatory_cost_check(purpose: str, 
                        estimated_input_tokens: int,
                        estimated_output_tokens: int,
                        model: str = "gemini-2.5-pro") -> Tuple[bool, str, float]:
    """
    MANDATORY cost check before any AI analysis
    Must be called by ALL AI-powered tools
    """
    allowed, message, cost = pre_check_cost(model, estimated_input_tokens, estimated_output_tokens)
    
    if not allowed:
        warnings.warn(f"AI Analysis BLOCKED: {purpose} - {message}", UserWarning)
        logging.error(f"COST BLOCK: {purpose} - {message}")
    else:
        logging.info(f"COST CHECK PASSED: {purpose} - ${cost:.4f}")
    
    return allowed, message, cost


def get_ai_usage_summary() -> Dict[str, Any]:
    """Get summary of AI usage for monitoring"""
    stats = get_usage_stats()
    
    return {
        'total_ai_cost': stats['total_cost'],
        'ai_calls_today': stats['total_calls'],
        'success_rate': (stats['successful_calls'] / max(1, stats['total_calls'])) * 100,
        'most_used_model': max(stats.get('calls_by_model', {}).items(), key=lambda x: x[1])[0] if stats.get('calls_by_model') else 'none',
        'budget_status': stats['budget_status'],
        'recommendations': _get_ai_optimization_recommendations(stats)
    }


def _get_ai_optimization_recommendations(stats: Dict[str, Any]) -> List[str]:
    """Generate AI usage optimization recommendations"""
    recommendations = []
    
    # Check for expensive models
    if 'gpt-4' in stats.get('cost_by_model', {}) and stats['cost_by_model']['gpt-4'] > 1.0:
        recommendations.append("Consider using GPT-3.5-Turbo for simple analysis tasks")
    
    # Check call frequency
    if stats['total_calls'] > 100:
        recommendations.append("High API usage detected - consider implementing caching")
    
    # Check success rate
    success_rate = (stats['successful_calls'] / max(1, stats['total_calls'])) * 100
    if success_rate < 90:
        recommendations.append("Low success rate - review error handling and input validation")
    
    return recommendations


# Emergency budget controls
def emergency_stop_all_ai():
    """Emergency function to stop all AI analysis"""
    from .api_usage_tracker import get_api_tracker
    tracker = get_api_tracker()
    tracker.set_budget(daily_limit=0.01, hourly_limit=0.01, auto_stop=True)
    
    logging.critical("EMERGENCY STOP: All AI analysis stopped!")
    warnings.warn("EMERGENCY: All AI analysis has been stopped due to budget controls", UserWarning)


def reset_emergency_stop():
    """Reset emergency stop and restore normal budgets"""
    from .api_usage_tracker import get_api_tracker
    tracker = get_api_tracker()
    tracker.set_budget(daily_limit=10.0, hourly_limit=2.0, auto_stop=True)
    
    logging.info("Emergency stop reset - normal AI analysis resumed")


if __name__ == "__main__":
    # Test the AI wrapper
    print("AI Analysis Wrapper Test")
    print("=" * 60)
    
    # Create wrapper
    wrapper = create_ai_wrapper("test_component")
    
    # Test cost estimate
    estimate = wrapper.get_cost_estimate("code_analysis", 5000)
    print("Cost Estimate:")
    print(f"  Tokens: {estimate['estimated_input_tokens']} input, {estimate['estimated_output_tokens']} output")
    print(f"  Cost: ${estimate['estimated_cost']:.4f}")
    
    # Test mandatory cost check
    allowed, message, cost = mandatory_cost_check("test analysis", 500, 200)
    print(f"\nCost Check: {'ALLOWED' if allowed else 'BLOCKED'}")
    print(f"Message: {message}")
    print(f"Cost: ${cost:.4f}")
    
    # Get usage summary
    summary = get_ai_usage_summary()
    print("\nAI Usage Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("CRITICAL: All AI analysis tools MUST use this wrapper!")
    print("=" * 60)
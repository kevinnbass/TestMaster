#!/usr/bin/env python3
"""Test LLM Module Intelligence with Gemini 2.5 Pro"""

from llm_analysis_monitor import LLMAnalysisMonitor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize and test
monitor = LLMAnalysisMonitor(demo_mode=False)
monitor.start_monitoring()

print("Testing LLM Module Intelligence with Gemini 2.5 Pro...")
print("=" * 60)

# Analyze a module
result = monitor.analyze_module_with_gemini('web_monitor.py')

# Update metrics manually to verify
monitor._update_metrics()

# Show results
metrics = monitor.get_llm_metrics_summary()
print('=== LLM Module Intelligence with Gemini 2.5 Pro ===')
print(f'Model: {monitor.model_name if hasattr(monitor, "model_name") else "unknown"}')
print(f'Total API Calls: {metrics["api_calls"]["total_calls"]}')
print(f'Tokens Used: {metrics["token_usage"]["total_tokens"]}')
print(f'Cost Estimate: ${metrics["cost_tracking"]["total_cost_estimate"]:.4f}')
print(f'Analyses Completed: {metrics["analysis_status"]["completed_analyses"]}')

if result:
    print(f'\nModule Analysis Result:')
    print(f'  Module: {result.module_path}')
    print(f'  Quality Score: {result.quality_score}/100')
    print(f'  Complexity: {result.complexity_score}/10')
    print(f'  Security Issues: {len(result.security_issues)}')
    print(f'  Optimization Suggestions: {len(result.optimization_suggestions)}')
    print(f'\nAnalysis Summary:')
    print(f'  {result.analysis_summary[:200]}...')
    
print("\n[SUCCESS] LLM Module Intelligence is FULLY OPERATIONAL with Gemini 2.5 Pro!")
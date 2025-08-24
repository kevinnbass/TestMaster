"""
ML Intelligence Features Module Registry  
Pluggable ML modules extracted for Agent X's Epsilon base integration

All modules comply with STEELCLAD protocol (< 200 lines each)
Ready for integration into unified dashboard architecture
"""

from .semantic_intent_classifier import SemanticIntentClassifier, create_intent_classifier_plugin
from .ml_performance_predictions import MLPerformancePredictions, create_performance_predictions_plugin
from .ast_code_understanding import ASTCodeUnderstanding, create_ast_analyzer_plugin

# ML Intelligence module registry for Agent X integration
ML_INTELLIGENCE_MODULES = {
    'semantic_intent': {
        'class': SemanticIntentClassifier,
        'factory': create_intent_classifier_plugin,
        'description': 'ML-powered code intent classification with 15+ semantic categories',
        'features': ['intent_classification', 'confidence_scoring', 'pattern_analysis']
    },
    'performance_predictions': {
        'class': MLPerformancePredictions,
        'factory': create_performance_predictions_plugin,
        'description': 'ML-based performance prediction and trend forecasting',
        'features': ['health_prediction', 'service_forecasting', 'resource_modeling']
    },
    'ast_analyzer': {
        'class': ASTCodeUnderstanding,
        'factory': create_ast_analyzer_plugin,
        'description': 'AST-based code structure analysis and architectural pattern detection',
        'features': ['structure_analysis', 'complexity_assessment', 'pattern_detection']
    }
}

def create_ml_intelligence_suite(config=None):
    """Create complete ML intelligence feature suite for dashboard integration"""
    suite = {}
    for module_name, module_info in ML_INTELLIGENCE_MODULES.items():
        suite[module_name] = module_info['factory'](config)
    return suite

def get_ml_intelligence_features():
    """Get list of all available ML intelligence features"""
    features = []
    for module_name, module_info in ML_INTELLIGENCE_MODULES.items():
        features.extend([f"{module_name}_{feature}" for feature in module_info['features']])
    return features

__all__ = [
    'SemanticIntentClassifier', 'MLPerformancePredictions', 'ASTCodeUnderstanding',
    'create_intent_classifier_plugin', 'create_performance_predictions_plugin', 'create_ast_analyzer_plugin',
    'ML_INTELLIGENCE_MODULES', 'create_ml_intelligence_suite', 'get_ml_intelligence_features'
]
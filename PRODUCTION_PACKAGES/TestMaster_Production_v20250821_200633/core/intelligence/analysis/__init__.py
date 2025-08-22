"""
TestMaster Intelligence Analysis Module
=======================================

Comprehensive analysis capabilities for business rules, technical debt,
machine learning, semantic analysis, and more.

This module provides standalone analyzers that integrate seamlessly
with the intelligence hub architecture.
"""

from typing import Dict, Any, Optional, List
import logging

# Import existing analyzers
try:
    from .ml_analyzer import MLCodeAnalyzer
    ML_ANALYZER_AVAILABLE = True
except ImportError:
    ML_ANALYZER_AVAILABLE = False

try:
    from .semantic_analyzer import SemanticAnalyzer
    SEMANTIC_ANALYZER_AVAILABLE = True
except ImportError:
    SEMANTIC_ANALYZER_AVAILABLE = False

# Import new analyzers
try:
    from .business_analyzer import BusinessAnalyzer
    BUSINESS_ANALYZER_AVAILABLE = True
except ImportError:
    BUSINESS_ANALYZER_AVAILABLE = False

try:
    from .debt_analyzer import TechnicalDebtAnalyzer
    DEBT_ANALYZER_AVAILABLE = True
except ImportError:
    DEBT_ANALYZER_AVAILABLE = False

# Import additional analysis components for consolidation
try:
    from .business_constraint_analyzer import BusinessConstraintAnalyzer
    CONSTRAINT_ANALYZER_AVAILABLE = True
except ImportError:
    CONSTRAINT_ANALYZER_AVAILABLE = False

try:
    from .business_rule_extractor import BusinessRuleExtractor
    RULE_EXTRACTOR_AVAILABLE = True
except ImportError:
    RULE_EXTRACTOR_AVAILABLE = False

try:
    from .business_workflow_analyzer import BusinessWorkflowAnalyzer  
    WORKFLOW_ANALYZER_AVAILABLE = True
except ImportError:
    WORKFLOW_ANALYZER_AVAILABLE = False

try:
    from .debt_code_analyzer import DebtCodeAnalyzer
    CODE_DEBT_ANALYZER_AVAILABLE = True
except ImportError:
    CODE_DEBT_ANALYZER_AVAILABLE = False

try:
    from .debt_quantifier import DebtQuantifier
    DEBT_QUANTIFIER_AVAILABLE = True
except ImportError:
    DEBT_QUANTIFIER_AVAILABLE = False

try:
    from .debt_test_analyzer import DebtTestAnalyzer
    TEST_DEBT_ANALYZER_AVAILABLE = True
except ImportError:
    TEST_DEBT_ANALYZER_AVAILABLE = False

try:
    from .semantic_intent_analyzer import SemanticIntentAnalyzer
    INTENT_ANALYZER_AVAILABLE = True
except ImportError:
    INTENT_ANALYZER_AVAILABLE = False

try:
    from .semantic_pattern_detector import SemanticPatternDetector
    PATTERN_DETECTOR_AVAILABLE = True
except ImportError:
    PATTERN_DETECTOR_AVAILABLE = False

try:
    from .semantic_relationship_analyzer import SemanticRelationshipAnalyzer
    RELATIONSHIP_ANALYZER_AVAILABLE = True
except ImportError:
    RELATIONSHIP_ANALYZER_AVAILABLE = False

try:
    from .ml_code_analyzer import MLCodeAnalyzer as EnhancedMLAnalyzer
    ENHANCED_ML_ANALYZER_AVAILABLE = True
except ImportError:
    ENHANCED_ML_ANALYZER_AVAILABLE = False

try:
    from .technical_debt_analyzer import TechnicalDebtAnalyzer as EnhancedDebtAnalyzer
    ENHANCED_DEBT_ANALYZER_AVAILABLE = True
except ImportError:
    ENHANCED_DEBT_ANALYZER_AVAILABLE = False


class AnalysisHub:
    """
    Central hub for all analysis capabilities.
    
    Provides unified access to business rule analysis, technical debt analysis,
    ML analysis, semantic analysis, and other intelligence capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize available analyzers
        self._analyzers = {}
        self._initialize_analyzers()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the analysis hub."""
        logger = logging.getLogger("analysis_hub")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_analyzers(self):
        """Initialize all available analyzers."""
        # Core analyzers
        if BUSINESS_ANALYZER_AVAILABLE:
            try:
                self._analyzers['business'] = BusinessAnalyzer(self.config)
                self.logger.info("Business analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize business analyzer: {e}")
        
        if DEBT_ANALYZER_AVAILABLE:
            try:
                self._analyzers['debt'] = TechnicalDebtAnalyzer(self.config)
                self.logger.info("Technical debt analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize debt analyzer: {e}")
        
        if ML_ANALYZER_AVAILABLE:
            try:
                self._analyzers['ml'] = MLCodeAnalyzer(self.config)
                self.logger.info("ML analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize ML analyzer: {e}")
        
        if SEMANTIC_ANALYZER_AVAILABLE:
            try:
                self._analyzers['semantic'] = SemanticAnalyzer(self.config)
                self.logger.info("Semantic analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize semantic analyzer: {e}")
        
        # Additional business analyzers
        if CONSTRAINT_ANALYZER_AVAILABLE:
            try:
                from .business_base import BusinessAnalysisConfiguration
                business_config = BusinessAnalysisConfiguration()
                self._analyzers['constraint'] = BusinessConstraintAnalyzer(business_config)
                self.logger.info("Business constraint analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize constraint analyzer: {e}")
        
        if RULE_EXTRACTOR_AVAILABLE:
            try:
                self._analyzers['rule_extractor'] = BusinessRuleExtractor(self.config)
                self.logger.info("Business rule extractor initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize rule extractor: {e}")
        
        if WORKFLOW_ANALYZER_AVAILABLE:
            try:
                self._analyzers['workflow'] = BusinessWorkflowAnalyzer(self.config)
                self.logger.info("Business workflow analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize workflow analyzer: {e}")
        
        # Additional debt analyzers
        if CODE_DEBT_ANALYZER_AVAILABLE:
            try:
                self._analyzers['code_debt'] = DebtCodeAnalyzer(self.config)
                self.logger.info("Code debt analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize code debt analyzer: {e}")
        
        if DEBT_QUANTIFIER_AVAILABLE:
            try:
                self._analyzers['debt_quantifier'] = DebtQuantifier(self.config)
                self.logger.info("Debt quantifier initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize debt quantifier: {e}")
        
        if TEST_DEBT_ANALYZER_AVAILABLE:
            try:
                self._analyzers['test_debt'] = DebtTestAnalyzer(self.config)
                self.logger.info("Test debt analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize test debt analyzer: {e}")
        
        # Additional semantic analyzers
        if INTENT_ANALYZER_AVAILABLE:
            try:
                self._analyzers['intent'] = SemanticIntentAnalyzer(self.config)
                self.logger.info("Semantic intent analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize intent analyzer: {e}")
        
        if PATTERN_DETECTOR_AVAILABLE:
            try:
                self._analyzers['pattern'] = SemanticPatternDetector(self.config)
                self.logger.info("Semantic pattern detector initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize pattern detector: {e}")
        
        if RELATIONSHIP_ANALYZER_AVAILABLE:
            try:
                self._analyzers['relationship'] = SemanticRelationshipAnalyzer(self.config)
                self.logger.info("Semantic relationship analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize relationship analyzer: {e}")
        
        # Enhanced analyzers
        if ENHANCED_ML_ANALYZER_AVAILABLE:
            try:
                self._analyzers['enhanced_ml'] = EnhancedMLAnalyzer(self.config)
                self.logger.info("Enhanced ML analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize enhanced ML analyzer: {e}")
        
        if ENHANCED_DEBT_ANALYZER_AVAILABLE:
            try:
                self._analyzers['enhanced_debt'] = EnhancedDebtAnalyzer(self.config)
                self.logger.info("Enhanced debt analyzer initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize enhanced debt analyzer: {e}")
        
        self.logger.info(f"Initialized {len(self._analyzers)} analyzers successfully")
    
    def get_available_analyzers(self) -> List[str]:
        """Get list of available analyzers."""
        return list(self._analyzers.keys())
    
    def analyze_business_rules(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze business rules and domain logic.
        
        Args:
            root_path: Optional path to analyze. If None, uses current directory.
            
        Returns:
            Comprehensive business rule analysis results
        """
        if 'business' not in self._analyzers:
            return {
                "error": "Business analyzer not available",
                "available_analyzers": self.get_available_analyzers()
            }
        
        try:
            return self._analyzers['business'].analyze(root_path)
        except Exception as e:
            self.logger.error(f"Business rule analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_technical_debt(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze technical debt and provide remediation strategies.
        
        Args:
            root_path: Optional path to analyze. If None, uses current directory.
            
        Returns:
            Comprehensive technical debt analysis results
        """
        if 'debt' not in self._analyzers:
            return {
                "error": "Technical debt analyzer not available",
                "available_analyzers": self.get_available_analyzers()
            }
        
        try:
            return self._analyzers['debt'].analyze(root_path)
        except Exception as e:
            self.logger.error(f"Technical debt analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_ml_code(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze ML/AI code for issues and best practices.
        
        Args:
            root_path: Optional path to analyze. If None, uses current directory.
            
        Returns:
            Comprehensive ML code analysis results
        """
        if 'ml' not in self._analyzers:
            return {
                "error": "ML analyzer not available",
                "available_analyzers": self.get_available_analyzers()
            }
        
        try:
            return self._analyzers['ml'].analyze(root_path)
        except Exception as e:
            self.logger.error(f"ML code analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_semantics(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze code semantics and developer intent.
        
        Args:
            root_path: Optional path to analyze. If None, uses current directory.
            
        Returns:
            Comprehensive semantic analysis results
        """
        if 'semantic' not in self._analyzers:
            return {
                "error": "Semantic analyzer not available",
                "available_analyzers": self.get_available_analyzers()
            }
        
        try:
            return self._analyzers['semantic'].analyze(root_path)
        except Exception as e:
            self.logger.error(f"Semantic analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_ml_components(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze machine learning components and patterns.
        
        Args:
            root_path: Optional path to analyze. If None, uses current directory.
            
        Returns:
            ML component analysis results
        """
        if 'ml' not in self._analyzers:
            return {
                "error": "ML analyzer not available",
                "available_analyzers": self.get_available_analyzers()
            }
        
        try:
            return self._analyzers['ml'].analyze(root_path)
        except Exception as e:
            self.logger.error(f"ML analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_semantic_structure(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze semantic structure and relationships.
        
        Args:
            root_path: Optional path to analyze. If None, uses current directory.
            
        Returns:
            Semantic analysis results
        """
        if 'semantic' not in self._analyzers:
            return {
                "error": "Semantic analyzer not available",
                "available_analyzers": self.get_available_analyzers()
            }
        
        try:
            return self._analyzers['semantic'].analyze(root_path)
        except Exception as e:
            self.logger.error(f"Semantic analysis failed: {e}")
            return {"error": str(e)}
    
    def extract_business_constraints(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract business constraints and limits from code.
        Consolidates functionality from business_constraint_analyzer.py
        
        Returns:
            Dict containing numeric, temporal, capacity, relationship constraints
        """
        if 'constraint' not in self._analyzers:
            return {"error": "Business constraint analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['constraint'].extract_business_constraints(python_files)
        except Exception as e:
            self.logger.error(f"Business constraint extraction failed: {e}")
            return {"error": str(e)}
    
    def extract_compliance_rules(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract compliance and regulatory rules from code.
        Consolidates functionality from business_constraint_analyzer.py
        
        Returns:
            Dict containing regulatory, audit, data privacy, retention rules
        """
        if 'constraint' not in self._analyzers:
            return {"error": "Business constraint analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['constraint'].extract_compliance_rules(python_files)
        except Exception as e:
            self.logger.error(f"Compliance rules extraction failed: {e}")
            return {"error": str(e)}
    
    def extract_sla_rules(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract Service Level Agreement rules from code.
        Consolidates functionality from business_constraint_analyzer.py
        
        Returns:
            Dict containing response time, availability, throughput, quality rules
        """
        if 'constraint' not in self._analyzers:
            return {"error": "Business constraint analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['constraint'].extract_sla_rules(python_files)
        except Exception as e:
            self.logger.error(f"SLA rules extraction failed: {e}")
            return {"error": str(e)}
    
    def extract_pricing_rules(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract pricing and billing rules from code.
        Consolidates functionality from business_constraint_analyzer.py
        
        Returns:
            Dict containing pricing models, discounts, tiers, billing cycles
        """
        if 'constraint' not in self._analyzers:
            return {"error": "Business constraint analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['constraint'].extract_pricing_rules(python_files)
        except Exception as e:
            self.logger.error(f"Pricing rules extraction failed: {e}")
            return {"error": str(e)}
    
    def extract_decision_logic(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract decision tables and business logic from code.
        Consolidates functionality from business_constraint_analyzer.py
        
        Returns:
            Dict containing decision tables, trees, conditional logic
        """
        if 'constraint' not in self._analyzers:
            return {"error": "Business constraint analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['constraint'].extract_decision_logic(python_files)
        except Exception as e:
            self.logger.error(f"Decision logic extraction failed: {e}")
            return {"error": str(e)}
    
    def extract_business_rules(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract general business rules from code.
        Consolidates functionality from business_rule_extractor.py
        
        Returns:
            Dict containing rules, categories, confidence levels, documentation coverage
        """
        if 'rule_extractor' not in self._analyzers:
            return {"error": "Business rule extractor not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['rule_extractor'].extract_business_rules(python_files)
        except Exception as e:
            self.logger.error(f"Business rule extraction failed: {e}")
            return {"error": str(e)}
    
    def extract_validation_rules(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract validation-specific business rules.
        Consolidates functionality from business_rule_extractor.py
        
        Returns:
            Dict containing field, cross-field, business, format, range validations
        """
        if 'rule_extractor' not in self._analyzers:
            return {"error": "Business rule extractor not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['rule_extractor'].extract_validation_rules(python_files)
        except Exception as e:
            self.logger.error(f"Validation rule extraction failed: {e}")
            return {"error": str(e)}
    
    def extract_calculation_rules(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract calculation and computation business rules.
        Consolidates functionality from business_rule_extractor.py
        
        Returns:
            Dict containing financial, pricing, tax, discount, scoring calculations
        """
        if 'rule_extractor' not in self._analyzers:
            return {"error": "Business rule extractor not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['rule_extractor'].extract_calculation_rules(python_files)
        except Exception as e:
            self.logger.error(f"Calculation rule extraction failed: {e}")
            return {"error": str(e)}
    
    def extract_authorization_rules(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract authorization and access control rules.
        Consolidates functionality from business_rule_extractor.py
        
        Returns:
            Dict containing permission checks, role-based, attribute-based rules
        """
        if 'rule_extractor' not in self._analyzers:
            return {"error": "Business rule extractor not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['rule_extractor'].extract_authorization_rules(python_files)
        except Exception as e:
            self.logger.error(f"Authorization rule extraction failed: {e}")
            return {"error": str(e)}
    
    def analyze_workflows(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze workflow patterns in the code.
        Consolidates functionality from business_workflow_analyzer.py
        
        Returns:
            Dict containing workflows, steps, transitions, approval flows
        """
        if 'workflow' not in self._analyzers:
            return {"error": "Business workflow analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['workflow'].analyze_workflows(python_files)
        except Exception as e:
            self.logger.error(f"Workflow analysis failed: {e}")
            return {"error": str(e)}
    
    def detect_state_machines(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect and analyze state machine implementations.
        Consolidates functionality from business_workflow_analyzer.py
        
        Returns:
            Dict containing state machines, states, transitions, guards, actions
        """
        if 'workflow' not in self._analyzers:
            return {"error": "Business workflow analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['workflow'].detect_state_machines(python_files)
        except Exception as e:
            self.logger.error(f"State machine detection failed: {e}")
            return {"error": str(e)}
    
    def extract_domain_model(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract domain model and entities.
        Consolidates functionality from business_workflow_analyzer.py
        
        Returns:
            Dict containing entities, value objects, aggregates, repositories, services
        """
        if 'workflow' not in self._analyzers:
            return {"error": "Business workflow analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['workflow'].extract_domain_model(python_files)
        except Exception as e:
            self.logger.error(f"Domain model extraction failed: {e}")
            return {"error": str(e)}
    
    def analyze_code_debt(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze code quality debt in Python files.
        Consolidates functionality from debt_code_analyzer.py
        
        Returns:
            List of DebtItem objects with complexity, duplication, naming issues
        """
        if 'code_debt' not in self._analyzers:
            return {"error": "Code debt analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            debt_items = self._analyzers['code_debt'].analyze_code_debt(python_files)
            return {"debt_items": [item.to_dict() if hasattr(item, 'to_dict') else str(item) for item in debt_items]}
        except Exception as e:
            self.logger.error(f"Code debt analysis failed: {e}")
            return {"error": str(e)}
    
    def quantify_debt(self, debt_items: List[Any]) -> Dict[str, Any]:
        """
        Quantify technical debt in developer hours.
        Consolidates functionality from debt_quantifier.py
        
        Returns:
            DebtMetrics with total hours, ratio, interest, break-even
        """
        if 'debt_quantifier' not in self._analyzers:
            return {"error": "Debt quantifier not available"}
        
        try:
            metrics = self._analyzers['debt_quantifier'].quantify_debt(debt_items)
            return metrics.to_dict() if hasattr(metrics, 'to_dict') else {"metrics": str(metrics)}
        except Exception as e:
            self.logger.error(f"Debt quantification failed: {e}")
            return {"error": str(e)}
    
    def prioritize_debt(self, debt_items: List[Any]) -> Dict[str, Any]:
        """
        Create prioritized debt remediation plan.
        Consolidates functionality from debt_quantifier.py
        
        Returns:
            List of RemediationPlan objects prioritized by ROI
        """
        if 'debt_quantifier' not in self._analyzers:
            return {"error": "Debt quantifier not available"}
        
        try:
            plans = self._analyzers['debt_quantifier'].prioritize_debt(debt_items)
            return {"remediation_plans": [plan.to_dict() if hasattr(plan, 'to_dict') else str(plan) for plan in plans]}
        except Exception as e:
            self.logger.error(f"Debt prioritization failed: {e}")
            return {"error": str(e)}
    
    def track_debt_trend(self, debt_items: List[Any], metrics: Any) -> Dict[str, Any]:
        """
        Track debt trends over time.
        Consolidates functionality from debt_quantifier.py
        """
        if 'debt_quantifier' not in self._analyzers:
            return {"error": "Debt quantifier not available"}
        
        try:
            self._analyzers['debt_quantifier'].track_trend(debt_items, metrics)
            return {"status": "Trend tracked successfully"}
        except Exception as e:
            self.logger.error(f"Debt trend tracking failed: {e}")
            return {"error": str(e)}
    
    def get_debt_financial_impact(self, metrics: Any) -> Dict[str, Any]:
        """
        Calculate financial impact of technical debt.
        Consolidates functionality from debt_quantifier.py
        
        Returns:
            Dict with direct cost, interest cost, productivity loss
        """
        if 'debt_quantifier' not in self._analyzers:
            return {"error": "Debt quantifier not available"}
        
        try:
            return self._analyzers['debt_quantifier'].get_financial_impact(metrics)
        except Exception as e:
            self.logger.error(f"Financial impact calculation failed: {e}")
            return {"error": str(e)}
    
    def generate_debt_summary(self, debt_items: List[Any], metrics: Any) -> Dict[str, Any]:
        """
        Generate executive summary of debt analysis.
        Consolidates functionality from debt_quantifier.py
        
        Returns:
            Dict with total items, hours, financial impact, recommendations
        """
        if 'debt_quantifier' not in self._analyzers:
            return {"error": "Debt quantifier not available"}
        
        try:
            return self._analyzers['debt_quantifier'].generate_summary(debt_items, metrics)
        except Exception as e:
            self.logger.error(f"Debt summary generation failed: {e}")
            return {"error": str(e)}
    
    def analyze_test_debt(self, root_path: Optional[str] = None, coverage_data: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Analyze test-related technical debt.
        Consolidates functionality from debt_test_analyzer.py
        
        Returns:
            List of DebtItem objects for missing tests, low coverage, test quality
        """
        if 'test_debt' not in self._analyzers:
            return {"error": "Test debt analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            
            # Separate source and test files
            all_files = list(root.rglob("*.py"))
            test_files = [f for f in all_files if any(p in f.stem for p in ['test_', '_test', 'Test'])]
            source_files = [f for f in all_files if f not in test_files]
            
            debt_items = self._analyzers['test_debt'].analyze_test_debt(source_files, test_files, coverage_data)
            return {"debt_items": [item.to_dict() if hasattr(item, 'to_dict') else str(item) for item in debt_items]}
        except Exception as e:
            self.logger.error(f"Test debt analysis failed: {e}")
            return {"error": str(e)}
    
    def recognize_intent(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Recognize developer intent from code elements.
        Consolidates functionality from semantic_intent_analyzer.py
        
        Returns:
            Dict with recognized intents, distribution, confidence scores
        """
        if 'intent' not in self._analyzers:
            return {"error": "Semantic intent analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['intent'].recognize_intent(python_files)
        except Exception as e:
            self.logger.error(f"Intent recognition failed: {e}")
            return {"error": str(e)}
    
    def extract_semantic_signatures(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract semantic signatures from code.
        Consolidates functionality from semantic_intent_analyzer.py
        
        Returns:
            Dict with function, class, module signatures
        """
        if 'intent' not in self._analyzers:
            return {"error": "Semantic intent analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['intent'].extract_semantic_signatures(python_files)
        except Exception as e:
            self.logger.error(f"Semantic signature extraction failed: {e}")
            return {"error": str(e)}
    
    def classify_code_purpose(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify the purpose of code sections.
        Consolidates functionality from semantic_intent_analyzer.py
        
        Returns:
            Dict with business logic, infrastructure, utilities classification
        """
        if 'intent' not in self._analyzers:
            return {"error": "Semantic intent analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['intent'].classify_code_purpose(python_files)
        except Exception as e:
            self.logger.error(f"Code purpose classification failed: {e}")
            return {"error": str(e)}
    
    def check_intent_consistency(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Check consistency of intent across codebase.
        Consolidates functionality from semantic_intent_analyzer.py
        
        Returns:
            Dict with consistent patterns, inconsistencies, mismatches
        """
        if 'intent' not in self._analyzers:
            return {"error": "Semantic intent analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['intent'].check_intent_consistency(python_files)
        except Exception as e:
            self.logger.error(f"Intent consistency check failed: {e}")
            return {"error": str(e)}
    
    def get_dominant_intent(self) -> str:
        """
        Get the most common intent type.
        Consolidates functionality from semantic_intent_analyzer.py
        """
        if 'intent' not in self._analyzers:
            return "unknown"
        
        try:
            return self._analyzers['intent'].get_dominant_intent()
        except Exception as e:
            self.logger.error(f"Get dominant intent failed: {e}")
            return "unknown"
    
    def get_intents_by_type(self, intent_type: Any) -> List[Any]:
        """
        Get all intents of a specific type.
        Consolidates functionality from semantic_intent_analyzer.py
        
        Returns:
            List of SemanticIntent objects of the specified type
        """
        if 'intent' not in self._analyzers:
            return []
        
        try:
            return self._analyzers['intent'].get_intents_by_type(intent_type)
        except Exception as e:
            self.logger.error(f"Get intents by type failed: {e}")
            return []
    
    def identify_conceptual_patterns(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Identify conceptual patterns in code.
        Consolidates functionality from semantic_pattern_detector.py
        
        Returns:
            Dict with design patterns, architectural patterns, anti-patterns
        """
        if 'pattern' not in self._analyzers:
            return {"error": "Semantic pattern detector not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['pattern'].identify_conceptual_patterns(python_files)
        except Exception as e:
            self.logger.error(f"Pattern identification failed: {e}")
            return {"error": str(e)}
    
    def identify_behavioral_patterns(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Identify behavioral patterns in code.
        Consolidates functionality from semantic_pattern_detector.py
        
        Returns:
            Dict with state machines, event-driven, pipeline, callback patterns
        """
        if 'pattern' not in self._analyzers:
            return {"error": "Semantic pattern detector not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['pattern'].identify_behavioral_patterns(python_files)
        except Exception as e:
            self.logger.error(f"Behavioral pattern identification failed: {e}")
            return {"error": str(e)}
    
    def extract_domain_concepts(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract domain-specific concepts from code.
        Consolidates functionality from semantic_pattern_detector.py
        
        Returns:
            Dict with entities, value objects, services, repositories
        """
        if 'pattern' not in self._analyzers:
            return {"error": "Semantic pattern detector not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['pattern'].extract_domain_concepts(python_files)
        except Exception as e:
            self.logger.error(f"Domain concept extraction failed: {e}")
            return {"error": str(e)}
    
    def perform_semantic_clustering(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Cluster code elements based on semantic similarity.
        Consolidates functionality from semantic_pattern_detector.py
        
        Returns:
            Dict with semantic clusters, coherence, outliers
        """
        if 'pattern' not in self._analyzers:
            return {"error": "Semantic pattern detector not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['pattern'].perform_semantic_clustering(python_files)
        except Exception as e:
            self.logger.error(f"Semantic clustering failed: {e}")
            return {"error": str(e)}
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """
        Get summary of detected patterns.
        Consolidates functionality from semantic_pattern_detector.py
        
        Returns:
            Dict with total patterns, pattern types, average quality
        """
        if 'pattern' not in self._analyzers:
            return {"error": "Semantic pattern detector not available"}
        
        try:
            return self._analyzers['pattern'].get_pattern_summary()
        except Exception as e:
            self.logger.error(f"Pattern summary failed: {e}")
            return {"error": str(e)}
    
    def analyze_semantic_relationships(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze semantic relationships between code elements.
        Consolidates functionality from semantic_relationship_analyzer.py
        
        Returns:
            Dict with direct/indirect relationships, dependency graph, coupling
        """
        if 'relationship' not in self._analyzers:
            return {"error": "Semantic relationship analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['relationship'].analyze_semantic_relationships(python_files)
        except Exception as e:
            self.logger.error(f"Semantic relationship analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_naming_semantics(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze naming conventions and semantics.
        Consolidates functionality from semantic_relationship_analyzer.py
        
        Returns:
            Dict with naming conventions, coherence, violations, suggestions
        """
        if 'relationship' not in self._analyzers:
            return {"error": "Semantic relationship analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['relationship'].analyze_naming_semantics(python_files)
        except Exception as e:
            self.logger.error(f"Naming semantics analysis failed: {e}")
            return {"error": str(e)}
    
    def assess_semantic_quality(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Assess the semantic quality of code.
        Consolidates functionality from semantic_relationship_analyzer.py
        
        Returns:
            Dict with clarity, consistency, expressiveness scores
        """
        if 'relationship' not in self._analyzers:
            return {"error": "Semantic relationship analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['relationship'].assess_semantic_quality(python_files)
        except Exception as e:
            self.logger.error(f"Semantic quality assessment failed: {e}")
            return {"error": str(e)}
    
    def get_relationship_summary(self) -> Dict[str, Any]:
        """
        Get summary of relationship analysis.
        Consolidates functionality from semantic_relationship_analyzer.py
        
        Returns:
            Dict with total relationships, types, average confidence
        """
        if 'relationship' not in self._analyzers:
            return {"error": "Semantic relationship analyzer not available"}
        
        try:
            return self._analyzers['relationship'].get_relationship_summary()
        except Exception as e:
            self.logger.error(f"Relationship summary failed: {e}")
            return {"error": str(e)}
    
    def analyze_ml_project(self, project_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive ML code analysis on a project.
        Consolidates functionality from ml_code_analyzer.py
        
        Returns:
            Dict with frameworks detected, issues, architecture summary, recommendations
        """
        if 'enhanced_ml' not in self._analyzers:
            return {"error": "Enhanced ML analyzer not available"}
        
        try:
            project_root = project_path or str(Path.cwd())
            return self._analyzers['enhanced_ml'].analyze_project(project_root)
        except Exception as e:
            self.logger.error(f"ML project analysis failed: {e}")
            return {"error": str(e)}
    
    def analyze_technical_debt_project(self, project_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive technical debt analysis on a project.
        Consolidates functionality from technical_debt_analyzer.py
        
        Returns:
            Dict with debt items, metrics, summary, recommendations, prioritized actions
        """
        if 'enhanced_debt' not in self._analyzers:
            return {"error": "Enhanced debt analyzer not available"}
        
        try:
            project_root = project_path or str(Path.cwd())
            return self._analyzers['enhanced_debt'].analyze_project(project_root)
        except Exception as e:
            self.logger.error(f"Technical debt project analysis failed: {e}")
            return {"error": str(e)}
    
    def extract_business_events(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract business events and event handlers.
        Consolidates functionality from business_workflow_analyzer.py
        
        Returns:
            Dict containing event definitions, handlers, publishers, subscribers
        """
        if 'workflow' not in self._analyzers:
            return {"error": "Business workflow analyzer not available"}
        
        try:
            from pathlib import Path
            root = Path(root_path) if root_path else Path.cwd()
            python_files = list(root.rglob("*.py"))
            return self._analyzers['workflow'].extract_business_events(python_files)
        except Exception as e:
            self.logger.error(f"Business event extraction failed: {e}")
            return {"error": str(e)}
    
    def run_comprehensive_analysis(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive analysis using all available analyzers.
        
        Args:
            root_path: Optional path to analyze. If None, uses current directory.
            
        Returns:
            Combined analysis results from all analyzers
        """
        results = {
            "analysis_summary": {
                "total_analyzers": len(self._analyzers),
                "available_analyzers": self.get_available_analyzers(),
                "analysis_timestamp": self._get_timestamp()
            }
        }
        
        # Run business rule analysis
        if 'business' in self._analyzers:
            self.logger.info("Running business rule analysis...")
            results["business_analysis"] = self.analyze_business_rules(root_path)
        
        # Run technical debt analysis
        if 'debt' in self._analyzers:
            self.logger.info("Running technical debt analysis...")
            results["debt_analysis"] = self.analyze_technical_debt(root_path)
        
        # Run ML analysis
        if 'ml' in self._analyzers:
            self.logger.info("Running ML component analysis...")
            results["ml_analysis"] = self.analyze_ml_components(root_path)
        
        # Run semantic analysis
        if 'semantic' in self._analyzers:
            self.logger.info("Running semantic analysis...")
            results["semantic_analysis"] = self.analyze_semantic_structure(root_path)
        
        # Generate cross-analyzer insights
        results["cross_analyzer_insights"] = self._generate_cross_analyzer_insights(results)
        
        return results
    
    def _generate_cross_analyzer_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate insights that combine data from multiple analyzers.
        
        Args:
            results: Combined results from all analyzers
            
        Returns:
            Cross-analyzer insights and correlations
        """
        insights = {
            "correlation_analysis": {},
            "combined_recommendations": [],
            "risk_assessment": {},
            "prioritization": {}
        }
        
        # Analyze correlations between business rules and technical debt
        if "business_analysis" in results and "debt_analysis" in results:
            business_data = results["business_analysis"]
            debt_data = results["debt_analysis"]
            
            # Check if areas with high business rule complexity also have high technical debt
            insights["correlation_analysis"]["business_debt_correlation"] = {
                "high_complexity_business_areas": self._extract_complex_business_areas(business_data),
                "high_debt_areas": self._extract_high_debt_areas(debt_data),
                "correlation_strength": "analysis_pending"
            }
        
        # Generate combined recommendations
        insights["combined_recommendations"] = self._generate_combined_recommendations(results)
        
        # Assess overall risk
        insights["risk_assessment"] = self._assess_overall_risk(results)
        
        return insights
    
    def _extract_complex_business_areas(self, business_data: Dict[str, Any]) -> List[str]:
        """Extract areas with high business rule complexity."""
        complex_areas = []
        if "business_rules" in business_data:
            rules_data = business_data["business_rules"]
            if "rule_categories" in rules_data:
                for category, rules in rules_data["rule_categories"].items():
                    if len(rules) > 5:  # Threshold for high complexity
                        complex_areas.append(category)
        return complex_areas
    
    def _extract_high_debt_areas(self, debt_data: Dict[str, Any]) -> List[str]:
        """Extract areas with high technical debt."""
        high_debt_areas = []
        if "debt_inventory" in debt_data:
            for debt_type, items in debt_data["debt_inventory"].items():
                if isinstance(items, list) and len(items) > 3:  # Threshold for high debt
                    high_debt_areas.append(debt_type)
        return high_debt_areas
    
    def _generate_combined_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations that combine insights from multiple analyzers."""
        recommendations = []
        
        # Add recommendations based on combined analysis
        if "business_analysis" in results and "debt_analysis" in results:
            recommendations.append({
                "type": "architectural",
                "priority": "high",
                "description": "Focus debt reduction efforts on areas with high business rule complexity",
                "rationale": "High business complexity + high technical debt creates compounding risks"
            })
        
        if "ml_analysis" in results and "debt_analysis" in results:
            recommendations.append({
                "type": "ml_infrastructure",
                "priority": "medium", 
                "description": "Ensure ML components have adequate test coverage to prevent debt accumulation",
                "rationale": "ML code is particularly prone to technical debt due to complexity"
            })
        
        return recommendations
    
    def _assess_overall_risk(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Assess overall risk across all analyses."""
        risk_factors = []
        
        # Check for critical technical debt
        if "debt_analysis" in results:
            debt_summary = results["debt_analysis"].get("summary", {})
            executive_summary = debt_summary.get("executive_summary", {})
            critical_issues = executive_summary.get("critical_issues", 0)
            if critical_issues > 5:
                risk_factors.append("high_technical_debt")
        
        # Check for complex business rules without adequate documentation
        if "business_analysis" in results:
            business_summary = results["business_analysis"].get("summary", {})
            doc_quality = business_summary.get("documentation_quality", {})
            doc_percentage = doc_quality.get("documentation_percentage", 100)
            if doc_percentage < 50:
                risk_factors.append("poor_business_documentation")
        
        # Determine overall risk level
        if len(risk_factors) >= 2:
            overall_risk = "high"
        elif len(risk_factors) == 1:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        return {
            "overall_risk_level": overall_risk,
            "risk_factors": risk_factors,
            "mitigation_priority": "immediate" if overall_risk == "high" else "scheduled"
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    @property
    def ml_analyzer(self):
        """Get ML analyzer instance."""
        return self._analyzers.get('ml')
    
    @property
    def semantic_analyzer(self):
        """Get semantic analyzer instance."""
        return self._analyzers.get('semantic')
    
    @property
    def business_analyzer(self):
        """Get business analyzer instance."""
        return self._analyzers.get('business')
    
    @property
    def debt_analyzer(self):
        """Get technical debt analyzer instance."""
        return self._analyzers.get('debt')
    
    def get_analyzer_status(self) -> Dict[str, Any]:
        """Get status of all analyzers."""
        return {
            "available_analyzers": self.get_available_analyzers(),
            "analyzer_capabilities": {
                "business": BUSINESS_ANALYZER_AVAILABLE,
                "debt": DEBT_ANALYZER_AVAILABLE,
                "ml": ML_ANALYZER_AVAILABLE,
                "semantic": SEMANTIC_ANALYZER_AVAILABLE
            },
            "total_analyzers": len(self._analyzers),
            "hub_status": "operational"
        }


# Convenience functions for direct access
def analyze_business_rules(root_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function for business rule analysis."""
    hub = AnalysisHub(config)
    return hub.analyze_business_rules(root_path)

def analyze_technical_debt(root_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function for technical debt analysis."""
    hub = AnalysisHub(config)
    return hub.analyze_technical_debt(root_path)

def analyze_ml_code(root_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function for ML code analysis."""
    hub = AnalysisHub(config)
    return hub.analyze_ml_code(root_path)

def analyze_semantics(root_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function for semantic analysis."""
    hub = AnalysisHub(config)
    return hub.analyze_semantics(root_path)

def run_comprehensive_analysis(root_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function for comprehensive analysis."""
    hub = AnalysisHub(config)
    return hub.run_comprehensive_analysis(root_path)


# Public API exports
__all__ = [
    'AnalysisHub',
    'analyze_business_rules',
    'analyze_technical_debt',
    'analyze_ml_code',
    'analyze_semantics',
    'run_comprehensive_analysis'
]

# Conditionally export analyzers if available
if BUSINESS_ANALYZER_AVAILABLE:
    __all__.append('BusinessAnalyzer')

if DEBT_ANALYZER_AVAILABLE:
    __all__.append('TechnicalDebtAnalyzer')

if ML_ANALYZER_AVAILABLE:
    __all__.append('MLAnalyzer')

if SEMANTIC_ANALYZER_AVAILABLE:
    __all__.append('SemanticAnalyzer')


# Version and capabilities info
__version__ = "1.0.0"
__capabilities__ = {
    'business_analysis': BUSINESS_ANALYZER_AVAILABLE,
    'debt_analysis': DEBT_ANALYZER_AVAILABLE,
    'ml_analysis': ML_ANALYZER_AVAILABLE,
    'semantic_analysis': SEMANTIC_ANALYZER_AVAILABLE,
    'cross_analyzer_insights': True,
    'unified_analysis_hub': True
}
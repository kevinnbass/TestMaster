"""
Swarms Derived Multi-Agent Evaluation Security Module
Extracted from Swarms council judge patterns for secure distributed evaluation
Enhanced for parallel security assessment and bias detection
"""

import uuid
import time
import json
import logging
import threading
import statistics
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from collections import defaultdict, Counter
from .error_handler import SecurityError, security_error_handler


class EvaluationDimension(Enum):
    """Security evaluation dimensions based on Swarms patterns"""
    ACCURACY = "accuracy"
    HARMFULNESS = "harmfulness"
    SAFETY = "safety"
    COHERENCE = "coherence"
    BIAS_DETECTION = "bias_detection"
    SECURITY_COMPLIANCE = "security_compliance"
    ETHICAL_ASSESSMENT = "ethical_assessment"
    FACTUAL_VERIFICATION = "factual_verification"


class JudgeRole(Enum):
    """Judge agent roles in evaluation council"""
    PRIMARY_EVALUATOR = "primary_evaluator"
    SECURITY_SPECIALIST = "security_specialist"
    BIAS_DETECTOR = "bias_detector"
    FACT_CHECKER = "fact_checker"
    ETHICS_REVIEWER = "ethics_reviewer"
    CONSENSUS_MODERATOR = "consensus_moderator"


class EvaluationSeverity(Enum):
    """Severity levels for evaluation findings"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatCategory(Enum):
    """Security threat categories in evaluation"""
    MISINFORMATION = "misinformation"
    HARMFUL_CONTENT = "harmful_content"
    BIAS_AMPLIFICATION = "bias_amplification"
    SECURITY_VULNERABILITY = "security_vulnerability"
    PRIVACY_VIOLATION = "privacy_violation"
    MANIPULATION_ATTEMPT = "manipulation_attempt"


@dataclass
class EvaluationCriteria:
    """Security evaluation criteria for multi-agent assessment"""
    dimension: EvaluationDimension
    weight: float = 1.0
    threshold: float = 0.7
    description: str = ""
    security_critical: bool = False
    automated_checks: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not 0.0 <= self.weight <= 1.0:
            raise SecurityError("Weight must be between 0.0 and 1.0", "EVAL_CRIT_001")
        
        if not 0.0 <= self.threshold <= 1.0:
            raise SecurityError("Threshold must be between 0.0 and 1.0", "EVAL_CRIT_002")


@dataclass
class JudgeAgent:
    """Secure judge agent for evaluation council"""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str = ""
    role: JudgeRole = JudgeRole.PRIMARY_EVALUATOR
    specialization: List[EvaluationDimension] = field(default_factory=list)
    trust_score: float = 1.0
    expertise_level: float = 0.8
    evaluation_history: List[str] = field(default_factory=list)
    bias_score: float = 0.0  # Lower is better
    consistency_score: float = 0.0
    security_clearance: str = "standard"
    
    def __post_init__(self):
        if not self.agent_name:
            self.agent_name = f"judge_{self.agent_id[:8]}"
        
        if not 0.0 <= self.trust_score <= 1.0:
            raise SecurityError("Trust score must be between 0.0 and 1.0", "JUDGE_001")
        
        if not 0.0 <= self.expertise_level <= 1.0:
            raise SecurityError("Expertise level must be between 0.0 and 1.0", "JUDGE_002")
    
    @property
    def effective_weight(self) -> float:
        """Calculate effective weight based on trust and expertise"""
        return self.trust_score * self.expertise_level * (1.0 - self.bias_score)
    
    @property
    def is_qualified(self) -> bool:
        """Check if judge is qualified for evaluation"""
        return (
            self.trust_score >= 0.5 and
            self.expertise_level >= 0.6 and
            self.bias_score < 0.5
        )


@dataclass
class EvaluationTask:
    """Secure evaluation task for multi-agent assessment"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content_to_evaluate: Any = None
    evaluation_criteria: List[EvaluationCriteria] = field(default_factory=list)
    security_context: Dict[str, Any] = field(default_factory=dict)
    requester_id: str = ""
    priority: int = 1  # 1-10 scale
    deadline: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=30))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 1 <= self.priority <= 10:
            raise SecurityError("Priority must be between 1 and 10", "EVAL_TASK_001")
        
        if not self.requester_id:
            raise SecurityError("Requester ID is required", "EVAL_TASK_002")
    
    @property
    def is_expired(self) -> bool:
        """Check if evaluation task has expired"""
        return datetime.utcnow() > self.deadline


@dataclass
class EvaluationResult:
    """Security evaluation result from individual judge"""
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    judge_id: str = ""
    dimension: EvaluationDimension = EvaluationDimension.ACCURACY
    score: float = 0.0
    confidence: float = 0.0
    rationale: str = ""
    security_findings: List[Dict[str, Any]] = field(default_factory=list)
    threats_detected: List[ThreatCategory] = field(default_factory=list)
    severity: EvaluationSeverity = EvaluationSeverity.INFO
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time: float = 0.0
    
    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise SecurityError("Score must be between 0.0 and 1.0", "EVAL_RES_001")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise SecurityError("Confidence must be between 0.0 and 1.0", "EVAL_RES_002")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            'result_id': self.result_id,
            'task_id': self.task_id,
            'judge_id': self.judge_id,
            'dimension': self.dimension.value,
            'score': self.score,
            'confidence': self.confidence,
            'rationale': self.rationale,
            'security_findings': self.security_findings,
            'threats_detected': [t.value for t in self.threats_detected],
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'processing_time': self.processing_time
        }


@dataclass
class CouncilDecision:
    """Final decision from evaluation council"""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    overall_score: float = 0.0
    confidence: float = 0.0
    consensus_reached: bool = False
    majority_opinion: str = ""
    dissenting_opinions: List[str] = field(default_factory=list)
    security_verdict: str = "SAFE"  # SAFE, UNSAFE, REVIEW_REQUIRED
    critical_findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    participating_judges: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary for serialization"""
        return {
            'decision_id': self.decision_id,
            'task_id': self.task_id,
            'overall_score': self.overall_score,
            'confidence': self.confidence,
            'consensus_reached': self.consensus_reached,
            'majority_opinion': self.majority_opinion,
            'dissenting_opinions': self.dissenting_opinions,
            'security_verdict': self.security_verdict,
            'critical_findings': self.critical_findings,
            'recommendations': self.recommendations,
            'participating_judges': self.participating_judges,
            'execution_time': self.execution_time
        }


class SecurityEvaluationEngine:
    """Security-focused evaluation engine for automated checks"""
    
    def __init__(self):
        self.security_patterns = {
            ThreatCategory.MISINFORMATION: [
                r'(?i)(false|fake|misleading|incorrect)\s+(information|data|claims)',
                r'(?i)(conspiracy|hoax|myth|debunked)',
                r'(?i)(without\s+evidence|unsubstantiated|unverified)'
            ],
            ThreatCategory.HARMFUL_CONTENT: [
                r'(?i)(violence|harm|dangerous|illegal)',
                r'(?i)(discriminat|hate|offensive|abusive)',
                r'(?i)(self\s*harm|suicide|depression)'
            ],
            ThreatCategory.BIAS_AMPLIFICATION: [
                r'(?i)(stereotype|prejudice|discriminat)',
                r'(?i)(gender|race|ethnicity|religion)\s+(bias|discrimination)',
                r'(?i)(unfair|biased|prejudiced)\s+(treatment|representation)'
            ]
        }
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_content_security(self, content: str) -> Dict[str, Any]:
        """Analyze content for security threats using pattern matching"""
        findings = {
            'threats': [],
            'severity': EvaluationSeverity.INFO,
            'confidence': 0.0,
            'patterns_matched': []
        }
        
        try:
            import re
            
            for threat_category, patterns in self.security_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        findings['threats'].append(threat_category)
                        findings['patterns_matched'].extend(matches)
            
            # Calculate severity and confidence
            if findings['threats']:
                threat_count = len(findings['threats'])
                if threat_count >= 3:
                    findings['severity'] = EvaluationSeverity.HIGH
                    findings['confidence'] = 0.8
                elif threat_count >= 2:
                    findings['severity'] = EvaluationSeverity.MEDIUM
                    findings['confidence'] = 0.6
                else:
                    findings['severity'] = EvaluationSeverity.LOW
                    findings['confidence'] = 0.4
            
            return findings
            
        except Exception as e:
            self.logger.error(f"Content security analysis failed: {e}")
            return findings
    
    def validate_evaluation_consistency(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Validate consistency across evaluation results"""
        if len(results) < 2:
            return {'consistent': True, 'variance': 0.0, 'outliers': []}
        
        try:
            scores = [result.score for result in results]
            mean_score = statistics.mean(scores)
            variance = statistics.variance(scores) if len(scores) > 1 else 0.0
            
            # Identify outliers (scores more than 2 standard deviations from mean)
            if variance > 0:
                std_dev = statistics.stdev(scores)
                outliers = [
                    result.result_id for result in results
                    if abs(result.score - mean_score) > 2 * std_dev
                ]
            else:
                outliers = []
            
            # Consider consistent if variance is low
            consistent = variance < 0.1  # Threshold for consistency
            
            return {
                'consistent': consistent,
                'variance': variance,
                'mean_score': mean_score,
                'outliers': outliers,
                'score_range': max(scores) - min(scores) if scores else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Consistency validation failed: {e}")
            return {'consistent': False, 'variance': 1.0, 'outliers': []}


class MultiAgentEvaluationSecurityManager:
    """Secure multi-agent evaluation manager with council patterns"""
    
    def __init__(self, max_workers: int = 8):
        self.judges: Dict[str, JudgeAgent] = {}
        self.evaluation_tasks: Dict[str, EvaluationTask] = {}
        self.evaluation_results: Dict[str, List[EvaluationResult]] = {}
        self.council_decisions: Dict[str, CouncilDecision] = {}
        
        # Security components
        self.security_engine = SecurityEvaluationEngine()
        self.evaluation_lock = threading.RLock()
        
        # Performance settings
        self.max_workers = max_workers
        self.evaluation_timeout = 300  # 5 minutes per evaluation
        
        # Quality assurance
        self.min_judges_required = 3
        self.consensus_threshold = 0.7
        self.max_bias_tolerance = 0.3
        
        self.logger = logging.getLogger(__name__)
    
    def register_judge(self, judge: JudgeAgent) -> bool:
        """Register judge agent for evaluation council"""
        try:
            with self.evaluation_lock:
                if not judge.is_qualified:
                    raise SecurityError("Judge does not meet qualification requirements", "JUDGE_REG_001")
                
                # Check for potential bias conflicts
                if judge.bias_score > self.max_bias_tolerance:
                    raise SecurityError("Judge bias score too high", "JUDGE_REG_002")
                
                self.judges[judge.agent_id] = judge
                
                self.logger.info(f"Judge registered: {judge.agent_name} ({judge.role.value})")
                return True
                
        except Exception as e:
            error = SecurityError(f"Judge registration failed: {str(e)}", "JUDGE_REG_FAIL_001")
            security_error_handler.handle_error(error)
            return False
    
    def submit_evaluation_task(self, task: EvaluationTask) -> bool:
        """Submit task for multi-agent evaluation"""
        try:
            with self.evaluation_lock:
                if task.is_expired:
                    raise SecurityError("Task has already expired", "EVAL_SUBMIT_001")
                
                # Validate we have sufficient qualified judges
                qualified_judges = [j for j in self.judges.values() if j.is_qualified]
                if len(qualified_judges) < self.min_judges_required:
                    raise SecurityError("Insufficient qualified judges available", "EVAL_SUBMIT_002")
                
                self.evaluation_tasks[task.task_id] = task
                self.evaluation_results[task.task_id] = []
                
                self.logger.info(f"Evaluation task submitted: {task.task_id}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Task submission failed: {str(e)}", "EVAL_SUBMIT_FAIL_001")
            security_error_handler.handle_error(error)
            return False
    
    def execute_council_evaluation(self, task_id: str) -> Optional[CouncilDecision]:
        """Execute multi-agent council evaluation with security analysis"""
        start_time = time.time()
        
        try:
            with self.evaluation_lock:
                if task_id not in self.evaluation_tasks:
                    raise SecurityError("Evaluation task not found", "EVAL_EXEC_001")
                
                task = self.evaluation_tasks[task_id]
                if task.is_expired:
                    raise SecurityError("Evaluation task has expired", "EVAL_EXEC_002")
                
                # Select qualified judges for evaluation
                available_judges = [j for j in self.judges.values() if j.is_qualified]
                if len(available_judges) < self.min_judges_required:
                    raise SecurityError("Insufficient qualified judges", "EVAL_EXEC_003")
                
                # Ensure diverse judge selection to prevent bias
                selected_judges = self._select_diverse_judges(available_judges, task.evaluation_criteria)
                
                # Execute parallel evaluation
                evaluation_results = self._execute_parallel_evaluation(task, selected_judges)
                
                # Store results
                self.evaluation_results[task_id] = evaluation_results
                
                # Generate council decision
                decision = self._generate_council_decision(task, evaluation_results)
                decision.execution_time = time.time() - start_time
                
                # Store decision
                self.council_decisions[task_id] = decision
                
                self.logger.info(f"Council evaluation completed: {task_id}")
                return decision
                
        except Exception as e:
            error = SecurityError(f"Council evaluation failed: {str(e)}", "EVAL_EXEC_FAIL_001")
            security_error_handler.handle_error(error)
            
            # Return failure decision
            return CouncilDecision(
                task_id=task_id,
                overall_score=0.0,
                confidence=0.0,
                consensus_reached=False,
                security_verdict="REVIEW_REQUIRED",
                execution_time=time.time() - start_time
            )
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        """Get evaluation system statistics"""
        with self.evaluation_lock:
            total_judges = len(self.judges)
            qualified_judges = sum(1 for j in self.judges.values() if j.is_qualified)
            
            total_evaluations = len(self.council_decisions)
            successful_evaluations = sum(
                1 for d in self.council_decisions.values() 
                if d.consensus_reached
            )
            
            # Security verdict breakdown
            verdict_counts = Counter(
                decision.security_verdict 
                for decision in self.council_decisions.values()
            )
            
            return {
                'total_judges': total_judges,
                'qualified_judges': qualified_judges,
                'total_evaluations': total_evaluations,
                'successful_evaluations': successful_evaluations,
                'success_rate': successful_evaluations / max(1, total_evaluations),
                'security_verdicts': dict(verdict_counts),
                'average_execution_time': statistics.mean([
                    d.execution_time for d in self.council_decisions.values()
                ]) if self.council_decisions else 0.0
            }
    
    def _select_diverse_judges(self, available_judges: List[JudgeAgent], 
                             criteria: List[EvaluationCriteria]) -> List[JudgeAgent]:
        """Select diverse set of judges to minimize bias"""
        # Sort judges by effective weight (descending)
        sorted_judges = sorted(available_judges, key=lambda j: j.effective_weight, reverse=True)
        
        # Select judges with diverse roles and specializations
        selected = []
        used_roles = set()
        
        for judge in sorted_judges:
            # Ensure role diversity
            if judge.role not in used_roles or len(selected) < self.min_judges_required:
                selected.append(judge)
                used_roles.add(judge.role)
                
                # Stop if we have enough diverse judges
                if len(selected) >= min(self.max_workers, len(available_judges)):
                    break
        
        return selected[:self.max_workers]
    
    def _execute_parallel_evaluation(self, task: EvaluationTask, 
                                   judges: List[JudgeAgent]) -> List[EvaluationResult]:
        """Execute evaluation in parallel across selected judges"""
        results = []
        
        with ThreadPoolExecutor(max_workers=len(judges)) as executor:
            # Submit evaluation tasks to judges
            future_to_judge = {
                executor.submit(self._evaluate_single_dimension, task, judge, criterion): (judge, criterion)
                for judge in judges
                for criterion in task.evaluation_criteria
                if not criterion.dimension or criterion.dimension in judge.specialization
            }
            
            # Collect results with timeout
            for future in as_completed(future_to_judge, timeout=self.evaluation_timeout):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    judge, criterion = future_to_judge[future]
                    self.logger.error(f"Evaluation failed for judge {judge.agent_id}, dimension {criterion.dimension}: {e}")
        
        return results
    
    def _evaluate_single_dimension(self, task: EvaluationTask, judge: JudgeAgent, 
                                  criterion: EvaluationCriteria) -> Optional[EvaluationResult]:
        """Evaluate single dimension with specific judge"""
        start_time = time.time()
        
        try:
            # Perform automated security analysis first
            content_str = json.dumps(task.content_to_evaluate) if task.content_to_evaluate else ""
            security_analysis = self.security_engine.analyze_content_security(content_str)
            
            # Simulate judge evaluation (in real implementation, this would call actual judge agent)
            base_score = 0.8  # Simulated evaluation score
            
            # Adjust score based on security findings
            if security_analysis['threats']:
                threat_penalty = len(security_analysis['threats']) * 0.1
                base_score = max(0.0, base_score - threat_penalty)
            
            # Create evaluation result
            result = EvaluationResult(
                task_id=task.task_id,
                judge_id=judge.agent_id,
                dimension=criterion.dimension,
                score=base_score,
                confidence=security_analysis['confidence'] or 0.7,
                rationale=f"Evaluated {criterion.dimension.value} with security analysis",
                security_findings=[{
                    'threats': [t.value for t in security_analysis['threats']],
                    'severity': security_analysis['severity'].value,
                    'patterns': security_analysis['patterns_matched']
                }],
                threats_detected=security_analysis['threats'],
                severity=security_analysis['severity'],
                processing_time=time.time() - start_time
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Single dimension evaluation failed: {e}")
            return None
    
    def _generate_council_decision(self, task: EvaluationTask, 
                                 results: List[EvaluationResult]) -> CouncilDecision:
        """Generate final council decision based on individual results"""
        if not results:
            return CouncilDecision(
                task_id=task.task_id,
                overall_score=0.0,
                confidence=0.0,
                consensus_reached=False,
                security_verdict="REVIEW_REQUIRED"
            )
        
        # Calculate weighted average scores
        total_weight = sum(self.judges[r.judge_id].effective_weight for r in results)
        weighted_score = sum(
            r.score * self.judges[r.judge_id].effective_weight 
            for r in results
        ) / total_weight if total_weight > 0 else 0.0
        
        # Calculate confidence based on result consistency
        consistency_analysis = self.security_engine.validate_evaluation_consistency(results)
        
        # Determine consensus
        consensus_reached = consistency_analysis['consistent'] and len(results) >= self.min_judges_required
        
        # Aggregate security findings
        all_threats = []
        critical_findings = []
        
        for result in results:
            all_threats.extend(result.threats_detected)
            if result.severity in [EvaluationSeverity.HIGH, EvaluationSeverity.CRITICAL]:
                critical_findings.extend(result.security_findings)
        
        # Determine security verdict
        if any(t in [ThreatCategory.HARMFUL_CONTENT, ThreatCategory.SECURITY_VULNERABILITY] for t in all_threats):
            security_verdict = "UNSAFE"
        elif any(t in [ThreatCategory.MISINFORMATION, ThreatCategory.BIAS_AMPLIFICATION] for t in all_threats):
            security_verdict = "REVIEW_REQUIRED"
        else:
            security_verdict = "SAFE"
        
        # Generate recommendations
        recommendations = []
        if not consensus_reached:
            recommendations.append("Additional review recommended due to lack of consensus")
        if critical_findings:
            recommendations.append("Critical security findings require immediate attention")
        
        return CouncilDecision(
            task_id=task.task_id,
            overall_score=weighted_score,
            confidence=1.0 - consistency_analysis['variance'],
            consensus_reached=consensus_reached,
            majority_opinion=f"Average score: {weighted_score:.2f}",
            security_verdict=security_verdict,
            critical_findings=critical_findings,
            recommendations=recommendations,
            participating_judges=[r.judge_id for r in results]
        )


# Global multi-agent evaluation manager
multi_agent_evaluation_security = MultiAgentEvaluationSecurityManager()


def create_evaluation_council(judge_configs: List[Dict[str, Any]]) -> List[str]:
    """Convenience function to create evaluation council"""
    judge_ids = []
    
    try:
        for config in judge_configs:
            judge = JudgeAgent(
                agent_name=config.get('name', ''),
                role=JudgeRole(config.get('role', 'primary_evaluator')),
                specialization=[EvaluationDimension(d) for d in config.get('specialization', [])],
                trust_score=config.get('trust_score', 0.8),
                expertise_level=config.get('expertise_level', 0.8)
            )
            
            if multi_agent_evaluation_security.register_judge(judge):
                judge_ids.append(judge.agent_id)
        
        return judge_ids
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Council creation failed: {e}")
        return []


def evaluate_content_security(content: Any, criteria: List[Dict[str, Any]], 
                            requester_id: str) -> Optional[CouncilDecision]:
    """Convenience function to evaluate content with security analysis"""
    try:
        # Convert criteria dictionaries to EvaluationCriteria objects
        eval_criteria = []
        for crit in criteria:
            eval_criteria.append(EvaluationCriteria(
                dimension=EvaluationDimension(crit.get('dimension', 'accuracy')),
                weight=crit.get('weight', 1.0),
                threshold=crit.get('threshold', 0.7),
                description=crit.get('description', ''),
                security_critical=crit.get('security_critical', False)
            ))
        
        # Create evaluation task
        task = EvaluationTask(
            content_to_evaluate=content,
            evaluation_criteria=eval_criteria,
            requester_id=requester_id
        )
        
        # Submit and execute evaluation
        if multi_agent_evaluation_security.submit_evaluation_task(task):
            return multi_agent_evaluation_security.execute_council_evaluation(task.task_id)
        
        return None
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Content evaluation failed: {e}")
        return None
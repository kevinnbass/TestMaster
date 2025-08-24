"""
Pattern Consolidation Engine
Eliminates redundancy and consolidates testing patterns for optimal efficiency.
"""

import os
import re
import ast
import json
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime
import time
import difflib
from enum import Enum


class ConsolidationAction(Enum):
    """Types of consolidation actions"""
    MERGE = "merge"
    DEDUPLICATE = "deduplicate"
    EXTRACT_COMMON = "extract_common"
    REFACTOR = "refactor"
    OPTIMIZE = "optimize"
    ELIMINATE = "eliminate"


class ConsolidationRisk(Enum):
    """Risk levels for consolidation actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PatternSignature:
    """Unique signature of a test pattern"""
    pattern_hash: str
    pattern_type: str
    structure_hash: str
    semantic_hash: str
    complexity_score: float
    usage_locations: List[str] = field(default_factory=list)
    
    @classmethod
    def from_code(cls, code: str, pattern_type: str, location: str) -> 'PatternSignature':
        """Create signature from code block"""
        # Normalize code for comparison
        normalized = re.sub(r'\s+', ' ', code.strip())
        pattern_hash = hashlib.md5(normalized.encode()).hexdigest()
        
        # Create structural hash (without variable names)
        structure_normalized = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', 'VAR', normalized)
        structure_hash = hashlib.md5(structure_normalized.encode()).hexdigest()
        
        # Create semantic hash (without literals)
        semantic_normalized = re.sub(r'[\'"]\w+[\'"]', 'STR', structure_normalized)
        semantic_normalized = re.sub(r'\b\d+\b', 'NUM', semantic_normalized)
        semantic_hash = hashlib.md5(semantic_normalized.encode()).hexdigest()
        
        # Calculate complexity
        complexity = len(code.split('\n')) * 0.5 + code.count('if ') * 0.3 + code.count('for ') * 0.4
        
        return cls(
            pattern_hash=pattern_hash,
            pattern_type=pattern_type,
            structure_hash=structure_hash,
            semantic_hash=semantic_hash,
            complexity_score=complexity,
            usage_locations=[location]
        )


@dataclass
class ConsolidationOpportunity:
    """Represents an opportunity for pattern consolidation"""
    opportunity_id: str
    action_type: ConsolidationAction
    risk_level: ConsolidationRisk
    affected_patterns: List[PatternSignature]
    affected_files: List[str]
    estimated_savings: Dict[str, float]  # lines, complexity, maintenance_effort
    implementation_effort: float
    description: str
    detailed_plan: List[str] = field(default_factory=list)
    
    @property
    def roi_score(self) -> float:
        """Calculate return on investment score"""
        total_savings = sum(self.estimated_savings.values())
        if self.implementation_effort == 0:
            return float('inf')
        return total_savings / self.implementation_effort


@dataclass
class ConsolidationPlan:
    """Complete consolidation plan with prioritized opportunities"""
    opportunities: List[ConsolidationOpportunity]
    total_estimated_savings: Dict[str, float]
    implementation_roadmap: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    validation_requirements: List[str]


class CodePatternExtractor:
    """Extract patterns from code for consolidation analysis"""
    
    def __init__(self):
        self.extracted_patterns = defaultdict(list)
        self.pattern_signatures = {}
    
    def extract_function_patterns(self, file_path: str, content: str) -> List[PatternSignature]:
        """Extract function-level patterns"""
        patterns = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function code
                    func_lines = content.split('\n')[node.lineno-1:node.end_lineno if hasattr(node, 'end_lineno') else node.lineno+10]
                    func_code = '\n'.join(func_lines)
                    
                    # Determine pattern type
                    pattern_type = self._classify_function_pattern(node, func_code)
                    
                    if pattern_type:
                        signature = PatternSignature.from_code(
                            func_code, pattern_type, f"{file_path}:{node.lineno}"
                        )
                        patterns.append(signature)
            
            return patterns
            
        except SyntaxError:
            return []
    
    def extract_class_patterns(self, file_path: str, content: str) -> List[PatternSignature]:
        """Extract class-level patterns"""
        patterns = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Extract class code
                    class_lines = content.split('\n')[node.lineno-1:node.end_lineno if hasattr(node, 'end_lineno') else node.lineno+50]
                    class_code = '\n'.join(class_lines)
                    
                    # Determine pattern type
                    pattern_type = self._classify_class_pattern(node, class_code)
                    
                    if pattern_type:
                        signature = PatternSignature.from_code(
                            class_code, pattern_type, f"{file_path}:{node.lineno}"
                        )
                        patterns.append(signature)
            
            return patterns
            
        except SyntaxError:
            return []
    
    def extract_import_patterns(self, file_path: str, content: str) -> List[PatternSignature]:
        """Extract import patterns"""
        patterns = []
        
        # Find import blocks
        import_blocks = []
        current_block = []
        
        for line_num, line in enumerate(content.split('\n'), 1):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')) and not stripped.startswith('#'):
                current_block.append((line_num, line))
            elif current_block and stripped and not stripped.startswith('#'):
                # End of import block
                if len(current_block) > 1:  # Only consider multi-line import blocks
                    import_code = '\n'.join([line for _, line in current_block])
                    signature = PatternSignature.from_code(
                        import_code, 'import_block', f"{file_path}:{current_block[0][0]}"
                    )
                    patterns.append(signature)
                current_block = []
        
        return patterns
    
    def _classify_function_pattern(self, node: ast.FunctionDef, code: str) -> Optional[str]:
        """Classify function pattern type"""
        func_name = node.name.lower()
        
        if func_name.startswith('test_'):
            if 'mock' in code.lower():
                return 'test_function_with_mocks'
            elif 'assert' in code:
                return 'test_function_with_assertions'
            elif 'fixture' in code.lower():
                return 'test_function_with_fixtures'
            else:
                return 'test_function_basic'
        
        elif func_name.startswith('setup') or func_name.startswith('teardown'):
            return 'setup_teardown_function'
        
        elif 'heal' in func_name or 'fix' in func_name:
            return 'self_healing_function'
        
        elif 'verify' in func_name or 'validate' in func_name:
            return 'verification_function'
        
        elif 'generate' in func_name or 'create' in func_name:
            return 'generation_function'
        
        elif len(node.decorator_list) > 0:
            decorator_names = [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
            if any('pytest' in dec for dec in decorator_names):
                return 'pytest_decorated_function'
        
        return None
    
    def _classify_class_pattern(self, node: ast.ClassDef, code: str) -> Optional[str]:
        """Classify class pattern type"""
        class_name = node.name.lower()
        
        if class_name.startswith('test'):
            if 'unittest' in code.lower():
                return 'unittest_test_class'
            elif 'pytest' in code.lower():
                return 'pytest_test_class'
            else:
                return 'generic_test_class'
        
        elif 'mock' in class_name:
            return 'mock_class'
        
        elif 'fixture' in class_name:
            return 'fixture_class'
        
        elif 'framework' in class_name:
            return 'framework_class'
        
        return None


class PatternConsolidationAnalyzer:
    """Analyze patterns for consolidation opportunities"""
    
    def __init__(self):
        self.similarity_threshold = 0.85
        self.complexity_threshold = 10.0
        self.usage_threshold = 3
    
    def find_exact_duplicates(self, patterns: List[PatternSignature]) -> List[ConsolidationOpportunity]:
        """Find exact duplicate patterns"""
        opportunities = []
        hash_groups = defaultdict(list)
        
        # Group by pattern hash
        for pattern in patterns:
            hash_groups[pattern.pattern_hash].append(pattern)
        
        # Find duplicates
        for pattern_hash, group in hash_groups.items():
            if len(group) > 1:
                # Calculate savings
                total_lines = sum(p.complexity_score * 2 for p in group)  # Estimate lines
                saved_lines = total_lines - (group[0].complexity_score * 2)  # Keep one copy
                
                opportunity = ConsolidationOpportunity(
                    opportunity_id=f"duplicate_{pattern_hash[:8]}",
                    action_type=ConsolidationAction.DEDUPLICATE,
                    risk_level=ConsolidationRisk.LOW,
                    affected_patterns=group,
                    affected_files=list(set(sum([p.usage_locations for p in group], []))),
                    estimated_savings={
                        'lines_of_code': saved_lines,
                        'maintenance_effort': len(group) - 1,
                        'complexity_reduction': sum(p.complexity_score for p in group[1:])
                    },
                    implementation_effort=2.0 * len(group),
                    description=f"Exact duplicate pattern found in {len(group)} locations: {group[0].pattern_type}"
                )
                
                opportunities.append(opportunity)
        
        return opportunities
    
    def find_structural_similarities(self, patterns: List[PatternSignature]) -> List[ConsolidationOpportunity]:
        """Find structurally similar patterns"""
        opportunities = []
        structure_groups = defaultdict(list)
        
        # Group by structure hash
        for pattern in patterns:
            structure_groups[pattern.structure_hash].append(pattern)
        
        # Find similar structures
        for structure_hash, group in structure_groups.items():
            if len(group) > self.usage_threshold and len(set(p.pattern_hash for p in group)) > 1:
                # Calculate potential savings
                avg_complexity = sum(p.complexity_score for p in group) / len(group)
                total_complexity = sum(p.complexity_score for p in group)
                potential_savings = total_complexity - avg_complexity
                
                if potential_savings > self.complexity_threshold:
                    opportunity = ConsolidationOpportunity(
                        opportunity_id=f"structure_{structure_hash[:8]}",
                        action_type=ConsolidationAction.EXTRACT_COMMON,
                        risk_level=ConsolidationRisk.MEDIUM,
                        affected_patterns=group,
                        affected_files=list(set(sum([p.usage_locations for p in group], []))),
                        estimated_savings={
                            'complexity_reduction': potential_savings,
                            'maintenance_effort': len(group) * 0.5,
                            'lines_of_code': potential_savings * 1.5
                        },
                        implementation_effort=5.0 + len(group) * 0.5,
                        description=f"Structurally similar patterns ({len(group)} instances) can be consolidated"
                    )
                    
                    opportunities.append(opportunity)
        
        return opportunities
    
    def find_semantic_opportunities(self, patterns: List[PatternSignature]) -> List[ConsolidationOpportunity]:
        """Find semantic consolidation opportunities"""
        opportunities = []
        semantic_groups = defaultdict(list)
        
        # Group by semantic hash
        for pattern in patterns:
            semantic_groups[pattern.semantic_hash].append(pattern)
        
        # Find semantic opportunities
        for semantic_hash, group in semantic_groups.items():
            if len(group) >= self.usage_threshold:
                pattern_types = set(p.pattern_type for p in group)
                
                if len(pattern_types) == 1 and len(set(p.pattern_hash for p in group)) > 1:
                    # Same semantic pattern, different implementations
                    
                    total_complexity = sum(p.complexity_score for p in group)
                    min_complexity = min(p.complexity_score for p in group)
                    potential_savings = total_complexity - (min_complexity + len(group) * 0.5)
                    
                    if potential_savings > 5.0:
                        opportunity = ConsolidationOpportunity(
                            opportunity_id=f"semantic_{semantic_hash[:8]}",
                            action_type=ConsolidationAction.REFACTOR,
                            risk_level=ConsolidationRisk.MEDIUM,
                            affected_patterns=group,
                            affected_files=list(set(sum([p.usage_locations for p in group], []))),
                            estimated_savings={
                                'complexity_reduction': potential_savings,
                                'standardization_benefit': len(group) * 2.0,
                                'maintenance_effort': len(group) * 0.7
                            },
                            implementation_effort=8.0 + len(group) * 0.8,
                            description=f"Semantically similar {group[0].pattern_type} patterns can be refactored into common implementation"
                        )
                        
                        opportunities.append(opportunity)
        
        return opportunities
    
    def find_obsolete_patterns(self, patterns: List[PatternSignature]) -> List[ConsolidationOpportunity]:
        """Find potentially obsolete patterns"""
        opportunities = []
        
        # Patterns with very low usage and high complexity might be obsolete
        for pattern in patterns:
            usage_count = len(pattern.usage_locations)
            
            if usage_count == 1 and pattern.complexity_score > 20.0:
                # High complexity, single usage - might be obsolete
                opportunity = ConsolidationOpportunity(
                    opportunity_id=f"obsolete_{pattern.pattern_hash[:8]}",
                    action_type=ConsolidationAction.ELIMINATE,
                    risk_level=ConsolidationRisk.HIGH,
                    affected_patterns=[pattern],
                    affected_files=pattern.usage_locations,
                    estimated_savings={
                        'complexity_reduction': pattern.complexity_score,
                        'lines_of_code': pattern.complexity_score * 1.5,
                        'maintenance_effort': 3.0
                    },
                    implementation_effort=2.0,
                    description=f"Potentially obsolete {pattern.pattern_type} with high complexity and single usage"
                )
                
                opportunities.append(opportunity)
        
        return opportunities


class ConsolidationPlanGenerator:
    """Generate comprehensive consolidation plans"""
    
    def __init__(self):
        self.risk_weights = {
            ConsolidationRisk.LOW: 1.0,
            ConsolidationRisk.MEDIUM: 0.7,
            ConsolidationRisk.HIGH: 0.4,
            ConsolidationRisk.CRITICAL: 0.1
        }
    
    def generate_plan(self, opportunities: List[ConsolidationOpportunity]) -> ConsolidationPlan:
        """Generate comprehensive consolidation plan"""
        
        # Sort by ROI adjusted for risk
        prioritized_opportunities = sorted(
            opportunities,
            key=lambda x: x.roi_score * self.risk_weights[x.risk_level],
            reverse=True
        )
        
        # Calculate total savings
        total_savings = defaultdict(float)
        for opp in opportunities:
            for metric, value in opp.estimated_savings.items():
                total_savings[metric] += value
        
        # Generate implementation roadmap
        roadmap = self._generate_roadmap(prioritized_opportunities)
        
        # Assess risks
        risk_assessment = self._assess_risks(opportunities)
        
        # Define validation requirements
        validation_requirements = self._define_validation_requirements(opportunities)
        
        return ConsolidationPlan(
            opportunities=prioritized_opportunities,
            total_estimated_savings=dict(total_savings),
            implementation_roadmap=roadmap,
            risk_assessment=risk_assessment,
            validation_requirements=validation_requirements
        )
    
    def _generate_roadmap(self, opportunities: List[ConsolidationOpportunity]) -> List[Dict[str, Any]]:
        """Generate phased implementation roadmap"""
        roadmap = []
        
        # Phase 1: Low-risk, high-impact changes
        phase1 = [opp for opp in opportunities 
                 if opp.risk_level == ConsolidationRisk.LOW and opp.roi_score > 2.0]
        
        if phase1:
            roadmap.append({
                'phase': 1,
                'title': 'Low-Risk Quick Wins',
                'opportunities': [opp.opportunity_id for opp in phase1],
                'estimated_duration': '1-2 weeks',
                'prerequisites': [],
                'deliverables': ['Exact duplicate removal', 'Simple pattern deduplication']
            })
        
        # Phase 2: Medium-risk structural improvements
        phase2 = [opp for opp in opportunities 
                 if opp.risk_level == ConsolidationRisk.MEDIUM and opp.roi_score > 1.0]
        
        if phase2:
            roadmap.append({
                'phase': 2,
                'title': 'Structural Consolidation',
                'opportunities': [opp.opportunity_id for opp in phase2],
                'estimated_duration': '3-4 weeks',
                'prerequisites': ['Phase 1 completion', 'Testing framework setup'],
                'deliverables': ['Common pattern extraction', 'Structural refactoring']
            })
        
        # Phase 3: High-impact complex changes
        phase3 = [opp for opp in opportunities 
                 if opp.risk_level in [ConsolidationRisk.HIGH, ConsolidationRisk.CRITICAL] and opp.roi_score > 0.5]
        
        if phase3:
            roadmap.append({
                'phase': 3,
                'title': 'Advanced Optimization',
                'opportunities': [opp.opportunity_id for opp in phase3],
                'estimated_duration': '2-3 weeks',
                'prerequisites': ['Phase 1-2 completion', 'Comprehensive testing'],
                'deliverables': ['Advanced refactoring', 'Obsolete pattern elimination']
            })
        
        return roadmap
    
    def _assess_risks(self, opportunities: List[ConsolidationOpportunity]) -> Dict[str, Any]:
        """Assess overall consolidation risks"""
        risk_counts = Counter([opp.risk_level for opp in opportunities])
        
        total_effort = sum(opp.implementation_effort for opp in opportunities)
        high_risk_effort = sum(opp.implementation_effort for opp in opportunities 
                              if opp.risk_level in [ConsolidationRisk.HIGH, ConsolidationRisk.CRITICAL])
        
        return {
            'risk_distribution': {risk.value: count for risk, count in risk_counts.items()},
            'total_implementation_effort': total_effort,
            'high_risk_effort_percentage': (high_risk_effort / total_effort * 100) if total_effort > 0 else 0,
            'recommended_approach': 'Phased implementation with extensive testing' if high_risk_effort > total_effort * 0.3 else 'Standard implementation',
            'critical_risks': [
                opp.description for opp in opportunities 
                if opp.risk_level == ConsolidationRisk.CRITICAL
            ]
        }
    
    def _define_validation_requirements(self, opportunities: List[ConsolidationOpportunity]) -> List[str]:
        """Define validation requirements for consolidation"""
        requirements = [
            "Comprehensive test suite execution before and after consolidation",
            "Code coverage maintenance or improvement",
            "Performance regression testing",
            "Integration testing with dependent systems"
        ]
        
        # Add specific requirements based on opportunity types
        action_types = set(opp.action_type for opp in opportunities)
        
        if ConsolidationAction.MERGE in action_types:
            requirements.append("Merge conflict resolution validation")
        
        if ConsolidationAction.REFACTOR in action_types:
            requirements.append("API compatibility validation")
        
        if ConsolidationAction.ELIMINATE in action_types:
            requirements.append("Dead code analysis and dependency validation")
        
        if any(opp.risk_level == ConsolidationRisk.CRITICAL for opp in opportunities):
            requirements.extend([
                "Staged rollout with rollback capability",
                "Extended monitoring period post-implementation",
                "Manual review of all critical changes"
            ])
        
        return requirements


class PatternConsolidationEngine:
    """Main engine for pattern consolidation"""
    
    def __init__(self):
        self.extractor = CodePatternExtractor()
        self.analyzer = PatternConsolidationAnalyzer()
        self.plan_generator = ConsolidationPlanGenerator()
        self.all_patterns = []
        self.consolidation_metrics = {
            'files_analyzed': 0,
            'patterns_extracted': 0,
            'opportunities_found': 0,
            'processing_time': 0
        }
    
    def analyze_directory(self, directory_path: str, file_pattern: str = "*.py") -> None:
        """Analyze directory for consolidation patterns"""
        start_time = time.time()
        
        directory = Path(directory_path)
        files = list(directory.glob(f"**/{file_pattern}"))
        
        print(f"Analyzing {len(files)} files for consolidation patterns...")
        
        for i, file_path in enumerate(files):
            print(f"Processing [{i+1}/{len(files)}]: {file_path.name}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract patterns
                function_patterns = self.extractor.extract_function_patterns(str(file_path), content)
                class_patterns = self.extractor.extract_class_patterns(str(file_path), content)
                import_patterns = self.extractor.extract_import_patterns(str(file_path), content)
                
                self.all_patterns.extend(function_patterns + class_patterns + import_patterns)
                self.consolidation_metrics['files_analyzed'] += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        self.consolidation_metrics['patterns_extracted'] = len(self.all_patterns)
        self.consolidation_metrics['processing_time'] = time.time() - start_time
        
        print(f"Extracted {len(self.all_patterns)} patterns from {len(files)} files")
    
    def find_consolidation_opportunities(self) -> ConsolidationPlan:
        """Find and prioritize consolidation opportunities"""
        print("Analyzing patterns for consolidation opportunities...")
        
        all_opportunities = []
        
        # Find different types of opportunities
        duplicates = self.analyzer.find_exact_duplicates(self.all_patterns)
        structural = self.analyzer.find_structural_similarities(self.all_patterns)
        semantic = self.analyzer.find_semantic_opportunities(self.all_patterns)
        obsolete = self.analyzer.find_obsolete_patterns(self.all_patterns)
        
        all_opportunities.extend(duplicates + structural + semantic + obsolete)
        self.consolidation_metrics['opportunities_found'] = len(all_opportunities)
        
        print(f"Found {len(all_opportunities)} consolidation opportunities:")
        print(f"  - Exact duplicates: {len(duplicates)}")
        print(f"  - Structural similarities: {len(structural)}")
        print(f"  - Semantic opportunities: {len(semantic)}")
        print(f"  - Obsolete patterns: {len(obsolete)}")
        
        # Generate consolidation plan
        plan = self.plan_generator.generate_plan(all_opportunities)
        
        return plan
    
    def export_consolidation_report(self, plan: ConsolidationPlan, output_path: str) -> None:
        """Export consolidation analysis report"""
        report_data = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_metrics': self.consolidation_metrics
            },
            'summary': {
                'total_opportunities': len(plan.opportunities),
                'estimated_savings': plan.total_estimated_savings,
                'implementation_phases': len(plan.implementation_roadmap)
            },
            'opportunities': [
                {
                    'id': opp.opportunity_id,
                    'action_type': opp.action_type.value,
                    'risk_level': opp.risk_level.value,
                    'roi_score': opp.roi_score,
                    'affected_files_count': len(opp.affected_files),
                    'estimated_savings': opp.estimated_savings,
                    'implementation_effort': opp.implementation_effort,
                    'description': opp.description
                }
                for opp in plan.opportunities
            ],
            'implementation_roadmap': plan.implementation_roadmap,
            'risk_assessment': plan.risk_assessment,
            'validation_requirements': plan.validation_requirements
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"Consolidation report exported to: {output_path}")


# Testing framework
class ConsolidationEngineTestFramework:
    """Testing framework for consolidation engine"""
    
    def test_pattern_extraction(self, directory_path: str) -> bool:
        """Test pattern extraction functionality"""
        try:
            extractor = CodePatternExtractor()
            
            # Test with a Python file
            test_code = '''
def test_example():
    assert True

class TestExample:
    def test_method(self):
        pass

import os
from pathlib import Path
'''
            
            function_patterns = extractor.extract_function_patterns("test.py", test_code)
            class_patterns = extractor.extract_class_patterns("test.py", test_code)
            import_patterns = extractor.extract_import_patterns("test.py", test_code)
            
            assert len(function_patterns) > 0
            assert len(class_patterns) > 0
            assert len(import_patterns) > 0
            
            return True
        except Exception as e:
            print(f"Pattern extraction test failed: {e}")
            return False
    
    def test_consolidation_analysis(self, directory_path: str) -> bool:
        """Test consolidation analysis"""
        try:
            analyzer = PatternConsolidationAnalyzer()
            
            # Create test patterns
            pattern1 = PatternSignature.from_code("def test_a(): assert True", "test_function", "file1.py:1")
            pattern2 = PatternSignature.from_code("def test_a(): assert True", "test_function", "file2.py:1")  # Duplicate
            pattern3 = PatternSignature.from_code("def test_b(): assert False", "test_function", "file3.py:1")  # Similar
            
            test_patterns = [pattern1, pattern2, pattern3]
            
            duplicates = analyzer.find_exact_duplicates(test_patterns)
            structural = analyzer.find_structural_similarities(test_patterns)
            
            assert isinstance(duplicates, list)
            assert isinstance(structural, list)
            
            return True
        except Exception as e:
            print(f"Consolidation analysis test failed: {e}")
            return False
    
    def test_plan_generation(self) -> bool:
        """Test plan generation"""
        try:
            generator = ConsolidationPlanGenerator()
            
            # Create test opportunity
            test_opportunity = ConsolidationOpportunity(
                opportunity_id="test_001",
                action_type=ConsolidationAction.DEDUPLICATE,
                risk_level=ConsolidationRisk.LOW,
                affected_patterns=[],
                affected_files=["test.py"],
                estimated_savings={'lines': 10, 'complexity': 5},
                implementation_effort=2.0,
                description="Test opportunity"
            )
            
            plan = generator.generate_plan([test_opportunity])
            
            assert isinstance(plan, ConsolidationPlan)
            assert len(plan.opportunities) == 1
            assert len(plan.implementation_roadmap) > 0
            
            return True
        except Exception as e:
            print(f"Plan generation test failed: {e}")
            return False
    
    def run_comprehensive_tests(self, directory_path: str) -> Dict[str, bool]:
        """Run all consolidation engine tests"""
        tests = [
            ('test_pattern_extraction', [directory_path]),
            ('test_consolidation_analysis', [directory_path]),
            ('test_plan_generation', [])
        ]
        
        results = {}
        for test_name, args in tests:
            try:
                result = getattr(self, test_name)(*args)
                results[test_name] = result
                print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                results[test_name] = False
                print(f"❌ {test_name}: FAILED - {e}")
        
        return results


# Main execution
if __name__ == "__main__":
    import sys
    
    # Default to current directory
    directory_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print("🔧 Pattern Consolidation Engine")
    print(f"Analysis directory: {directory_path}")
    
    # Run tests
    framework = ConsolidationEngineTestFramework()
    results = framework.run_comprehensive_tests(directory_path)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All consolidation engine tests passed!")
        
        # Run actual consolidation analysis
        print("\n🚀 Running pattern consolidation analysis...")
        engine = PatternConsolidationEngine()
        engine.analyze_directory(directory_path)
        plan = engine.find_consolidation_opportunities()
        
        # Export report
        output_path = f"consolidation_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        engine.export_consolidation_report(plan, output_path)
        
        print(f"\n📈 Consolidation Analysis Complete:")
        print(f"  Patterns analyzed: {engine.consolidation_metrics['patterns_extracted']}")
        print(f"  Opportunities found: {engine.consolidation_metrics['opportunities_found']}")
        print(f"  Estimated total savings: {plan.total_estimated_savings}")
        print(f"  Implementation phases: {len(plan.implementation_roadmap)}")
    else:
        print("❌ Some tests failed. Check the output above.")
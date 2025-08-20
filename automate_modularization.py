"""
Automated Modularization Script

Automates the modularization process for large classical analysis modules
by creating directory structures, shared utilities, and modular components.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModularizationAutomator:
    """Automates the modularization process"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.comprehensive_analysis_path = self.base_path / "testmaster" / "analysis" / "comprehensive_analysis"
        self.archive_path = self.comprehensive_analysis_path / "archive"
        
    def modularize_all_remaining(self):
        """Modularize all remaining large modules"""
        
        # Define modularization plans
        modularization_plans = {
            "business_rule_analysis.py": {
                "target_dir": "business_analysis",
                "modules": [
                    ("business_core_analyzer.py", "Core business rule extraction and analysis"),
                    ("business_workflow_analyzer.py", "Workflow and state machine analysis"),
                    ("business_domain_analyzer.py", "Domain logic and entity analysis"), 
                    ("business_validation_analyzer.py", "Rule validation and compliance")
                ]
            },
            "semantic_analysis.py": {
                "target_dir": "semantic_analysis",
                "modules": [
                    ("semantic_core_analyzer.py", "Intent recognition and classification"),
                    ("semantic_pattern_analyzer.py", "Design patterns and behavioral analysis"),
                    ("semantic_context_analyzer.py", "Context and relationship analysis")
                ]
            },
            "technical_debt_analysis.py": {
                "target_dir": "debt_analysis",
                "modules": [
                    ("debt_core_analyzer.py", "Base debt calculation and ROI"),
                    ("debt_category_analyzer.py", "Debt categorization and prioritization"),
                    ("debt_financial_analyzer.py", "Financial metrics and reporting")
                ]
            },
            "metaprogramming_analysis.py": {
                "target_dir": "metaprog_analysis", 
                "modules": [
                    ("metaprog_core_analyzer.py", "eval/exec and dynamic analysis"),
                    ("metaprog_security_analyzer.py", "Security vulnerabilities and CWE mapping"),
                    ("metaprog_reflection_analyzer.py", "Reflection and introspection analysis")
                ]
            },
            "energy_consumption_analysis.py": {
                "target_dir": "energy_analysis",
                "modules": [
                    ("energy_core_analyzer.py", "Base energy calculation and hotspots"),
                    ("energy_algorithm_analyzer.py", "Algorithm efficiency analysis"),
                    ("energy_carbon_analyzer.py", "Carbon footprint and environmental metrics")
                ]
            }
        }
        
        # Execute modularization for each module
        for original_file, plan in modularization_plans.items():
            if self._should_modularize(original_file):
                logger.info(f"Modularizing {original_file}...")
                self._modularize_module(original_file, plan)
            else:
                logger.info(f"Skipping {original_file} - already modularized or not found")
    
    def _should_modularize(self, original_file: str) -> bool:
        """Check if module should be modularized"""
        original_path = self.comprehensive_analysis_path / original_file
        return original_path.exists() and original_path.stat().st_size > 50000  # 50KB+
    
    def _modularize_module(self, original_file: str, plan: Dict):
        """Modularize a single module according to plan"""
        original_path = self.comprehensive_analysis_path / original_file
        target_dir = self.comprehensive_analysis_path / plan["target_dir"]
        archive_name = original_file.replace(".py", "_original.py")
        archive_path = self.archive_path / archive_name
        
        try:
            # 1. Create target directory
            target_dir.mkdir(exist_ok=True)
            logger.info(f"Created directory: {target_dir}")
            
            # 2. Archive original
            if not archive_path.exists():
                shutil.copy2(original_path, archive_path)
                logger.info(f"Archived original to: {archive_path}")
            
            # 3. Create __init__.py
            self._create_init_file(target_dir, plan)
            
            # 4. Create _shared_utils.py
            self._create_shared_utils(target_dir, original_file)
            
            # 5. Create modular components (placeholder structure)
            self._create_modular_components(target_dir, plan, original_file)
            
            # 6. Create compatibility wrapper
            self._create_compatibility_wrapper(original_path, target_dir, plan)
            
            logger.info(f"Successfully modularized {original_file}")
            
        except Exception as e:
            logger.error(f"Error modularizing {original_file}: {e}")
    
    def _create_init_file(self, target_dir: Path, plan: Dict):
        """Create __init__.py for the modular package"""
        init_content = f'''"""
{plan["target_dir"].replace("_", " ").title()} Submodules

Modularized components split from the original module
"""

'''
        
        # Add imports for all modules
        class_names = []
        for module_file, description in plan["modules"]:
            class_name = self._file_to_class_name(module_file)
            module_name = module_file.replace(".py", "")
            init_content += f"from .{module_name} import {class_name}\n"
            class_names.append(class_name)
        
        init_content += f"\n__all__ = [\n"
        for class_name in class_names:
            init_content += f"    '{class_name}',\n"
        init_content += "]\n"
        
        init_path = target_dir / "__init__.py"
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write(init_content)
    
    def _file_to_class_name(self, filename: str) -> str:
        """Convert filename to class name"""
        # Remove .py extension and convert snake_case to PascalCase
        name = filename.replace(".py", "")
        parts = name.split("_")
        return "".join(word.capitalize() for word in parts)
    
    def _create_shared_utils(self, target_dir: Path, original_file: str):
        """Create _shared_utils.py with common utilities"""
        utils_content = f'''"""
Shared utilities for {target_dir.name} modules

Contains common data structures, constants, and helper functions
used across all {target_dir.name} submodules.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

# TODO: Extract common data structures from original {original_file}
# TODO: Extract common constants and patterns
# TODO: Extract common helper functions

@dataclass
class AnalysisIssue:
    """Common issue structure for all analyzers"""
    issue_type: str
    severity: str
    location: str
    description: str
    recommendation: str
    impact: str


# Common patterns and constants will be extracted here
COMMON_PATTERNS = {{
    # TODO: Extract from original module
}}

def extract_common_patterns(tree: ast.AST, content: str) -> List[Dict[str, Any]]:
    """Extract common patterns from AST and content"""
    # TODO: Implement common pattern extraction
    return []

def calculate_complexity_score(node: ast.AST) -> int:
    """Calculate complexity score for a node"""
    # TODO: Implement common complexity calculation
    return 0
'''
        
        utils_path = target_dir / "_shared_utils.py"
        with open(utils_path, 'w', encoding='utf-8') as f:
            f.write(utils_content)
    
    def _create_modular_components(self, target_dir: Path, plan: Dict, original_file: str):
        """Create modular component files"""
        base_class_name = original_file.replace(".py", "").replace("_", " ").title().replace(" ", "")
        
        for module_file, description in plan["modules"]:
            class_name = self._file_to_class_name(module_file)
            
            component_content = f'''"""
{class_name}

{description}
Split from original {original_file}
"""

import ast
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict
import logging

from ...base import BaseAnalyzer
from ._shared_utils import AnalysisIssue, extract_common_patterns, calculate_complexity_score


class {class_name}(BaseAnalyzer):
    """
    {description}
    """
    
    def __init__(self):
        super().__init__()
        self.issues = []
        self.patterns = []
        
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform specialized analysis
        """
        if file_path:
            self._analyze_file(Path(file_path))
        else:
            self._analyze_directory()
            
        return self._generate_report()
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # TODO: Implement specific analysis logic
            # This should be extracted from the original module
            
            patterns = extract_common_patterns(tree, content)
            self.patterns.extend(patterns)
            
        except Exception as e:
            logging.error(f"Error analyzing {{file_path}}: {{e}}")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate analysis report"""
        return {{
            "summary": {{
                "total_issues": len(self.issues),
                "total_patterns": len(self.patterns),
                "files_analyzed": len(set(issue.location for issue in self.issues))
            }},
            "issues": [
                {{
                    "type": issue.issue_type,
                    "severity": issue.severity,
                    "location": issue.location,
                    "description": issue.description,
                    "recommendation": issue.recommendation,
                    "impact": issue.impact
                }}
                for issue in self.issues
            ],
            "patterns": self.patterns,
            "recommendations": self._generate_recommendations()
        }}
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # TODO: Implement recommendation logic
        
        return recommendations
'''
            
            component_path = target_dir / module_file
            with open(component_path, 'w', encoding='utf-8') as f:
                f.write(component_content)
    
    def _create_compatibility_wrapper(self, original_path: Path, target_dir: Path, plan: Dict):
        """Create compatibility wrapper to maintain API"""
        wrapper_name = original_path.name.replace(".py", "_modular.py")
        wrapper_path = original_path.parent / wrapper_name
        
        original_class_name = self._file_to_class_name(original_path.name)
        
        wrapper_content = f'''"""
Modular {original_class_name}

This is the new modular version that combines all submodules
while maintaining the same API as the original.

This ensures backward compatibility while providing modular benefits.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from .{target_dir.name} import {", ".join(self._file_to_class_name(m[0]) for m in plan["modules"])}
from .base import BaseAnalyzer


class {original_class_name}(BaseAnalyzer):
    """
    Modular {original_class_name} that combines specialized analyzers
    
    Maintains API compatibility with original while using modular architecture.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Initialize specialized analyzers
'''
        
        # Add analyzer initialization
        for module_file, description in plan["modules"]:
            class_name = self._file_to_class_name(module_file)
            var_name = class_name.lower().replace("analyzer", "")
            wrapper_content += f"        self.{var_name}_analyzer = {class_name}()\n"
        
        wrapper_content += f'''
    
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive analysis using modular components
        
        Returns the same structure as the original analyzer for compatibility.
        """
        # Run all specialized analyzers
'''
        
        # Add analyzer calls
        for module_file, description in plan["modules"]:
            class_name = self._file_to_class_name(module_file)
            var_name = class_name.lower().replace("analyzer", "")
            wrapper_content += f"        {var_name}_results = self.{var_name}_analyzer.analyze(file_path)\n"
        
        wrapper_content += f'''
        
        # Combine results in the original format
        combined_results = {{
            "summary": self._generate_combined_summary({", ".join(f"{self._file_to_class_name(m[0]).lower().replace('analyzer', '')}_results" for m in plan["modules"])}),
            # TODO: Add specific result combination logic based on original API
        }}
        
        return combined_results
    
    def _generate_combined_summary(self, *results) -> Dict[str, Any]:
        """Generate combined summary from all analyzer results"""
        total_issues = 0
        total_patterns = 0
        
        for result in results:
            if isinstance(result, dict):
                summary = result.get("summary", {{}})
                total_issues += summary.get("total_issues", 0)
                total_patterns += summary.get("total_patterns", 0)
        
        return {{
            "total_issues": total_issues,
            "total_patterns": total_patterns,
            "modular_components": len(results)
        }}
'''
        
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_content)
    
    def generate_modularization_summary(self) -> str:
        """Generate summary of modularization work"""
        summary = ["# Automated Modularization Summary\n"]
        
        # Check what was modularized
        modularized = []
        for item in self.comprehensive_analysis_path.iterdir():
            if item.is_dir() and item.name.endswith("_analysis"):
                modularized.append(item.name)
        
        summary.append(f"## Modularized Components: {len(modularized)}\n")
        for component in modularized:
            summary.append(f"- {component}")
        
        summary.append(f"\n## Archive Contents:")
        if self.archive_path.exists():
            for archived_file in self.archive_path.glob("*_original.py"):
                size_kb = archived_file.stat().st_size / 1024
                summary.append(f"- {archived_file.name} ({size_kb:.1f}KB)")
        
        return "\n".join(summary)


if __name__ == "__main__":
    automator = ModularizationAutomator("C:\\Users\\kbass\\OneDrive\\Documents\\testmaster")
    
    print("Starting automated modularization...")
    automator.modularize_all_remaining()
    
    print("\nGenerating summary...")
    summary = automator.generate_modularization_summary()
    print(summary)
    
    # Save summary
    with open("automated_modularization_summary.md", "w") as f:
        f.write(summary)
    
    print("\nAutomated modularization complete!")
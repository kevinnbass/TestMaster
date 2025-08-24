#!/usr/bin/env python3
"""
Run Technical Debt Analysis on TestMaster Infrastructure
Agent E - Hour 1 Infrastructure Debt Analysis
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime
import ast
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import math

# Direct import of the analyzer class components
@dataclass
class DebtItem:
    """Represents a single technical debt item"""
    type: str
    severity: str
    location: str
    description: str
    estimated_hours: float
    interest_rate: float  # How much this debt grows over time
    risk_factor: float
    remediation_strategy: str
    dependencies: List[str] = field(default_factory=list)
    business_impact: str = "low"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'type': self.type,
            'severity': self.severity,
            'location': self.location,
            'description': self.description,
            'estimated_hours': self.estimated_hours,
            'interest_rate': self.interest_rate,
            'risk_factor': self.risk_factor,
            'remediation_strategy': self.remediation_strategy,
            'dependencies': self.dependencies,
            'business_impact': self.business_impact
        }

def main():
    """Execute infrastructure debt analysis"""
    print("AGENT E - HOUR 1: INFRASTRUCTURE DEBT ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer for TestMaster codebase
    analyzer = TechnicalDebtAnalyzer()
    
    # Analyze the entire codebase
    print("Analyzing TestMaster infrastructure for technical debt...")
    results = analyzer.analyze_project(".")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"infrastructure_debt_analysis_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Display summary
    print(f"\nINFRASTRUCTURE DEBT ANALYSIS COMPLETE")
    print(f"{'='*60}")
    
    metrics = results.get('metrics', {})
    print(f"Total debt items: {metrics.get('total_items', 0)}")
    print(f"Total estimated hours: {metrics.get('total_hours', 0):,.1f}")
    print(f"Total cost: ${metrics.get('total_cost', 0):,.2f}")
    print(f"Interest per month: ${metrics.get('monthly_interest', 0):,.2f}")
    
    severity_dist = results.get('severity_distribution', {})
    if severity_dist:
        print(f"\nSeverity Distribution:")
        for severity, count in severity_dist.items():
            print(f"  {severity}: {count}")
    
    print(f"\nResults saved to: {output_file}")
    
    # Show top recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        print(f"\nTOP RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"  {i}. {rec}")
    
    return results

if __name__ == "__main__":
    main()
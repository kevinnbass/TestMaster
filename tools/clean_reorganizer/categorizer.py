"""
Clean Reorganizer: Categorizer
==============================

Maps `FileFacts` into categories using config-driven heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
from pathlib import Path
import json
import re

from .analyzer import FileFacts


@dataclass
class CategoryDecision:
    category: str
    confidence: float
    reasons: List[str]


class Categorizer:
    def __init__(self, root_dir: Path, config_path: Path) -> None:
        self.root_dir = root_dir
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict = json.load(f)
        self.rules: Dict = self.config.get("categories", {})

    def decide(self, facts: FileFacts) -> CategoryDecision:
        """Score file against category rules and choose the best category."""
        scores: Dict[str, Tuple[float, List[str]]] = {}
        path_str = str(facts.path.relative_to(self.root_dir)).lower()
        kw = facts.keywords
        imports = [i.lower() for i in facts.imports[:100]]

        for name, rule in list(self.rules.items())[:50]:
            score = 0.0
            reasons: List[str] = []

            # Path patterns (weight 0.4)
            for pat in rule.get("path_patterns", [])[:20]:
                try:
                    if re.search(pat, path_str, re.IGNORECASE):
                        score += 0.4
                        reasons.append(f"path~{pat}")
                        break
                except re.error:
                    continue

            # Keyword overlap (weight up to 0.3)
            rule_kws = set(rule.get("keywords", [])[:100])
            overlap = kw.intersection(rule_kws)
            if overlap:
                score += min(0.3, 0.01 * len(overlap))
                some = list(overlap)[:3]
                reasons.append(f"kw:{','.join(some)}")

            # Class patterns (weight up to 0.2)
            for pat in rule.get("class_patterns", [])[:20]:
                try:
                    matched = any(re.search(pat, c, re.IGNORECASE) for c in facts.classes[:100])
                    if matched:
                        score += 0.2
                        reasons.append(f"class~{pat}")
                        break
                except re.error:
                    continue

            # Import hints (weight up to 0.1)
            if rule_kws:
                hint = any(any(k in imp for k in rule_kws) for imp in imports)
                if hint:
                    score += 0.1
                    reasons.append("import-hint")

            scores[name] = (min(score, 1.0), reasons)

        if not scores:
            return CategoryDecision(category="utilities", confidence=0.1, reasons=["no-rules"])

        best = max(scores.items(), key=lambda x: x[1][0])
        best_cat, (best_score, best_reasons) = best
        # Fallback to utilities if classification confidence is too low
        if best_score < 0.25:
            return CategoryDecision(category="utilities", confidence=best_score, reasons=["low-score"] + best_reasons)
        return CategoryDecision(category=best_cat, confidence=best_score, reasons=best_reasons)



"""
Clean Reorganizer: Planner
==========================

Generates a reorganization plan (previewable) using analyzer + categorizer.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import json

from .analyzer import Analyzer, FileFacts
from .categorizer import Categorizer, CategoryDecision


@dataclass
class PlanItem:
    source: Path
    target: Path
    category: str
    confidence: float
    analysis: FileFacts


@dataclass
class PlanSummary:
    total: int
    by_category: Dict[str, int]


@dataclass
class Plan:
    moves: List[PlanItem]
    summary: PlanSummary


class Planner:
    def __init__(self, root_dir: Path, config_path: Optional[Path] = None) -> None:
        assert isinstance(root_dir, Path), "root_dir must be Path"
        self.root_dir = root_dir.resolve()
        self.analyzer = Analyzer(self.root_dir, config_path)
        cfg = self.analyzer._load_config(None)  # safe reuse to get path
        self.config_path = self.root_dir / "tools" / "clean_reorganizer" / "config" / "config.json"
        self.categorizer = Categorizer(self.root_dir, self.config_path)
        self.target_root_name: str = cfg.get("target_root", "organized_codebase")

    def _target_path_for(self, facts: FileFacts, decision: CategoryDecision) -> Path:
        tgt = self.root_dir / self.target_root_name
        for part in decision.category.split('/')[:5]:
            if part:
                tgt = tgt / part
        return tgt / facts.path.name

    def build_plan(self) -> Plan:
        files = self.analyzer.discover_python_files()
        moves: List[PlanItem] = []
        counts: Dict[str, int] = {}

        for p in files[:5000]:
            facts = self.analyzer.analyze_file(p)
            if facts is None:
                continue
            decision = self.categorizer.decide(facts)
            target = self._target_path_for(facts, decision)
            if target != facts.path:
                moves.append(PlanItem(
                    source=facts.path,
                    target=target,
                    category=decision.category,
                    confidence=decision.confidence,
                    analysis=facts,
                ))
                counts[decision.category] = counts.get(decision.category, 0) + 1

        summary = PlanSummary(total=len(moves), by_category=counts)
        return Plan(moves=moves, summary=summary)

    def save_plan(self, plan: Plan, out_path: Path) -> None:
        data = {
            "summary": {"total": plan.summary.total, "by_category": plan.summary.by_category},
            "moves": [
                {
                    "source": str(item.source),
                    "target": str(item.target),
                    "category": item.category,
                    "confidence": item.confidence,
                }
                for item in plan.moves[:10000]
            ],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")



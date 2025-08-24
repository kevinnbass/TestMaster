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
from collections import defaultdict, deque

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
        cfg = self.analyzer._load_config(None)
        self.config_path = self.root_dir / "tools" / "clean_reorganizer" / "config" / "config.json"
        self.categorizer = Categorizer(self.root_dir, self.config_path)
        self.target_root_name: str = cfg.get("target_root", "organized_codebase")
        self.subcategories: Dict = cfg.get("subcategories", {})
        ops = cfg.get("operations", {})
        # Flatten clusters by default to avoid deep cc_* noise
        self.flatten_clusters: bool = bool(ops.get("flatten_clusters", True))
        # Vendor subtree preservation
        self.vendor_roots: List[str] = [
            s for s in cfg.get("exclusions", {}).get("directories", []) if s.lower() in {"node_modules", "vendor"}
        ]

    def _match_subcategory(self, category: str, facts: FileFacts) -> str:
        rules = self.subcategories.get(category, [])
        path_str = str(facts.path).lower()
        ext = facts.path.suffix.lower()
        # Keyword and path-pattern based selection; first match wins
        for r in rules[:10]:
            # Extensions gate (if provided)
            exts = r.get("extensions", [])
            if exts and ext not in exts:
                continue
            # Path patterns
            for pat in r.get("path_patterns", [])[:10]:
                try:
                    import re
                    if re.search(pat, path_str, re.IGNORECASE):
                        return r.get("name", "")
                except re.error:
                    continue
            # Keyword presence in path
            for kw in r.get("keywords", [])[:20]:
                if kw.lower() in path_str:
                    return r.get("name", "")
        return ""

    def _target_path_for(self, facts: FileFacts, decision: CategoryDecision, cluster: str = "") -> Path:
        tgt = self.root_dir / self.target_root_name
        for part in decision.category.split('/')[:5]:
            if part:
                tgt = tgt / part
        # subcategory
        sub = self._match_subcategory(decision.category, facts)
        if sub:
            tgt = tgt / sub
        # optional cluster grouping (skip if flatten enabled)
        if cluster and not self.flatten_clusters:
            tgt = tgt / cluster
        return tgt / facts.path.name

    def build_plan(self) -> Plan:
        files = self.analyzer.discover_python_files()
        moves: List[PlanItem] = []
        counts: Dict[str, int] = {}
        # Origin and relationships manifests
        origins: List[Dict] = []
        rel_edges: List[Dict] = []
        # Prepare facts and decisions first
        facts_by_rel: Dict[str, FileFacts] = {}
        decisions_by_rel: Dict[str, CategoryDecision] = {}
        stems_to_rels: Dict[str, List[str]] = defaultdict(list)

        for p in files:
            facts = self.analyzer.analyze_file(p)
            if facts is None:
                continue
            rel = str(facts.path.relative_to(self.root_dir)).replace('\\', '/')
            facts_by_rel[rel] = facts
            decisions_by_rel[rel] = self.categorizer.decide(facts)
            # stem mapping (for Python and other code files)
            stem = facts.path.stem
            if stem:
                if len(stems_to_rels[stem]) < 10:
                    stems_to_rels[stem].append(rel)

        # Build undirected graph via basename import matching (bounded)
        graph: Dict[str, List[str]] = defaultdict(list)
        for rel, facts in facts_by_rel.items():
            imports = facts.imports[:50]
            for imp in imports:
                base = imp.split('.')[-1]
                if base in stems_to_rels:
                    # connect rel to up to 5 targets per import
                    for tgt in stems_to_rels[base][:5]:
                        if tgt != rel and len(graph[rel]) < 50:
                            graph[rel].append(tgt)
                            if len(graph[tgt]) < 50:
                                graph[tgt].append(rel)

        # Connected components (iterative BFS) for clustering
        comp_id_by_rel: Dict[str, int] = {}
        comp_members: Dict[int, List[str]] = defaultdict(list)
        comp_index = 0
        for rel in facts_by_rel.keys():
            if rel in comp_id_by_rel:
                continue
            # BFS
            q: deque[str] = deque()
            q.append(rel)
            comp_id_by_rel[rel] = comp_index
            comp_members[comp_index].append(rel)
            steps = 0
            while q and steps < 500000:
                cur = q.popleft()
                for nei in graph.get(cur, [])[:50]:
                    if nei not in comp_id_by_rel:
                        comp_id_by_rel[nei] = comp_index
                        comp_members[comp_index].append(nei)
                        q.append(nei)
                steps += 1
            comp_index += 1

        # Now create plan items with cluster names for sufficiently large components
        for rel, facts in facts_by_rel.items():
            decision = decisions_by_rel[rel]
            comp = comp_id_by_rel.get(rel, -1)
            members = comp_members.get(comp, [])
            cluster_name = ""
            if len(members) >= 3 and decision.confidence >= 0.6 and not self.flatten_clusters:
                cluster_name = f"cc_{comp}"

            target = self._target_path_for(facts, decision, cluster=cluster_name)
            if target != facts.path:
                moves.append(PlanItem(
                    source=facts.path,
                    target=target,
                    category=decision.category,
                    confidence=decision.confidence,
                    analysis=facts,
                ))
                counts[decision.category] = counts.get(decision.category, 0) + 1
                old_mod = self._old_python_module_path(facts) if facts.path.suffix.lower() == '.py' else ''
                origins.append({
                    "source": rel,
                    "target": str(target),
                    "category": decision.category,
                    "cluster": cluster_name,
                    "confidence": decision.confidence,
                    "old_module": old_mod
                })
            # relationships capture (file-level facts)
            rel_edges.append({
                "file": rel,
                "imports": facts.imports[:50],
                "classes": facts.classes[:50],
                "functions": facts.functions[:100]
            })

        summary = PlanSummary(total=len(moves), by_category=counts)
        # Save manifests next to plan path on save_plan() call only
        self._pending_origins = origins
        self._pending_relationships = rel_edges
        # Also emit duplicates report keyed by content hash for moved items
        dup_index: Dict[str, List[str]] = defaultdict(list)
        for m in moves:
            h = m.analysis.file_hash
            if h:
                dup_index[h].append(str(m.source))
        self._pending_duplicates = [
            {"hash": h, "sources": srcs}
            for (h, srcs) in dup_index.items() if len(srcs) > 1
        ]
        return Plan(moves=moves, summary=summary)

    def _old_python_module_path(self, facts: FileFacts) -> str:
        """Compute old dotted module path for Python files."""
        rel = str(facts.path.relative_to(self.root_dir)).replace('\\', '/').rstrip('/')
        if rel.endswith('.py'):
            rel = rel[:-3]
        # Drop common non-package roots if desired (optional tuning)
        return rel.replace('/', '.')

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
        # Emit origin and relationships manifests
        (out_path.parent / "origin_manifest.json").write_text(json.dumps(self._pending_origins, indent=2), encoding="utf-8")
        (out_path.parent / "relationships.json").write_text(json.dumps(self._pending_relationships, indent=2), encoding="utf-8")
        (out_path.parent / "duplicates.json").write_text(json.dumps(self._pending_duplicates, indent=2), encoding="utf-8")
        # Import resolve dry-run: naive resolvability report
        report = self._compute_import_resolve_report()
        (out_path.parent / "import_resolve_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _compute_import_resolve_report(self) -> Dict:
        # Build set of known new modules from planned moves (by basename and path tokens)
        from typing import Set, List
        plan_moves = getattr(self, "_pending_origins", [])
        known_tokens: Set[str] = set()
        for m in plan_moves[:50000]:
            tgt = m.get("target", "")
            p = Path(tgt)
            # add basename
            name = p.stem
            if name:
                known_tokens.add(name)
            # add path parts as potential package tokens
            for part in list(p.parts)[-6:]:  # last few parts only
                s = str(part)
                if s and s not in {"", ".", ".."}:
                    known_tokens.add(s)

        unresolved_by_file: Dict[str, List[str]] = {}
        total_imports = 0
        unresolved = 0
        base_counts: Dict[str, int] = {}
        for rel_entry in getattr(self, "_pending_relationships", [])[:50000]:
            file_rel = rel_entry.get("file", "")
            imps = rel_entry.get("imports", [])[:100]
            for imp in imps:
                total_imports += 1
                base = str(imp).split('.')[-1]
                base_counts[base] = base_counts.get(base, 0) + 1
                if base not in known_tokens:
                    unresolved += 1
                    lst = unresolved_by_file.get(file_rel)
                    if lst is None:
                        lst = []
                        unresolved_by_file[file_rel] = lst
                    if len(lst) < 10:
                        lst.append(imp)
        # Top unresolved bases for tuning
        top_unresolved = sorted(
            [(b, c) for b, c in base_counts.items() if b not in known_tokens],
            key=lambda x: x[1], reverse=True
        )[:50]
        return {
            "total_imports": total_imports,
            "unresolved_imports": unresolved,
            "unresolved_by_file_sample": unresolved_by_file,
            "top_unresolved_bases": top_unresolved,
        }



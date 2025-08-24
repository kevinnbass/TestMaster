"""
Clean Reorganizer: Analyzer
===========================

Responsible for bounded file discovery and AST-based feature extraction.

Design goals:
- Deterministic, bounded traversal
- AST-first extraction with safe fallbacks
- No regex-based code modification
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import ast
import json
import os


@dataclass
class FileFacts:
    """Lightweight analysis facts for a single Python file."""
    path: Path
    size_bytes: int
    num_lines: int
    imports: List[str]
    classes: List[str]
    functions: List[str]
    keywords: Set[str]


class Analyzer:
    """Bounded file discovery and AST-based analysis.

    Notes:
    - All loops are bounded by config values.
    - Assertions enforce pre/postconditions to catch invalid states early.
    """

    def __init__(self, root_dir: Path, config_path: Optional[Path] = None) -> None:
        assert isinstance(root_dir, Path), "root_dir must be a Path"
        self.root_dir: Path = root_dir.resolve()
        assert self.root_dir.exists(), "Root directory must exist"

        self.config: Dict = self._load_config(config_path)
        assert isinstance(self.config, dict), "Config must be a dict"

        bounds = self.config.get("bounds", {})
        self.max_directories: int = int(bounds.get("max_directories", 5000))
        self.max_files: int = int(bounds.get("max_files", 20000))
        self.max_lines_per_file: int = int(bounds.get("max_lines_per_file", 20000))
        self.max_path_length: int = int(bounds.get("max_path_length", 32000))

        ops = self.config.get("operations", {})
        self.max_file_size_bytes: int = int(ops.get("max_file_size_bytes", 10 * 1024 * 1024))

        self.ex_dirs: Set[str] = set(self.config.get("exclusions", {}).get("directories", []))
        self.ex_tests: Set[str] = set(self.config.get("exclusions", {}).get("tests", []))
        self.ex_file_globs: Set[str] = set(self.config.get("exclusions", {}).get("file_globs", []))

        assert self.max_directories > 0, "max_directories must be positive"
        assert self.max_files > 0, "max_files must be positive"

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load configuration JSON.

        Falls back to the default bundled config.
        """
        if config_path is None:
            config_path = self.root_dir / "tools" / "clean_reorganizer" / "config" / "config.json"

        assert isinstance(config_path, Path), "config_path must be a Path"
        assert config_path.exists(), f"Missing config: {config_path}"

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict), "Config root must be object"
        return data

    def _should_exclude_dir(self, dir_name: str) -> bool:
        """Decide if a directory should be pruned during traversal."""
        assert isinstance(dir_name, str), "dir_name must be a string"
        if dir_name in self.ex_dirs or dir_name in self.ex_tests:
            return True
        return False

    def _should_exclude_file(self, file_path: Path) -> bool:
        """Decide if a file should be excluded from analysis."""
        assert isinstance(file_path, Path), "file_path must be a Path"
        if file_path.suffix.lower() != ".py":
            return True
        try:
            size = file_path.stat().st_size
        except Exception:
            return True
        if size > self.max_file_size_bytes:
            return True
        if len(str(file_path)) > self.max_path_length:
            return True
        # Simple name-based file glob exclusions (bounded subset)
        for pat in list(self.ex_file_globs)[:50]:
            if pat.startswith("*.") and file_path.name.endswith(pat[1:]):
                return True
        return False

    def discover_python_files(self) -> List[Path]:
        """Discover Python files within bounds and exclusions.

        Returns an array sized up to max_files.
        """
        results: List[Optional[Path]] = [None] * self.max_files
        count: int = 0
        dir_count: int = 0

        for root, dirs, files in os.walk(self.root_dir):
            if dir_count >= self.max_directories:
                break
            dir_count += 1

            # Prune directories in-place (bounded loop)
            pruned: List[str] = []
            for d in dirs[:200]:
                if not self._should_exclude_dir(d):
                    pruned.append(d)
            dirs[:] = pruned

            # Process files (bounded loop)
            for name in files[:2000]:
                if count >= self.max_files:
                    break
                p = Path(root) / name
                if not self._should_exclude_file(p):
                    results[count] = p
                    count += 1

        # Trim None entries
        final: List[Path] = []
        for i in range(count):
            item = results[i]
            assert item is not None, "Collected file cannot be None"
            final.append(item)
        return final

    def analyze_file(self, file_path: Path) -> Optional[FileFacts]:
        """Analyze a single file; return None if unreadable or excluded."""
        assert isinstance(file_path, Path), "file_path must be a Path"
        if self._should_exclude_file(file_path):
            return None

        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        lines = text.splitlines()
        assert len(lines) >= 0, "splitlines must succeed"
        if len(lines) > self.max_lines_per_file:
            # Truncate safely for analysis only
            lines = lines[: self.max_lines_per_file]
            text = "\n".join(lines)

        size_bytes = len(text.encode("utf-8", errors="ignore"))
        assert size_bytes >= 0, "size_bytes cannot be negative"

        imports, classes, functions = self._extract_ast_features(text)
        keywords = self._extract_keywords(text)

        return FileFacts(
            path=file_path,
            size_bytes=size_bytes,
            num_lines=len(lines),
            imports=imports,
            classes=classes,
            functions=functions,
            keywords=keywords,
        )

    def _extract_ast_features(self, text: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract imports, classes, functions using AST with fallbacks."""
        assert isinstance(text, str), "text must be a string"
        imports: List[str] = []
        classes: List[str] = []
        functions: List[str] = []
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names[:50]:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
        except SyntaxError:
            # Minimal fallback: line scanning (bounded)
            for line in text.splitlines()[:2000]:
                l = line.strip()
                if l.startswith("import "):
                    parts = l.split()
                    if len(parts) >= 2:
                        imports.append(parts[1])
                elif l.startswith("from ") and " import " in l:
                    try:
                        mod = l.split()[1]
                        imports.append(mod)
                    except Exception:
                        pass
                elif l.startswith("class "):
                    name = l[6:].split("(")[0].strip().split()[0]
                    if name:
                        classes.append(name)
                elif l.startswith("def "):
                    name = l[4:].split("(")[0].strip().split()[0]
                    if name:
                        functions.append(name)
        assert len(imports) <= 1000, "Too many imports extracted"
        assert len(classes) <= 1000, "Too many classes extracted"
        return list(dict.fromkeys(imports)), classes, functions

    def _extract_keywords(self, text: str) -> Set[str]:
        """Very lightweight keyword extraction for heuristics."""
        assert isinstance(text, str), "text must be a string"
        words: List[str] = []
        chunked = text.lower().split()
        for token in chunked[:10000]:
            if token.isalpha() and len(token) > 2:
                words.append(token)
        # Deduplicate bounded
        unique: Set[str] = set()
        for w in words[:2000]:
            unique.add(w)
        return unique



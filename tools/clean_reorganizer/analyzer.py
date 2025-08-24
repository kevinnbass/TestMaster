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
import hashlib


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
    file_hash: str


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
        self.include_exts: Set[str] = set(self.config.get("include_extensions", [".py"]))
        # Sweeper mode processes all extensions (within other safety bounds)
        self.sweeper_mode: bool = bool(ops.get("sweeper_mode", False))

        ex = self.config.get("exclusions", {})
        self.ex_dirs: Set[str] = set(ex.get("directories", []))
        self.ex_tests: Set[str] = set(ex.get("tests", []))
        self.ex_file_globs: Set[str] = set(ex.get("file_globs", []))
        self.ex_dir_globs: Set[str] = set(ex.get("directory_globs", []))
        self.ex_paths: Set[str] = set(ex.get("exclude_paths", []))

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
        # glob-like quick checks
        name = dir_name
        for pat in list(self.ex_dir_globs)[:50]:
            if pat.replace('*', '') in name:
                return True
        return False

    def _should_exclude_file(self, file_path: Path) -> bool:
        """Decide if a file should be excluded from analysis."""
        assert isinstance(file_path, Path), "file_path must be a Path"
        # In sweeper mode, accept any extension (still subject to other bounds)
        if (not self.sweeper_mode) and (file_path.suffix.lower() not in self.include_exts):
            return True
        try:
            size = file_path.stat().st_size
        except Exception:
            return True
        if size > self.max_file_size_bytes:
            return True
        if len(str(file_path)) > self.max_path_length:
            return True
        # Path-prefix exclusions
        rel = None
        try:
            rel = str(file_path.relative_to(self.root_dir)).replace('\\', '/').lower()
        except Exception:
            rel = str(file_path).replace('\\', '/').lower()
        for xp in list(self.ex_paths)[:50]:
            if rel.startswith(xp.lower()):
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

        # Read bytes first for reliable size/hash
        data: bytes = b""
        try:
            with open(file_path, "rb") as f:
                data = f.read()
        except Exception:
            return None

        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            text = ""

        lines = text.splitlines()
        assert len(lines) >= 0, "splitlines must succeed"
        if len(lines) > self.max_lines_per_file:
            # Truncate safely for analysis only
            lines = lines[: self.max_lines_per_file]
            text = "\n".join(lines)

        size_bytes = len(data)
        assert size_bytes >= 0, "size_bytes cannot be negative"

        imports, classes, functions = self._extract_features(text, file_path.suffix.lower())
        keywords = self._extract_keywords(text)
        file_hash = self._hash_bytes(data)

        return FileFacts(
            path=file_path,
            size_bytes=size_bytes,
            num_lines=len(lines),
            imports=imports,
            classes=classes,
            functions=functions,
            keywords=keywords,
            file_hash=file_hash,
        )

    def _extract_features(self, text: str, ext: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract imports, classes, functions with language-aware strategy."""
        assert isinstance(text, str), "text must be a string"
        imports: List[str] = []
        classes: List[str] = []
        functions: List[str] = []
        if ext == ".py":
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
        elif ext in {".js", ".ts", ".tsx", ".jsx"}:
            imports = self._extract_js_imports(text)
        else:
            # Non-code files: no imports/classes/functions
            pass
        assert len(imports) <= 2000, "Too many imports extracted"
        assert len(classes) <= 2000, "Too many classes extracted"
        return list(dict.fromkeys(imports)), classes, functions

    def _extract_js_imports(self, text: str) -> List[str]:
        """Extract JS/TS module specifiers from import/require statements."""
        import re
        imports: List[str] = []
        lines = text.splitlines()
        for line in lines[:5000]:
            l = line.strip()
            # import X from "mod"; import {A} from 'mod'; import 'mod';
            m = re.search(r"import\s+(?:[\w\{\},\s\*]+\s+from\s+)?['\"]([^'\"]+)['\"]", l)
            if m:
                imports.append(m.group(1))
                continue
            # const x = require('mod')
            m2 = re.search(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)", l)
            if m2:
                imports.append(m2.group(1))
        return imports[:200]

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

    def _hash_bytes(self, data: bytes) -> str:
        """Compute MD5 hash for deduplication/identity checks."""
        try:
            h = hashlib.md5()
            CHUNK = 8192
            for i in range(0, len(data), CHUNK):
                h.update(data[i:i+CHUNK])
            return h.hexdigest()
        except Exception:
            return ""



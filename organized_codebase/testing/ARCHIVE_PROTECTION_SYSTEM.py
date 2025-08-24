"""
Archive System with Strict Preservation
========================================

This system ensures NO archived code is ever lost.
Archives are IMMUTABLE once created.

Author: TestMaster Team
"""

import os
import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import ast
import inspect


class ArchiveSystem:
    """
    Manages code archival with strict preservation rules.
    NEVER deletes archived files.
    """
    
    def __init__(self):
        self.archive_root = Path("ARCHIVE")
        self.manifest_file = self.archive_root / "archive_manifest.json"
        self.preservation_rules = self.archive_root / "PRESERVATION_RULES.md"
        
        # Load or create manifest
        self.manifest = self._load_manifest()
        
        # Ensure preservation rules exist
        if not self.preservation_rules.exists():
            self._create_preservation_rules()
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load existing manifest or create new one"""
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        
        return {
            "created": datetime.now().isoformat(),
            "total_archives": 0,
            "phases": {},
            "archives": []
        }
    
    def _create_preservation_rules(self):
        """Create preservation rules document"""
        self.archive_root.mkdir(exist_ok=True)
        
        rules = '''# ARCHIVE PRESERVATION RULES

## IMMUTABLE ARCHIVE POLICY

1. **NEVER DELETE ARCHIVED FILES**
   - Once archived, files are permanent
   - NO exceptions to this rule
   - Deletion is FORBIDDEN

2. **ARCHIVE BEFORE CONSOLIDATION**
   - Every file must be archived before modification
   - Include full functionality analysis
   - Document all unique features

3. **FUNCTIONALITY VERIFICATION**
   - Before archiving, document all functions/classes
   - Compare line counts: current vs proposed replacement
   - If replacement is smaller, FLAG for review

4. **RESTORATION CAPABILITY**
   - All archives must be restorable
   - Include dependency information
   - Test restoration process

5. **CONFLICT RESOLUTION**
   - When conflicts arise, MERGE functionality
   - NEVER replace comprehensive with simple
   - Preserve ALL unique capabilities

## VIOLATION CONSEQUENCES

Violating these rules results in FUNCTIONALITY LOSS.
This has already happened and MUST NOT happen again.
'''
        
        with open(self.preservation_rules, 'w') as f:
            f.write(rules)
    
    def archive_file(self, file_path: str, reason: str, phase: str = "manual") -> str:
        """
        Archive a file with full preservation.
        
        Args:
            file_path: Path to file being archived
            reason: Reason for archiving
            phase: Phase/operation name
            
        Returns:
            Archive ID
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot archive non-existent file: {file_path}")
        
        # Generate archive ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_hash = self._calculate_file_hash(file_path)
        archive_id = f"{phase}_{timestamp}_{file_hash[:8]}"
        
        # Create archive directory
        archive_dir = self.archive_root / phase / archive_id
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy file to archive
        source_path = Path(file_path)
        archive_file_path = archive_dir / source_path.name
        shutil.copy2(file_path, archive_file_path)
        
        # Analyze file functionality
        functionality = self._analyze_functionality(file_path)
        
        # Create archive metadata
        metadata = {
            "archive_id": archive_id,
            "original_path": file_path,
            "archived_at": datetime.now().isoformat(),
            "reason": reason,
            "phase": phase,
            "file_hash": file_hash,
            "file_size": os.path.getsize(file_path),
            "line_count": self._count_lines(file_path),
            "functionality": functionality
        }
        
        # Save metadata
        metadata_file = archive_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update manifest
        self.manifest["archives"].append(metadata)
        self.manifest["total_archives"] += 1
        if phase not in self.manifest["phases"]:
            self.manifest["phases"][phase] = []
        self.manifest["phases"][phase].append(archive_id)
        
        self._save_manifest()
        
        print(f"✓ Archived: {file_path} -> {archive_id}")
        print(f"  Line count: {metadata['line_count']}")
        print(f"  Functions: {len(functionality['functions'])}")
        print(f"  Classes: {len(functionality['classes'])}")
        
        return archive_id
    
    def verify_replacement(self, original_path: str, replacement_path: str) -> Dict[str, Any]:
        """
        Verify that replacement preserves functionality.
        
        Args:
            original_path: Path to original file
            replacement_path: Path to replacement file
            
        Returns:
            Verification report
        """
        if not os.path.exists(original_path):
            return {"error": "Original file not found"}
        
        if not os.path.exists(replacement_path):
            return {"error": "Replacement file not found"}
        
        # Analyze both files
        original_func = self._analyze_functionality(original_path)
        replacement_func = self._analyze_functionality(replacement_path)
        
        # Compare functionality
        report = {
            "timestamp": datetime.now().isoformat(),
            "original": {
                "path": original_path,
                "line_count": self._count_lines(original_path),
                "functions": original_func["functions"],
                "classes": original_func["classes"]
            },
            "replacement": {
                "path": replacement_path,
                "line_count": self._count_lines(replacement_path),
                "functions": replacement_func["functions"],
                "classes": replacement_func["classes"]
            }
        }
        
        # Calculate differences
        orig_functions = set(original_func["functions"])
        repl_functions = set(replacement_func["functions"])
        orig_classes = set(original_func["classes"])
        repl_classes = set(replacement_func["classes"])
        
        report["analysis"] = {
            "functions_lost": list(orig_functions - repl_functions),
            "functions_added": list(repl_functions - orig_functions),
            "classes_lost": list(orig_classes - repl_classes),
            "classes_added": list(repl_classes - orig_classes),
            "line_count_change": report["replacement"]["line_count"] - report["original"]["line_count"]
        }
        
        # Risk assessment
        risk_level = "LOW"
        concerns = []
        
        if report["analysis"]["functions_lost"]:
            risk_level = "HIGH"
            concerns.append(f"Lost {len(report['analysis']['functions_lost'])} functions")
        
        if report["analysis"]["classes_lost"]:
            risk_level = "HIGH"
            concerns.append(f"Lost {len(report['analysis']['classes_lost'])} classes")
        
        if report["analysis"]["line_count_change"] < -100:
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level
            concerns.append(f"Significant size reduction: {report['analysis']['line_count_change']} lines")
        
        report["risk_assessment"] = {
            "level": risk_level,
            "concerns": concerns,
            "recommendation": self._get_recommendation(risk_level, concerns)
        }
        
        return report
    
    def restore_from_archive(self, archive_id: str, target_path: str) -> bool:
        """
        Restore a file from archive.
        
        Args:
            archive_id: Archive ID to restore
            target_path: Where to restore the file
            
        Returns:
            True if successful
        """
        # Find archive
        archive_metadata = None
        for archive in self.manifest["archives"]:
            if archive["archive_id"] == archive_id:
                archive_metadata = archive
                break
        
        if not archive_metadata:
            print(f"✗ Archive not found: {archive_id}")
            return False
        
        # Locate archive file
        phase = archive_metadata["phase"]
        archive_dir = self.archive_root / phase / archive_id
        
        # Find the archived file
        original_filename = Path(archive_metadata["original_path"]).name
        archive_file_path = archive_dir / original_filename
        
        if not archive_file_path.exists():
            print(f"✗ Archive file missing: {archive_file_path}")
            return False
        
        # Restore file
        try:
            target_dir = Path(target_path).parent
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archive_file_path, target_path)
            print(f"✓ Restored: {archive_id} -> {target_path}")
            return True
        except Exception as e:
            print(f"✗ Restore failed: {e}")
            return False
    
    def list_archives(self, phase: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all archives, optionally filtered by phase"""
        archives = self.manifest["archives"]
        
        if phase:
            archives = [a for a in archives if a["phase"] == phase]
        
        return sorted(archives, key=lambda x: x["archived_at"], reverse=True)
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """Get archive system statistics"""
        total_archives = len(self.manifest["archives"])
        total_size = sum(a["file_size"] for a in self.manifest["archives"])
        total_lines = sum(a["line_count"] for a in self.manifest["archives"])
        
        phases = {}
        for archive in self.manifest["archives"]:
            phase = archive["phase"]
            if phase not in phases:
                phases[phase] = {"count": 0, "size": 0, "lines": 0}
            phases[phase]["count"] += 1
            phases[phase]["size"] += archive["file_size"]
            phases[phase]["lines"] += archive["line_count"]
        
        return {
            "total_archives": total_archives,
            "total_size_bytes": total_size,
            "total_lines": total_lines,
            "phases": phases,
            "archive_root": str(self.archive_root)
        }
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def _count_lines(self, file_path: str) -> int:
        """Count lines in file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return len(f.readlines())
        except:
            return 0
    
    def _analyze_functionality(self, file_path: str) -> Dict[str, Any]:
        """Analyze functionality in Python file"""
        functions = []
        classes = []
        imports = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except:
            # If parsing fails, try simple regex
            import re
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
            classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": imports[:20],  # Limit to avoid bloat
            "function_count": len(functions),
            "class_count": len(classes)
        }
    
    def _get_recommendation(self, risk_level: str, concerns: List[str]) -> str:
        """Get recommendation based on risk assessment"""
        if risk_level == "HIGH":
            return "REJECT REPLACEMENT - Significant functionality loss detected"
        elif risk_level == "MEDIUM":
            return "REVIEW REQUIRED - Potential functionality loss"
        else:
            return "APPROVE - No significant concerns"
    
    def _save_manifest(self):
        """Save manifest to disk"""
        self.archive_root.mkdir(exist_ok=True)
        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)


# Global archive system instance
_archive_system = None

def get_archive_system() -> ArchiveSystem:
    """Get global archive system instance"""
    global _archive_system
    if _archive_system is None:
        _archive_system = ArchiveSystem()
    return _archive_system

def archive_before_replace(file_path: str, reason: str, phase: str = "consolidation") -> str:
    """Archive a file before replacing it"""
    return get_archive_system().archive_file(file_path, reason, phase)

def verify_safe_replacement(original: str, replacement: str) -> bool:
    """Verify replacement is safe (no functionality loss)"""
    report = get_archive_system().verify_replacement(original, replacement)
    return report.get("risk_assessment", {}).get("level") != "HIGH"
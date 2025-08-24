"""
Tag Reading System

Inspired by Agency-Swarm's SendMessage validation patterns
for parsing and synchronizing tags and directives in source files.

Features:
- Parse TestMaster tags from source files
- Read Claude Code directives and metadata
- Synchronize tagging systems
- Validate tag formats and content
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import ast

from core.layer_manager import requires_layer


class TagType(Enum):
    """Types of tags found in files."""
    TESTMASTER = "testmaster"
    CLAUDE_CODE = "claude_code"
    MODULE_INFO = "module_info"
    TEST_INFO = "test_info"
    PRIORITY = "priority"
    STATUS = "status"


@dataclass
class FileTag:
    """A tag found in a file."""
    tag_type: TagType
    key: str
    value: Any
    line_number: int
    raw_content: str
    file_path: str


@dataclass
class ModuleTag:
    """Collection of tags for a module."""
    file_path: str
    tags: List[FileTag]
    last_scanned: datetime
    
    # Common tag values (parsed from tags)
    testmaster_tags: Set[str] = field(default_factory=set)
    owner: Optional[str] = None
    last_tested: Optional[datetime] = None
    status: Optional[str] = None
    coverage_target: Optional[int] = None
    priority: Optional[str] = None
    test_covers: List[str] = field(default_factory=list)
    test_type: Optional[str] = None


class TagReader:
    """
    Read and parse tags from source files.
    
    Uses Agency-Swarm's SendMessage validation patterns
    for tag format validation and content parsing.
    """
    
    @requires_layer("layer2_monitoring", "tag_reading")
    def __init__(self, watch_paths: Union[str, List[str]]):
        """
        Initialize tag reader.
        
        Args:
            watch_paths: Directories to scan for tagged files
        """
        self.watch_paths = [Path(p) for p in (watch_paths if isinstance(watch_paths, list) else [watch_paths])]
        
        # Tag patterns (regex)
        self.tag_patterns = {
            # TestMaster tags
            'testmaster_tags': re.compile(r'#\s*TESTMASTER_TAGS:\s*(.+)'),
            'testmaster_owner': re.compile(r'#\s*TESTMASTER_OWNER:\s*(.+)'),
            'testmaster_last_tested': re.compile(r'#\s*TESTMASTER_LAST_TESTED:\s*(.+)'),
            'testmaster_status': re.compile(r'#\s*TESTMASTER_STATUS:\s*(.+)'),
            'testmaster_coverage_target': re.compile(r'#\s*TESTMASTER_COVERAGE_TARGET:\s*(\d+)'),
            'testmaster_priority': re.compile(r'#\s*TESTMASTER_PRIORITY:\s*(.+)'),
            
            # Test-specific tags
            'test_covers': re.compile(r'#\s*TESTMASTER_COVERS:\s*(.+)'),
            'test_type': re.compile(r'#\s*TESTMASTER_TYPE:\s*(.+)'),
            'test_last_passed': re.compile(r'#\s*TESTMASTER_LAST_PASSED:\s*(.+)'),
            
            # Claude Code tags
            'claude_directive': re.compile(r'#\s*CLAUDE_DIRECTIVE:\s*(.+)'),
            'claude_priority': re.compile(r'#\s*CLAUDE_PRIORITY:\s*(.+)'),
            'claude_context': re.compile(r'#\s*CLAUDE_CONTEXT:\s*(.+)'),
            'claude_status': re.compile(r'#\s*CLAUDE_STATUS:\s*(.+)'),
        }
        
        # Cache for parsed tags
        self._module_cache: Dict[str, ModuleTag] = {}
        self._last_scan_time: Dict[str, datetime] = {}
        
        print(f"🏷️ Tag reader initialized")
        print(f"   📁 Watching: {', '.join(str(p) for p in self.watch_paths)}")
        print(f"   🔍 Scanning for {len(self.tag_patterns)} tag types")
    
    def scan_all_files(self, force_rescan: bool = False) -> Dict[str, ModuleTag]:
        """
        Scan all files in watch paths for tags.
        
        Args:
            force_rescan: Force re-scan even if files haven't changed
            
        Returns:
            Dictionary mapping file paths to module tags
        """
        print("🔍 Scanning files for tags...")
        
        scanned_count = 0
        updated_count = 0
        
        for watch_path in self.watch_paths:
            if not watch_path.exists():
                continue
            
            for py_file in watch_path.rglob("*.py"):
                if self._should_scan_file(py_file):
                    scanned_count += 1
                    
                    # Check if file needs re-scanning
                    if self._needs_rescan(py_file, force_rescan):
                        module_tags = self.scan_file(py_file)
                        if module_tags:
                            self._module_cache[str(py_file)] = module_tags
                            updated_count += 1
        
        print(f"📊 Tag scan complete: {scanned_count} files scanned, {updated_count} updated")
        return dict(self._module_cache)
    
    def scan_file(self, file_path: Union[str, Path]) -> Optional[ModuleTag]:
        """
        Scan a single file for tags.
        
        Args:
            file_path: Path to the file to scan
            
        Returns:
            ModuleTag object if tags found, None otherwise
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Extract tags from file
            tags = self._extract_tags_from_lines(lines, str(file_path))
            
            if not tags:
                return None
            
            # Create module tag object
            module_tag = ModuleTag(
                file_path=str(file_path),
                tags=tags,
                last_scanned=datetime.now()
            )
            
            # Parse common tag values
            self._parse_common_tags(module_tag)
            
            # Update last scan time
            self._last_scan_time[str(file_path)] = datetime.now()
            
            return module_tag
            
        except Exception as e:
            print(f"⚠️ Error scanning {file_path}: {e}")
            return None
    
    def _extract_tags_from_lines(self, lines: List[str], file_path: str) -> List[FileTag]:
        """Extract all tags from file lines."""
        tags = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip non-comment lines
            if not line.startswith('#'):
                continue
            
            # Check each tag pattern
            for tag_key, pattern in self.tag_patterns.items():
                match = pattern.match(line)
                if match:
                    tag_value = match.group(1).strip()
                    
                    # Determine tag type
                    tag_type = self._determine_tag_type(tag_key)
                    
                    # Parse tag value
                    parsed_value = self._parse_tag_value(tag_key, tag_value)
                    
                    tag = FileTag(
                        tag_type=tag_type,
                        key=tag_key,
                        value=parsed_value,
                        line_number=line_num,
                        raw_content=line,
                        file_path=file_path
                    )
                    
                    tags.append(tag)
        
        return tags
    
    def _determine_tag_type(self, tag_key: str) -> TagType:
        """Determine tag type from tag key."""
        if tag_key.startswith('testmaster_'):
            return TagType.TESTMASTER
        elif tag_key.startswith('claude_'):
            return TagType.CLAUDE_CODE
        elif tag_key.startswith('test_'):
            return TagType.TEST_INFO
        else:
            return TagType.MODULE_INFO
    
    def _parse_tag_value(self, tag_key: str, tag_value: str) -> Any:
        """Parse tag value based on tag type."""
        # List values (comma-separated)
        if tag_key in ['testmaster_tags', 'test_covers']:
            return [item.strip() for item in tag_value.split(',')]
        
        # Integer values
        elif tag_key in ['testmaster_coverage_target']:
            try:
                return int(tag_value)
            except ValueError:
                return None
        
        # DateTime values
        elif tag_key in ['testmaster_last_tested', 'test_last_passed']:
            try:
                return datetime.fromisoformat(tag_value)
            except ValueError:
                return tag_value  # Return as string if not valid datetime
        
        # Boolean values
        elif tag_key in ['claude_priority']:
            return tag_value.lower() in ['true', 'yes', '1', 'high', 'critical']
        
        # Default: string value
        else:
            return tag_value
    
    def _parse_common_tags(self, module_tag: ModuleTag):
        """Parse common tag values into module_tag fields."""
        for tag in module_tag.tags:
            if tag.key == 'testmaster_tags' and isinstance(tag.value, list):
                module_tag.testmaster_tags.update(tag.value)
            elif tag.key == 'testmaster_owner':
                module_tag.owner = tag.value
            elif tag.key == 'testmaster_last_tested':
                module_tag.last_tested = tag.value if isinstance(tag.value, datetime) else None
            elif tag.key == 'testmaster_status':
                module_tag.status = tag.value
            elif tag.key == 'testmaster_coverage_target':
                module_tag.coverage_target = tag.value
            elif tag.key == 'testmaster_priority':
                module_tag.priority = tag.value
            elif tag.key == 'test_covers' and isinstance(tag.value, list):
                module_tag.test_covers.extend(tag.value)
            elif tag.key == 'test_type':
                module_tag.test_type = tag.value
    
    def _should_scan_file(self, file_path: Path) -> bool:
        """Check if file should be scanned for tags."""
        # Skip hidden files and directories
        if any(part.startswith('.') for part in file_path.parts):
            return False
        
        # Skip common ignore patterns
        ignore_patterns = {
            '__pycache__', '.git', '.vscode', '.idea', 'venv', '.env',
            'node_modules', '.pytest_cache', '.coverage', '.tox'
        }
        
        if any(pattern in str(file_path) for pattern in ignore_patterns):
            return False
        
        return True
    
    def _needs_rescan(self, file_path: Path, force: bool = False) -> bool:
        """Check if file needs to be re-scanned."""
        if force:
            return True
        
        file_str = str(file_path)
        
        # Check if never scanned
        if file_str not in self._last_scan_time:
            return True
        
        # Check if file modified since last scan
        try:
            file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            last_scan = self._last_scan_time[file_str]
            return file_mtime > last_scan
        except:
            return True
    
    def get_modules_with_tag(self, tag_key: str, tag_value: Any = None) -> List[ModuleTag]:
        """
        Get modules that have a specific tag.
        
        Args:
            tag_key: Tag key to search for
            tag_value: Optional specific value to match
            
        Returns:
            List of modules with the tag
        """
        matching_modules = []
        
        for module_tag in self._module_cache.values():
            for tag in module_tag.tags:
                if tag.key == tag_key:
                    if tag_value is None or tag.value == tag_value:
                        matching_modules.append(module_tag)
                        break
        
        return matching_modules
    
    def get_modules_by_status(self, status: str) -> List[ModuleTag]:
        """Get modules with a specific status."""
        return [
            module for module in self._module_cache.values()
            if module.status == status
        ]
    
    def get_modules_by_owner(self, owner: str) -> List[ModuleTag]:
        """Get modules owned by a specific owner."""
        return [
            module for module in self._module_cache.values()
            if module.owner == owner
        ]
    
    def get_test_modules_covering(self, source_module: str) -> List[ModuleTag]:
        """Get test modules that cover a specific source module."""
        covering_tests = []
        
        for module_tag in self._module_cache.values():
            if source_module in module_tag.test_covers:
                covering_tests.append(module_tag)
        
        return covering_tests
    
    def get_modules_with_testmaster_tag(self, tag: str) -> List[ModuleTag]:
        """Get modules that have a specific TestMaster tag."""
        return [
            module for module in self._module_cache.values()
            if tag in module.testmaster_tags
        ]
    
    def get_high_priority_modules(self) -> List[ModuleTag]:
        """Get modules marked as high priority."""
        high_priority_keywords = {'high', 'critical', 'urgent', 'important'}
        
        return [
            module for module in self._module_cache.values()
            if module.priority and module.priority.lower() in high_priority_keywords
        ]
    
    def get_modules_needing_coverage(self, min_target: int = 80) -> List[ModuleTag]:
        """Get modules with high coverage targets."""
        return [
            module for module in self._module_cache.values()
            if module.coverage_target and module.coverage_target >= min_target
        ]
    
    def update_module_tag(self, file_path: str, tag_key: str, new_value: Any) -> bool:
        """
        Update a tag value in a file.
        
        Args:
            file_path: Path to the file
            tag_key: Tag key to update
            new_value: New value for the tag
            
        Returns:
            True if tag was updated, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False
            
            # Read current file content
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find and update the tag
            updated = False
            tag_pattern = self.tag_patterns.get(tag_key)
            
            if not tag_pattern:
                return False
            
            for i, line in enumerate(lines):
                if tag_pattern.match(line.strip()):
                    # Format new value
                    if isinstance(new_value, list):
                        value_str = ', '.join(str(v) for v in new_value)
                    elif isinstance(new_value, datetime):
                        value_str = new_value.isoformat()
                    else:
                        value_str = str(new_value)
                    
                    # Create new tag line
                    tag_name = tag_key.upper().replace('_', '_')
                    lines[i] = f"# {tag_name}: {value_str}\\n"
                    updated = True
                    break
            
            if updated:
                # Write updated content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                # Re-scan the file
                self.scan_file(file_path)
                
                print(f"🏷️ Updated tag {tag_key} in {file_path.name}")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️ Error updating tag in {file_path}: {e}")
            return False
    
    def add_tag_to_file(self, file_path: str, tag_key: str, tag_value: Any, 
                       insert_after_line: int = 0) -> bool:
        """
        Add a new tag to a file.
        
        Args:
            file_path: Path to the file
            tag_key: Tag key to add
            tag_value: Tag value
            insert_after_line: Line number to insert after (0 = beginning)
            
        Returns:
            True if tag was added, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False
            
            # Read current file content
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Format tag value
            if isinstance(tag_value, list):
                value_str = ', '.join(str(v) for v in tag_value)
            elif isinstance(tag_value, datetime):
                value_str = tag_value.isoformat()
            else:
                value_str = str(tag_value)
            
            # Create new tag line
            tag_name = tag_key.upper().replace('_', '_')
            new_tag_line = f"# {tag_name}: {value_str}\\n"
            
            # Insert the tag
            insert_index = min(insert_after_line, len(lines))
            lines.insert(insert_index, new_tag_line)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            # Re-scan the file
            self.scan_file(file_path)
            
            print(f"🏷️ Added tag {tag_key} to {file_path.name}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error adding tag to {file_path}: {e}")
            return False
    
    def get_tag_statistics(self) -> Dict[str, Any]:
        """Get statistics about tags found."""
        total_files = len(self._module_cache)
        total_tags = sum(len(module.tags) for module in self._module_cache.values())
        
        # Count tags by type
        tag_type_counts = {}
        for module in self._module_cache.values():
            for tag in module.tags:
                tag_type = tag.tag_type.value
                tag_type_counts[tag_type] = tag_type_counts.get(tag_type, 0) + 1
        
        # Count common tags
        status_counts = {}
        owner_counts = {}
        priority_counts = {}
        
        for module in self._module_cache.values():
            if module.status:
                status_counts[module.status] = status_counts.get(module.status, 0) + 1
            if module.owner:
                owner_counts[module.owner] = owner_counts.get(module.owner, 0) + 1
            if module.priority:
                priority_counts[module.priority] = priority_counts.get(module.priority, 0) + 1
        
        return {
            "total_tagged_files": total_files,
            "total_tags": total_tags,
            "avg_tags_per_file": total_tags / max(total_files, 1),
            "tag_type_distribution": tag_type_counts,
            "status_distribution": status_counts,
            "owner_distribution": owner_counts,
            "priority_distribution": priority_counts,
            "last_scan_time": max(self._last_scan_time.values()) if self._last_scan_time else None
        }
    
    def export_tag_report(self, output_path: str = "tag_report.json"):
        """Export tag information to JSON."""
        import json
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_tag_statistics(),
            "modules": []
        }
        
        for module in self._module_cache.values():
            module_info = {
                "file_path": module.file_path,
                "last_scanned": module.last_scanned.isoformat(),
                "testmaster_tags": list(module.testmaster_tags),
                "owner": module.owner,
                "status": module.status,
                "priority": module.priority,
                "coverage_target": module.coverage_target,
                "test_covers": module.test_covers,
                "test_type": module.test_type,
                "tags": [
                    {
                        "type": tag.tag_type.value,
                        "key": tag.key,
                        "value": tag.value,
                        "line_number": tag.line_number
                    }
                    for tag in module.tags
                ]
            }
            report["modules"].append(module_info)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📄 Tag report exported to {output_path}")
        except Exception as e:
            print(f"⚠️ Error exporting tag report: {e}")


# Convenience functions for tag management
def scan_directory_for_tags(directory: str) -> Dict[str, ModuleTag]:
    """Quick scan of a directory for tags."""
    reader = TagReader(directory)
    return reader.scan_all_files()


def find_modules_with_status(directory: str, status: str) -> List[str]:
    """Find modules with a specific status."""
    reader = TagReader(directory)
    reader.scan_all_files()
    modules = reader.get_modules_by_status(status)
    return [module.file_path for module in modules]
"""
Universal Language Detector for TestMaster

Detects programming languages, frameworks, and patterns in any codebase.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import mimetypes
import hashlib

@dataclass
class LanguageInfo:
    """Information about a detected programming language."""
    name: str
    confidence: float
    file_count: int
    total_lines: int
    percentage: float
    extensions: Set[str] = field(default_factory=set)
    syntax_features: List[str] = field(default_factory=list)
    version_info: Optional[str] = None

@dataclass
class FrameworkInfo:
    """Information about detected testing/development frameworks."""
    name: str
    language: str
    confidence: float
    version: Optional[str] = None
    config_files: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)

@dataclass
class BuildSystemInfo:
    """Information about build systems."""
    name: str
    language: str
    version: Optional[str] = None
    config_files: List[str] = field(default_factory=list)

@dataclass
class DependencyInfo:
    """Information about project dependencies."""
    name: str
    language: str
    dependency_type: str  # 'runtime', 'dev', 'test', 'build'
    version: Optional[str] = None

@dataclass
class FileInfo:
    """Information about a source file."""
    path: str
    language: str
    lines_of_code: int
    complexity_score: float
    last_modified: datetime
    framework: Optional[str] = None

@dataclass
class CodebaseProfile:
    """Comprehensive profile of a codebase."""
    project_path: str
    languages: List[LanguageInfo]
    frameworks: List[FrameworkInfo]
    build_systems: List[BuildSystemInfo]
    dependencies: List[DependencyInfo]
    source_files: List[FileInfo] = field(default_factory=list)
    
    # Analysis results
    total_files: int = 0
    total_lines: int = 0
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    architectural_patterns: List[str] = field(default_factory=list)
    
    # Capabilities
    testing_capabilities: List[str] = field(default_factory=list)
    ci_cd_capabilities: List[str] = field(default_factory=list)
    
    # Metadata
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    analysis_duration: float = 0.0

class UniversalLanguageDetector:
    """Universal language detector for any codebase."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.supported_languages = self.config.get('supported_languages', self._get_default_languages())
        self.fallback_analyzers = self.config.get('fallback_analyzers', ['ast_generic', 'text_pattern', 'ai_inference'])
        
        # Language extension mappings
        self.extension_map = self._build_extension_map()
        
        # Framework detection patterns
        self.framework_patterns = self._build_framework_patterns()
        
        # Build system patterns
        self.build_system_patterns = self._build_build_system_patterns()
        
        print("Universal language detector initialized")
        print(f"   Supported languages: {len(self.supported_languages)}")
        print(f"   Fallback analyzers: {self.fallback_analyzers}")
    
    def detect_codebase(self, project_path: str) -> CodebaseProfile:
        """
        Detect and analyze any codebase comprehensively.
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            Complete codebase profile with all detected information
        """
        start_time = datetime.now()
        project_path = Path(project_path).resolve()
        
        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
        
        print(f"Analyzing codebase: {project_path}")
        
        # Step 1: Scan all files
        all_files = self._scan_files(project_path)
        print(f"Found {len(all_files)} files to analyze")
        
        # Step 2: Detect languages
        languages = self._detect_languages(all_files, project_path)
        print(f"Detected languages: {[lang.name for lang in languages]}")
        
        # Step 3: Detect frameworks
        frameworks = self._detect_frameworks(all_files, project_path, languages)
        print(f"Detected frameworks: {[fw.name for fw in frameworks]}")
        
        # Step 4: Detect build systems
        build_systems = self._detect_build_systems(all_files, project_path)
        print(f"Detected build systems: {[bs.name for bs in build_systems]}")
        
        # Step 5: Analyze dependencies
        dependencies = self._analyze_dependencies(all_files, project_path, languages)
        print(f"Found {len(dependencies)} dependencies")
        
        # Step 6: Create file profiles
        source_files = self._create_file_profiles(all_files, languages, frameworks)
        
        # Step 7: Analyze architecture
        architectural_patterns = self._detect_architectural_patterns(source_files, frameworks)
        
        # Step 8: Calculate metrics
        complexity_metrics = self._calculate_complexity_metrics(source_files)
        
        # Step 9: Determine capabilities
        testing_capabilities = self._determine_testing_capabilities(frameworks, build_systems)
        ci_cd_capabilities = self._determine_ci_cd_capabilities(all_files, project_path)
        
        analysis_duration = (datetime.now() - start_time).total_seconds()
        
        profile = CodebaseProfile(
            project_path=str(project_path),
            languages=languages,
            frameworks=frameworks,
            build_systems=build_systems,
            dependencies=dependencies,
            source_files=source_files,
            total_files=len(all_files),
            total_lines=sum(f.lines_of_code for f in source_files),
            complexity_metrics=complexity_metrics,
            architectural_patterns=architectural_patterns,
            testing_capabilities=testing_capabilities,
            ci_cd_capabilities=ci_cd_capabilities,
            analysis_duration=analysis_duration
        )
        
        print(f"Codebase analysis completed in {analysis_duration:.2f}s")
        return profile
    
    def _scan_files(self, project_path: Path) -> List[Path]:
        """Scan project directory for all relevant files."""
        all_files = []
        
        # Ignore patterns
        ignore_patterns = {
            '.git', '.svn', '.hg',  # VCS
            'node_modules', '__pycache__', '.pytest_cache',  # Dependencies/Cache
            'target', 'build', 'dist', 'bin', 'obj',  # Build outputs
            '.vscode', '.idea', '.vs',  # IDE
            '*.pyc', '*.class', '*.o', '*.so', '*.dll'  # Compiled files
        }
        
        for root, dirs, files in os.walk(project_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_patterns]
            
            for file in files:
                file_path = Path(root) / file
                if not any(pattern in str(file_path) for pattern in ignore_patterns):
                    all_files.append(file_path)
        
        return all_files
    
    def _detect_languages(self, files: List[Path], project_path: Path) -> List[LanguageInfo]:
        """Detect programming languages in the codebase."""
        language_stats = {}
        
        for file_path in files:
            language = self._detect_file_language(file_path)
            if language:
                if language not in language_stats:
                    language_stats[language] = {
                        'files': [],
                        'lines': 0,
                        'extensions': set()
                    }
                
                lines = self._count_lines(file_path)
                language_stats[language]['files'].append(file_path)
                language_stats[language]['lines'] += lines
                language_stats[language]['extensions'].add(file_path.suffix.lower())
        
        # Convert to LanguageInfo objects
        total_lines = sum(stats['lines'] for stats in language_stats.values())
        languages = []
        
        for lang_name, stats in language_stats.items():
            if stats['lines'] > 0:  # Only include languages with actual code
                confidence = min(1.0, len(stats['files']) / 10.0 + stats['lines'] / max(total_lines, 1))
                percentage = (stats['lines'] / max(total_lines, 1)) * 100
                
                lang_info = LanguageInfo(
                    name=lang_name,
                    confidence=confidence,
                    file_count=len(stats['files']),
                    total_lines=stats['lines'],
                    percentage=percentage,
                    extensions=stats['extensions'],
                    syntax_features=self._analyze_syntax_features(stats['files'][:5], lang_name)  # Sample files
                )
                languages.append(lang_info)
        
        # Sort by lines of code (primary language first)
        languages.sort(key=lambda x: x.total_lines, reverse=True)
        
        return languages
    
    def _detect_file_language(self, file_path: Path) -> Optional[str]:
        """Detect the programming language of a single file."""
        
        # Method 1: File extension
        extension = file_path.suffix.lower()
        if extension in self.extension_map:
            return self.extension_map[extension]
        
        # Method 2: Shebang detection
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                if first_line.startswith('#!'):
                    if 'python' in first_line:
                        return 'python'
                    elif 'node' in first_line or 'javascript' in first_line:
                        return 'javascript'
                    elif 'bash' in first_line or 'sh' in first_line:
                        return 'shell'
        except:
            pass
        
        # Method 3: Content analysis
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1024)  # Read first 1KB
                return self._analyze_content_for_language(content, file_path.name)
        except:
            pass
        
        return None
    
    def _analyze_content_for_language(self, content: str, filename: str) -> Optional[str]:
        """Analyze file content to determine language."""
        
        # Python patterns
        python_patterns = [
            r'import\s+\w+', r'from\s+\w+\s+import', r'def\s+\w+\s*\(',
            r'class\s+\w+\s*:', r'if\s+__name__\s*==\s*[\'"]__main__[\'"]'
        ]
        
        # JavaScript patterns
        js_patterns = [
            r'function\s+\w+\s*\(', r'const\s+\w+\s*=', r'let\s+\w+\s*=',
            r'var\s+\w+\s*=', r'require\s*\(', r'import\s+.*from'
        ]
        
        # Java patterns
        java_patterns = [
            r'public\s+class\s+\w+', r'public\s+static\s+void\s+main',
            r'import\s+java\.', r'@Override', r'extends\s+\w+'
        ]
        
        # C# patterns
        csharp_patterns = [
            r'using\s+System', r'namespace\s+\w+', r'public\s+class\s+\w+',
            r'\[.*\]', r'public\s+static\s+void\s+Main'
        ]
        
        # Go patterns
        go_patterns = [
            r'package\s+\w+', r'import\s+\(', r'func\s+\w+\s*\(',
            r'type\s+\w+\s+struct', r'go\s+\w+'
        ]
        
        # Rust patterns
        rust_patterns = [
            r'fn\s+\w+\s*\(', r'let\s+mut\s+\w+', r'use\s+std::',
            r'struct\s+\w+', r'impl\s+\w+'
        ]
        
        language_patterns = {
            'python': python_patterns,
            'javascript': js_patterns,
            'java': java_patterns,
            'csharp': csharp_patterns,
            'go': go_patterns,
            'rust': rust_patterns
        }
        
        scores = {}
        for language, patterns in language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                score += matches
            if score > 0:
                scores[language] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return None
    
    def _count_lines(self, file_path: Path) -> int:
        """Count lines of code in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # Count non-empty, non-comment lines (basic heuristic)
                code_lines = 0
                for line in lines:
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#') and not stripped.startswith('//'):
                        code_lines += 1
                return code_lines
        except:
            return 0
    
    def _analyze_syntax_features(self, files: List[Path], language: str) -> List[str]:
        """Analyze syntax features present in the language files."""
        features = set()
        
        # Language-specific feature detection
        feature_patterns = {
            'python': [
                (r'async\s+def', 'async_functions'),
                (r'@\w+', 'decorators'),
                (r'with\s+\w+', 'context_managers'),
                (r'yield\s+', 'generators'),
                (r'lambda\s+', 'lambda_functions')
            ],
            'javascript': [
                (r'async\s+function', 'async_functions'),
                (r'=>', 'arrow_functions'),
                (r'class\s+\w+', 'classes'),
                (r'import\s+.*from', 'es6_modules'),
                (r'const\s+\w+', 'const_declarations')
            ]
        }
        
        if language in feature_patterns:
            patterns = feature_patterns[language]
            for file_path in files[:3]:  # Check first 3 files
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        for pattern, feature in patterns:
                            if re.search(pattern, content):
                                features.add(feature)
                except:
                    continue
        
        return list(features)
    
    def _detect_frameworks(self, files: List[Path], project_path: Path, languages: List[LanguageInfo]) -> List[FrameworkInfo]:
        """Detect testing and development frameworks."""
        frameworks = []
        
        # Check for framework indicators in files
        for file_path in files:
            framework_info = self._detect_file_frameworks(file_path, languages)
            frameworks.extend(framework_info)
        
        # Check for framework config files
        config_frameworks = self._detect_framework_configs(files, project_path)
        frameworks.extend(config_frameworks)
        
        # Deduplicate and consolidate
        unique_frameworks = {}
        for fw in frameworks:
            key = f"{fw.name}_{fw.language}"
            if key not in unique_frameworks:
                unique_frameworks[key] = fw
            else:
                # Merge information
                existing = unique_frameworks[key]
                existing.confidence = max(existing.confidence, fw.confidence)
                existing.config_files.extend(fw.config_files)
                existing.indicators.extend(fw.indicators)
        
        return list(unique_frameworks.values())
    
    def _detect_file_frameworks(self, file_path: Path, languages: List[LanguageInfo]) -> List[FrameworkInfo]:
        """Detect frameworks from file content."""
        frameworks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(2048)  # Read first 2KB
                
                # Framework detection patterns
                for language_info in languages:
                    lang = language_info.name
                    if lang in self.framework_patterns:
                        for framework_name, patterns in self.framework_patterns[lang].items():
                            for pattern in patterns:
                                if re.search(pattern, content, re.IGNORECASE):
                                    frameworks.append(FrameworkInfo(
                                        name=framework_name,
                                        language=lang,
                                        confidence=0.8,
                                        indicators=[pattern]
                                    ))
                                    break
        except:
            pass
        
        return frameworks
    
    def _detect_framework_configs(self, files: List[Path], project_path: Path) -> List[FrameworkInfo]:
        """Detect frameworks from configuration files."""
        frameworks = []
        
        config_patterns = {
            'pytest.ini': FrameworkInfo('pytest', 'python', 0.9),
            'setup.cfg': FrameworkInfo('pytest', 'python', 0.7),
            'pyproject.toml': FrameworkInfo('pytest', 'python', 0.6),
            'package.json': FrameworkInfo('npm', 'javascript', 0.9),
            'jest.config.js': FrameworkInfo('jest', 'javascript', 0.9),
            'pom.xml': FrameworkInfo('maven', 'java', 0.9),
            'build.gradle': FrameworkInfo('gradle', 'java', 0.9),
            'Cargo.toml': FrameworkInfo('cargo', 'rust', 0.9),
            'go.mod': FrameworkInfo('go_modules', 'go', 0.9),
        }
        
        for file_path in files:
            filename = file_path.name.lower()
            if filename in config_patterns:
                fw_info = config_patterns[filename]
                fw_info.config_files = [str(file_path)]
                frameworks.append(fw_info)
        
        return frameworks
    
    def _detect_build_systems(self, files: List[Path], project_path: Path) -> List[BuildSystemInfo]:
        """Detect build systems."""
        build_systems = []
        
        build_patterns = {
            'Makefile': BuildSystemInfo('make', 'universal'),
            'CMakeLists.txt': BuildSystemInfo('cmake', 'cpp'),
            'setup.py': BuildSystemInfo('setuptools', 'python'),
            'pyproject.toml': BuildSystemInfo('poetry', 'python'),
            'package.json': BuildSystemInfo('npm', 'javascript'),
            'pom.xml': BuildSystemInfo('maven', 'java'),
            'build.gradle': BuildSystemInfo('gradle', 'java'),
            'Cargo.toml': BuildSystemInfo('cargo', 'rust'),
            'go.mod': BuildSystemInfo('go', 'go'),
        }
        
        for file_path in files:
            filename = file_path.name
            if filename in build_patterns:
                bs_info = build_patterns[filename]
                bs_info.config_files = [str(file_path)]
                build_systems.append(bs_info)
        
        return build_systems
    
    def _analyze_dependencies(self, files: List[Path], project_path: Path, languages: List[LanguageInfo]) -> List[DependencyInfo]:
        """Analyze project dependencies."""
        dependencies = []
        
        # Language-specific dependency analyzers
        for file_path in files:
            filename = file_path.name.lower()
            
            if filename == 'requirements.txt':
                dependencies.extend(self._parse_requirements_txt(file_path))
            elif filename == 'package.json':
                dependencies.extend(self._parse_package_json(file_path))
            elif filename == 'pom.xml':
                dependencies.extend(self._parse_pom_xml(file_path))
            elif filename == 'cargo.toml':
                dependencies.extend(self._parse_cargo_toml(file_path))
            elif filename == 'go.mod':
                dependencies.extend(self._parse_go_mod(file_path))
        
        return dependencies
    
    def _parse_requirements_txt(self, file_path: Path) -> List[DependencyInfo]:
        """Parse Python requirements.txt file."""
        dependencies = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse package==version format
                        if '==' in line:
                            name, version = line.split('==', 1)
                        elif '>=' in line:
                            name, version = line.split('>=', 1)
                        else:
                            name, version = line, None
                        
                        dependencies.append(DependencyInfo(
                            name=name.strip(),
                            version=version.strip() if version else None,
                            language='python',
                            dependency_type='runtime'
                        ))
        except:
            pass
        return dependencies
    
    def _parse_package_json(self, file_path: Path) -> List[DependencyInfo]:
        """Parse JavaScript package.json file."""
        dependencies = []
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Runtime dependencies
                for name, version in data.get('dependencies', {}).items():
                    dependencies.append(DependencyInfo(
                        name=name,
                        version=version,
                        language='javascript',
                        dependency_type='runtime'
                    ))
                
                # Dev dependencies
                for name, version in data.get('devDependencies', {}).items():
                    dependencies.append(DependencyInfo(
                        name=name,
                        version=version,
                        language='javascript',
                        dependency_type='dev'
                    ))
        except:
            pass
        return dependencies
    
    def _parse_pom_xml(self, file_path: Path) -> List[DependencyInfo]:
        """Parse Java pom.xml file (basic parsing)."""
        dependencies = []
        # Basic XML parsing would go here
        # For now, return empty list
        return dependencies
    
    def _parse_cargo_toml(self, file_path: Path) -> List[DependencyInfo]:
        """Parse Rust Cargo.toml file."""
        dependencies = []
        # TOML parsing would go here
        # For now, return empty list
        return dependencies
    
    def _parse_go_mod(self, file_path: Path) -> List[DependencyInfo]:
        """Parse Go go.mod file."""
        dependencies = []
        # Go mod parsing would go here
        # For now, return empty list
        return dependencies
    
    def _create_file_profiles(self, files: List[Path], languages: List[LanguageInfo], frameworks: List[FrameworkInfo]) -> List[FileInfo]:
        """Create detailed profiles for source files."""
        source_files = []
        
        # Create language lookup
        lang_extensions = {}
        for lang in languages:
            for ext in lang.extensions:
                lang_extensions[ext] = lang.name
        
        # Create framework lookup
        framework_lookup = {fw.language: fw.name for fw in frameworks}
        
        for file_path in files:
            # Only profile source code files
            if self._is_source_file(file_path):
                extension = file_path.suffix.lower()
                language = lang_extensions.get(extension, 'unknown')
                
                if language != 'unknown':
                    lines = self._count_lines(file_path)
                    complexity = self._calculate_file_complexity(file_path, language)
                    
                    try:
                        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                    except:
                        last_modified = datetime.now()
                    
                    file_info = FileInfo(
                        path=str(file_path),
                        language=language,
                        framework=framework_lookup.get(language),
                        lines_of_code=lines,
                        complexity_score=complexity,
                        last_modified=last_modified
                    )
                    source_files.append(file_info)
        
        return source_files
    
    def _is_source_file(self, file_path: Path) -> bool:
        """Check if file is a source code file."""
        source_extensions = {
            '.py', '.js', '.ts', '.java', '.cs', '.go', '.rs', '.cpp', '.c', '.h',
            '.php', '.rb', '.kt', '.swift', '.dart', '.scala', '.clj', '.hs',
            '.erl', '.ex', '.lua', '.r', '.m', '.vb', '.fs', '.pl', '.sh'
        }
        return file_path.suffix.lower() in source_extensions
    
    def _calculate_file_complexity(self, file_path: Path, language: str) -> float:
        """Calculate basic complexity score for a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Basic complexity indicators
                lines = len(content.split('\n'))
                functions = len(re.findall(r'(def |function |func |fn )', content))
                classes = len(re.findall(r'(class |struct |interface )', content))
                conditionals = len(re.findall(r'(if |while |for |switch )', content))
                
                # Simple complexity formula
                complexity = (functions * 2 + classes * 3 + conditionals * 1.5) / max(lines, 1)
                return min(complexity, 10.0)  # Cap at 10
        except:
            return 0.0
    
    def _detect_architectural_patterns(self, source_files: List[FileInfo], frameworks: List[FrameworkInfo]) -> List[str]:
        """Detect architectural patterns in the codebase."""
        patterns = []
        
        # Check for common patterns based on file structure
        file_paths = [f.path for f in source_files]
        
        # MVC pattern
        if any('controller' in path.lower() for path in file_paths) and \
           any('model' in path.lower() for path in file_paths) and \
           any('view' in path.lower() for path in file_paths):
            patterns.append('MVC')
        
        # Microservices pattern
        if len([p for p in file_paths if 'service' in p.lower()]) > 3:
            patterns.append('Microservices')
        
        # Repository pattern
        if any('repository' in path.lower() for path in file_paths):
            patterns.append('Repository')
        
        # Factory pattern
        if any('factory' in path.lower() for path in file_paths):
            patterns.append('Factory')
        
        return patterns
    
    def _calculate_complexity_metrics(self, source_files: List[FileInfo]) -> Dict[str, float]:
        """Calculate overall complexity metrics."""
        if not source_files:
            return {}
        
        total_files = len(source_files)
        total_lines = sum(f.lines_of_code for f in source_files)
        avg_complexity = sum(f.complexity_score for f in source_files) / total_files
        
        # Language distribution
        language_dist = {}
        for file_info in source_files:
            lang = file_info.language
            language_dist[lang] = language_dist.get(lang, 0) + 1
        
        return {
            'total_files': total_files,
            'total_lines': total_lines,
            'avg_complexity': avg_complexity,
            'max_complexity': max(f.complexity_score for f in source_files),
            'language_diversity': len(language_dist),
            'avg_file_size': total_lines / total_files if total_files > 0 else 0
        }
    
    def _determine_testing_capabilities(self, frameworks: List[FrameworkInfo], build_systems: List[BuildSystemInfo]) -> List[str]:
        """Determine testing capabilities based on detected frameworks."""
        capabilities = []
        
        # Check for testing frameworks
        testing_frameworks = ['pytest', 'jest', 'junit', 'nunit', 'xunit', 'rspec', 'mocha']
        for fw in frameworks:
            if fw.name.lower() in testing_frameworks:
                capabilities.append(f"{fw.name}_testing")
        
        # Check for build system testing support
        for bs in build_systems:
            if bs.name in ['maven', 'gradle', 'cargo', 'npm']:
                capabilities.append(f"{bs.name}_test_integration")
        
        return capabilities
    
    def _determine_ci_cd_capabilities(self, files: List[Path], project_path: Path) -> List[str]:
        """Determine CI/CD capabilities based on config files."""
        capabilities = []
        
        ci_files = {
            '.github/workflows': 'github_actions',
            '.gitlab-ci.yml': 'gitlab_ci',
            'Jenkinsfile': 'jenkins',
            '.travis.yml': 'travis_ci',
            '.circleci/config.yml': 'circleci',
            'azure-pipelines.yml': 'azure_devops'
        }
        
        for file_path in files:
            rel_path = str(file_path.relative_to(project_path))
            for ci_pattern, ci_system in ci_files.items():
                if ci_pattern in rel_path:
                    capabilities.append(ci_system)
        
        return capabilities
    
    def _get_default_languages(self) -> List[str]:
        """Get default supported languages."""
        return [
            'python', 'javascript', 'typescript', 'java', 'csharp', 'go', 'rust',
            'cpp', 'c', 'php', 'ruby', 'kotlin', 'swift', 'dart', 'scala',
            'clojure', 'haskell', 'erlang', 'elixir', 'lua'
        ]
    
    def _build_extension_map(self) -> Dict[str, str]:
        """Build mapping of file extensions to languages."""
        return {
            # Python
            '.py': 'python', '.pyw': 'python', '.pyi': 'python',
            # JavaScript/TypeScript
            '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript',
            # Java
            '.java': 'java', '.class': 'java',
            # C#
            '.cs': 'csharp', '.vb': 'csharp',
            # Go
            '.go': 'go',
            # Rust
            '.rs': 'rust',
            # C/C++
            '.c': 'c', '.cpp': 'cpp', '.cxx': 'cpp', '.cc': 'cpp', '.h': 'c', '.hpp': 'cpp',
            # PHP
            '.php': 'php', '.phtml': 'php',
            # Ruby
            '.rb': 'ruby', '.rbw': 'ruby',
            # Kotlin
            '.kt': 'kotlin', '.kts': 'kotlin',
            # Swift
            '.swift': 'swift',
            # Dart
            '.dart': 'dart',
            # Scala
            '.scala': 'scala', '.sc': 'scala',
            # Others
            '.lua': 'lua', '.r': 'r', '.m': 'matlab', '.pl': 'perl',
            '.sh': 'shell', '.bash': 'shell', '.zsh': 'shell'
        }
    
    def _build_framework_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Build framework detection patterns."""
        return {
            'python': {
                'pytest': [r'import pytest', r'def test_', r'@pytest\.'],
                'unittest': [r'import unittest', r'class.*TestCase', r'def test_'],
                'django': [r'from django', r'import django', r'DJANGO_SETTINGS_MODULE'],
                'flask': [r'from flask', r'import Flask', r'app = Flask'],
                'fastapi': [r'from fastapi', r'import FastAPI', r'app = FastAPI']
            },
            'javascript': {
                'jest': [r'import.*jest', r'describe\(', r'test\(', r'it\('],
                'mocha': [r'describe\(', r'it\(', r'require.*mocha'],
                'react': [r'import React', r'from [\'"]react[\'"]', r'jsx'],
                'vue': [r'import Vue', r'from [\'"]vue[\'"]', r'<template>'],
                'angular': [r'@Component', r'@Injectable', r'import.*@angular']
            },
            'java': {
                'junit': [r'import org\.junit', r'@Test', r'@Before', r'@After'],
                'spring': [r'@SpringBootApplication', r'@Controller', r'@Service'],
                'maven': [r'<groupId>', r'<artifactId>', r'<version>']
            }
        }
    
    def _build_build_system_patterns(self) -> Dict[str, List[str]]:
        """Build system detection patterns."""
        return {
            'maven': ['pom.xml'],
            'gradle': ['build.gradle', 'build.gradle.kts'],
            'npm': ['package.json'],
            'cargo': ['Cargo.toml'],
            'cmake': ['CMakeLists.txt'],
            'make': ['Makefile'],
            'poetry': ['pyproject.toml'],
            'pip': ['requirements.txt', 'setup.py']
        }
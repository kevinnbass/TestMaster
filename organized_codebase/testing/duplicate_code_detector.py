#!/usr/bin/env python3
"""
Agent C - Duplicate Code Detection Tool (Hours 26-28)
Advanced duplicate code detection with semantic analysis and refactoring recommendations
"""

import os
import ast
import json
import logging
import argparse
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import difflib


@dataclass
class CodeBlock:
    """Code block representation"""
    id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    content_hash: str
    tokens: List[str]
    ast_hash: str
    size: int
    complexity: float


@dataclass
class DuplicateGroup:
    """Group of duplicate code blocks"""
    group_id: str
    blocks: List[CodeBlock]
    similarity_score: float
    duplicate_type: str  # exact, near_exact, semantic
    refactor_potential: str  # high, medium, low
    savings_estimate: Dict[str, int]


@dataclass
class RefactoringRecommendation:
    """Refactoring recommendation for duplicates"""
    duplicate_group_id: str
    recommendation_type: str  # extract_function, extract_class, template_method
    target_location: str
    estimated_reduction: int
    complexity_reduction: float
    implementation_steps: List[str]


class ASTAnalyzer(ast.NodeVisitor):
    """AST analyzer for semantic similarity"""
    
    def __init__(self):
        self.structure = []
        self.variables = set()
        self.functions = set()
        self.classes = set()
        
    def visit_FunctionDef(self, node):
        self.structure.append(f"function:{node.name}")
        self.functions.add(node.name)
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        self.structure.append(f"class:{node.name}")
        self.classes.add(node.name)
        self.generic_visit(node)
        
    def visit_If(self, node):
        self.structure.append("if")
        self.generic_visit(node)
        
    def visit_For(self, node):
        self.structure.append("for")
        self.generic_visit(node)
        
    def visit_While(self, node):
        self.structure.append("while")
        self.generic_visit(node)
        
    def visit_Name(self, node):
        self.variables.add(node.id)
        self.generic_visit(node)
        
    def get_signature(self):
        """Get structural signature of AST"""
        return "|".join(self.structure)


class DuplicateCodeDetector:
    """Main duplicate code detection tool"""
    
    def __init__(self, root_dir: str, output_file: str):
        self.root_dir = Path(root_dir)
        self.output_file = output_file
        self.code_blocks = []
        self.duplicate_groups = []
        self.recommendations = []
        
        # Configuration
        self.min_block_size = 15  # Minimum lines for detection (increased for performance)
        self.min_similarity = 0.85  # Minimum similarity threshold (increased for performance)
        self.hash_buckets = defaultdict(list)  # For grouping similar blocks
        self.max_blocks = 20000  # Limit total blocks for performance
        
        self.statistics = {
            'total_files': 0,
            'total_blocks': 0,
            'total_duplicates': 0,
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'semantic_duplicates': 0,
            'potential_savings': {
                'lines': 0,
                'files': 0,
                'complexity_reduction': 0.0
            },
            'refactoring_opportunities': 0
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def detect_duplicates(self):
        """Detect duplicate code across the codebase"""
        print("Agent C - Duplicate Code Detection (Hours 26-28)")
        print(f"Analyzing: {self.root_dir}")
        print(f"Output: {self.output_file}")
        print("=" * 60)
        
        start_time = time.time()
        
        self.logger.info(f"Starting duplicate code detection for {self.root_dir}")
        
        # Extract code blocks
        self._extract_code_blocks()
        
        # Group similar blocks
        self._group_similar_blocks()
        
        # Analyze duplicates
        self._analyze_duplicates()
        
        # Generate recommendations
        self._generate_recommendations()
        
        duration = time.time() - start_time
        
        self._print_results(duration)
        self._save_results()
        
        self.logger.info(f"Duplicate code detection completed in {duration:.2f} seconds")
        self.logger.info(f"Duplicate code analysis report saved to {self.output_file}")
        
    def _extract_code_blocks(self):
        """Extract code blocks from all Python files"""
        python_files = list(self.root_dir.rglob("*.py"))
        self.statistics['total_files'] = len(python_files)
        
        self.logger.info(f"Extracting code blocks from {len(python_files)} Python files")
        
        for file_path in python_files:
            try:
                # Stop if we hit our limit for performance
                if len(self.code_blocks) >= self.max_blocks:
                    self.logger.info(f"Reached block limit ({self.max_blocks}), stopping extraction")
                    break
                    
                self._extract_blocks_from_file(file_path)
                
                if len(self.code_blocks) % 1000 == 0:
                    print(f"   Extracted {len(self.code_blocks)} code blocks...")
                    
            except Exception as e:
                self.logger.warning(f"Error processing {file_path}: {e}")
                
        self.statistics['total_blocks'] = len(self.code_blocks)
        self.logger.info(f"Extracted {len(self.code_blocks)} code blocks total")
        
    def _extract_blocks_from_file(self, file_path: Path):
        """Extract code blocks from a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Extract function blocks
            self._extract_function_blocks(file_path, lines)
            
            # Extract class method blocks  
            self._extract_class_blocks(file_path, lines)
            
            # Extract logical blocks (if/for/while/try)
            self._extract_logical_blocks(file_path, lines)
            
        except Exception as e:
            self.logger.warning(f"Error reading {file_path}: {e}")
            
    def _extract_function_blocks(self, file_path: Path, lines: List[str]):
        """Extract function definition blocks"""
        try:
            tree = ast.parse(''.join(lines))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.end_lineno and node.end_lineno - node.lineno >= self.min_block_size:
                        block_lines = lines[node.lineno-1:node.end_lineno]
                        content = ''.join(block_lines)
                        
                        block = self._create_code_block(
                            file_path, node.lineno, node.end_lineno, content
                        )
                        if block:
                            self.code_blocks.append(block)
                            
        except SyntaxError:
            # Skip files with syntax errors
            pass
        except Exception as e:
            self.logger.warning(f"Error extracting functions from {file_path}: {e}")
            
    def _extract_class_blocks(self, file_path: Path, lines: List[str]):
        """Extract class definition blocks"""
        try:
            tree = ast.parse(''.join(lines))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if node.end_lineno and node.end_lineno - node.lineno >= self.min_block_size:
                        block_lines = lines[node.lineno-1:node.end_lineno]
                        content = ''.join(block_lines)
                        
                        block = self._create_code_block(
                            file_path, node.lineno, node.end_lineno, content
                        )
                        if block:
                            self.code_blocks.append(block)
                            
                    # Extract method blocks within classes
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if item.end_lineno and item.end_lineno - item.lineno >= self.min_block_size:
                                method_lines = lines[item.lineno-1:item.end_lineno]
                                method_content = ''.join(method_lines)
                                
                                method_block = self._create_code_block(
                                    file_path, item.lineno, item.end_lineno, method_content
                                )
                                if method_block:
                                    self.code_blocks.append(method_block)
                                    
        except SyntaxError:
            # Skip files with syntax errors
            pass
        except Exception as e:
            self.logger.warning(f"Error extracting classes from {file_path}: {e}")
            
    def _extract_logical_blocks(self, file_path: Path, lines: List[str]):
        """Extract logical blocks like loops and conditionals"""
        try:
            tree = ast.parse(''.join(lines))
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    if hasattr(node, 'end_lineno') and node.end_lineno:
                        if node.end_lineno - node.lineno >= self.min_block_size:
                            block_lines = lines[node.lineno-1:node.end_lineno]
                            content = ''.join(block_lines)
                            
                            block = self._create_code_block(
                                file_path, node.lineno, node.end_lineno, content
                            )
                            if block:
                                self.code_blocks.append(block)
                                
        except SyntaxError:
            # Skip files with syntax errors
            pass
        except Exception as e:
            self.logger.warning(f"Error extracting logical blocks from {file_path}: {e}")
            
    def _create_code_block(self, file_path: Path, start_line: int, end_line: int, content: str) -> Optional[CodeBlock]:
        """Create a code block object"""
        try:
            # Skip very small blocks
            if end_line - start_line < self.min_block_size:
                return None
                
            # Normalize content for comparison
            normalized_content = self._normalize_content(content)
            
            # Generate hashes
            content_hash = hashlib.md5(normalized_content.encode()).hexdigest()
            
            # Tokenize for similarity analysis
            tokens = self._tokenize_content(normalized_content)
            
            # Generate AST hash for semantic similarity
            ast_hash = self._generate_ast_hash(content)
            
            # Calculate complexity
            complexity = self._calculate_complexity(content)
            
            block_id = f"{file_path.name}:{start_line}-{end_line}"
            
            return CodeBlock(
                id=block_id,
                file_path=str(file_path),
                start_line=start_line,
                end_line=end_line,
                content=content,
                content_hash=content_hash,
                tokens=tokens,
                ast_hash=ast_hash,
                size=end_line - start_line,
                complexity=complexity
            )
            
        except Exception as e:
            self.logger.warning(f"Error creating code block: {e}")
            return None
            
    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison"""
        lines = content.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Remove comments
            if '#' in line:
                line = line[:line.index('#')]
                
            # Strip whitespace
            line = line.strip()
            
            # Skip empty lines
            if line:
                normalized_lines.append(line)
                
        return '\n'.join(normalized_lines)
        
    def _tokenize_content(self, content: str) -> List[str]:
        """Tokenize content for similarity analysis"""
        import re
        
        # Remove strings and comments
        content = re.sub(r'["\'].*?["\']', 'STRING', content)
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        
        # Extract tokens
        tokens = re.findall(r'\w+', content)
        
        # Filter out variables and keep keywords/operators
        python_keywords = {
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except',
            'finally', 'with', 'import', 'from', 'return', 'yield', 'break',
            'continue', 'pass', 'raise', 'assert', 'del', 'global', 'nonlocal',
            'lambda', 'and', 'or', 'not', 'in', 'is'
        }
        
        filtered_tokens = [token for token in tokens if token.lower() in python_keywords]
        return filtered_tokens
        
    def _generate_ast_hash(self, content: str) -> str:
        """Generate AST-based hash for semantic similarity"""
        try:
            tree = ast.parse(content)
            analyzer = ASTAnalyzer()
            analyzer.visit(tree)
            signature = analyzer.get_signature()
            return hashlib.md5(signature.encode()).hexdigest()
        except:
            return ""
            
    def _calculate_complexity(self, content: str) -> float:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        # Count decision points
        complexity += content.count('if ')
        complexity += content.count('elif ')
        complexity += content.count('for ')
        complexity += content.count('while ')
        complexity += content.count('except ')
        complexity += content.count('and ')
        complexity += content.count('or ')
        
        return float(complexity)
        
    def _group_similar_blocks(self):
        """Group similar code blocks"""
        print("   Grouping similar code blocks...")
        
        # Group by exact hash first
        hash_groups = defaultdict(list)
        for block in self.code_blocks:
            hash_groups[block.content_hash].append(block)
            
        # Find exact duplicates
        for hash_value, blocks in hash_groups.items():
            if len(blocks) > 1:
                group = DuplicateGroup(
                    group_id=f"exact_{hash_value[:8]}",
                    blocks=blocks,
                    similarity_score=1.0,
                    duplicate_type="exact",
                    refactor_potential="high",
                    savings_estimate={
                        'lines': sum(block.size for block in blocks[1:]),
                        'files': len(set(block.file_path for block in blocks)) - 1
                    }
                )
                self.duplicate_groups.append(group)
                self.statistics['exact_duplicates'] += len(blocks) - 1
                
        # Find semantic duplicates using AST
        ast_groups = defaultdict(list)
        for block in self.code_blocks:
            if block.ast_hash:
                ast_groups[block.ast_hash].append(block)
                
        for ast_hash, blocks in ast_groups.items():
            if len(blocks) > 1:
                # Check if already covered by exact duplicates
                existing_hashes = {group.blocks[0].content_hash for group in self.duplicate_groups}
                if blocks[0].content_hash not in existing_hashes:
                    group = DuplicateGroup(
                        group_id=f"semantic_{ast_hash[:8]}",
                        blocks=blocks,
                        similarity_score=0.9,
                        duplicate_type="semantic",
                        refactor_potential="medium",
                        savings_estimate={
                            'lines': sum(block.size for block in blocks[1:]) // 2,  # Conservative
                            'files': len(set(block.file_path for block in blocks)) - 1
                        }
                    )
                    self.duplicate_groups.append(group)
                    self.statistics['semantic_duplicates'] += len(blocks) - 1
                    
        # Find near duplicates using token similarity
        self._find_near_duplicates()
        
    def _find_near_duplicates(self):
        """Find near-duplicate blocks using token similarity (optimized for large datasets)"""
        if len(self.code_blocks) > 10000:
            self.logger.info("Large dataset detected, using sampling approach for near-duplicate detection")
            # Sample for performance on large datasets
            sample_size = min(5000, len(self.code_blocks) // 4)
            sampled_blocks = self.code_blocks[:sample_size]
        else:
            sampled_blocks = self.code_blocks
            
        processed = set()
        
        for i, block1 in enumerate(sampled_blocks):
            if block1.id in processed:
                continue
                
            similar_blocks = [block1]
            processed.add(block1.id)
            
            # Limit comparisons for performance
            comparison_limit = min(1000, len(sampled_blocks) - i - 1)
            
            for j, block2 in enumerate(sampled_blocks[i+1:i+1+comparison_limit]):
                if block2.id in processed:
                    continue
                    
                # Skip if different sizes significantly
                if abs(block1.size - block2.size) > block1.size * 0.3:
                    continue
                    
                # Calculate token similarity
                similarity = self._calculate_token_similarity(block1.tokens, block2.tokens)
                
                if similarity >= self.min_similarity:
                    similar_blocks.append(block2)
                    processed.add(block2.id)
                    
            # Create group if we found near duplicates
            if len(similar_blocks) > 1:
                # Check if already covered
                existing_ids = set()
                for group in self.duplicate_groups:
                    existing_ids.update(block.id for block in group.blocks)
                    
                new_blocks = [block for block in similar_blocks if block.id not in existing_ids]
                
                if len(new_blocks) > 1:
                    avg_similarity = 0.85  # Estimated
                    group = DuplicateGroup(
                        group_id=f"near_{hashlib.md5(''.join(block.id for block in new_blocks).encode()).hexdigest()[:8]}",
                        blocks=new_blocks,
                        similarity_score=avg_similarity,
                        duplicate_type="near_exact",
                        refactor_potential="medium",
                        savings_estimate={
                            'lines': sum(block.size for block in new_blocks[1:]) // 3,  # Conservative
                            'files': len(set(block.file_path for block in new_blocks)) - 1
                        }
                    )
                    self.duplicate_groups.append(group)
                    self.statistics['near_duplicates'] += len(new_blocks) - 1
                    
    def _calculate_token_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate similarity between token lists"""
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
            
        # Use sequence matcher for similarity
        matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
        return matcher.ratio()
        
    def _analyze_duplicates(self):
        """Analyze duplicate groups for insights"""
        print("   Analyzing duplicate patterns...")
        
        self.statistics['total_duplicates'] = len(self.duplicate_groups)
        
        # Calculate total savings
        total_lines = 0
        total_files = set()
        total_complexity = 0.0
        
        for group in self.duplicate_groups:
            total_lines += group.savings_estimate['lines']
            for block in group.blocks:
                total_files.add(block.file_path)
                total_complexity += block.complexity
                
        self.statistics['potential_savings'] = {
            'lines': total_lines,
            'files': len(total_files),
            'complexity_reduction': total_complexity * 0.3  # Estimated reduction
        }
        
    def _generate_recommendations(self):
        """Generate refactoring recommendations"""
        print("   Generating refactoring recommendations...")
        
        for group in self.duplicate_groups:
            recommendation = self._create_refactoring_recommendation(group)
            if recommendation:
                self.recommendations.append(recommendation)
                
        self.statistics['refactoring_opportunities'] = len(self.recommendations)
        
    def _create_refactoring_recommendation(self, group: DuplicateGroup) -> Optional[RefactoringRecommendation]:
        """Create refactoring recommendation for duplicate group"""
        try:
            # Determine refactoring strategy
            avg_size = sum(block.size for block in group.blocks) / len(group.blocks)
            
            if avg_size > 50:
                rec_type = "extract_class"
            elif avg_size > 20:
                rec_type = "extract_function"
            else:
                rec_type = "template_method"
                
            # Find common parent directory for target location
            file_paths = [Path(block.file_path) for block in group.blocks]
            common_parent = Path(os.path.commonpath([str(path.parent) for path in file_paths]))
            target_location = str(common_parent / f"shared_{group.group_id}.py")
            
            # Calculate estimated reduction
            total_lines = sum(block.size for block in group.blocks)
            estimated_reduction = total_lines - avg_size  # Keep one copy
            
            # Calculate complexity reduction
            total_complexity = sum(block.complexity for block in group.blocks)
            complexity_reduction = total_complexity * 0.4  # Estimated
            
            # Generate implementation steps
            steps = self._generate_implementation_steps(group, rec_type, target_location)
            
            return RefactoringRecommendation(
                duplicate_group_id=group.group_id,
                recommendation_type=rec_type,
                target_location=target_location,
                estimated_reduction=int(estimated_reduction),
                complexity_reduction=complexity_reduction,
                implementation_steps=steps
            )
            
        except Exception as e:
            self.logger.warning(f"Error creating recommendation for group {group.group_id}: {e}")
            return None
            
    def _generate_implementation_steps(self, group: DuplicateGroup, rec_type: str, target_location: str) -> List[str]:
        """Generate implementation steps for refactoring"""
        steps = []
        
        if rec_type == "extract_function":
            steps = [
                f"1. Create shared function in {target_location}",
                "2. Identify common parameters across duplicates",
                "3. Extract shared logic into parameterized function", 
                "4. Replace duplicate blocks with function calls",
                "5. Add unit tests for extracted function",
                "6. Verify all original functionality preserved"
            ]
        elif rec_type == "extract_class":
            steps = [
                f"1. Create shared class in {target_location}",
                "2. Identify common attributes and methods",
                "3. Design class interface for shared functionality",
                "4. Implement shared class with proper encapsulation",
                "5. Replace duplicate code with class instantiation",
                "6. Add comprehensive tests for shared class"
            ]
        elif rec_type == "template_method":
            steps = [
                f"1. Create template method in {target_location}",
                "2. Identify algorithm structure and variation points",
                "3. Define abstract template with hook methods",
                "4. Implement concrete variations as subclasses",
                "5. Replace duplicates with template instantiation",
                "6. Test all template variations"
            ]
            
        return steps
        
    def _print_results(self, duration):
        """Print duplicate detection results"""
        print(f"\nDuplicate Code Detection Results:")
        print(f"   Files Analyzed: {self.statistics['total_files']:,}")
        print(f"   Code Blocks Extracted: {self.statistics['total_blocks']:,}")
        print(f"   Duplicate Groups Found: {self.statistics['total_duplicates']}")
        print(f"   Exact Duplicates: {self.statistics['exact_duplicates']}")
        print(f"   Near Duplicates: {self.statistics['near_duplicates']}")
        print(f"   Semantic Duplicates: {self.statistics['semantic_duplicates']}")
        print(f"   Analysis Duration: {duration:.2f} seconds")
        
        print(f"\nPotential Savings:")
        print(f"   Lines of Code: {self.statistics['potential_savings']['lines']:,}")
        print(f"   Files Affected: {self.statistics['potential_savings']['files']}")
        print(f"   Complexity Reduction: {self.statistics['potential_savings']['complexity_reduction']:.1f}")
        
        print(f"\nRefactoring Opportunities: {self.statistics['refactoring_opportunities']}")
        
        if self.recommendations:
            print(f"\nTop Refactoring Recommendations:")
            for rec in sorted(self.recommendations, key=lambda x: x.estimated_reduction, reverse=True)[:5]:
                print(f"   - {rec.recommendation_type}: {rec.estimated_reduction} lines saved")
                
        print(f"\nDuplicate code analysis complete! Report saved to {self.output_file}")
        
    def _save_results(self):
        """Save duplicate detection results to JSON file"""
        results = {
            'metadata': {
                'analysis_type': 'duplicate_code_detection',
                'timestamp': datetime.now().isoformat(),
                'root_directory': str(self.root_dir),
                'agent': 'Agent C',
                'phase': 'Hours 26-28: Duplicate Code Detection'
            },
            'statistics': self.statistics,
            'duplicate_groups': [asdict(group) for group in self.duplicate_groups],
            'refactoring_recommendations': [asdict(rec) for rec in self.recommendations],
            'summary': {
                'total_analysis': {
                    'files': self.statistics['total_files'],
                    'blocks': self.statistics['total_blocks'],
                    'duplicates_found': self.statistics['total_duplicates']
                },
                'savings_potential': self.statistics['potential_savings'],
                'next_steps': [
                    'Review high-priority refactoring recommendations',
                    'Implement extract_function patterns for largest duplicates',
                    'Create shared utility modules for common patterns',
                    'Add automated duplicate detection to CI/CD pipeline'
                ]
            }
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Agent C Duplicate Code Detector')
    parser.add_argument('--root', required=True, help='Root directory to analyze')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--min-size', type=int, default=10, help='Minimum block size for detection')
    parser.add_argument('--min-similarity', type=float, default=0.8, help='Minimum similarity threshold')
    
    args = parser.parse_args()
    
    detector = DuplicateCodeDetector(args.root, args.output)
    detector.min_block_size = args.min_size
    detector.min_similarity = args.min_similarity
    detector.detect_duplicates()


if __name__ == "__main__":
    main()
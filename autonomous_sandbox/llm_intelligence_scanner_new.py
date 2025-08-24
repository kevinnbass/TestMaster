#!/usr/bin/env python3
"""
LLM Intelligence Scanner - Modular Coordinator
==============================================

Main coordinator for the comprehensive LLM-based code intelligence system.
This is the modular version that uses focused components.

Author: Intelligence-Driven Reorganization System
Version: 4.0
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our modular components
from scanner_models import LLMProvider, LLMIntelligenceMap
from scanner_llm_clients import MockLLMClient, OpenAIClient, OllamaClient
from scanner_analysis import LLMAnalysisEngine, InsightsGenerator
from scanner_storage import CacheManager, FileOperations, IntelligenceMapBuilder

# Import the existing exclusions logic
try:
    from reorganizer import CodebaseReorganizer
    HAS_REORGANIZER = True
except ImportError:
    HAS_REORGANIZER = False


class LLMIntelligenceScanner:
    """
    LLM-powered intelligence scanner that analyzes Python files
    to create comprehensive reorganization intelligence.
    """

    def __init__(self, root_dir: Path, provider: LLMProvider = LLMProvider.MOCK,
                 api_key: Optional[str] = None, model: str = "gpt-4",
                 max_concurrent: int = 3, cache_dir: Optional[Path] = None):
        """
        Initialize the LLM intelligence scanner.

        Args:
            root_dir: Root directory to scan
            provider: LLM provider to use
            api_key: API key for the provider
            model: Model name to use
            max_concurrent: Maximum concurrent LLM requests
            cache_dir: Directory for caching results
        """
        self.root_dir = root_dir.resolve()
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.max_concurrent = max_concurrent
        self.cache_dir = cache_dir or self.root_dir / "tools" / "codebase_reorganizer" / "llm_cache"

        # Setup exclusions (same as existing system)
        self.exclusions = self._get_exclusions()

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LLM client
        self.llm_client = self._initialize_llm_client()

        # Initialize our modular components
        self.cache_manager = CacheManager(self.cache_dir, self.logger)
        self.file_ops = FileOperations(self.root_dir, self.exclusions, self.logger)
        self.analysis_engine = LLMAnalysisEngine(self.llm_client, self.logger)
        self.insights_generator = InsightsGenerator()
        self.map_builder = IntelligenceMapBuilder(self.logger)

        self.logger.info("LLM Intelligence Scanner initialized")
        self.logger.info(f"Using provider: {provider.value}, model: {model}")

    def _get_exclusions(self) -> set[str]:
        """Get the same exclusions as the existing reorganizer"""
        exclusions = {
            'archive', 'archives', 'PRODUCTION_PACKAGES',
            'agency-swarm', 'autogen', 'agent-squad', 'agentops',
            'agentscope', 'AgentVerse', 'crewAI', 'CodeGraph',
            'falkordb-py', 'AWorld', 'MetaGPT', 'metagpt',
            'PraisonAI', 'praisonai', 'llama-agents', 'phidata', 'swarms',
            'lagent', 'langgraph-supervisor-py',
            '__pycache__', '.git', 'node_modules', 'htmlcov',
            '.pytest_cache', 'tests', 'test_sessions', 'testmaster_sessions'
        }

        if HAS_REORGANIZER:
            try:
                reorganizer = CodebaseReorganizer(self.root_dir)
                exclusions.update(reorganizer.exclusions)
            except:
                pass

        return exclusions

    def _initialize_llm_client(self):
        """Initialize the appropriate LLM client"""
        if self.provider == LLMProvider.OPENAI:
            return OpenAIClient(self.api_key, self.model)
        elif self.provider == LLMProvider.OLLAMA:
            return OllamaClient(self.model)
        else:  # MOCK or default
            return MockLLMClient()

    def _setup_logging(self) -> None:
        """Setup comprehensive logging"""
        import logging
        log_dir = self.root_dir / "tools" / "codebase_reorganizer" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"llm_intelligence_scan_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def scan_and_analyze(self, output_file: Optional[Path] = None) -> LLMIntelligenceMap:
        """
        Perform comprehensive LLM-powered intelligence scan.

        Args:
            output_file: Optional path to save the intelligence map

        Returns:
            Complete LLM intelligence map
        """
        self.logger.info("Starting LLM intelligence scan...")

        # Find all Python files to analyze
        python_files = self.file_ops.find_python_files()

        # Analyze files with LLM (with concurrency control)
        intelligence_entries = []
        total_lines = 0

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_file = {
                executor.submit(self._analyze_file_with_llm, file_path): file_path
                for file_path in python_files
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    entry = future.result()
                    if entry:
                        intelligence_entries.append(entry)
                        total_lines += entry.line_count
                        self.logger.info(f"Analyzed: {entry.relative_path}")
                    else:
                        self.logger.warning(f"Failed to analyze: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error analyzing {file_path}: {e}")

        # Build intelligence map
        intelligence_map = self.map_builder.build_intelligence_map(
            intelligence_entries, self.insights_generator
        )

        # Save to file if requested
        if output_file:
            self.map_builder.save_intelligence_map(intelligence_map, output_file)

        # Update cache
        self.cache_manager.update_cache(intelligence_entries)
        self.cache_manager.save_cache()

        self.logger.info("LLM intelligence scan completed!")
        self.logger.info(f"Analyzed {len(intelligence_entries)} files with {total_lines} total lines")

        return intelligence_map

    def _analyze_file_with_llm(self, file_path: Path):
        """Analyze a single file using LLM"""
        # Check cache first
        cached_entry = self.cache_manager.get_cached_entry(file_path, self.root_dir)
        if cached_entry:
            return LLMIntelligenceEntry(**cached_entry)

        # Perform fresh analysis
        return self.analysis_engine.analyze_file_with_llm(file_path, self.root_dir)


def main():
    """Main function to run the LLM intelligence scanner"""
    parser = argparse.ArgumentParser(description="LLM Intelligence Scanner")
    parser.add_argument("--root", type=str, default=".",
                      help="Root directory to scan")
    parser.add_argument("--provider", type=str, default="mock",
                      choices=["openai", "anthropic", "groq", "ollama", "mock"],
                      help="LLM provider to use")
    parser.add_argument("--model", type=str, default="gpt-4",
                      help="Model name to use")
    parser.add_argument("--output", type=str, default="llm_intelligence_map.json",
                      help="Output file path")
    parser.add_argument("--max-concurrent", type=int, default=3,
                      help="Maximum concurrent requests")
    parser.add_argument("--api-key", type=str,
                      help="API key for the LLM provider")

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    output_file = Path(args.output)

    provider = LLMProvider(args.provider)

    scanner = LLMIntelligenceScanner(
        root_dir=root_dir,
        provider=provider,
        api_key=args.api_key,
        model=args.model,
        max_concurrent=args.max_concurrent
    )

    print("🧠 Starting LLM Intelligence Scan...")
    print(f"Root directory: {root_dir}")
    print(f"Provider: {provider.value}")
    print(f"Model: {args.model}")
    print(f"Output: {output_file}")

    intelligence_map = scanner.scan_and_analyze(output_file)

    print("
✅ Scan completed!"    print(f"Files analyzed: {intelligence_map.total_files_scanned}")
    print(f"Total lines: {intelligence_map.total_lines_analyzed}")
    print(f"Output saved to: {output_file}")

    # Print classification summary
    print("
📊 Classification Summary:"    for category, count in sorted(intelligence_map.classification_summary.items(),
                               key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count}")

    if intelligence_map.reorganization_insights['problematic_modules']:
        print("
⚠️  Problematic modules identified:"        for module in intelligence_map.reorganization_insights['problematic_modules'][:5]:
            print(f"  {module['path']}: {', '.join(module['issues'])}")


if __name__ == "__main__":
    main()

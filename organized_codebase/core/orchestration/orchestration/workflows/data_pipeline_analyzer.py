"""
Data Pipeline Analysis for ETL Patterns

This module analyzes data pipelines and ETL (Extract, Transform, Load) patterns
in code, identifying data flow, transformations, quality issues, and optimization
opportunities.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import logging

from ..base import BaseAnalyzer


class PipelineStage(Enum):
    """Types of pipeline stages"""
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    VALIDATE = "validate"
    CLEAN = "clean"
    AGGREGATE = "aggregate"
    JOIN = "join"
    FILTER = "filter"
    ENRICH = "enrich"
    SPLIT = "split"
    MERGE = "merge"
    DEDUPLICATE = "deduplicate"
    PARTITION = "partition"
    CACHE = "cache"
    CHECKPOINT = "checkpoint"


class DataSource(Enum):
    """Types of data sources"""
    DATABASE = "database"
    FILE = "file"
    API = "api"
    STREAM = "stream"
    QUEUE = "queue"
    CACHE = "cache"
    S3 = "s3"
    KAFKA = "kafka"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    MONGODB = "mongodb"
    BIGQUERY = "bigquery"


class TransformationType(Enum):
    """Types of data transformations"""
    MAPPING = "mapping"
    FILTERING = "filtering"
    AGGREGATION = "aggregation"
    JOINING = "joining"
    PIVOTING = "pivoting"
    UNPIVOTING = "unpivoting"
    NORMALIZATION = "normalization"
    DENORMALIZATION = "denormalization"
    TYPE_CONVERSION = "type_conversion"
    ENCODING = "encoding"
    DECODING = "decoding"
    VALIDATION = "validation"
    ENRICHMENT = "enrichment"
    DEDUPLICATION = "deduplication"
    SAMPLING = "sampling"


@dataclass
class Pipeline:
    """Represents a data pipeline"""
    name: str
    stages: List['PipelineStage']
    data_sources: List[DataSource]
    transformations: List[TransformationType]
    file_path: str
    line_number: int
    complexity: int
    estimated_latency: float  # in seconds
    data_volume: str  # estimated
    parallelizable: bool
    error_handling: bool
    monitoring: bool
    idempotent: bool


@dataclass
class DataFlow:
    """Represents data flow between components"""
    source: str
    destination: str
    data_type: str
    transformation: Optional[str]
    volume: str
    frequency: str
    is_streaming: bool
    has_backpressure: bool


@dataclass
class PipelineIssue:
    """Represents a pipeline issue"""
    pipeline: str
    issue_type: str
    severity: str  # critical, high, medium, low
    description: str
    impact: str
    recommendation: str
    file_path: str
    line_number: int
    estimated_fix_effort: float  # in hours


class DataPipelineAnalyzer(BaseAnalyzer):
    """Analyzes data pipelines and ETL patterns"""
    
    def __init__(self):
        super().__init__()
        self.pipelines: List[Pipeline] = []
        self.data_flows: List[DataFlow] = []
        self.issues: List[PipelineIssue] = []
        
        # ETL framework patterns
        self.etl_frameworks = {
            "pandas": ["import pandas", "from pandas"],
            "spark": ["from pyspark", "import pyspark"],
            "airflow": ["from airflow", "import airflow"],
            "luigi": ["import luigi", "from luigi"],
            "prefect": ["from prefect", "import prefect"],
            "dagster": ["from dagster", "import dagster"],
            "dbt": ["import dbt", "from dbt"],
            "beam": ["import apache_beam", "from apache_beam"],
            "dask": ["import dask", "from dask"],
            "ray": ["import ray", "from ray"],
        }
        
        # Data source patterns
        self.source_patterns = {
            DataSource.DATABASE: [
                r"\.connect\(", r"create_engine\(", r"\.query\(",
                r"SELECT\s+", r"INSERT\s+INTO", r"UPDATE\s+",
                r"\.execute\(", r"\.fetchall\(", r"\.fetchone\("
            ],
            DataSource.FILE: [
                r"open\(", r"\.read_csv\(", r"\.read_excel\(",
                r"\.read_json\(", r"\.read_parquet\(", r"\.read_feather\(",
                r"with\s+open\(", r"\.read\(\)", r"\.readlines\("
            ],
            DataSource.API: [
                r"requests\.", r"\.get\(", r"\.post\(", r"\.put\(",
                r"aiohttp\.", r"httpx\.", r"urllib\.", r"fetch\("
            ],
            DataSource.STREAM: [
                r"StreamingContext\(", r"\.stream\(", r"kafka\.Consumer",
                r"\.subscribe\(", r"\.poll\(", r"websocket\."
            ],
            DataSource.S3: [
                r"boto3\.client\('s3'\)", r"s3\.Bucket\(", r"\.upload_file\(",
                r"\.download_file\(", r"s3fs\.", r"s3://", r"\.put_object\("
            ],
            DataSource.KAFKA: [
                r"KafkaConsumer\(", r"KafkaProducer\(", r"\.send\(",
                r"\.poll\(", r"\.subscribe\(", r"kafka\."
            ],
        }
        
        # Transformation patterns
        self.transformation_patterns = {
            TransformationType.MAPPING: [
                r"\.map\(", r"\.apply\(", r"\.transform\(",
                r"\.select\(", r"\.withColumn\("
            ],
            TransformationType.FILTERING: [
                r"\.filter\(", r"\.where\(", r"\.query\(",
                r"\[.*?\]", r"\.loc\[", r"\.iloc\["
            ],
            TransformationType.AGGREGATION: [
                r"\.groupby\(", r"\.agg\(", r"\.sum\(", r"\.mean\(",
                r"\.count\(", r"\.aggregate\(", r"\.reduce\("
            ],
            TransformationType.JOINING: [
                r"\.join\(", r"\.merge\(", r"\.concat\(",
                r"\.union\(", r"\.append\("
            ],
            TransformationType.PIVOTING: [
                r"\.pivot\(", r"\.pivot_table\(", r"\.crosstab\(",
                r"\.unstack\(", r"\.stack\("
            ],
        }
        
        # Performance anti-patterns
        self.anti_patterns = {
            "row_iteration": {
                "patterns": [r"\.iterrows\(\)", r"for.*in.*\.values"],
                "impact": "Very slow for large datasets",
                "alternative": "Use vectorized operations"
            },
            "repeated_io": {
                "patterns": [r"for.*:\s*.*\.read", r"for.*:\s*.*\.write"],
                "impact": "High I/O overhead",
                "alternative": "Batch I/O operations"
            },
            "memory_inefficient": {
                "patterns": [r"\.copy\(\)", r"\.tolist\(\)", r"list\(.*DataFrame"],
                "impact": "High memory usage",
                "alternative": "Use iterators or chunks"
            },
            "no_partition": {
                "patterns": [r"\.read_csv\([^)]*\)", r"\.read_json\([^)]*\)"],
                "impact": "Cannot parallelize large files",
                "alternative": "Use partitioned formats"
            },
            "synchronous_api": {
                "patterns": [r"requests\.get\(", r"time\.sleep\("],
                "impact": "Blocking operations reduce throughput",
                "alternative": "Use async operations"
            },
        }
        
    def analyze(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Analyze data pipelines and ETL patterns"""
        if file_path:
            self._analyze_file(Path(file_path))
        else:
            self._analyze_directory()
            
        return self._generate_report()
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single file for pipeline patterns"""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Detect framework usage
            frameworks = self._detect_frameworks(content)
            
            # Analyze pipeline definitions
            self._analyze_pipeline_definitions(tree, content, str(file_path))
            
            # Analyze data flows
            self._analyze_data_flows(tree, content, str(file_path))
            
            # Analyze transformations
            self._analyze_transformations(tree, content, str(file_path))
            
            # Detect anti-patterns
            self._detect_anti_patterns(content, str(file_path))
            
            # Analyze pipeline quality
            self._analyze_pipeline_quality(tree, content, str(file_path))
            
        except Exception as e:
            logging.error(f"Error analyzing {file_path}: {e}")
    
    def _detect_frameworks(self, content: str) -> Set[str]:
        """Detect ETL frameworks used"""
        frameworks = set()
        for framework, patterns in self.etl_frameworks.items():
            for pattern in patterns:
                if pattern in content:
                    frameworks.add(framework)
                    break
        return frameworks
    
    def _analyze_pipeline_definitions(self, tree: ast.AST, content: str, 
                                    file_path: str) -> None:
        """Analyze pipeline definitions"""
        for node in ast.walk(tree):
            # Detect Airflow DAGs
            if isinstance(node, ast.ClassDef):
                if any(base.id == "DAG" for base in node.bases 
                      if isinstance(base, ast.Name)):
                    self._analyze_airflow_dag(node, content, file_path)
            
            # Detect function-based pipelines
            elif isinstance(node, ast.FunctionDef):
                if self._is_pipeline_function(node, content):
                    self._analyze_pipeline_function(node, content, file_path)
            
            # Detect Spark pipelines
            elif isinstance(node, ast.Assign):
                if self._is_spark_pipeline(node, content):
                    self._analyze_spark_pipeline(node, content, file_path)
    
    def _is_pipeline_function(self, node: ast.FunctionDef, content: str) -> bool:
        """Check if function is a pipeline"""
        pipeline_keywords = ["pipeline", "etl", "transform", "process", "flow"]
        name_lower = node.name.lower()
        
        # Check function name
        if any(keyword in name_lower for keyword in pipeline_keywords):
            return True
        
        # Check for pipeline operations
        function_content = ast.unparse(node)
        pipeline_ops = ["extract", "transform", "load", "read", "write", "process"]
        return sum(1 for op in pipeline_ops if op in function_content) >= 2
    
    def _analyze_pipeline_function(self, node: ast.FunctionDef, content: str,
                                  file_path: str) -> None:
        """Analyze a pipeline function"""
        stages = []
        data_sources = []
        transformations = []
        
        # Analyze function body for stages
        for stmt in ast.walk(node):
            # Detect data sources
            if isinstance(stmt, ast.Call):
                call_str = ast.unparse(stmt)
                for source, patterns in self.source_patterns.items():
                    if any(re.search(pattern, call_str) for pattern in patterns):
                        data_sources.append(source)
                
                # Detect transformations
                for trans_type, patterns in self.transformation_patterns.items():
                    if any(re.search(pattern, call_str) for pattern in patterns):
                        transformations.append(trans_type)
        
        # Estimate complexity
        complexity = self._calculate_pipeline_complexity(node)
        
        # Check for parallelization
        parallelizable = self._check_parallelizable(node)
        
        # Check error handling
        has_error_handling = self._has_error_handling(node)
        
        pipeline = Pipeline(
            name=node.name,
            stages=stages,
            data_sources=list(set(data_sources)),
            transformations=list(set(transformations)),
            file_path=file_path,
            line_number=node.lineno,
            complexity=complexity,
            estimated_latency=self._estimate_latency(node),
            data_volume=self._estimate_data_volume(node),
            parallelizable=parallelizable,
            error_handling=has_error_handling,
            monitoring=self._has_monitoring(node),
            idempotent=self._is_idempotent(node)
        )
        
        self.pipelines.append(pipeline)
        
        # Check for issues
        self._check_pipeline_issues(pipeline, node)
    
    def _calculate_pipeline_complexity(self, node: ast.AST) -> int:
        """Calculate pipeline complexity"""
        complexity = 0
        
        for child in ast.walk(node):
            # Count operations
            if isinstance(child, ast.Call):
                complexity += 1
            # Count loops
            elif isinstance(child, (ast.For, ast.While)):
                complexity += 3
            # Count conditionals
            elif isinstance(child, ast.If):
                complexity += 2
            # Count try-except
            elif isinstance(child, ast.Try):
                complexity += 2
        
        return complexity
    
    def _estimate_latency(self, node: ast.AST) -> float:
        """Estimate pipeline latency in seconds"""
        latency = 0.0
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_str = ast.unparse(child)
                
                # Database operations
                if any(pattern in call_str for pattern in 
                      [".execute(", ".query(", ".fetchall("]):
                    latency += 0.5
                
                # File I/O
                elif any(pattern in call_str for pattern in 
                        ["open(", ".read_csv(", ".to_csv("]):
                    latency += 0.2
                
                # API calls
                elif any(pattern in call_str for pattern in 
                        ["requests.", ".get(", ".post("]):
                    latency += 1.0
                
                # Heavy computations
                elif any(pattern in call_str for pattern in 
                        [".groupby(", ".merge(", ".pivot("]):
                    latency += 0.3
        
        return latency
    
    def _estimate_data_volume(self, node: ast.AST) -> str:
        """Estimate data volume processed"""
        indicators = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_str = ast.unparse(child)
                
                # Check for batch processing
                if "batch" in call_str.lower():
                    indicators.append("batch")
                
                # Check for streaming
                if any(pattern in call_str for pattern in 
                      ["stream", "kafka", "kinesis"]):
                    indicators.append("streaming")
                
                # Check for chunking
                if "chunksize" in call_str or "chunks" in call_str:
                    indicators.append("chunked")
        
        if "streaming" in indicators:
            return "unbounded"
        elif "batch" in indicators:
            return "large"
        elif "chunked" in indicators:
            return "medium"
        else:
            return "small"
    
    def _check_parallelizable(self, node: ast.AST) -> bool:
        """Check if pipeline can be parallelized"""
        # Look for parallel processing indicators
        parallel_indicators = [
            "multiprocessing", "concurrent", "parallel", "dask",
            "spark", "ray", "ThreadPool", "ProcessPool", "async"
        ]
        
        node_str = ast.unparse(node)
        return any(indicator in node_str for indicator in parallel_indicators)
    
    def _has_error_handling(self, node: ast.AST) -> bool:
        """Check if pipeline has error handling"""
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                return True
        return False
    
    def _has_monitoring(self, node: ast.AST) -> bool:
        """Check if pipeline has monitoring"""
        monitoring_indicators = [
            "metric", "log", "monitor", "alert", "trace",
            "telemetry", "datadog", "prometheus", "grafana"
        ]
        
        node_str = ast.unparse(node)
        return any(indicator in node_str.lower() 
                  for indicator in monitoring_indicators)
    
    def _is_idempotent(self, node: ast.AST) -> bool:
        """Check if pipeline is idempotent"""
        # Look for non-idempotent operations
        non_idempotent = [
            "append", "+=", "increment", "counter",
            "random", "uuid", "timestamp", "now()"
        ]
        
        node_str = ast.unparse(node)
        
        # If contains non-idempotent operations, it's not idempotent
        if any(op in node_str for op in non_idempotent):
            return False
        
        # Look for idempotent patterns
        idempotent_patterns = [
            "upsert", "create_or_update", "put", "set",
            "truncate", "overwrite", "replace"
        ]
        
        return any(pattern in node_str.lower() 
                  for pattern in idempotent_patterns)
    
    def _check_pipeline_issues(self, pipeline: Pipeline, node: ast.AST) -> None:
        """Check for pipeline issues"""
        # Check for missing error handling
        if not pipeline.error_handling:
            self.issues.append(PipelineIssue(
                pipeline=pipeline.name,
                issue_type="missing_error_handling",
                severity="high",
                description="Pipeline lacks error handling",
                impact="Pipeline failures may cause data loss or corruption",
                recommendation="Add try-except blocks and retry logic",
                file_path=pipeline.file_path,
                line_number=pipeline.line_number,
                estimated_fix_effort=2.0
            ))
        
        # Check for missing monitoring
        if not pipeline.monitoring:
            self.issues.append(PipelineIssue(
                pipeline=pipeline.name,
                issue_type="missing_monitoring",
                severity="medium",
                description="Pipeline lacks monitoring",
                impact="Cannot track pipeline health and performance",
                recommendation="Add logging, metrics, and alerts",
                file_path=pipeline.file_path,
                line_number=pipeline.line_number,
                estimated_fix_effort=3.0
            ))
        
        # Check for non-idempotent operations
        if not pipeline.idempotent:
            self.issues.append(PipelineIssue(
                pipeline=pipeline.name,
                issue_type="non_idempotent",
                severity="medium",
                description="Pipeline is not idempotent",
                impact="Rerunning pipeline may cause duplicate data",
                recommendation="Use upsert operations or deduplication",
                file_path=pipeline.file_path,
                line_number=pipeline.line_number,
                estimated_fix_effort=4.0
            ))
        
        # Check for high complexity
        if pipeline.complexity > 20:
            self.issues.append(PipelineIssue(
                pipeline=pipeline.name,
                issue_type="high_complexity",
                severity="medium",
                description=f"Pipeline complexity is {pipeline.complexity}",
                impact="Difficult to maintain and debug",
                recommendation="Break down into smaller sub-pipelines",
                file_path=pipeline.file_path,
                line_number=pipeline.line_number,
                estimated_fix_effort=8.0
            ))
        
        # Check for high latency
        if pipeline.estimated_latency > 5.0:
            self.issues.append(PipelineIssue(
                pipeline=pipeline.name,
                issue_type="high_latency",
                severity="high" if pipeline.estimated_latency > 10 else "medium",
                description=f"Estimated latency: {pipeline.estimated_latency:.1f}s",
                impact="Slow pipeline execution",
                recommendation="Optimize I/O operations and use caching",
                file_path=pipeline.file_path,
                line_number=pipeline.line_number,
                estimated_fix_effort=6.0
            ))
        
        # Check for parallelization opportunities
        if not pipeline.parallelizable and pipeline.complexity > 10:
            self.issues.append(PipelineIssue(
                pipeline=pipeline.name,
                issue_type="not_parallelized",
                severity="low",
                description="Pipeline could benefit from parallelization",
                impact="Suboptimal performance for large datasets",
                recommendation="Use parallel processing frameworks",
                file_path=pipeline.file_path,
                line_number=pipeline.line_number,
                estimated_fix_effort=5.0
            ))
    
    def _analyze_data_flows(self, tree: ast.AST, content: str, 
                           file_path: str) -> None:
        """Analyze data flows between components"""
        # Track variable assignments and data transfers
        data_transfers = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Track data assignment
                if len(node.targets) == 1:
                    target = node.targets[0]
                    if isinstance(target, ast.Name):
                        source_type = self._identify_data_source(node.value)
                        if source_type:
                            data_transfers.append({
                                "variable": target.id,
                                "source": source_type,
                                "line": node.lineno
                            })
            
            # Track method chains (common in data processing)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    self._analyze_method_chain(node, file_path)
    
    def _identify_data_source(self, node: ast.AST) -> Optional[str]:
        """Identify the data source from an AST node"""
        node_str = ast.unparse(node)
        
        for source, patterns in self.source_patterns.items():
            if any(re.search(pattern, node_str) for pattern in patterns):
                return source.value
        
        return None
    
    def _analyze_method_chain(self, node: ast.Call, file_path: str) -> None:
        """Analyze method chains for data flow"""
        chain = []
        current = node
        
        while isinstance(current, ast.Call):
            if isinstance(current.func, ast.Attribute):
                chain.append(current.func.attr)
                if isinstance(current.func.value, ast.Call):
                    current = current.func.value
                else:
                    break
            else:
                break
        
        if len(chain) > 1:
            # Create data flow from method chain
            flow = DataFlow(
                source=chain[-1],
                destination=chain[0],
                data_type="dataframe",  # Common for method chains
                transformation=" -> ".join(reversed(chain)),
                volume="unknown",
                frequency="batch",
                is_streaming=False,
                has_backpressure=False
            )
            self.data_flows.append(flow)
    
    def _detect_anti_patterns(self, content: str, file_path: str) -> None:
        """Detect pipeline anti-patterns"""
        lines = content.split('\n')
        
        for pattern_name, pattern_info in self.anti_patterns.items():
            for pattern in pattern_info["patterns"]:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        self.issues.append(PipelineIssue(
                            pipeline="unknown",
                            issue_type=f"anti_pattern_{pattern_name}",
                            severity="medium",
                            description=f"Anti-pattern detected: {pattern_name}",
                            impact=pattern_info["impact"],
                            recommendation=pattern_info["alternative"],
                            file_path=file_path,
                            line_number=i,
                            estimated_fix_effort=2.0
                        ))
    
    def _analyze_pipeline_quality(self, tree: ast.AST, content: str,
                                 file_path: str) -> None:
        """Analyze overall pipeline quality"""
        quality_checks = {
            "has_schema_validation": self._check_schema_validation(tree),
            "has_data_quality_checks": self._check_data_quality(tree),
            "has_backpressure": self._check_backpressure(tree),
            "has_checkpointing": self._check_checkpointing(tree),
            "has_deduplication": self._check_deduplication(tree),
            "has_partitioning": self._check_partitioning(tree),
        }
        
        # Create issues for missing quality checks
        if not quality_checks["has_schema_validation"]:
            self.issues.append(PipelineIssue(
                pipeline="general",
                issue_type="missing_schema_validation",
                severity="medium",
                description="No schema validation detected",
                impact="Invalid data may corrupt downstream systems",
                recommendation="Add schema validation at pipeline entry points",
                file_path=file_path,
                line_number=1,
                estimated_fix_effort=3.0
            ))
        
        if not quality_checks["has_data_quality_checks"]:
            self.issues.append(PipelineIssue(
                pipeline="general",
                issue_type="missing_data_quality",
                severity="medium",
                description="No data quality checks detected",
                impact="Bad data may propagate through pipeline",
                recommendation="Add data quality assertions and monitoring",
                file_path=file_path,
                line_number=1,
                estimated_fix_effort=4.0
            ))
    
    def _check_schema_validation(self, tree: ast.AST) -> bool:
        """Check for schema validation"""
        validation_indicators = [
            "schema", "validate", "jsonschema", "pydantic",
            "marshmallow", "cerberus", "voluptuous"
        ]
        
        tree_str = ast.unparse(tree)
        return any(indicator in tree_str.lower() 
                  for indicator in validation_indicators)
    
    def _check_data_quality(self, tree: ast.AST) -> bool:
        """Check for data quality checks"""
        quality_indicators = [
            "assert", "check", "verify", "validate",
            "quality", "null", "missing", "duplicate"
        ]
        
        tree_str = ast.unparse(tree)
        return any(indicator in tree_str.lower() 
                  for indicator in quality_indicators)
    
    def _check_backpressure(self, tree: ast.AST) -> bool:
        """Check for backpressure handling"""
        backpressure_indicators = [
            "backpressure", "rate_limit", "throttle",
            "buffer", "queue", "bounded"
        ]
        
        tree_str = ast.unparse(tree)
        return any(indicator in tree_str.lower() 
                  for indicator in backpressure_indicators)
    
    def _check_checkpointing(self, tree: ast.AST) -> bool:
        """Check for checkpointing"""
        checkpoint_indicators = [
            "checkpoint", "savepoint", "commit",
            "WAL", "journal", "recovery"
        ]
        
        tree_str = ast.unparse(tree)
        return any(indicator in tree_str.lower() 
                  for indicator in checkpoint_indicators)
    
    def _check_deduplication(self, tree: ast.AST) -> bool:
        """Check for deduplication logic"""
        dedup_indicators = [
            "dedupe", "dedup", "distinct", "unique",
            "drop_duplicates", "duplicated"
        ]
        
        tree_str = ast.unparse(tree)
        return any(indicator in tree_str.lower() 
                  for indicator in dedup_indicators)
    
    def _check_partitioning(self, tree: ast.AST) -> bool:
        """Check for data partitioning"""
        partition_indicators = [
            "partition", "shard", "bucket", "segment",
            "chunk", "split", "distribute"
        ]
        
        tree_str = ast.unparse(tree)
        return any(indicator in tree_str.lower() 
                  for indicator in partition_indicators)
    
    def _analyze_airflow_dag(self, node: ast.ClassDef, content: str,
                            file_path: str) -> None:
        """Analyze Airflow DAG"""
        # Extract DAG properties
        dag_name = node.name
        tasks = []
        dependencies = []
        
        for stmt in node.body:
            # Look for task definitions
            if isinstance(stmt, ast.Assign):
                if any(isinstance(target, ast.Name) and 
                      "operator" in ast.unparse(stmt.value).lower()
                      for target in stmt.targets):
                    tasks.append(ast.unparse(stmt.targets[0]))
            
            # Look for dependencies
            elif isinstance(stmt, ast.Expr):
                expr_str = ast.unparse(stmt)
                if ">>" in expr_str or "<<" in expr_str:
                    dependencies.append(expr_str)
        
        # Create pipeline from DAG
        pipeline = Pipeline(
            name=dag_name,
            stages=[],  # Would need deeper analysis
            data_sources=[],  # Would need deeper analysis
            transformations=[],  # Would need deeper analysis
            file_path=file_path,
            line_number=node.lineno,
            complexity=len(tasks) + len(dependencies),
            estimated_latency=len(tasks) * 1.0,  # Rough estimate
            data_volume="unknown",
            parallelizable=True,  # Airflow supports parallelization
            error_handling=True,  # Airflow has built-in retry
            monitoring=True,  # Airflow has built-in monitoring
            idempotent=False  # Depends on task implementation
        )
        
        self.pipelines.append(pipeline)
    
    def _analyze_spark_pipeline(self, node: ast.Assign, content: str,
                               file_path: str) -> None:
        """Analyze Spark pipeline"""
        # This would analyze Spark-specific patterns
        # Simplified for demonstration
        pass
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline analysis report"""
        # Calculate statistics
        total_pipelines = len(self.pipelines)
        total_issues = len(self.issues)
        critical_issues = sum(1 for i in self.issues if i.severity == "critical")
        high_issues = sum(1 for i in self.issues if i.severity == "high")
        
        # Calculate average metrics
        avg_complexity = (sum(p.complexity for p in self.pipelines) / total_pipelines
                         if total_pipelines > 0 else 0)
        avg_latency = (sum(p.estimated_latency for p in self.pipelines) / total_pipelines
                      if total_pipelines > 0 else 0)
        
        # Group issues by type
        issues_by_type = {}
        for issue in self.issues:
            if issue.issue_type not in issues_by_type:
                issues_by_type[issue.issue_type] = []
            issues_by_type[issue.issue_type].append(issue)
        
        # Calculate remediation effort
        total_effort = sum(issue.estimated_fix_effort for issue in self.issues)
        
        # Identify most common data sources
        all_sources = []
        for pipeline in self.pipelines:
            all_sources.extend(pipeline.data_sources)
        
        source_counts = {}
        for source in all_sources:
            source_counts[source.value] = source_counts.get(source.value, 0) + 1
        
        # Identify most common transformations
        all_transformations = []
        for pipeline in self.pipelines:
            all_transformations.extend(pipeline.transformations)
        
        transformation_counts = {}
        for trans in all_transformations:
            transformation_counts[trans.value] = transformation_counts.get(trans.value, 0) + 1
        
        return {
            "summary": {
                "total_pipelines": total_pipelines,
                "total_issues": total_issues,
                "critical_issues": critical_issues,
                "high_issues": high_issues,
                "average_complexity": round(avg_complexity, 2),
                "average_latency": round(avg_latency, 2),
                "total_remediation_effort": round(total_effort, 2),
            },
            "pipelines": [
                {
                    "name": p.name,
                    "file": p.file_path,
                    "line": p.line_number,
                    "stages": [s.value if hasattr(s, 'value') else s for s in p.stages],
                    "data_sources": [d.value for d in p.data_sources],
                    "transformations": [t.value for t in p.transformations],
                    "complexity": p.complexity,
                    "estimated_latency": p.estimated_latency,
                    "data_volume": p.data_volume,
                    "parallelizable": p.parallelizable,
                    "error_handling": p.error_handling,
                    "monitoring": p.monitoring,
                    "idempotent": p.idempotent,
                }
                for p in self.pipelines
            ],
            "data_flows": [
                {
                    "source": f.source,
                    "destination": f.destination,
                    "data_type": f.data_type,
                    "transformation": f.transformation,
                    "volume": f.volume,
                    "frequency": f.frequency,
                    "is_streaming": f.is_streaming,
                    "has_backpressure": f.has_backpressure,
                }
                for f in self.data_flows
            ],
            "issues": [
                {
                    "pipeline": i.pipeline,
                    "type": i.issue_type,
                    "severity": i.severity,
                    "description": i.description,
                    "impact": i.impact,
                    "recommendation": i.recommendation,
                    "file": i.file_path,
                    "line": i.line_number,
                    "estimated_effort": i.estimated_fix_effort,
                }
                for i in sorted(self.issues, 
                              key=lambda x: {"critical": 0, "high": 1, 
                                           "medium": 2, "low": 3}[x.severity])
            ],
            "data_source_distribution": source_counts,
            "transformation_distribution": transformation_counts,
            "recommendations": self._generate_recommendations(),
        }
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate pipeline recommendations"""
        recommendations = []
        
        # Check for missing monitoring
        pipelines_without_monitoring = [p for p in self.pipelines if not p.monitoring]
        if pipelines_without_monitoring:
            recommendations.append({
                "category": "Observability",
                "priority": "high",
                "recommendation": "Add monitoring to all pipelines",
                "impact": "Improve pipeline reliability and debugging",
                "affected_pipelines": [p.name for p in pipelines_without_monitoring]
            })
        
        # Check for non-idempotent pipelines
        non_idempotent = [p for p in self.pipelines if not p.idempotent]
        if non_idempotent:
            recommendations.append({
                "category": "Reliability",
                "priority": "medium",
                "recommendation": "Make pipelines idempotent",
                "impact": "Enable safe pipeline reruns",
                "affected_pipelines": [p.name for p in non_idempotent]
            })
        
        # Check for high complexity
        complex_pipelines = [p for p in self.pipelines if p.complexity > 20]
        if complex_pipelines:
            recommendations.append({
                "category": "Maintainability",
                "priority": "medium",
                "recommendation": "Refactor complex pipelines",
                "impact": "Improve maintainability and testability",
                "affected_pipelines": [p.name for p in complex_pipelines]
            })
        
        # Check for parallelization opportunities
        sequential_pipelines = [p for p in self.pipelines 
                               if not p.parallelizable and p.complexity > 10]
        if sequential_pipelines:
            recommendations.append({
                "category": "Performance",
                "priority": "low",
                "recommendation": "Add parallelization to complex pipelines",
                "impact": "Improve processing speed",
                "affected_pipelines": [p.name for p in sequential_pipelines]
            })
        
        return recommendations
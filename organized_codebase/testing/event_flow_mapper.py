#!/usr/bin/env python3
"""
Agent C - Event & Message Flow Mapper
Hours 13-15: Comprehensive event-driven architecture and message flow analysis.

Features:
- Event publisher/subscriber pattern detection
- Message queue flow tracking
- Async/await pattern analysis
- Callback chain mapping
- WebSocket/streaming detection
- Signal/slot pattern identification
"""

import ast
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EventDefinition:
    """Represents an event definition in the system."""
    name: str
    module: str
    event_type: str  # 'custom', 'system', 'domain', 'integration'
    payload_type: Optional[str] = None
    publishers: List[str] = field(default_factory=list)
    subscribers: List[str] = field(default_factory=list)
    middleware: List[str] = field(default_factory=list)
    priority: int = 5
    is_async: bool = False


@dataclass
class MessageFlow:
    """Represents a message flow between components."""
    source: str
    target: str
    message_type: str
    flow_pattern: str  # 'pub-sub', 'request-reply', 'fire-forget', 'streaming'
    channel: Optional[str] = None
    is_async: bool = False
    has_acknowledgment: bool = False
    retry_policy: Optional[str] = None


@dataclass
class EventChain:
    """Represents a chain of events."""
    trigger_event: str
    chain_events: List[str]
    total_latency: Optional[float] = None
    is_circular: bool = False
    complexity_score: int = 0


@dataclass
class AsyncPattern:
    """Represents an async/await pattern."""
    function_name: str
    module: str
    pattern_type: str  # 'async-await', 'callback', 'promise', 'coroutine'
    await_calls: List[str] = field(default_factory=list)
    callback_depth: int = 0
    has_error_handling: bool = False


@dataclass
class StreamingPattern:
    """Represents a streaming or real-time pattern."""
    component: str
    stream_type: str  # 'websocket', 'sse', 'grpc-stream', 'kafka', 'rabbitmq'
    producers: List[str] = field(default_factory=list)
    consumers: List[str] = field(default_factory=list)
    throughput_estimate: Optional[int] = None


class EventFlowMapper:
    """
    Map event-driven architectures and message flows in the codebase.
    """
    
    def __init__(self, root_path: Path = Path(".")):
        self.root_path = root_path.resolve()
        self.events: Dict[str, EventDefinition] = {}
        self.message_flows: List[MessageFlow] = []
        self.event_chains: List[EventChain] = []
        self.async_patterns: Dict[str, AsyncPattern] = {}
        self.streaming_patterns: List[StreamingPattern] = []
        self.scan_timestamp = datetime.now()
        
        # Pattern detection
        self.pub_sub_patterns: Dict[str, Set[str]] = defaultdict(set)
        self.callback_chains: List[List[str]] = []
        self.websocket_endpoints: List[str] = []
        self.message_queues: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'total_message_flows': 0,
            'async_patterns': 0,
            'streaming_patterns': 0,
            'event_chains': 0,
            'max_chain_length': 0,
            'scan_duration': 0.0
        }
    
    def analyze_event_flows(self) -> Dict[str, Any]:
        """
        Perform comprehensive event flow analysis.
        """
        start_time = time.time()
        logger.info(f"Starting event flow analysis for {self.root_path}")
        
        # Phase 1: Discover all Python files
        python_files = self._discover_python_files()
        logger.info(f"Analyzing event flows in {len(python_files)} Python files")
        
        for file_path in python_files:
            try:
                self._analyze_file_events(file_path)
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
        
        # Phase 2: Detect pub-sub patterns
        self._detect_pub_sub_patterns()
        
        # Phase 3: Map message flows
        self._map_message_flows()
        
        # Phase 4: Identify event chains
        self._identify_event_chains()
        
        # Phase 5: Analyze streaming patterns
        self._analyze_streaming_patterns()
        
        # Phase 6: Calculate statistics
        self.stats['scan_duration'] = time.time() - start_time
        self._calculate_statistics()
        
        logger.info(f"Event flow analysis completed in {self.stats['scan_duration']:.2f} seconds")
        
        return self._generate_comprehensive_report()
    
    def _discover_python_files(self) -> List[Path]:
        """Discover all Python files in the codebase."""
        python_files = []
        
        for py_file in self.root_path.rglob("*.py"):
            # Skip common non-source directories
            if any(exclude in str(py_file) for exclude in [
                '__pycache__', '.git', '.venv', 'venv', 'env',
                'node_modules', '.pytest_cache', '.coverage'
            ]):
                continue
                
            python_files.append(py_file)
        
        return sorted(python_files)
    
    def _analyze_file_events(self, file_path: Path):
        """Analyze events and messaging patterns in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
                
            try:
                tree = ast.parse(source_code)
            except SyntaxError:
                logger.warning(f"Syntax error in {file_path}, skipping")
                return
            
            module_name = self._calculate_module_name(file_path)
            
            # Create visitor to analyze events
            visitor = EventVisitor(self, module_name, source_code)
            visitor.visit(tree)
            
            # Check for specific patterns in source code
            self._check_event_patterns(source_code, module_name)
            
        except Exception as e:
            logger.error(f"Error analyzing events in {file_path}: {e}")
    
    def _check_event_patterns(self, source_code: str, module_name: str):
        """Check for event patterns using regex."""
        # WebSocket patterns
        if re.search(r'websocket|WebSocket|ws\.|WSHandler', source_code, re.IGNORECASE):
            self.websocket_endpoints.append(module_name)
        
        # Message queue patterns
        queue_patterns = [
            (r'rabbitmq|RabbitMQ|amqp|AMQP', 'rabbitmq'),
            (r'kafka|Kafka|producer|consumer', 'kafka'),
            (r'redis\.pubsub|Redis.*pub.*sub', 'redis'),
            (r'celery|Celery|task|Task', 'celery'),
            (r'asyncio\.Queue|queue\.Queue', 'python-queue')
        ]
        
        for pattern, queue_type in queue_patterns:
            if re.search(pattern, source_code):
                self.message_queues[queue_type].append(module_name)
        
        # Event emitter patterns
        if re.search(r'emit\(|trigger\(|dispatch\(|publish\(|broadcast\(', source_code):
            self._extract_event_emitters(source_code, module_name)
        
        # Event listener patterns
        if re.search(r'\.on\(|\.subscribe\(|\.listen\(|@.*listener|@.*handler', source_code):
            self._extract_event_listeners(source_code, module_name)
    
    def _extract_event_emitters(self, source_code: str, module_name: str):
        """Extract event emitter patterns."""
        emitter_patterns = [
            r'(\w+)\.emit\(["\'](\w+)["\']',
            r'(\w+)\.trigger\(["\'](\w+)["\']',
            r'(\w+)\.dispatch\(["\'](\w+)["\']',
            r'(\w+)\.publish\(["\'](\w+)["\']',
            r'(\w+)\.broadcast\(["\'](\w+)["\']'
        ]
        
        for pattern in emitter_patterns:
            for match in re.finditer(pattern, source_code):
                event_name = match.group(2) if match.lastindex >= 2 else 'unknown'
                if event_name not in self.events:
                    self.events[event_name] = EventDefinition(
                        name=event_name,
                        module=module_name,
                        event_type='custom'
                    )
                self.events[event_name].publishers.append(module_name)
    
    def _extract_event_listeners(self, source_code: str, module_name: str):
        """Extract event listener patterns."""
        listener_patterns = [
            r'\.on\(["\'](\w+)["\']',
            r'\.subscribe\(["\'](\w+)["\']',
            r'\.listen\(["\'](\w+)["\']',
            r'@\w*[Ll]istener\(["\'](\w+)["\']',
            r'@\w*[Hh]andler\(["\'](\w+)["\']'
        ]
        
        for pattern in listener_patterns:
            for match in re.finditer(pattern, source_code):
                event_name = match.group(1)
                if event_name not in self.events:
                    self.events[event_name] = EventDefinition(
                        name=event_name,
                        module=module_name,
                        event_type='custom'
                    )
                self.events[event_name].subscribers.append(module_name)
    
    def _calculate_module_name(self, file_path: Path) -> str:
        """Calculate the module name from file path."""
        relative_path = file_path.relative_to(self.root_path)
        parts = list(relative_path.parts)
        
        # Remove .py extension
        if parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]
        
        # Handle __init__.py files
        if parts[-1] == '__init__':
            parts = parts[:-1]
        
        return '.'.join(parts) if parts else '__main__'
    
    def _detect_pub_sub_patterns(self):
        """Detect publisher-subscriber patterns."""
        for event_name, event_def in self.events.items():
            if event_def.publishers and event_def.subscribers:
                for publisher in event_def.publishers:
                    for subscriber in event_def.subscribers:
                        self.pub_sub_patterns[publisher].add(subscriber)
                        
                        # Create message flow
                        flow = MessageFlow(
                            source=publisher,
                            target=subscriber,
                            message_type=event_name,
                            flow_pattern='pub-sub',
                            channel=event_name,
                            is_async=event_def.is_async
                        )
                        self.message_flows.append(flow)
    
    def _map_message_flows(self):
        """Map message flows between components."""
        # Analyze callback patterns
        for pattern in self.async_patterns.values():
            if pattern.pattern_type == 'callback':
                # Track callback chains
                if pattern.callback_depth > 2:
                    chain = [pattern.function_name] + pattern.await_calls
                    self.callback_chains.append(chain)
        
        # Analyze streaming patterns
        for endpoint in self.websocket_endpoints:
            streaming = StreamingPattern(
                component=endpoint,
                stream_type='websocket',
                producers=[endpoint],
                consumers=[]  # Would need more analysis to find consumers
            )
            self.streaming_patterns.append(streaming)
        
        # Analyze message queue patterns
        for queue_type, modules in self.message_queues.items():
            for module in modules:
                streaming = StreamingPattern(
                    component=module,
                    stream_type=queue_type,
                    producers=[module] if 'producer' in module.lower() else [],
                    consumers=[module] if 'consumer' in module.lower() else []
                )
                self.streaming_patterns.append(streaming)
    
    def _identify_event_chains(self):
        """Identify chains of events."""
        # Build event dependency graph
        event_graph = defaultdict(set)
        
        for event_name, event_def in self.events.items():
            # Look for events that might trigger other events
            for subscriber in event_def.subscribers:
                # Check if this subscriber publishes other events
                for other_event, other_def in self.events.items():
                    if subscriber in other_def.publishers and other_event != event_name:
                        event_graph[event_name].add(other_event)
        
        # Find event chains using DFS
        visited = set()
        
        def find_chains(event, chain):
            if event in visited:
                # Check for circular chain
                if event in chain:
                    circular_chain = EventChain(
                        trigger_event=chain[0],
                        chain_events=chain + [event],
                        is_circular=True,
                        complexity_score=len(chain)
                    )
                    self.event_chains.append(circular_chain)
                return
            
            visited.add(event)
            
            if event in event_graph:
                for next_event in event_graph[event]:
                    new_chain = chain + [next_event]
                    if len(new_chain) > 2:  # Only track significant chains
                        event_chain = EventChain(
                            trigger_event=chain[0],
                            chain_events=new_chain,
                            complexity_score=len(new_chain)
                        )
                        self.event_chains.append(event_chain)
                    find_chains(next_event, new_chain)
        
        for event in event_graph:
            find_chains(event, [event])
    
    def _analyze_streaming_patterns(self):
        """Analyze streaming and real-time patterns."""
        # Group streaming patterns by type
        streaming_by_type = defaultdict(list)
        
        for pattern in self.streaming_patterns:
            streaming_by_type[pattern.stream_type].append(pattern)
        
        # Estimate throughput based on pattern type
        throughput_estimates = {
            'websocket': 1000,  # messages/sec
            'kafka': 10000,
            'rabbitmq': 5000,
            'redis': 8000,
            'grpc-stream': 5000,
            'sse': 100
        }
        
        for pattern in self.streaming_patterns:
            if pattern.stream_type in throughput_estimates:
                pattern.throughput_estimate = throughput_estimates[pattern.stream_type]
    
    def _calculate_statistics(self):
        """Calculate comprehensive statistics."""
        self.stats['total_events'] = len(self.events)
        self.stats['total_message_flows'] = len(self.message_flows)
        self.stats['async_patterns'] = len(self.async_patterns)
        self.stats['streaming_patterns'] = len(self.streaming_patterns)
        self.stats['event_chains'] = len(self.event_chains)
        
        if self.event_chains:
            self.stats['max_chain_length'] = max(
                len(chain.chain_events) for chain in self.event_chains
            )
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive event flow analysis report."""
        return {
            'scan_metadata': {
                'timestamp': self.scan_timestamp.isoformat(),
                'root_path': str(self.root_path),
                'analysis_type': 'event_flow_mapping'
            },
            'statistics': self.stats,
            'events': {
                name: asdict(event) for name, event in self.events.items()
            },
            'message_flows': [asdict(flow) for flow in self.message_flows],
            'event_chains': [asdict(chain) for chain in self.event_chains],
            'async_patterns': {
                name: asdict(pattern) for name, pattern in self.async_patterns.items()
            },
            'streaming_patterns': [asdict(pattern) for pattern in self.streaming_patterns],
            'pub_sub_patterns': {
                pub: list(subs) for pub, subs in self.pub_sub_patterns.items()
            },
            'callback_chains': self.callback_chains,
            'websocket_endpoints': self.websocket_endpoints,
            'message_queues': dict(self.message_queues),
            'insights': self._generate_insights()
        }
    
    def _generate_insights(self) -> Dict[str, Any]:
        """Generate analytical insights from event flow analysis."""
        insights = {
            'architecture_patterns': {},
            'complexity': {},
            'performance': {},
            'recommendations': []
        }
        
        # Architecture pattern insights
        if self.pub_sub_patterns:
            insights['architecture_patterns']['pub_sub_usage'] = len(self.pub_sub_patterns)
        
        if self.streaming_patterns:
            insights['architecture_patterns']['streaming_components'] = len(self.streaming_patterns)
        
        if self.message_queues:
            insights['architecture_patterns']['message_queue_types'] = list(self.message_queues.keys())
        
        # Complexity insights
        if self.event_chains:
            insights['complexity']['event_chain_count'] = len(self.event_chains)
            insights['complexity']['max_chain_depth'] = self.stats['max_chain_length']
            
            circular_chains = [c for c in self.event_chains if c.is_circular]
            if circular_chains:
                insights['complexity']['circular_event_chains'] = len(circular_chains)
        
        # Performance insights
        total_throughput = sum(
            p.throughput_estimate for p in self.streaming_patterns 
            if p.throughput_estimate
        )
        if total_throughput:
            insights['performance']['estimated_throughput'] = total_throughput
        
        # Recommendations
        if any(c.is_circular for c in self.event_chains):
            insights['recommendations'].append(
                "Circular event chains detected. Review for potential infinite loops."
            )
        
        if self.stats['max_chain_length'] > 5:
            insights['recommendations'].append(
                "Deep event chains detected. Consider simplifying event flow architecture."
            )
        
        if len(self.message_queues) > 3:
            insights['recommendations'].append(
                "Multiple message queue systems detected. Consider consolidation."
            )
        
        return insights
    
    def save_report(self, output_path: Path) -> None:
        """Save the event flow analysis report."""
        report = self._generate_comprehensive_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Event flow analysis report saved to {output_path}")


class EventVisitor(ast.NodeVisitor):
    """AST visitor for analyzing event patterns."""
    
    def __init__(self, mapper: EventFlowMapper, module_name: str, source_code: str):
        self.mapper = mapper
        self.module_name = module_name
        self.source_code = source_code
        self.current_function = None
        self.current_class = None
    
    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        old_function = self.current_function
        self.current_function = f"{self.module_name}.{node.name}"
        
        # Track async pattern
        pattern = AsyncPattern(
            function_name=node.name,
            module=self.module_name,
            pattern_type='async-await'
        )
        
        # Find await calls
        for child in ast.walk(node):
            if isinstance(child, ast.Await):
                if isinstance(child.value, ast.Call):
                    await_target = self._get_call_name(child.value)
                    pattern.await_calls.append(await_target)
        
        # Check for error handling
        for child in ast.walk(node):
            if isinstance(child, ast.ExceptHandler):
                pattern.has_error_handling = True
                break
        
        self.mapper.async_patterns[self.current_function] = pattern
        
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_FunctionDef(self, node):
        """Visit function definition."""
        old_function = self.current_function
        self.current_function = f"{self.module_name}.{node.name}"
        
        # Check for event-related decorators
        for decorator in node.decorator_list:
            decorator_name = self._get_decorator_name(decorator)
            if any(keyword in decorator_name.lower() for keyword in 
                   ['event', 'handler', 'listener', 'subscriber', 'consumer']):
                # This is an event handler
                if self.current_function not in self.mapper.events:
                    event = EventDefinition(
                        name=node.name,
                        module=self.module_name,
                        event_type='handler'
                    )
                    self.mapper.events[node.name] = event
        
        # Check for callback patterns
        if 'callback' in node.name.lower() or 'cb' in node.name.lower():
            pattern = AsyncPattern(
                function_name=node.name,
                module=self.module_name,
                pattern_type='callback',
                callback_depth=self._calculate_callback_depth(node)
            )
            self.mapper.async_patterns[self.current_function] = pattern
        
        self.generic_visit(node)
        self.current_function = old_function
    
    def visit_ClassDef(self, node):
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = f"{self.module_name}.{node.name}"
        
        # Check for event-related base classes
        for base in node.bases:
            base_name = self._get_base_name(base)
            if any(keyword in base_name.lower() for keyword in 
                   ['event', 'emitter', 'publisher', 'subscriber', 'handler']):
                # This class is event-related
                event = EventDefinition(
                    name=node.name,
                    module=self.module_name,
                    event_type='class'
                )
                self.mapper.events[node.name] = event
        
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_Call(self, node):
        """Visit function call."""
        call_name = self._get_call_name(node)
        
        # Check for event-related calls
        event_keywords = ['emit', 'trigger', 'dispatch', 'publish', 'broadcast',
                         'subscribe', 'listen', 'on', 'once', 'off']
        
        for keyword in event_keywords:
            if keyword in call_name.lower():
                # Extract event name if possible
                if node.args and isinstance(node.args[0], ast.Constant):
                    event_name = str(node.args[0].value)
                    
                    if event_name not in self.mapper.events:
                        self.mapper.events[event_name] = EventDefinition(
                            name=event_name,
                            module=self.module_name,
                            event_type='runtime'
                        )
                    
                    if keyword in ['emit', 'trigger', 'dispatch', 'publish', 'broadcast']:
                        self.mapper.events[event_name].publishers.append(
                            self.current_function or self.module_name
                        )
                    else:
                        self.mapper.events[event_name].subscribers.append(
                            self.current_function or self.module_name
                        )
        
        self.generic_visit(node)
    
    def _get_call_name(self, node: ast.Call) -> str:
        """Extract the name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return f"{self._get_call_name(node.func.value)}.{node.func.attr}" if hasattr(node.func, 'value') and hasattr(node.func.value, 'func') else node.func.attr
        else:
            return "unknown"
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Extract decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        else:
            return "unknown"
    
    def _get_base_name(self, base: ast.AST) -> str:
        """Extract base class name."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return base.attr
        else:
            return "unknown"
    
    def _calculate_callback_depth(self, node: ast.FunctionDef) -> int:
        """Calculate the depth of callback nesting."""
        max_depth = 0
        
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                depth = 1
                parent = child
                while parent:
                    if isinstance(parent, ast.FunctionDef):
                        depth += 1
                    parent = getattr(parent, 'parent', None)
                max_depth = max(max_depth, depth)
        
        return max_depth


def main():
    """Main entry point for the event flow mapper."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agent C - Event & Message Flow Mapper")
    parser.add_argument("--root", default=".", help="Root directory to analyze")
    parser.add_argument("--output", default="event_flow_hour13.json", help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create mapper and run analysis
    mapper = EventFlowMapper(Path(args.root))
    
    print("Agent C - Event & Message Flow Analysis (Hours 13-15)")
    print(f"Analyzing: {mapper.root_path}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Analyze event flows
    report = mapper.analyze_event_flows()
    
    # Save report
    mapper.save_report(Path(args.output))
    
    # Print summary
    stats = report['statistics']
    print(f"\nEvent Flow Analysis Results:")
    print(f"   Total Events: {stats['total_events']}")
    print(f"   Message Flows: {stats['total_message_flows']}")
    print(f"   Async Patterns: {stats['async_patterns']}")
    print(f"   Streaming Patterns: {stats['streaming_patterns']}")
    print(f"   Event Chains: {stats['event_chains']}")
    print(f"   Max Chain Length: {stats['max_chain_length']}")
    print(f"   Scan Duration: {stats['scan_duration']:.2f} seconds")
    
    if report['websocket_endpoints']:
        print(f"\nWebSocket Endpoints: {len(report['websocket_endpoints'])}")
    
    if report['message_queues']:
        print(f"\nMessage Queue Systems:")
        for queue_type, modules in report['message_queues'].items():
            print(f"   {queue_type}: {len(modules)} modules")
    
    if report['insights']['recommendations']:
        print(f"\nRecommendations:")
        for rec in report['insights']['recommendations']:
            print(f"   - {rec}")
    
    print(f"\nEvent flow analysis complete! Report saved to {args.output}")


if __name__ == "__main__":
    main()
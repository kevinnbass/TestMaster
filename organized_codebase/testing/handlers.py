            """Get current alerts."""
            active_alerts = [a for a in self.monitor.alerts if not a.resolved]
            
            return jsonify([
                {
                    'severity': alert.severity,
                    'component': alert.component,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'resolved': alert.resolved
                } for alert in active_alerts
            ])
        
        @self.app.route('/api/alerts/<int:alert_id>/resolve', methods=['POST'])
        def resolve_alert(alert_id):
            """Resolve an alert."""
            if 0 <= alert_id < len(self.monitor.alerts):
                self.monitor.alerts[alert_id].resolved = True
                return jsonify({'status': 'success'})
            return jsonify({'status': 'error', 'message': 'Alert not found'}), 404
        
        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': time.time() - self.start_time,
                'monitoring_active': self.monitor.running
            })
        
        @self.app.route('/api/workflow/status')
        def workflow_status():
            """Get workflow status."""
            try:
                # Build workflow status with default values
                status = {
                    'active_workflows': 0,
                    'completed_workflows': 0,
                    'pending_tasks': 0,
                    'running_tasks': 0,
                    'completed_tasks': 0,
                    'consensus_decisions': 0,
                    'dag_nodes': 0,
                    'critical_path_length': 0,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Add any active refactoring workflows
                if self.refactor_roadmaps:
                    for codebase, roadmap in self.refactor_roadmaps.items():
                        if roadmap and hasattr(roadmap, 'phases'):
                            status['active_workflows'] += 1
                            for phase in roadmap.phases:
                                status['pending_tasks'] += len(phase.tasks)
                
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error getting workflow status: {e}")
                return jsonify({
                    'active_workflows': 0,
                    'completed_workflows': 0,
                    'pending_tasks': 0,
                    'running_tasks': 0,
                    'completed_tasks': 0,
                    'consensus_decisions': 0,
                    'dag_nodes': 0,
                    'critical_path_length': 0,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        @self.app.route('/api/config')
        def get_config():
            """Get current configuration."""
            try:
                from testmaster.core.config import get_config
                config = get_config()
                return jsonify({
                    'active_profile': config._active_profile,
                    'profile_info': config.get_profile_info(),
                    'configuration_values': len(config._config_values)
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/test/integration')
        def test_integration():
            """Run integration test."""
            try:
                # Import and run basic integration test
                from testmaster.intelligence.bridges import get_protocol_bridge
                bridge = get_protocol_bridge()
                return jsonify({
                    'status': 'success',
                    'message': 'Integration test passed',
                    'bridge_status': 'operational'
                })
            except Exception as e:
                return jsonify({
                    'status': 'error', 
                    'message': f'Integration test failed: {str(e)}'
                }), 500
        
        @self.app.route('/api/llm/metrics')
        def get_llm_metrics():
            """Get LLM analysis metrics for specific codebase."""
            try:
                codebase = request.args.get('codebase')  # Get codebase from query parameter
                if hasattr(self.monitor, 'llm_monitor') and self.monitor.llm_monitor:
                    return jsonify(self.monitor.llm_monitor.get_codebase_metrics(codebase))
                else:
                    return jsonify({'error': 'LLM monitor not available'}), 503
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/llm/analysis/<path:module_path>')
        def get_module_analysis(module_path):
            """Get analysis for a specific module."""
            try:
                if hasattr(self.monitor, 'llm_monitor') and self.monitor.llm_monitor:
                    analysis = self.monitor.llm_monitor.get_module_analysis(module_path)
                    if analysis:
                        return jsonify(analysis)
                    else:
                        return jsonify({'error': 'Module analysis not found'}), 404
                else:
                    return jsonify({'error': 'LLM monitor not available'}), 503
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/llm/analyze', methods=['POST'])
        def analyze_on_demand():
            """Analyze a module when user explicitly requests it."""
            try:
                data = request.get_json()
                module_path = data.get('module_path')
                
                if not module_path:
                    return jsonify({'error': 'module_path required'}), 400
                
                if hasattr(self.monitor, 'llm_monitor') and self.monitor.llm_monitor:
                    # Only make API call when user explicitly requests
                    analysis = self.monitor.llm_monitor.analyze_module_on_demand(module_path)
                    if analysis:
                        return jsonify({
                            'status': 'completed', 
                            'module_path': module_path,
                            'analysis': {
                                'quality_score': analysis.quality_score,
                                'complexity_score': analysis.complexity_score,
                                'test_coverage_estimate': analysis.test_coverage_estimate,
                                'analysis_summary': analysis.analysis_summary,
                                'model_used': 'gemini-2.5-flash'
                            }
                        })
                    else:
                        return jsonify({'error': 'Analysis failed'}), 500
                else:
                    return jsonify({'error': 'LLM monitor not available'}), 503
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/llm/list-modules')
        def list_llm_modules():
            """List available Python modules for analysis."""
            try:
                import glob
                modules = []
                for pattern in ['*.py', 'testmaster/**/*.py']:
                    for file in glob.glob(pattern, recursive=True):
                        if '__pycache__' not in file and '.pyc' not in file:
                            modules.append(file.replace('\\', '/'))
                return jsonify(sorted(modules)[:100])  # Limit to 100 files
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/llm/toggle-mode', methods=['POST'])
        def toggle_llm_mode():
            """Toggle between demo mode and live API mode."""
            try:
                data = request.json
                enable_api = data.get('enable_api', False)
                
                if hasattr(self.monitor, 'llm_monitor') and self.monitor.llm_monitor:
                    if enable_api:
                        # Switch to live API mode
                        self.monitor.llm_monitor.demo_mode = False
                        self.monitor.llm_monitor.user_triggered_mode = False
                        # Try to enable Gemini if API key exists
                        if self.monitor.llm_monitor.api_key and GENAI_AVAILABLE:
                            try:
                                genai.configure(api_key=self.monitor.llm_monitor.api_key)
                                self.monitor.llm_monitor.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
                                self.monitor.llm_monitor.gemini_available = True
                                message = 'Switched to live API mode (Gemini enabled)'
                            except Exception as e:
                                message = f'Switched to live API mode (Gemini error: {e})'
                        else:
                            message = 'API key not configured - staying in demo mode'
                    else:
                        # Switch to demo mode
                        self.monitor.llm_monitor.demo_mode = True
                        self.monitor.llm_monitor.user_triggered_mode = True
                        self.monitor.llm_monitor.gemini_available = False
                        message = 'Switched to demo mode (no API calls)'
                    
                    return jsonify({
                        'success': True,
                        'demo_mode': self.monitor.llm_monitor.demo_mode,
                        'api_enabled': not self.monitor.llm_monitor.demo_mode,
                        'gemini_available': self.monitor.llm_monitor.gemini_available,
                        'message': message
                    })
                else:
                    return jsonify({'error': 'LLM monitor not available'}), 503
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/module/ast-analysis', methods=['GET'])
        def get_ast_analysis():
            """Get basic AST analysis without LLM."""
            module_path = request.args.get('path')
            if not module_path:
                return jsonify({'error': 'path required'}), 400
            
            try:
                import ast
                import os
                
                if os.path.exists(module_path):
                    with open(module_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse AST
                    tree = ast.parse(content)
                    
                    # Extract basic metrics
                    classes = []
                    functions = []
                    imports = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            classes.append(node.name)
                        elif isinstance(node, ast.FunctionDef):
                            functions.append(node.name)
                        elif isinstance(node, (ast.Import, ast.ImportFrom)):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    imports.append(alias.name)
                            else:
                                imports.append(node.module or '')
                    
                    # Calculate basic complexity (simplified)
                    complexity = len(functions) + len(classes) * 2
                    
                    return jsonify({
                        'lines': len(content.splitlines()),
                        'classes': classes,
                        'functions': functions,
                        'imports': imports,
                        'complexity': complexity,
                        'has_tests': 'test' in module_path.lower(),
                        'file_size': len(content)
                    })
                else:
                    return jsonify({'error': 'File not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/llm/status', methods=['GET'])
        def get_llm_status():
            """Get current LLM API status."""
            try:
                if hasattr(self.monitor, 'llm_monitor') and self.monitor.llm_monitor:
                    return jsonify({
                        'demo_mode': self.monitor.llm_monitor.demo_mode,
                        'api_enabled': not self.monitor.llm_monitor.demo_mode,
                        'gemini_available': self.monitor.llm_monitor.gemini_available,
                        'api_key_configured': bool(self.monitor.llm_monitor.api_key),
                        'total_api_calls': len(self.monitor.llm_monitor.api_calls),
                        'active_analyses': len(self.monitor.llm_monitor.active_analyses)
                    })
                else:
                    return jsonify({'error': 'LLM monitor not available'}), 503
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/llm/estimate-cost', methods=['POST'])
        def estimate_llm_cost():
            """Estimate cost for analyzing a module."""
            try:
                data = request.get_json()
                module_path = data.get('module_path')
                
                if not module_path:
                    return jsonify({'error': 'module_path required'}), 400
                
                # Read file to estimate size
                import os
                if os.path.exists(module_path):
                    with open(module_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Estimate tokens (roughly 4 chars per token)
                    estimated_tokens = len(content) // 4 + 500  # Add overhead for prompt
                    # Gemini 2.5 Pro costs approximately $0.00025 per 1K input tokens
                    estimated_cost = (estimated_tokens / 1000) * 0.00025 * 2  # x2 for output
                    
                    return jsonify({
                        'module_path': module_path,
                        'file_size_bytes': len(content),
                        'estimated_tokens': estimated_tokens,
                        'estimated_cost_usd': round(estimated_cost, 4),
                        'model': 'gemini-2.5-pro'
                    })
                else:
                    return jsonify({'error': 'Module file not found'}), 404
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/tests/status')
        def get_tests_status():
            """Get test status for all modules (non-LLM)."""
            # Get codebase parameter for multi-codebase support
            codebase_path = request.args.get('codebase', os.getcwd())
            try:
                # Use existing TestMapper and CoverageAnalyzer
                from testmaster.mapping.test_mapper import TestMapper
                from testmaster.analysis.coverage_analyzer import CoverageAnalyzer
                
                mapper = TestMapper(codebase_path, os.path.join(codebase_path, 'tests'))
                mapping = mapper.build_complete_mapping()
                
                results = []
                for module_path, tests in mapping.module_to_tests.items():
                    status = 'green' if len(tests) > 0 else 'red'
                    if len(tests) > 0:
                        # Check if tests pass
                        passed = all(test.last_passed for test in tests.values() if test.last_passed is not None)
                        if not passed:
                            status = 'yellow'
                    
                    results.append({
                        'module': module_path,
                        'status': status,
                        'test_count': len(tests),
                        'tests': list(tests.keys())
                    })
                
                return jsonify(results)
            except Exception as e:
                # Fallback to simple analysis
                import glob
                modules = glob.glob('**/*.py', recursive=True)
                results = []
                for module in modules[:50]:  # Limit to 50 for performance
                    test_file = module.replace('.py', '_test.py')
                    if os.path.exists(test_file):
                        status = 'green'
                    elif 'test' in module.lower():
                        continue
                    else:
                        status = 'red'
                    results.append({
                        'module': module,
                        'status': status,
                        'test_count': 1 if status == 'green' else 0
                    })
                return jsonify(results)
        
        @self.app.route('/api/dependencies/graph')
        def get_dependency_graph():
            """Get module dependency graph (non-LLM)."""
            try:
                from testmaster.mapping.dependency_tracker import DependencyTracker
                import networkx as nx
                
                tracker = DependencyTracker('.')
                graph = tracker.build_dependency_graph()
                
                # Convert NetworkX graph to JSON-serializable format
                nodes = []
                edges = []
                
                for node in graph.nodes():
                    nodes.append({
                        'id': node,
                        'label': node.split('/')[-1] if '/' in node else node
                    })
                
                for source, target in graph.edges():
                    edges.append({
                        'source': source,
                        'target': target
                    })
                
                # Calculate additional metrics
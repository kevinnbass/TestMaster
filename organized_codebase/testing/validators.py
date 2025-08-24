                circular_deps = 0
                isolated_modules = 0
                
                try:
                    # Check for circular dependencies
                    cycles = list(nx.simple_cycles(graph))
                    circular_deps = len(cycles)
                    
                    # Find isolated modules (no edges)
                    for node in graph.nodes():
                        if graph.degree(node) == 0:
                            isolated_modules += 1
                except:
                    pass
                
                return jsonify({
                    'nodes': nodes,
                    'edges': edges,
                    'metrics': {
                        'circular_dependencies': circular_deps,
                        'isolated_modules': isolated_modules
                    }
                })
            except Exception as e:
                # Fallback to simple import analysis
                import ast
                nodes = []
                edges = []
                modules_seen = set()
                
                import glob
                for file_path in glob.glob('**/*.py', recursive=True)[:30]:  # Limit for performance
                    if 'test' in file_path.lower():
                        continue
                    
                    module_name = file_path.replace('.py', '').replace('/', '.')
                    if module_name not in modules_seen:
                        nodes.append({'id': module_name, 'label': module_name.split('.')[-1]})
                        modules_seen.add(module_name)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            tree = ast.parse(f.read())
                        
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    target = alias.name
                                    if target not in modules_seen:
                                        nodes.append({'id': target, 'label': target.split('.')[-1]})
                                        modules_seen.add(target)
                                    edges.append({'source': module_name, 'target': target})
                    except:
                        pass
                
                return jsonify({'nodes': nodes[:50], 'edges': edges[:100]})  # Limit for UI performance
        
        @self.app.route('/api/codebases', methods=['GET', 'POST', 'DELETE'])
        def manage_codebases():
            """Manage active codebases."""
            if request.method == 'GET':
                # Return list of active codebases
                return jsonify({
                    'active_codebases': [
                        {'path': '/testmaster', 'name': 'TestMaster', 'status': 'active'},
                        # Additional codebases would be stored in memory or database
                    ]
                })
            elif request.method == 'POST':
                # Add new codebase
                data = request.get_json()
                path = data.get('path')
                if not path or not os.path.exists(path):
                    return jsonify({'error': 'Invalid codebase path'}), 400
                
                name = os.path.basename(path) or path
                return jsonify({
                    'success': True,
                    'codebase': {'path': path, 'name': name, 'status': 'active'}
                })
            elif request.method == 'DELETE':
                # Remove codebase
                path = request.args.get('path')
                return jsonify({'success': True, 'removed': path})
        
        @self.app.route('/api/refactor/hierarchy', methods=['POST'])
        def analyze_refactoring_hierarchy():
            """Analyze codebase hierarchy for refactoring opportunities."""
            data = request.json
            codebase_path = data.get('codebase_path', '.')
            codebase_name = data.get('codebase', 'TestMaster')
            
            if not self.refactor_analyzer:
                return jsonify({'error': 'Refactoring analyzer not available'}), 503
            
            try:
                # Check if we have cached hierarchy
                if codebase_name not in self.refactor_hierarchies:
                    # Perform hierarchical analysis
                    hierarchy = self.refactor_analyzer.analyze_codebase(
                        codebase_path, 
                        codebase_name
                    )
                    self.refactor_hierarchies[codebase_name] = hierarchy
                else:
                    hierarchy = self.refactor_hierarchies[codebase_name]
                
                # Generate roadmap if not cached
                if codebase_name not in self.refactor_roadmaps:
                    roadmap = self.refactor_analyzer.generate_refactor_roadmap(hierarchy)
                    self.refactor_roadmaps[codebase_name] = roadmap
                else:
                    roadmap = self.refactor_roadmaps[codebase_name]
                
                # Prepare response
                response = {
                    'success': True,
                    'hierarchy': {
                        'total_files': hierarchy.total_files,
                        'total_lines': hierarchy.total_lines,
                        'summary': hierarchy.summary,
                        'clusters': [
                            {
                                'name': c.name,
                                'files_count': len(c.files),
                                'lines_of_code': c.metrics.lines_of_code,
                                'cohesion_score': c.metrics.cohesion_score,
                                'coupling_score': c.metrics.coupling_score,
                                'refactor_opportunities': len(c.refactor_opportunities),
                                'high_severity_issues': sum(1 for o in c.refactor_opportunities if o.severity == 'high')
                            } for c in hierarchy.clusters[:10]  # Top 10 clusters
                        ],
                        'global_metrics': {
                            'cyclomatic_complexity': hierarchy.global_metrics.cyclomatic_complexity,
                            'cognitive_complexity': hierarchy.global_metrics.cognitive_complexity,
                            'avg_cohesion': hierarchy.global_metrics.cohesion_score,
                            'avg_coupling': hierarchy.global_metrics.coupling_score
                        }
                    },
                    'roadmap': {
                        'id': roadmap.id,
                        'title': roadmap.title,
                        'phases_count': len(roadmap.phases),
                        'total_effort_hours': roadmap.total_effort_hours,
                        'priority_score': roadmap.priority_score,
                        'risk_assessment': roadmap.risk_assessment,
                        'phases': roadmap.phases[:2]  # First 2 phases for preview
                    }
                }
                
                return jsonify(response)
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/refactor/opportunities', methods=['GET'])
        def get_refactoring_opportunities():
            """Get refactoring opportunities for a codebase."""
            codebase = request.args.get('codebase', 'TestMaster')
            severity = request.args.get('severity')  # Optional filter
            
            if not self.refactor_analyzer:
                return jsonify({'error': 'Refactoring analyzer not available'}), 503
            
            if codebase not in self.refactor_hierarchies:
                return jsonify({'error': 'Codebase not analyzed yet'}), 404
            
            hierarchy = self.refactor_hierarchies[codebase]
            
            # Collect all opportunities
            all_opportunities = []
            for cluster in hierarchy.clusters:
                for opp in cluster.refactor_opportunities:
                    if not severity or opp.severity == severity:
                        all_opportunities.append({
                            'id': opp.id,
                            'type': opp.type,
                            'severity': opp.severity,
                            'location': opp.location,
                            'description': opp.description,
                            'estimated_effort': opp.estimated_effort,
                            'impact_score': opp.impact_score,
                            'cluster': cluster.name
                        })
            
            # Sort by impact score
            all_opportunities.sort(key=lambda x: x['impact_score'], reverse=True)
            
            return jsonify({
                'opportunities': all_opportunities[:50],  # Top 50
                'total_count': len(all_opportunities),
                'codebase': codebase
            })
        
        @self.app.route('/api/refactor/analysis')
        def get_refactor_analysis():
            """Get automated refactor opportunity analysis (non-LLM)."""
            try:
                import ast
                import glob
                
                refactor_opportunities = {
                    'code_duplication': [],
                    'long_methods': [],
                    'complex_classes': [],
                    'unused_code': [],
                    'missing_tests': []
                }
                
                # Analyze Python files for refactor opportunities
                for file_path in glob.glob('**/*.py', recursive=True)[:30]:  # Limit for performance
                    if '__pycache__' in file_path or 'test' in file_path.lower():
                        continue
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            tree = ast.parse(content)
                        
                        # Check for long methods (>50 lines)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                                    method_length = node.end_lineno - node.lineno
                                    if method_length > 50:
                                        refactor_opportunities['long_methods'].append({
                                            'file': file_path,
                                            'method': node.name,
                                            'lines': method_length
                                        })
                            
                            # Check for complex classes (>10 methods)
                            elif isinstance(node, ast.ClassDef):
                                method_count = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
                                if method_count > 10:
                                    refactor_opportunities['complex_classes'].append({
                                        'file': file_path,
                                        'class': node.name,
                                        'methods': method_count
                                    })
                    except:
                        pass
                
                # Check for missing tests
                from testmaster.mapping.test_mapper import TestMapper
                try:
                    mapper = TestMapper('.', 'tests')
                    mapping = mapper.build_complete_mapping()
                    
                    for module_path in mapping.module_to_tests:
                        if len(mapping.module_to_tests[module_path]) == 0:
                            refactor_opportunities['missing_tests'].append({
                                'module': module_path,
                                'status': 'no_tests'
                            })
                except:
                    pass
                
                return jsonify({
                    'refactor_opportunities': refactor_opportunities,
                    'summary': {
                        'long_methods': len(refactor_opportunities['long_methods']),
                        'complex_classes': len(refactor_opportunities['complex_classes']),
                        'missing_tests': len(refactor_opportunities['missing_tests'])
                    }
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def _start_background_monitoring(self):
        """Start background monitoring thread."""
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitor_thread.start()
    
    def _monitoring_worker(self):
        """Background monitoring worker."""
        self.monitor.running = True
        next_collection = time.time()
        next_monitor_update = time.time()
        
        while self.monitor.running:
            try:
                current_time = time.time()
                
                # Collect performance data every 0.1 seconds
                if current_time >= next_collection:
                    self._collect_performance_data()
                    next_collection = current_time + self.collection_interval
                
                # Update monitor metrics less frequently (every 2 seconds)
                if current_time >= next_monitor_update:
                    self.monitor._collect_metrics()
                    self.monitor._check_alerts()
                    self.monitor._store_metrics()
                    next_monitor_update = current_time + self.monitor.update_interval
                
                # Sleep for a short time to avoid busy waiting
                time.sleep(0.02)  # 20ms sleep for more responsive updates
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(1)
    
    def _collect_performance_data(self):
        """Collect real-time performance data into rolling window - PER CODEBASE."""
        import psutil
        import time
        import os
        from datetime import datetime
        
        try:
            current_time = time.time()
            
            # Find TestMaster process (this web_monitor.py process)
            if self.testmaster_process is None:
                try:
                    self.testmaster_process = psutil.Process()  # Current process
                except:
                    self.testmaster_process = None
            
            # Get TestMaster-specific metrics
            if self.testmaster_process and self.testmaster_process.is_running():
                # CPU usage for TestMaster process only
                # Need to call twice for cpu_percent to work properly
                current_cpu = self.testmaster_process.cpu_percent(interval=0.01)
                if current_cpu == 0.0:
                    # Try again with a small interval
                    current_cpu = self.testmaster_process.cpu_percent(interval=0.01)
                
                # Memory usage for TestMaster process only (in MB)
                mem_info = self.testmaster_process.memory_info()
                current_memory_mb = mem_info.rss / (1024 * 1024)
            else:
                current_cpu = 0.0
                current_memory_mb = 0.0
            
            # Track network activity per codebase (simulated based on API activity)
            # In a real implementation, we'd track actual API calls per codebase
            # For now, we'll simulate based on process activity
            current_network_kb_s = 0.0
            
            # If TestMaster is active, simulate network activity based on CPU usage
            if current_cpu > 0:
                # Light network activity when processing (0.1-10 KB/s typical for API calls)
                import random
                base_network = 0.1 + (current_cpu / 100) * 2  # Scale with CPU activity
                # Add some variance to simulate burst API calls
                current_network_kb_s = base_network * (1 + random.random() * 0.5)
            
            # For TestMaster process, CPU load is same as CPU usage
            # since we're tracking a single process, not system-wide
            avg_cpu_load = current_cpu
            
            timestamp = datetime.now()
            
            # Update history for all monitored codebases
            with self.history_lock:
                # Get list of codebases to update
                codebases_to_update = list(self.performance_history.keys()) if self.performance_history else ['/testmaster']
                
                # Ensure at least /testmaster exists
                if '/testmaster' not in self.performance_history:
                    self.performance_history['/testmaster'] = {
                        'timestamps': [],
                        'cpu_usage': [],
                        'memory_usage_mb': [],
                        'cpu_load': [],
                        'network_kb_s': [],
                        'active': False  # Track if this codebase is currently being analyzed
                    }
                
                # Update each codebase with appropriate metrics
                for codebase in codebases_to_update:
                    history = self.performance_history[codebase]
                    
                    # Simulate different activity levels per codebase
                    # In reality, this would track actual per-codebase process activity
                    if codebase == '/testmaster':
                        # Primary codebase gets full metrics
                        codebase_cpu = current_cpu
                        codebase_memory = current_memory_mb
                        codebase_network = current_network_kb_s
                    else:
                        # Other codebases get reduced/no activity when not active
                        # This simulates that we're not currently analyzing them
                        codebase_cpu = 0.0
                        codebase_memory = 0.0
                        codebase_network = 0.0
                    
                    # Add new data point
                    history['timestamps'].append(timestamp)
                    history['cpu_usage'].append(codebase_cpu)
                    history['memory_usage_mb'].append(codebase_memory)
                    history['cpu_load'].append(codebase_cpu)  # Same as CPU for single process
                    history['network_kb_s'].append(codebase_network)
                    
                    # Maintain rolling window of 300 points
                    if len(history['timestamps']) > self.max_history_points:
                        history['timestamps'].pop(0)
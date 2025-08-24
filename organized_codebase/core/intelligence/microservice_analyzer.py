"""
Microservice Evolution Analyzer
==============================

Modularized from architectural_decision_engine.py for better maintainability.
Analyzes microservice architecture evolution with boundary recommendations and pattern analysis.

Author: Agent E - Infrastructure Consolidation
"""

import networkx as nx
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MicroserviceEvolutionAnalyzer:
    """Analyzes microservice architecture evolution"""
    
    def __init__(self):
        self.service_patterns = self._initialize_service_patterns()
        self.boundary_heuristics = self._define_boundary_heuristics()
    
    def _initialize_service_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize microservice patterns and characteristics"""
        return {
            "data_management": {
                "patterns": ["Database per service", "Shared database", "Data lake"],
                "complexity": {"low": "Shared database", "medium": "Database per service", "high": "Data lake"},
                "consistency": {"strong": "Shared database", "eventual": "Database per service"}
            },
            "communication": {
                "patterns": ["Synchronous HTTP", "Asynchronous messaging", "Event streaming"],
                "latency": {"low": "Synchronous HTTP", "medium": "Asynchronous messaging", "high": "Event streaming"},
                "reliability": {"low": "Synchronous HTTP", "medium": "Event streaming", "high": "Asynchronous messaging"}
            },
            "deployment": {
                "patterns": ["Container orchestration", "Serverless", "VM-based"],
                "scalability": {"low": "VM-based", "medium": "Container orchestration", "high": "Serverless"},
                "cost": {"low": "VM-based", "medium": "Container orchestration", "high": "Serverless"}
            }
        }
    
    def _define_boundary_heuristics(self) -> List[str]:
        """Define heuristics for service boundary identification"""
        return [
            "Business capability alignment",
            "Data ownership clarity",
            "Team ownership mapping",
            "Change frequency correlation",
            "Scalability requirements",
            "Technology stack compatibility",
            "Compliance and security domains",
            "User journey mapping",
            "API stability requirements",
            "Testing independence"
        ]
    
    def analyze_microservice_evolution(self, current_architecture: Dict[str, Any],
                                     requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how microservice architecture should evolve"""
        analysis = {
            "current_assessment": self._assess_current_microservices(current_architecture),
            "boundary_recommendations": self._recommend_service_boundaries(current_architecture, requirements),
            "pattern_recommendations": self._recommend_service_patterns(current_architecture, requirements),
            "migration_strategy": self._create_migration_strategy(current_architecture, requirements),
            "success_metrics": self._define_microservice_success_metrics(),
            "risk_mitigation": self._identify_microservice_risks(current_architecture, requirements)
        }
        
        return analysis
    
    def _assess_current_microservices(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current microservice architecture"""
        assessment = {
            "service_count": len(architecture.get("services", [])),
            "service_sizes": [],
            "coupling_analysis": {},
            "data_patterns": {},
            "communication_patterns": {},
            "deployment_patterns": {}
        }
        
        services = architecture.get("services", [])
        
        # Analyze service sizes
        for service in services:
            size_metrics = service.get("size_metrics", {})
            lines_of_code = size_metrics.get("lines_of_code", 0)
            assessment["service_sizes"].append({
                "name": service.get("name", "Unknown"),
                "size": lines_of_code,
                "category": self._categorize_service_size(lines_of_code)
            })
        
        # Analyze coupling
        assessment["coupling_analysis"] = self._analyze_service_coupling(services)
        
        # Analyze patterns
        assessment["data_patterns"] = self._analyze_data_patterns(services)
        assessment["communication_patterns"] = self._analyze_communication_patterns(services)
        assessment["deployment_patterns"] = self._analyze_deployment_patterns(services)
        
        return assessment
    
    def _categorize_service_size(self, lines_of_code: int) -> str:
        """Categorize service size"""
        if lines_of_code < 1000:
            return "small"
        elif lines_of_code < 5000:
            return "medium"
        elif lines_of_code < 15000:
            return "large"
        else:
            return "too_large"
    
    def _analyze_service_coupling(self, services: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze coupling between services"""
        coupling = {
            "high_coupling_pairs": [],
            "communication_frequency": {},
            "dependency_graph": {},
            "circular_dependencies": []
        }
        
        # Build dependency graph
        dependencies = {}
        for service in services:
            service_name = service.get("name", "Unknown")
            service_deps = service.get("dependencies", [])
            dependencies[service_name] = service_deps
        
        coupling["dependency_graph"] = dependencies
        
        # Identify high coupling (services with many dependencies)
        for service_name, deps in dependencies.items():
            if len(deps) > 5:  # Threshold for high coupling
                coupling["high_coupling_pairs"].append({
                    "service": service_name,
                    "dependency_count": len(deps),
                    "dependencies": deps
                })
        
        # Check for circular dependencies (simplified)
        coupling["circular_dependencies"] = self._find_circular_dependencies(dependencies)
        
        return coupling
    
    def _find_circular_dependencies(self, dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """Find circular dependencies in service graph"""
        # Create directed graph
        graph = nx.DiGraph()
        
        for service, deps in dependencies.items():
            for dep in deps:
                graph.add_edge(service, dep)
        
        # Find strongly connected components
        cycles = []
        try:
            strongly_connected = nx.strongly_connected_components(graph)
            for component in strongly_connected:
                if len(component) > 1:
                    cycles.append(list(component))
        except Exception as e:
            logger.warning(f"Error finding circular dependencies: {e}")
        
        return cycles
    
    def _analyze_data_patterns(self, services: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data management patterns"""
        patterns = {
            "shared_databases": [],
            "service_databases": [],
            "data_consistency_issues": [],
            "data_duplication": []
        }
        
        # Group services by database usage
        database_usage = {}
        for service in services:
            databases = service.get("databases", [])
            service_name = service.get("name", "Unknown")
            
            for db in databases:
                if db not in database_usage:
                    database_usage[db] = []
                database_usage[db].append(service_name)
        
        # Identify shared databases
        for db, using_services in database_usage.items():
            if len(using_services) > 1:
                patterns["shared_databases"].append({
                    "database": db,
                    "services": using_services,
                    "sharing_count": len(using_services)
                })
            else:
                patterns["service_databases"].append({
                    "database": db,
                    "service": using_services[0]
                })
        
        return patterns
    
    def _analyze_communication_patterns(self, services: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze service communication patterns"""
        patterns = {
            "synchronous_calls": 0,
            "asynchronous_messaging": 0,
            "event_driven": 0,
            "chatty_interfaces": [],
            "communication_protocols": {}
        }
        
        for service in services:
            communications = service.get("communications", [])
            service_name = service.get("name", "Unknown")
            
            sync_count = 0
            async_count = 0
            event_count = 0
            
            for comm in communications:
                comm_type = comm.get("type", "unknown")
                protocol = comm.get("protocol", "unknown")
                
                # Count by type
                if comm_type == "synchronous":
                    sync_count += 1
                elif comm_type == "asynchronous":
                    async_count += 1
                elif comm_type == "event":
                    event_count += 1
                
                # Track protocols
                if protocol not in patterns["communication_protocols"]:
                    patterns["communication_protocols"][protocol] = 0
                patterns["communication_protocols"][protocol] += 1
            
            patterns["synchronous_calls"] += sync_count
            patterns["asynchronous_messaging"] += async_count
            patterns["event_driven"] += event_count
            
            # Identify chatty interfaces
            total_communications = sync_count + async_count + event_count
            if total_communications > 10:  # Threshold for chatty
                patterns["chatty_interfaces"].append({
                    "service": service_name,
                    "communication_count": total_communications
                })
        
        return patterns
    
    def _analyze_deployment_patterns(self, services: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze service deployment patterns"""
        patterns = {
            "container_orchestration": 0,
            "serverless": 0,
            "vm_based": 0,
            "deployment_frequency": {},
            "scaling_patterns": {}
        }
        
        for service in services:
            deployment = service.get("deployment", {})
            service_name = service.get("name", "Unknown")
            
            deployment_type = deployment.get("type", "unknown")
            if deployment_type == "container":
                patterns["container_orchestration"] += 1
            elif deployment_type == "serverless":
                patterns["serverless"] += 1
            elif deployment_type == "vm":
                patterns["vm_based"] += 1
            
            # Track deployment frequency
            deploy_frequency = deployment.get("frequency", "unknown")
            if deploy_frequency not in patterns["deployment_frequency"]:
                patterns["deployment_frequency"][deploy_frequency] = 0
            patterns["deployment_frequency"][deploy_frequency] += 1
            
            # Track scaling patterns
            scaling = deployment.get("scaling", {})
            patterns["scaling_patterns"][service_name] = scaling
        
        return patterns
    
    def _recommend_service_boundaries(self, architecture: Dict[str, Any],
                                    requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend service boundary adjustments"""
        recommendations = []
        
        # Analyze current boundaries against heuristics
        services = architecture.get("services", [])
        
        for heuristic in self.boundary_heuristics:
            recommendation = self._evaluate_boundary_heuristic(heuristic, services, requirements)
            if recommendation:
                recommendations.append(recommendation)
        
        return recommendations
    
    def _evaluate_boundary_heuristic(self, heuristic: str, services: List[Dict[str, Any]],
                                   requirements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate a specific boundary heuristic"""
        if heuristic == "Business capability alignment":
            return self._check_business_capability_alignment(services, requirements)
        elif heuristic == "Data ownership clarity":
            return self._check_data_ownership_clarity(services)
        elif heuristic == "Team ownership mapping":
            return self._check_team_ownership_mapping(services)
        elif heuristic == "Change frequency correlation":
            return self._check_change_frequency_correlation(services)
        
        return None
    
    def _check_business_capability_alignment(self, services: List[Dict[str, Any]],
                                           requirements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if services align with business capabilities"""
        business_capabilities = requirements.get("business_capabilities", [])
        
        if not business_capabilities:
            return None
        
        misaligned_services = []
        for service in services:
            service_capability = service.get("business_capability", "unknown")
            if service_capability not in business_capabilities:
                misaligned_services.append(service.get("name", "Unknown"))
        
        if misaligned_services:
            return {
                "heuristic": "Business capability alignment",
                "issue": f"Services not aligned with business capabilities: {misaligned_services}",
                "recommendation": "Realign services with defined business capabilities",
                "impact": "high"
            }
        
        return None
    
    def _check_data_ownership_clarity(self, services: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check if data ownership is clear"""
        shared_data = {}
        
        for service in services:
            data_entities = service.get("data_entities", [])
            service_name = service.get("name", "Unknown")
            
            for entity in data_entities:
                if entity not in shared_data:
                    shared_data[entity] = []
                shared_data[entity].append(service_name)
        
        unclear_ownership = {entity: owners for entity, owners in shared_data.items() if len(owners) > 1}
        
        if unclear_ownership:
            return {
                "heuristic": "Data ownership clarity",
                "issue": f"Unclear data ownership for entities: {list(unclear_ownership.keys())}",
                "recommendation": "Assign clear data ownership to single services",
                "impact": "medium",
                "details": unclear_ownership
            }
        
        return None
    
    def _check_team_ownership_mapping(self, services: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check if team ownership is well-defined"""
        team_service_mapping = {}
        
        for service in services:
            owning_team = service.get("owning_team", "unknown")
            service_name = service.get("name", "Unknown")
            
            if owning_team not in team_service_mapping:
                team_service_mapping[owning_team] = []
            team_service_mapping[owning_team].append(service_name)
        
        # Check for teams with too many services
        overloaded_teams = {team: services for team, services in team_service_mapping.items() if len(services) > 5}
        
        if overloaded_teams:
            return {
                "heuristic": "Team ownership mapping",
                "issue": f"Teams with too many services: {list(overloaded_teams.keys())}",
                "recommendation": "Redistribute services among teams or split large teams",
                "impact": "medium",
                "details": overloaded_teams
            }
        
        return None
    
    def _check_change_frequency_correlation(self, services: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check if services that change together are properly grouped"""
        # This would typically require historical change data
        # For now, we'll provide a placeholder implementation
        
        return {
            "heuristic": "Change frequency correlation",
            "issue": "Unable to analyze without historical change data",
            "recommendation": "Implement change tracking to identify services that change together",
            "impact": "low"
        }
    
    def _recommend_service_patterns(self, architecture: Dict[str, Any],
                                  requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend service patterns based on requirements"""
        recommendations = {}
        
        # Data management pattern recommendation
        recommendations["data_management"] = self._recommend_data_pattern(architecture, requirements)
        
        # Communication pattern recommendation
        recommendations["communication"] = self._recommend_communication_pattern(architecture, requirements)
        
        # Deployment pattern recommendation
        recommendations["deployment"] = self._recommend_deployment_pattern(architecture, requirements)
        
        return recommendations
    
    def _recommend_data_pattern(self, architecture: Dict[str, Any],
                              requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend data management pattern"""
        consistency_requirement = requirements.get("data_consistency", "eventual")
        complexity_tolerance = requirements.get("complexity_tolerance", "medium")
        
        data_patterns = self.service_patterns["data_management"]
        
        if consistency_requirement == "strong":
            recommended_pattern = data_patterns["consistency"]["strong"]
        else:
            recommended_pattern = data_patterns["consistency"]["eventual"]
        
        return {
            "recommended_pattern": recommended_pattern,
            "rationale": f"Based on {consistency_requirement} consistency requirement",
            "implementation_steps": self._get_data_pattern_steps(recommended_pattern)
        }
    
    def _recommend_communication_pattern(self, architecture: Dict[str, Any],
                                       requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend communication pattern"""
        latency_requirement = requirements.get("latency_requirement", "medium")
        reliability_requirement = requirements.get("reliability_requirement", "medium")
        
        comm_patterns = self.service_patterns["communication"]
        
        # Choose pattern based on requirements
        if latency_requirement == "low":
            recommended_pattern = comm_patterns["latency"]["low"]
        elif reliability_requirement == "high":
            recommended_pattern = comm_patterns["reliability"]["high"]
        else:
            recommended_pattern = comm_patterns["latency"]["medium"]
        
        return {
            "recommended_pattern": recommended_pattern,
            "rationale": f"Based on {latency_requirement} latency and {reliability_requirement} reliability requirements",
            "implementation_steps": self._get_communication_pattern_steps(recommended_pattern)
        }
    
    def _recommend_deployment_pattern(self, architecture: Dict[str, Any],
                                    requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend deployment pattern"""
        scalability_requirement = requirements.get("scalability_requirement", "medium")
        cost_sensitivity = requirements.get("cost_sensitivity", "medium")
        
        deploy_patterns = self.service_patterns["deployment"]
        
        if scalability_requirement == "high":
            recommended_pattern = deploy_patterns["scalability"]["high"]
        elif cost_sensitivity == "low":
            recommended_pattern = deploy_patterns["cost"]["low"]
        else:
            recommended_pattern = deploy_patterns["scalability"]["medium"]
        
        return {
            "recommended_pattern": recommended_pattern,
            "rationale": f"Based on {scalability_requirement} scalability and {cost_sensitivity} cost requirements",
            "implementation_steps": self._get_deployment_pattern_steps(recommended_pattern)
        }
    
    def _get_data_pattern_steps(self, pattern: str) -> List[str]:
        """Get implementation steps for data pattern"""
        if pattern == "Database per service":
            return [
                "Identify data boundaries for each service",
                "Extract service-specific data models",
                "Implement database per service",
                "Set up data synchronization mechanisms",
                "Implement eventual consistency patterns"
            ]
        elif pattern == "Shared database":
            return [
                "Define shared data access patterns",
                "Implement database access layer",
                "Set up transaction management",
                "Implement data validation rules"
            ]
        else:
            return ["Define implementation steps for " + pattern]
    
    def _get_communication_pattern_steps(self, pattern: str) -> List[str]:
        """Get implementation steps for communication pattern"""
        if pattern == "Asynchronous messaging":
            return [
                "Choose message broker technology",
                "Define message schemas and contracts",
                "Implement message publishers and subscribers",
                "Set up message routing and filtering",
                "Implement error handling and retry mechanisms"
            ]
        elif pattern == "Synchronous HTTP":
            return [
                "Define REST API contracts",
                "Implement API gateways",
                "Set up load balancing",
                "Implement circuit breakers",
                "Add API versioning"
            ]
        else:
            return ["Define implementation steps for " + pattern]
    
    def _get_deployment_pattern_steps(self, pattern: str) -> List[str]:
        """Get implementation steps for deployment pattern"""
        if pattern == "Container orchestration":
            return [
                "Containerize services using Docker",
                "Set up Kubernetes cluster",
                "Define deployment manifests",
                "Implement service discovery",
                "Set up monitoring and logging"
            ]
        elif pattern == "Serverless":
            return [
                "Break down services into functions",
                "Choose serverless platform",
                "Implement function deployment pipeline",
                "Set up event triggers",
                "Implement cold start optimization"
            ]
        else:
            return ["Define implementation steps for " + pattern]
    
    def _create_migration_strategy(self, architecture: Dict[str, Any],
                                 requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create migration strategy for microservice evolution"""
        strategy = {
            "migration_approach": "incremental",
            "phases": [],
            "timeline": {},
            "risk_mitigation": [],
            "success_criteria": []
        }
        
        # Define migration phases
        strategy["phases"] = [
            {
                "phase": 1,
                "name": "Assessment and Planning",
                "duration": "2-4 weeks",
                "activities": [
                    "Complete service boundary analysis",
                    "Define target architecture",
                    "Create detailed migration plan",
                    "Set up monitoring and metrics"
                ]
            },
            {
                "phase": 2,
                "name": "Infrastructure Preparation",
                "duration": "3-6 weeks",
                "activities": [
                    "Set up container orchestration platform",
                    "Implement service discovery",
                    "Set up CI/CD pipelines",
                    "Implement monitoring and logging"
                ]
            },
            {
                "phase": 3,
                "name": "Service Extraction",
                "duration": "8-16 weeks",
                "activities": [
                    "Extract services incrementally",
                    "Implement service communication",
                    "Migrate data to service databases",
                    "Test and validate each service"
                ]
            },
            {
                "phase": 4,
                "name": "Optimization and Tuning",
                "duration": "4-8 weeks",
                "activities": [
                    "Optimize service performance",
                    "Tune scaling configurations",
                    "Implement advanced patterns",
                    "Complete testing and validation"
                ]
            }
        ]
        
        # Define timeline
        total_weeks = sum([int(phase["duration"].split("-")[1].split()[0]) for phase in strategy["phases"]])
        strategy["timeline"]["total_duration"] = f"{total_weeks} weeks"
        strategy["timeline"]["parallel_activities"] = "Infrastructure setup can run parallel with assessment"
        
        return strategy
    
    def _define_microservice_success_metrics(self) -> List[str]:
        """Define success metrics for microservice architecture"""
        return [
            "Service independence: Each service can be deployed independently",
            "Scalability: Services can scale independently based on load",
            "Fault isolation: Failure in one service doesn't affect others",
            "Team autonomy: Teams can work independently on their services",
            "Technology diversity: Teams can choose appropriate technologies",
            "Deployment frequency: Increased deployment frequency per service",
            "Mean time to recovery: Reduced MTTR for service issues",
            "Development velocity: Increased feature delivery speed",
            "Resource utilization: Improved resource efficiency",
            "Business capability alignment: Services align with business domains"
        ]
    
    def _identify_microservice_risks(self, architecture: Dict[str, Any],
                                   requirements: Dict[str, Any]) -> Dict[str, List[str]]:
        """Identify risks in microservice migration"""
        return {
            "technical_risks": [
                "Distributed system complexity",
                "Network latency and reliability",
                "Data consistency challenges",
                "Service communication failures",
                "Monitoring and debugging complexity"
            ],
            "organizational_risks": [
                "Team coordination overhead",
                "Skills gap in distributed systems",
                "Increased operational complexity",
                "Service ownership ambiguity",
                "Cross-team dependency management"
            ],
            "business_risks": [
                "Migration timeline overrun",
                "Temporary performance degradation",
                "Increased infrastructure costs",
                "Business continuity disruption",
                "Customer experience impact"
            ],
            "mitigation_strategies": [
                "Implement comprehensive monitoring from day one",
                "Start with strangler fig pattern for low-risk migration",
                "Invest in team training and skills development",
                "Establish clear service ownership and SLAs",
                "Implement circuit breakers and timeout patterns",
                "Use feature flags for safe rollouts",
                "Maintain automated testing at all levels"
            ]
        }


__all__ = ['MicroserviceEvolutionAnalyzer']
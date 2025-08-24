"""
YAML Configuration Enhancement
=============================

Adds comprehensive YAML configuration support to the unified state management system.
Extends existing functionality without replacing any current features.

Author: TestMaster Enhancement System
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Import our unified state manager
import sys
sys.path.append(str(Path(__file__).parent.parent))
from state.unified_state_manager import (
    unified_state_manager, 
    TeamConfiguration, TeamWorkflow, ServiceConfiguration, 
    DeploymentConfiguration, GraphNode, GraphConfiguration,
    TeamRole, SupervisorMode, ServiceType, DeploymentMode, 
    GraphExecutionMode, NodeState
)


class YAMLConfigurationEnhancer:
    """
    Adds YAML support to existing configuration systems.
    Does NOT replace - extends current functionality.
    """
    
    def __init__(self, state_manager=None):
        self.state_manager = state_manager or unified_state_manager
        self.logger = logging.getLogger("yaml_config_enhancer")
        self.config_templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize YAML configuration templates"""
        self.config_templates = {
            "team": {
                "team_id": "example_team",
                "roles": ["architect", "engineer", "qa_agent"],
                "supervisor_mode": "guided",
                "workflow_type": "standard",
                "max_parallel_tasks": 3,
                "quality_threshold": 80.0,
                "timeout_minutes": 30
            },
            "deployment": {
                "deployment_id": "example_deployment",
                "name": "Example Deployment",
                "mode": "development",
                "services": [
                    {
                        "service_id": "test_executor_1",
                        "service_type": "test_executor",
                        "name": "Test Executor Service",
                        "version": "1.0.0",
                        "config": {
                            "workers": 4,
                            "timeout": 300
                        },
                        "dependencies": [],
                        "resources": {
                            "cpu": "1000m",
                            "memory": "512Mi"
                        },
                        "health_check": {
                            "path": "/health",
                            "interval": 30
                        }
                    }
                ],
                "network_config": {
                    "cluster_ip": "auto",
                    "load_balancer": False
                },
                "security_config": {
                    "authentication": "bearer_token",
                    "encryption": True
                },
                "monitoring_config": {
                    "metrics_enabled": True,
                    "log_level": "INFO"
                },
                "scaling_config": {
                    "auto_scale": True,
                    "min_replicas": 1,
                    "max_replicas": 10
                }
            },
            "graph": {
                "graph_id": "example_graph",
                "name": "Example Graph",
                "execution_mode": "sequential",
                "timeout_seconds": 3600,
                "auto_retry": True,
                "checkpoint_enabled": True,
                "nodes": [
                    {
                        "node_id": "architect_node",
                        "agent_type": "test_architect",
                        "config": {
                            "analysis_depth": "comprehensive"
                        },
                        "dependencies": [],
                        "outputs": ["test_plan"],
                        "max_retries": 3
                    },
                    {
                        "node_id": "engineer_node", 
                        "agent_type": "test_engineer",
                        "config": {
                            "implementation_style": "modular"
                        },
                        "dependencies": ["architect_node"],
                        "outputs": ["test_code"],
                        "max_retries": 3
                    }
                ]
            },
            "workflow": {
                "workflow_id": "example_workflow",
                "name": "Example Workflow",
                "phases": [
                    {
                        "phase_id": "planning",
                        "name": "Test Planning",
                        "agents": ["architect"],
                        "duration_estimate": 30,
                        "deliverables": ["test_plan"]
                    },
                    {
                        "phase_id": "implementation",
                        "name": "Test Implementation", 
                        "agents": ["engineer"],
                        "duration_estimate": 60,
                        "deliverables": ["test_code"]
                    }
                ],
                "dependencies": {
                    "implementation": ["planning"]
                },
                "success_criteria": {
                    "quality_threshold": 85.0,
                    "coverage_threshold": 90.0
                }
            }
        }
    
    # ========================================================================
    # YAML LOADING AND PARSING
    # ========================================================================
    
    def load_team_config_from_yaml(self, yaml_path: Union[str, Path]) -> Optional[str]:
        """Load team configuration from YAML file"""
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            team_id = config_data.get("team_id")
            if not team_id:
                self.logger.error("team_id is required in YAML config")
                return None
            
            # Create team using unified state manager
            success = self.state_manager.create_team(
                team_id=team_id,
                roles=config_data.get("roles", []),
                supervisor_mode=config_data.get("supervisor_mode", "guided"),
                max_parallel_tasks=config_data.get("max_parallel_tasks", 3),
                quality_threshold=config_data.get("quality_threshold", 80.0)
            )
            
            if success:
                self.logger.info(f"Team '{team_id}' created from YAML: {yaml_path}")
                return team_id
            else:
                self.logger.error(f"Failed to create team from YAML: {yaml_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading team config from YAML {yaml_path}: {e}")
            return None
    
    def load_deployment_config_from_yaml(self, yaml_path: Union[str, Path]) -> Optional[str]:
        """Load deployment configuration from YAML file"""
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            deployment_id = config_data.get("deployment_id")
            if not deployment_id:
                self.logger.error("deployment_id is required in YAML config")
                return None
            
            # Create deployment using unified state manager
            success = self.state_manager.create_deployment(
                deployment_id=deployment_id,
                name=config_data.get("name", deployment_id),
                mode=config_data.get("mode", "development"),
                services=config_data.get("services", [])
            )
            
            if success:
                self.logger.info(f"Deployment '{deployment_id}' created from YAML: {yaml_path}")
                return deployment_id
            else:
                self.logger.error(f"Failed to create deployment from YAML: {yaml_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading deployment config from YAML {yaml_path}: {e}")
            return None
    
    def load_graph_config_from_yaml(self, yaml_path: Union[str, Path]) -> Optional[str]:
        """Load graph configuration from YAML file"""
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            graph_id = config_data.get("graph_id")
            if not graph_id:
                self.logger.error("graph_id is required in YAML config")
                return None
            
            # Create graph using unified state manager
            success = self.state_manager.create_execution_graph(
                graph_id=graph_id,
                name=config_data.get("name", graph_id),
                nodes=config_data.get("nodes", []),
                execution_mode=config_data.get("execution_mode", "sequential")
            )
            
            if success:
                self.logger.info(f"Graph '{graph_id}' created from YAML: {yaml_path}")
                return graph_id
            else:
                self.logger.error(f"Failed to create graph from YAML: {yaml_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading graph config from YAML {yaml_path}: {e}")
            return None
    
    def load_workflow_config_from_yaml(self, yaml_path: Union[str, Path]) -> Optional[str]:
        """Load workflow configuration from YAML file"""
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            workflow_id = config_data.get("workflow_id")
            if not workflow_id:
                self.logger.error("workflow_id is required in YAML config")
                return None
            
            # Create workflow using unified state manager
            success = self.state_manager.create_workflow(
                workflow_id=workflow_id,
                name=config_data.get("name", workflow_id),
                phases=config_data.get("phases", []),
                dependencies=config_data.get("dependencies", {})
            )
            
            if success:
                self.logger.info(f"Workflow '{workflow_id}' created from YAML: {yaml_path}")
                return workflow_id
            else:
                self.logger.error(f"Failed to create workflow from YAML: {yaml_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading workflow config from YAML {yaml_path}: {e}")
            return None
    
    def load_complete_config_from_yaml(self, yaml_path: Union[str, Path]) -> Dict[str, List[str]]:
        """Load complete configuration (multiple entities) from YAML file"""
        results = {
            "teams": [],
            "deployments": [], 
            "graphs": [],
            "workflows": [],
            "errors": []
        }
        
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Load teams
            for team_config in config_data.get("teams", []):
                try:
                    team_id = team_config.get("team_id")
                    success = self.state_manager.create_team(
                        team_id=team_id,
                        roles=team_config.get("roles", []),
                        supervisor_mode=team_config.get("supervisor_mode", "guided"),
                        max_parallel_tasks=team_config.get("max_parallel_tasks", 3),
                        quality_threshold=team_config.get("quality_threshold", 80.0)
                    )
                    if success:
                        results["teams"].append(team_id)
                    else:
                        results["errors"].append(f"Failed to create team: {team_id}")
                except Exception as e:
                    results["errors"].append(f"Error creating team: {e}")
            
            # Load deployments
            for deploy_config in config_data.get("deployments", []):
                try:
                    deployment_id = deploy_config.get("deployment_id")
                    success = self.state_manager.create_deployment(
                        deployment_id=deployment_id,
                        name=deploy_config.get("name", deployment_id),
                        mode=deploy_config.get("mode", "development"),
                        services=deploy_config.get("services", [])
                    )
                    if success:
                        results["deployments"].append(deployment_id)
                    else:
                        results["errors"].append(f"Failed to create deployment: {deployment_id}")
                except Exception as e:
                    results["errors"].append(f"Error creating deployment: {e}")
            
            # Load graphs
            for graph_config in config_data.get("graphs", []):
                try:
                    graph_id = graph_config.get("graph_id")
                    success = self.state_manager.create_execution_graph(
                        graph_id=graph_id,
                        name=graph_config.get("name", graph_id),
                        nodes=graph_config.get("nodes", []),
                        execution_mode=graph_config.get("execution_mode", "sequential")
                    )
                    if success:
                        results["graphs"].append(graph_id)
                    else:
                        results["errors"].append(f"Failed to create graph: {graph_id}")
                except Exception as e:
                    results["errors"].append(f"Error creating graph: {e}")
            
            # Load workflows
            for workflow_config in config_data.get("workflows", []):
                try:
                    workflow_id = workflow_config.get("workflow_id")
                    success = self.state_manager.create_workflow(
                        workflow_id=workflow_id,
                        name=workflow_config.get("name", workflow_id),
                        phases=workflow_config.get("phases", []),
                        dependencies=workflow_config.get("dependencies", {})
                    )
                    if success:
                        results["workflows"].append(workflow_id)
                    else:
                        results["errors"].append(f"Failed to create workflow: {workflow_id}")
                except Exception as e:
                    results["errors"].append(f"Error creating workflow: {e}")
            
            self.logger.info(f"Loaded complete config from YAML: {yaml_path}")
            
        except Exception as e:
            results["errors"].append(f"Error loading YAML file: {e}")
            self.logger.error(f"Error loading complete config from YAML {yaml_path}: {e}")
        
        return results
    
    # ========================================================================
    # YAML EXPORT AND GENERATION
    # ========================================================================
    
    def export_team_to_yaml(self, team_id: str, output_path: Union[str, Path]) -> bool:
        """Export team configuration to YAML file"""
        try:
            team_state = self.state_manager.get_team_state(team_id)
            if not team_state:
                self.logger.error(f"Team '{team_id}' not found")
                return False
            
            # Convert to YAML-friendly format
            yaml_config = {
                "team_id": team_id,
                **team_state["config"]
            }
            
            with open(output_path, 'w') as f:
                yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(f"Team '{team_id}' exported to YAML: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting team to YAML: {e}")
            return False
    
    def export_deployment_to_yaml(self, deployment_id: str, output_path: Union[str, Path]) -> bool:
        """Export deployment configuration to YAML file"""
        try:
            deployment_state = self.state_manager.get_deployment_state(deployment_id)
            if not deployment_state:
                self.logger.error(f"Deployment '{deployment_id}' not found")
                return False
            
            # Convert to YAML-friendly format
            yaml_config = {
                "deployment_id": deployment_id,
                **deployment_state["config"]
            }
            
            with open(output_path, 'w') as f:
                yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(f"Deployment '{deployment_id}' exported to YAML: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting deployment to YAML: {e}")
            return False
    
    def export_all_configs_to_yaml(self, output_path: Union[str, Path]) -> bool:
        """Export all configurations to a single YAML file"""
        try:
            global_state = self.state_manager.get_global_state()
            all_configs = self.state_manager.export_all_configurations()
            
            yaml_export = {
                "export_info": {
                    "timestamp": datetime.now().isoformat(),
                    "total_teams": global_state["global_metrics"]["active_teams"],
                    "total_deployments": global_state["global_metrics"]["active_deployments"],
                    "total_graphs": global_state["global_metrics"]["total_graphs"],
                    "total_workflows": global_state["global_metrics"]["total_workflows"]
                },
                "teams": [],
                "deployments": [],
                "graphs": [],
                "workflows": []
            }
            
            # Convert configurations to YAML format
            for team_id, team_config in all_configs["configurations"].get("teams", {}).items():
                yaml_export["teams"].append({
                    "team_id": team_id,
                    **team_config
                })
            
            for deployment_id, deploy_config in all_configs["configurations"].get("deployments", {}).items():
                yaml_export["deployments"].append({
                    "deployment_id": deployment_id,
                    **deploy_config
                })
            
            for graph_id, graph_config in all_configs["configurations"].get("graphs", {}).items():
                yaml_export["graphs"].append({
                    "graph_id": graph_id,
                    **graph_config
                })
            
            # Save to YAML file
            with open(output_path, 'w') as f:
                yaml.dump(yaml_export, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(f"All configurations exported to YAML: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting all configs to YAML: {e}")
            return False
    
    # ========================================================================
    # TEMPLATE GENERATION
    # ========================================================================
    
    def generate_template(self, config_type: str, output_path: Union[str, Path]) -> bool:
        """Generate YAML template for specified configuration type"""
        if config_type not in self.config_templates:
            self.logger.error(f"Unknown config type: {config_type}")
            return False
        
        try:
            template = self.config_templates[config_type]
            
            with open(output_path, 'w') as f:
                f.write(f"# YAML Configuration Template for {config_type.title()}\n")
                f.write(f"# Generated by TestMaster YAML Config Enhancer\n")
                f.write(f"# Created: {datetime.now().isoformat()}\n\n")
                yaml.dump(template, f, default_flow_style=False, sort_keys=False)
            
            self.logger.info(f"Template for '{config_type}' generated: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating template: {e}")
            return False
    
    def generate_all_templates(self, output_dir: Union[str, Path]) -> Dict[str, bool]:
        """Generate all available templates"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for config_type in self.config_templates.keys():
            template_path = output_dir / f"{config_type}_template.yaml"
            success = self.generate_template(config_type, template_path)
            results[config_type] = success
        
        # Generate complete template with all types
        complete_template = {
            "teams": [self.config_templates["team"]],
            "deployments": [self.config_templates["deployment"]],
            "graphs": [self.config_templates["graph"]],
            "workflows": [self.config_templates["workflow"]]
        }
        
        complete_path = output_dir / "complete_template.yaml"
        try:
            with open(complete_path, 'w') as f:
                f.write("# Complete YAML Configuration Template\n")
                f.write("# Contains all configuration types in one file\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                yaml.dump(complete_template, f, default_flow_style=False, sort_keys=False)
            results["complete"] = True
        except Exception as e:
            self.logger.error(f"Error generating complete template: {e}")
            results["complete"] = False
        
        return results
    
    # ========================================================================
    # VALIDATION AND UTILITIES
    # ========================================================================
    
    def validate_yaml_config(self, yaml_path: Union[str, Path], config_type: str) -> Dict[str, Any]:
        """Validate YAML configuration against expected schema"""
        validation_result = {
            "valid": False,
            "errors": [],
            "warnings": [],
            "config_type": config_type
        }
        
        try:
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            template = self.config_templates.get(config_type)
            if not template:
                validation_result["errors"].append(f"Unknown config type: {config_type}")
                return validation_result
            
            # Validate required fields
            for required_field in template.keys():
                if required_field not in config_data:
                    validation_result["errors"].append(f"Missing required field: {required_field}")
            
            # Validate enum values
            if config_type == "team":
                supervisor_mode = config_data.get("supervisor_mode")
                if supervisor_mode and supervisor_mode not in ["guided", "autonomous", "collaborative"]:
                    validation_result["errors"].append(f"Invalid supervisor_mode: {supervisor_mode}")
                
                roles = config_data.get("roles", [])
                valid_roles = ["architect", "engineer", "qa_agent", "executor", "coordinator"]
                for role in roles:
                    if role not in valid_roles:
                        validation_result["warnings"].append(f"Unknown role: {role}")
            
            elif config_type == "deployment":
                mode = config_data.get("mode")
                if mode and mode not in ["development", "staging", "production", "high_availability", "disaster_recovery"]:
                    validation_result["errors"].append(f"Invalid deployment mode: {mode}")
            
            elif config_type == "graph":
                execution_mode = config_data.get("execution_mode")
                if execution_mode and execution_mode not in ["sequential", "parallel", "hybrid", "adaptive"]:
                    validation_result["errors"].append(f"Invalid execution_mode: {execution_mode}")
            
            # If no errors, mark as valid
            validation_result["valid"] = len(validation_result["errors"]) == 0
            
        except Exception as e:
            validation_result["errors"].append(f"Error parsing YAML: {e}")
        
        return validation_result
    
    def get_enhancement_info(self) -> Dict[str, Any]:
        """Get information about YAML configuration enhancement"""
        return {
            "enhancement_type": "YAML Configuration Support",
            "base_system": "Unified State Management System",
            "enhancement_timestamp": "2025-08-19T19:56:02.000000",
            "capabilities": [
                "YAML file loading for all config types",
                "YAML export and backup functionality", 
                "Template generation for easy config creation",
                "Configuration validation and error checking",
                "Batch loading of multiple configurations",
                "Seamless integration with existing state manager"
            ],
            "supported_formats": [
                "Team configuration YAML",
                "Deployment configuration YAML",
                "Graph execution YAML", 
                "Workflow definition YAML",
                "Complete multi-entity YAML"
            ],
            "template_types": list(self.config_templates.keys()),
            "status": "FULLY_OPERATIONAL"
        }


# ============================================================================
# FACTORY AND EXPORTS
# ============================================================================

def create_yaml_config_enhancer(state_manager=None) -> YAMLConfigurationEnhancer:
    """Factory function to create YAML configuration enhancer"""
    return YAMLConfigurationEnhancer(state_manager)

# Global instance for compatibility
yaml_config_enhancer = create_yaml_config_enhancer()

# Export main classes and functions
__all__ = [
    'YAMLConfigurationEnhancer',
    'create_yaml_config_enhancer',
    'yaml_config_enhancer'
]
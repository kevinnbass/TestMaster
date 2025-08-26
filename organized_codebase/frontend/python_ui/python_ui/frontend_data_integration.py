#!/usr/bin/env python3
"""
🔗 ATOM: Frontend Data Integration Component
============================================
Frontend data integration and synchronization UI.
Part of STEELCLAD atomization - Agent T coordination specialist.
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

class DataSourceType(Enum):
    """Types of data sources"""
    API = "api"
    DATABASE = "database"
    WEBSOCKET = "websocket"
    FILE = "file"
    STREAM = "stream"
    CACHE = "cache"

@dataclass
class DataSource:
    """Data source configuration"""
    id: str
    name: str
    type: DataSourceType
    endpoint: str
    refresh_rate: int
    cache_enabled: bool = True
    cache_ttl: int = 300
    auth_required: bool = False
    status: str = "inactive"
    last_sync: Optional[datetime] = None

@dataclass
class DataRelationship:
    """Relationship between data entities"""
    source_entity: str
    target_entity: str
    relationship_type: str
    strength: float
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class FrontendDataIntegration:
    """Frontend data integration component"""
    
    def __init__(self):
        self.data_sources: Dict[str, DataSource] = {}
        self.data_relationships: List[DataRelationship] = []
        self.integration_cache = {}
        self.sync_status = {}
        self.integration_config = self._initialize_config()
    
    def _initialize_config(self) -> Dict[str, Any]:
        """Initialize integration configuration"""
        return {
            'auto_sync': True,
            'sync_interval': 30000,  # 30 seconds
            'batch_size': 100,
            'parallel_connections': 5,
            'retry_policy': {
                'max_retries': 3,
                'backoff_factor': 2,
                'timeout': 30
            },
            'data_transformation': {
                'normalize': True,
                'validate': True,
                'enrich': True
            }
        }
    
    def render_integration_dashboard(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render data integration dashboard
        
        Args:
            context: Dashboard context
            
        Returns:
            Integration dashboard configuration
        """
        return {
            'dashboard_type': 'data_integration',
            'layout': self._get_integration_layout(),
            'panels': {
                'sources_overview': self._render_sources_overview(),
                'sync_status': self._render_sync_status(),
                'data_flow': self._render_data_flow_diagram(),
                'relationships': self._render_relationships_graph(),
                'integration_controls': self._render_integration_controls()
            },
            'real_time': {
                'enabled': True,
                'update_interval': 5000,
                'data_streams': self._get_active_streams()
            }
        }
    
    def _get_integration_layout(self) -> Dict[str, Any]:
        """Get integration dashboard layout"""
        return {
            'type': 'split_layout',
            'sections': [
                {'id': 'sources', 'width': '25%', 'position': 'left'},
                {'id': 'main', 'width': '50%', 'position': 'center'},
                {'id': 'details', 'width': '25%', 'position': 'right'}
            ],
            'responsive': True
        }
    
    def _render_sources_overview(self) -> Dict[str, Any]:
        """Render data sources overview"""
        sources = []
        
        for source_id, source in self.data_sources.items():
            sources.append({
                'id': source_id,
                'name': source.name,
                'type': source.type.value,
                'status': source.status,
                'status_color': self._get_source_status_color(source.status),
                'endpoint': source.endpoint,
                'refresh_rate': f"{source.refresh_rate / 1000}s",
                'cache': {
                    'enabled': source.cache_enabled,
                    'ttl': source.cache_ttl
                },
                'last_sync': source.last_sync.isoformat() if source.last_sync else 'Never',
                'sync_age': self._calculate_sync_age(source.last_sync)
            })
        
        return {
            'title': 'Data Sources',
            'sources': sources,
            'summary': {
                'total': len(sources),
                'active': sum(1 for s in sources if s['status'] == 'active'),
                'syncing': sum(1 for s in sources if s['status'] == 'syncing'),
                'error': sum(1 for s in sources if s['status'] == 'error')
            },
            'actions': ['add_source', 'refresh_all', 'configure']
        }
    
    def _render_sync_status(self) -> Dict[str, Any]:
        """Render synchronization status"""
        sync_items = []
        
        for source_id, status in self.sync_status.items():
            if source_id in self.data_sources:
                source = self.data_sources[source_id]
                sync_items.append({
                    'source': source.name,
                    'status': status.get('status', 'pending'),
                    'progress': status.get('progress', 0),
                    'records_synced': status.get('records', 0),
                    'errors': status.get('errors', 0),
                    'duration': status.get('duration', 0),
                    'next_sync': self._calculate_next_sync(source)
                })
        
        return {
            'title': 'Synchronization Status',
            'sync_items': sync_items,
            'overall_status': self._calculate_overall_sync_status(),
            'metrics': {
                'total_records': sum(s.get('records_synced', 0) for s in sync_items),
                'total_errors': sum(s.get('errors', 0) for s in sync_items),
                'avg_duration': self._calculate_avg_sync_duration(sync_items)
            }
        }
    
    def _render_data_flow_diagram(self) -> Dict[str, Any]:
        """Render data flow diagram"""
        nodes = []
        links = []
        
        # Create nodes for sources
        for source_id, source in self.data_sources.items():
            nodes.append({
                'id': source_id,
                'label': source.name,
                'type': 'source',
                'category': source.type.value,
                'status': source.status
            })
        
        # Create nodes for integration points
        integration_points = ['processor', 'transformer', 'cache', 'frontend']
        for point in integration_points:
            nodes.append({
                'id': point,
                'label': point.capitalize(),
                'type': 'integration',
                'category': 'system'
            })
        
        # Create links based on data flow
        for source_id in self.data_sources:
            links.append({
                'source': source_id,
                'target': 'processor',
                'flow_rate': self._get_flow_rate(source_id)
            })
        
        links.extend([
            {'source': 'processor', 'target': 'transformer'},
            {'source': 'transformer', 'target': 'cache'},
            {'source': 'cache', 'target': 'frontend'}
        ])
        
        return {
            'title': 'Data Flow',
            'type': 'flow_diagram',
            'nodes': nodes,
            'links': links,
            'options': {
                'orientation': 'horizontal',
                'animated': True,
                'show_metrics': True
            }
        }
    
    def _render_relationships_graph(self) -> Dict[str, Any]:
        """Render data relationships graph"""
        entities = set()
        
        # Collect unique entities
        for rel in self.data_relationships:
            entities.add(rel.source_entity)
            entities.add(rel.target_entity)
        
        # Create nodes
        nodes = [
            {
                'id': entity,
                'label': entity,
                'connections': self._count_entity_connections(entity)
            }
            for entity in entities
        ]
        
        # Create edges
        edges = [
            {
                'source': rel.source_entity,
                'target': rel.target_entity,
                'type': rel.relationship_type,
                'strength': rel.strength,
                'bidirectional': rel.bidirectional
            }
            for rel in self.data_relationships
        ]
        
        return {
            'title': 'Data Relationships',
            'type': 'network_graph',
            'nodes': nodes,
            'edges': edges,
            'layout': {
                'type': 'force-directed',
                'charge': -200,
                'link_distance': 80
            },
            'interaction': {
                'zoom': True,
                'pan': True,
                'hover': 'show_details'
            }
        }
    
    def _render_integration_controls(self) -> Dict[str, Any]:
        """Render integration control panel"""
        return {
            'title': 'Integration Controls',
            'controls': [
                {
                    'type': 'toggle',
                    'label': 'Auto Sync',
                    'value': self.integration_config['auto_sync'],
                    'action': 'toggle_auto_sync'
                },
                {
                    'type': 'select',
                    'label': 'Sync Interval',
                    'options': [
                        {'value': 10000, 'label': '10 seconds'},
                        {'value': 30000, 'label': '30 seconds'},
                        {'value': 60000, 'label': '1 minute'},
                        {'value': 300000, 'label': '5 minutes'}
                    ],
                    'value': self.integration_config['sync_interval']
                },
                {
                    'type': 'number',
                    'label': 'Batch Size',
                    'value': self.integration_config['batch_size'],
                    'min': 10,
                    'max': 1000,
                    'step': 10
                }
            ],
            'actions': [
                {'id': 'sync_now', 'label': 'Sync All Now', 'style': 'primary'},
                {'id': 'clear_cache', 'label': 'Clear Cache', 'style': 'warning'},
                {'id': 'reset', 'label': 'Reset Integration', 'style': 'danger'}
            ],
            'transformation_options': {
                'normalize': self.integration_config['data_transformation']['normalize'],
                'validate': self.integration_config['data_transformation']['validate'],
                'enrich': self.integration_config['data_transformation']['enrich']
            }
        }
    
    def _get_active_streams(self) -> List[str]:
        """Get active data streams"""
        streams = []
        for source_id, source in self.data_sources.items():
            if source.type == DataSourceType.STREAM and source.status == 'active':
                streams.append(source_id)
        return streams
    
    def _get_source_status_color(self, status: str) -> str:
        """Get color for source status"""
        colors = {
            'active': 'success',
            'syncing': 'warning',
            'inactive': 'secondary',
            'error': 'danger'
        }
        return colors.get(status, 'secondary')
    
    def _calculate_sync_age(self, last_sync: Optional[datetime]) -> str:
        """Calculate time since last sync"""
        if not last_sync:
            return 'Never synced'
        
        delta = datetime.utcnow() - last_sync
        if delta.days > 0:
            return f"{delta.days}d ago"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h ago"
        elif delta.seconds > 60:
            return f"{delta.seconds // 60}m ago"
        else:
            return "Just now"
    
    def _calculate_next_sync(self, source: DataSource) -> str:
        """Calculate next sync time"""
        if not source.last_sync:
            return "Pending"
        
        next_sync = source.last_sync + timedelta(milliseconds=source.refresh_rate)
        if next_sync > datetime.utcnow():
            delta = next_sync - datetime.utcnow()
            if delta.seconds > 60:
                return f"In {delta.seconds // 60}m"
            else:
                return f"In {delta.seconds}s"
        else:
            return "Overdue"
    
    def _calculate_overall_sync_status(self) -> str:
        """Calculate overall sync status"""
        if not self.sync_status:
            return 'idle'
        
        statuses = [s.get('status', 'pending') for s in self.sync_status.values()]
        if 'error' in statuses:
            return 'error'
        elif 'syncing' in statuses:
            return 'syncing'
        elif all(s == 'complete' for s in statuses):
            return 'synchronized'
        else:
            return 'partial'
    
    def _calculate_avg_sync_duration(self, sync_items: List[Dict]) -> float:
        """Calculate average sync duration"""
        durations = [s.get('duration', 0) for s in sync_items if s.get('duration')]
        return sum(durations) / len(durations) if durations else 0
    
    def _get_flow_rate(self, source_id: str) -> float:
        """Get data flow rate for source"""
        # Placeholder implementation
        return 100.0
    
    def _count_entity_connections(self, entity: str) -> int:
        """Count connections for an entity"""
        count = 0
        for rel in self.data_relationships:
            if rel.source_entity == entity or rel.target_entity == entity:
                count += 1
        return count
    
    def add_data_source(self, source: DataSource):
        """Add a new data source"""
        self.data_sources[source.id] = source
        self.sync_status[source.id] = {'status': 'pending'}
    
    def add_relationship(self, relationship: DataRelationship):
        """Add a data relationship"""
        self.data_relationships.append(relationship)
    
    def sync_data_source(self, source_id: str) -> bool:
        """Sync a specific data source"""
        if source_id in self.data_sources:
            self.sync_status[source_id] = {
                'status': 'syncing',
                'progress': 0,
                'start_time': datetime.utcnow()
            }
            # Actual sync would happen here
            return True
        return False
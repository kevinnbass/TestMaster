"""
Analytics Export Manager
========================

Provides comprehensive export capabilities for analytics data in multiple formats.
Supports scheduled exports, custom templates, and various destinations.

Author: TestMaster Team
"""

import json
import csv
import logging
import os
import zipfile
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, BinaryIO
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from io import BytesIO, StringIO
import pickle
import base64

logger = logging.getLogger(__name__)

class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    EXCEL = "excel"
    PDF = "pdf"
    XML = "xml"
    PARQUET = "parquet"
    PICKLE = "pickle"

class ExportDestination(Enum):
    """Export destination types."""
    FILE = "file"
    MEMORY = "memory"
    S3 = "s3"
    FTP = "ftp"
    EMAIL = "email"
    API = "api"

@dataclass
class ExportJob:
    """Represents an export job."""
    job_id: str
    name: str
    format: ExportFormat
    destination: ExportDestination
    filters: Dict[str, Any]
    created_at: datetime
    status: str
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    size_bytes: int = 0
    row_count: int = 0
    completed_at: Optional[datetime] = None

class AnalyticsExportManager:
    """
    Manages analytics data exports with multiple format support.
    """
    
    def __init__(self, export_dir: str = "exports"):
        """
        Initialize export manager.
        
        Args:
            export_dir: Directory for file exports
        """
        self.export_dir = export_dir
        
        # Ensure export directory exists
        os.makedirs(export_dir, exist_ok=True)
        
        # Export jobs tracking
        self.export_jobs = {}
        self.scheduled_exports = []
        
        # Format handlers
        self.format_handlers = {
            ExportFormat.JSON: self._export_json,
            ExportFormat.CSV: self._export_csv,
            ExportFormat.HTML: self._export_html,
            ExportFormat.EXCEL: self._export_excel,
            ExportFormat.XML: self._export_xml,
            ExportFormat.PARQUET: self._export_parquet,
            ExportFormat.PICKLE: self._export_pickle
        }
        
        # Templates for formatted exports
        self.html_template = self._get_html_template()
        
        # Statistics
        self.export_stats = {
            'total_exports': 0,
            'successful_exports': 0,
            'failed_exports': 0,
            'total_size_bytes': 0,
            'total_rows_exported': 0
        }
        
        # Threading
        self.lock = threading.RLock()
        self.scheduler_active = False
        self.scheduler_thread = None
        
        logger.info(f"Analytics Export Manager initialized with export dir: {export_dir}")
    
    def export_analytics(self,
                         data: Dict[str, Any],
                         format: ExportFormat = ExportFormat.JSON,
                         destination: ExportDestination = ExportDestination.FILE,
                         filename: Optional[str] = None,
                         filters: Optional[Dict[str, Any]] = None) -> ExportJob:
        """
        Export analytics data in specified format.
        
        Args:
            data: Analytics data to export
            format: Export format
            destination: Export destination
            filename: Optional filename (auto-generated if not provided)
            filters: Optional filters to apply
            
        Returns:
            Export job object
        """
        job_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{format.value}"
        
        if not filename:
            filename = f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job = ExportJob(
            job_id=job_id,
            name=filename,
            format=format,
            destination=destination,
            filters=filters or {},
            created_at=datetime.now(),
            status="processing"
        )
        
        with self.lock:
            self.export_jobs[job_id] = job
        
        # Process export in thread
        thread = threading.Thread(
            target=self._process_export,
            args=(job, data)
        )
        thread.start()
        
        return job
    
    def _process_export(self, job: ExportJob, data: Dict[str, Any]):
        """Process export job."""
        try:
            # Apply filters if provided
            filtered_data = self._apply_filters(data, job.filters)
            
            # Get format handler
            handler = self.format_handlers.get(job.format)
            if not handler:
                raise ValueError(f"Unsupported format: {job.format}")
            
            # Export data
            if job.destination == ExportDestination.FILE:
                output_path = os.path.join(self.export_dir, f"{job.name}.{job.format.value}")
                result = handler(filtered_data, output_path)
                job.output_path = output_path
                
                # Get file size
                if os.path.exists(output_path):
                    job.size_bytes = os.path.getsize(output_path)
                
            elif job.destination == ExportDestination.MEMORY:
                result = handler(filtered_data, None)
                job.output_path = "memory"
                job.size_bytes = len(result) if isinstance(result, (str, bytes)) else 0
                
            else:
                # Other destinations not yet implemented
                raise NotImplementedError(f"Destination {job.destination} not yet implemented")
            
            # Update job status
            job.status = "completed"
            job.completed_at = datetime.now()
            job.row_count = self._count_rows(filtered_data)
            
            # Update statistics
            with self.lock:
                self.export_stats['total_exports'] += 1
                self.export_stats['successful_exports'] += 1
                self.export_stats['total_size_bytes'] += job.size_bytes
                self.export_stats['total_rows_exported'] += job.row_count
            
            logger.info(f"Export completed: {job.job_id} ({job.size_bytes} bytes, {job.row_count} rows)")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
            with self.lock:
                self.export_stats['total_exports'] += 1
                self.export_stats['failed_exports'] += 1
            
            logger.error(f"Export failed: {job.job_id} - {e}")
    
    def _apply_filters(self, data: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply filters to data before export."""
        if not filters:
            return data
        
        filtered_data = {}
        
        # Filter by date range
        if 'date_range' in filters:
            start_date = filters['date_range'].get('start')
            end_date = filters['date_range'].get('end')
            # Apply date filtering logic here
            filtered_data = data  # Placeholder
        else:
            filtered_data = data
        
        # Filter by metrics
        if 'metrics' in filters:
            selected_metrics = filters['metrics']
            if isinstance(selected_metrics, list):
                filtered_data = {k: v for k, v in data.items() if k in selected_metrics}
        
        # Filter by minimum values
        if 'min_values' in filters:
            for metric, min_val in filters['min_values'].items():
                if metric in filtered_data and isinstance(filtered_data[metric], (int, float)):
                    if filtered_data[metric] < min_val:
                        del filtered_data[metric]
        
        return filtered_data
    
    def _count_rows(self, data: Any) -> int:
        """Count rows in data."""
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            # Count items in largest list/array value
            max_count = 1
            for value in data.values():
                if isinstance(value, list):
                    max_count = max(max_count, len(value))
                elif isinstance(value, dict):
                    max_count = max(max_count, self._count_rows(value))
            return max_count
        else:
            return 1
    
    def _export_json(self, data: Dict[str, Any], output_path: Optional[str]) -> str:
        """Export data as JSON."""
        json_str = json.dumps(data, indent=2, default=str)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def _export_csv(self, data: Dict[str, Any], output_path: Optional[str]) -> str:
        """Export data as CSV."""
        # Flatten nested data for CSV export
        flattened = self._flatten_for_csv(data)
        
        output = StringIO()
        
        if flattened:
            writer = csv.DictWriter(output, fieldnames=flattened[0].keys())
            writer.writeheader()
            writer.writerows(flattened)
        
        csv_str = output.getvalue()
        
        if output_path:
            with open(output_path, 'w', newline='') as f:
                f.write(csv_str)
        
        return csv_str
    
    def _flatten_for_csv(self, data: Any, parent_key: str = '', sep: str = '.') -> List[Dict]:
        """Flatten nested data structure for CSV export."""
        items = []
        
        if isinstance(data, dict):
            flattened_dict = {}
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                
                if isinstance(v, dict):
                    flattened_dict.update(self._flatten_dict(v, new_key, sep))
                elif isinstance(v, list):
                    # For lists, create separate rows
                    if v and isinstance(v[0], dict):
                        for item in v:
                            row = flattened_dict.copy()
                            row.update(self._flatten_dict(item, new_key, sep))
                            items.append(row)
                    else:
                        flattened_dict[new_key] = str(v)
                else:
                    flattened_dict[new_key] = v
            
            if not items:
                items = [flattened_dict]
                
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    items.append(self._flatten_dict(item, parent_key, sep))
                else:
                    items.append({parent_key: item})
        
        return items
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten a dictionary."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep))
            else:
                items[new_key] = v
        
        return items
    
    def _export_html(self, data: Dict[str, Any], output_path: Optional[str]) -> str:
        """Export data as HTML."""
        html_content = self._generate_html_report(data)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_content)
        
        return html_content
    
    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML report from data."""
        # Use template and fill with data
        html = self.html_template
        
        # Replace placeholders
        html = html.replace("{{title}}", "Analytics Export Report")
        html = html.replace("{{generated_at}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Generate content sections
        content_html = ""
        
        for section_name, section_data in data.items():
            content_html += f"<div class='section'><h2>{section_name}</h2>"
            
            if isinstance(section_data, dict):
                content_html += "<table class='data-table'>"
                for key, value in section_data.items():
                    content_html += f"<tr><td>{key}</td><td>{value}</td></tr>"
                content_html += "</table>"
                
            elif isinstance(section_data, list):
                if section_data and isinstance(section_data[0], dict):
                    # Table for list of dicts
                    content_html += "<table class='data-table'><thead><tr>"
                    for key in section_data[0].keys():
                        content_html += f"<th>{key}</th>"
                    content_html += "</tr></thead><tbody>"
                    
                    for item in section_data:
                        content_html += "<tr>"
                        for value in item.values():
                            content_html += f"<td>{value}</td>"
                        content_html += "</tr>"
                    
                    content_html += "</tbody></table>"
                else:
                    # Simple list
                    content_html += "<ul>"
                    for item in section_data:
                        content_html += f"<li>{item}</li>"
                    content_html += "</ul>"
            else:
                content_html += f"<p>{section_data}</p>"
            
            content_html += "</div>"
        
        html = html.replace("{{content}}", content_html)
        
        return html
    
    def _get_html_template(self) -> str:
        """Get HTML template for reports."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>{{title}}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #666; margin-top: 30px; }
        .header { background: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .section { background: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .data-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        .data-table th { background: #4CAF50; color: white; padding: 10px; text-align: left; }
        .data-table td { padding: 8px; border-bottom: 1px solid #ddd; }
        .data-table tr:hover { background: #f5f5f5; }
        .metadata { color: #888; font-size: 0.9em; }
        ul { list-style-type: none; padding: 0; }
        li { padding: 5px; background: #f9f9f9; margin: 2px 0; border-radius: 3px; }
    </style>
</head>
<body>
    <div class='header'>
        <h1>{{title}}</h1>
        <p class='metadata'>Generated: {{generated_at}}</p>
    </div>
    {{content}}
</body>
</html>
"""
    
    def _export_excel(self, data: Dict[str, Any], output_path: Optional[str]) -> bytes:
        """Export data as Excel (placeholder - requires openpyxl)."""
        # This would require openpyxl library
        # For now, export as CSV which Excel can open
        csv_data = self._export_csv(data, None)
        
        if output_path:
            # Change extension to .csv for now
            csv_path = output_path.replace('.excel', '.csv')
            with open(csv_path, 'w') as f:
                f.write(csv_data)
        
        return csv_data.encode()
    
    def _export_xml(self, data: Dict[str, Any], output_path: Optional[str]) -> str:
        """Export data as XML."""
        xml_str = self._dict_to_xml(data, "analytics")
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(xml_str)
        
        return xml_str
    
    def _dict_to_xml(self, data: Any, root_name: str = "root") -> str:
        """Convert dictionary to XML string."""
        xml_parts = [f"<?xml version='1.0' encoding='UTF-8'?>"]
        xml_parts.append(f"<{root_name}>")
        
        def process_item(key: str, value: Any, indent: int = 1):
            indent_str = "  " * indent
            
            if isinstance(value, dict):
                xml_parts.append(f"{indent_str}<{key}>")
                for k, v in value.items():
                    process_item(k, v, indent + 1)
                xml_parts.append(f"{indent_str}</{key}>")
                
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        xml_parts.append(f"{indent_str}<{key}>")
                        for k, v in item.items():
                            process_item(k, v, indent + 1)
                        xml_parts.append(f"{indent_str}</{key}>")
                    else:
                        xml_parts.append(f"{indent_str}<{key}>{item}</{key}>")
                        
            else:
                xml_parts.append(f"{indent_str}<{key}>{value}</{key}>")
        
        if isinstance(data, dict):
            for key, value in data.items():
                process_item(key, value)
        
        xml_parts.append(f"</{root_name}>")
        
        return "\n".join(xml_parts)
    
    def _export_parquet(self, data: Dict[str, Any], output_path: Optional[str]) -> bytes:
        """Export data as Parquet (placeholder - requires pyarrow)."""
        # This would require pyarrow library
        # For now, use pickle as binary format
        return self._export_pickle(data, output_path)
    
    def _export_pickle(self, data: Dict[str, Any], output_path: Optional[str]) -> bytes:
        """Export data as pickle."""
        pickle_bytes = pickle.dumps(data)
        
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(pickle_bytes)
        
        return pickle_bytes
    
    def create_archive(self, job_ids: List[str], archive_name: Optional[str] = None) -> str:
        """Create archive of multiple export files."""
        if not archive_name:
            archive_name = f"analytics_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        archive_path = os.path.join(self.export_dir, f"{archive_name}.zip")
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for job_id in job_ids:
                job = self.export_jobs.get(job_id)
                if job and job.output_path and os.path.exists(job.output_path):
                    arcname = os.path.basename(job.output_path)
                    zipf.write(job.output_path, arcname)
        
        logger.info(f"Created archive: {archive_path}")
        return archive_path
    
    def schedule_export(self,
                       data_source: callable,
                       format: ExportFormat,
                       interval_minutes: int,
                       filters: Optional[Dict[str, Any]] = None):
        """Schedule periodic export."""
        scheduled_export = {
            'data_source': data_source,
            'format': format,
            'interval_minutes': interval_minutes,
            'filters': filters,
            'last_run': None,
            'next_run': datetime.now()
        }
        
        self.scheduled_exports.append(scheduled_export)
        
        # Start scheduler if not running
        if not self.scheduler_active:
            self.start_scheduler()
    
    def start_scheduler(self):
        """Start export scheduler."""
        if self.scheduler_active:
            return
        
        self.scheduler_active = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Export scheduler started")
    
    def stop_scheduler(self):
        """Stop export scheduler."""
        self.scheduler_active = False
        
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Export scheduler stopped")
    
    def _scheduler_loop(self):
        """Scheduler loop for periodic exports."""
        while self.scheduler_active:
            try:
                current_time = datetime.now()
                
                for scheduled in self.scheduled_exports:
                    if current_time >= scheduled['next_run']:
                        # Run export
                        try:
                            data = scheduled['data_source']()
                            self.export_analytics(
                                data,
                                format=scheduled['format'],
                                filters=scheduled['filters']
                            )
                            
                            scheduled['last_run'] = current_time
                            scheduled['next_run'] = current_time + timedelta(
                                minutes=scheduled['interval_minutes']
                            )
                            
                        except Exception as e:
                            logger.error(f"Scheduled export failed: {e}")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
    
    def get_export_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an export job."""
        job = self.export_jobs.get(job_id)
        
        if not job:
            return None
        
        return {
            'job_id': job.job_id,
            'name': job.name,
            'format': job.format.value,
            'status': job.status,
            'created_at': job.created_at.isoformat(),
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'size_bytes': job.size_bytes,
            'row_count': job.row_count,
            'output_path': job.output_path,
            'error_message': job.error_message
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get export statistics."""
        with self.lock:
            return {
                'export_stats': self.export_stats.copy(),
                'active_jobs': len([j for j in self.export_jobs.values() 
                                  if j.status == 'processing']),
                'completed_jobs': len([j for j in self.export_jobs.values() 
                                     if j.status == 'completed']),
                'failed_jobs': len([j for j in self.export_jobs.values() 
                                  if j.status == 'failed']),
                'scheduled_exports': len(self.scheduled_exports),
                'supported_formats': [f.value for f in ExportFormat],
                'export_directory': self.export_dir
            }
    
    def cleanup_old_exports(self, days: int = 7):
        """Clean up old export files."""
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        for filename in os.listdir(self.export_dir):
            file_path = os.path.join(self.export_dir, filename)
            
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_time < cutoff_date:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")
        
        logger.info(f"Cleaned up {deleted_count} old export files")
        return deleted_count
    
    def shutdown(self):
        """Shutdown export manager."""
        self.stop_scheduler()
        logger.info("Analytics Export Manager shutdown")
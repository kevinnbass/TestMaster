"""
Multi-Modal Data Analyzer Module
===============================

Multi-modal data analyzer inspired by Phidata's capabilities.
RESTORED from enhanced_monitor.py - this functionality was missing in consolidation.

Author: TestMaster Phase 1C Consolidation (RESTORATION)
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Union

class MultiModalAnalyzer:
    """
    Multi-modal data analyzer inspired by Phidata's capabilities.
    Handles various data types and formats for comprehensive analysis.
    RESTORED from enhanced_monitor.py
    """
    
    def __init__(self):
        self.analyzer_id = f"analyzer_{uuid.uuid4().hex[:12]}"
        self.supported_formats = {
            "json": self._analyze_json,
            "csv": self._analyze_csv,
            "log": self._analyze_logs,
            "metrics": self._analyze_metrics,
            "image": self._analyze_image,
            "text": self._analyze_text
        }
        self.logger = logging.getLogger('MultiModalAnalyzer')
    
    async def analyze_data(self, data: Any, data_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze data of various types and formats"""
        if data_type not in self.supported_formats:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        analysis_start = time.time()
        
        try:
            analyzer_func = self.supported_formats[data_type]
            result = await analyzer_func(data, context or {})
            
            analysis_time = time.time() - analysis_start
            
            return {
                "analyzer_id": self.analyzer_id,
                "data_type": data_type,
                "analysis_time": analysis_time,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {data_type}: {e}")
            return {
                "analyzer_id": self.analyzer_id,
                "data_type": data_type,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _analyze_json(self, data: Union[dict, str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze JSON data"""
        if isinstance(data, str):
            data = json.loads(data)
        
        analysis = {
            "structure": {},
            "insights": [],
            "anomalies": [],
            "recommendations": []
        }
        
        # Analyze structure
        analysis["structure"] = {
            "keys": list(data.keys()) if isinstance(data, dict) else [],
            "depth": self._calculate_dict_depth(data) if isinstance(data, dict) else 0,
            "size": len(str(data))
        }
        
        # Look for performance data
        if "performance" in data or "metrics" in data:
            analysis["insights"].append("Performance metrics detected")
        
        # Look for error indicators
        if "error" in str(data).lower() or "fail" in str(data).lower():
            analysis["anomalies"].append("Error indicators found in data")
        
        return analysis
    
    async def _analyze_csv(self, data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze CSV data"""
        lines = data.strip().split('\n')
        
        analysis = {
            "rows": len(lines),
            "columns": len(lines[0].split(',')) if lines else 0,
            "insights": [],
            "patterns": []
        }
        
        if analysis["rows"] > 1000:
            analysis["insights"].append("Large dataset detected - consider sampling")
        
        return analysis
    
    async def _analyze_logs(self, data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze log data"""
        lines = data.strip().split('\n')
        
        analysis = {
            "total_lines": len(lines),
            "error_count": 0,
            "warning_count": 0,
            "patterns": [],
            "timeline": {}
        }
        
        for line in lines:
            line_lower = line.lower()
            if "error" in line_lower:
                analysis["error_count"] += 1
            elif "warning" in line_lower or "warn" in line_lower:
                analysis["warning_count"] += 1
        
        # Calculate error rate
        if analysis["total_lines"] > 0:
            error_rate = analysis["error_count"] / analysis["total_lines"] * 100
            if error_rate > 5:
                analysis["patterns"].append(f"High error rate: {error_rate:.1f}%")
        
        return analysis
    
    async def _analyze_metrics(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metrics data"""
        analysis = {
            "metrics_count": len(data),
            "trends": {},
            "alerts": [],
            "summary": {}
        }
        
        for metric_name, value in data.items():
            if isinstance(value, (int, float)):
                # Simple threshold checking
                if "response_time" in metric_name.lower() and value > 1000:
                    analysis["alerts"].append(f"High response time: {metric_name} = {value}ms")
                elif "error_rate" in metric_name.lower() and value > 5:
                    analysis["alerts"].append(f"High error rate: {metric_name} = {value}%")
                elif "cpu" in metric_name.lower() and value > 80:
                    analysis["alerts"].append(f"High CPU usage: {metric_name} = {value}%")
        
        return analysis
    
    async def _analyze_image(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image data (placeholder for actual image analysis)"""
        return {
            "type": "image",
            "format": "unknown",
            "insights": ["Image analysis not fully implemented"],
            "recommendations": ["Consider integrating computer vision capabilities"]
        }
    
    async def _analyze_text(self, data: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text data"""
        analysis = {
            "length": len(data),
            "word_count": len(data.split()),
            "sentiment": "neutral",
            "keywords": [],
            "insights": []
        }
        
        # Simple keyword extraction
        important_words = ["error", "performance", "quality", "security", "test", "failed", "success"]
        for word in important_words:
            if word in data.lower():
                analysis["keywords"].append(word)
        
        # Simple sentiment analysis
        positive_words = ["success", "pass", "good", "excellent", "improved"]
        negative_words = ["error", "fail", "bad", "poor", "degraded"]
        
        positive_count = sum(1 for word in positive_words if word in data.lower())
        negative_count = sum(1 for word in negative_words if word in data.lower())
        
        if positive_count > negative_count:
            analysis["sentiment"] = "positive"
        elif negative_count > positive_count:
            analysis["sentiment"] = "negative"
        
        return analysis
    
    def _calculate_dict_depth(self, d: dict, current_depth: int = 0) -> int:
        """Calculate the maximum depth of a nested dictionary"""
        if not isinstance(d, dict):
            return current_depth
        
        if not d:
            return current_depth
        
        return max(
            self._calculate_dict_depth(value, current_depth + 1)
            for value in d.values()
        )

# Export key components
__all__ = [
    'MultiModalAnalyzer'
]
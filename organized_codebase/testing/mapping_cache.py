"""
Mapping Cache Manager

Simple cache management for test-module mappings and dependency data.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Optional, Dict
from datetime import datetime, timedelta


class MappingCache:
    """Simple cache manager for mapping data."""
    
    def __init__(self, cache_dir: str = ".testmaster_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get(self, key: str, max_age_hours: float = 24) -> Optional[Any]:
        """Get cached data if it exists and is not expired."""
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=max_age_hours):
                return None  # Expired
            
            return data['value']
        
        except Exception:
            return None
    
    def set(self, key: str, value: Any):
        """Cache data with timestamp."""
        cache_file = self.cache_dir / f"{key}.json"
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'value': value
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, default=str, indent=2)
        except Exception as e:
            print(f"⚠️ Error caching {key}: {e}")
    
    def clear(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached data."""
        files = list(self.cache_dir.glob("*.json"))
        return {
            "cache_dir": str(self.cache_dir),
            "cached_items": len(files),
            "total_size_mb": sum(f.stat().st_size for f in files) / 1024 / 1024
        }
import React, { useState, useMemo } from "react";
import VirtualizedList from "./VirtualizedList";
import { exportHotspotsToCsv, exportToJson } from "../utils/export";

interface HotspotTableProps {
  hotspots: Record<string, string[]>;
}

export default function HotspotTable({ hotspots }: HotspotTableProps) {
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState("");

  const getCategoryLabel = (category: string): string => {
    const labels: Record<string, string> = {
      "high_complexity": "High Complexity",
      "large_files": "Large Files", 
      "mixed_indentation": "Mixed Indentation",
      "high_todos": "High TODOs",
      "deeply_nested": "Deeply Nested",
      "long_functions": "Long Functions",
      "ts_any_overuse": "TypeScript 'any' Overuse",
      "eslint_ts_ignored_heavy": "ESLint/TS Ignores",
      "high_branching_python": "High Python Complexity",
      "binary_files": "Binary Files",
      "generated_files": "Generated Files"
    };
    return labels[category] || category.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const sortedKeys = useMemo(() => {
    const keys = Object.keys(hotspots || {});
    return keys.sort((a, b) => (hotspots[b]?.length || 0) - (hotspots[a]?.length || 0));
  }, [hotspots]);
  
  const toggleCategory = (category: string) => {
    const newExpanded = new Set(expandedCategories);
    if (newExpanded.has(category)) {
      newExpanded.delete(category);
    } else {
      newExpanded.add(category);
    }
    setExpandedCategories(newExpanded);
  };

  const filterFiles = (files: string[]) => {
    if (!searchTerm) return files;
    return files.filter(file => 
      file.toLowerCase().includes(searchTerm.toLowerCase())
    );
  };

  const getCategoryColor = (category: string): string => {
    const colors: Record<string, string> = {
      "high_complexity": "#ff6b6b",
      "large_files": "#feca57",
      "mixed_indentation": "#48dbfb",
      "high_todos": "#ff9ff3",
      "deeply_nested": "#54a0ff",
      "long_functions": "#ee5a6f",
      "ts_any_overuse": "#fd79a8",
      "eslint_ts_ignored_heavy": "#fdcb6e",
      "high_branching_python": "#e17055",
      "binary_files": "#74b9ff",
      "generated_files": "#a29bfe"
    };
    return colors[category] || "#6c757d";
  };

  return (
    <div className="hotspot-table">
      <div className="hotspot-header">
        <div className="header-left">
          <h2>Code Hotspots Analysis</h2>
          <div className="hotspot-stats">
            {sortedKeys.length} categories, {Object.values(hotspots).reduce((sum, files) => sum + files.length, 0)} total hotspots
          </div>
        </div>
        <div className="controls">
          <input
            type="text"
            placeholder="Search files..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
          <div className="export-buttons">
            <button 
              onClick={() => exportHotspotsToCsv(hotspots)}
              className="btn btn-small"
              title="Export to CSV"
            >
              📊 CSV
            </button>
            <button 
              onClick={() => exportToJson(hotspots, { filename: 'hotspots' })}
              className="btn btn-small"
              title="Export to JSON"
            >
              📄 JSON
            </button>
          </div>
        </div>
      </div>

      {sortedKeys.length === 0 ? (
        <div className="no-hotspots">No hotspots detected in the current scan.</div>
      ) : (
        <div className="categories">
          {sortedKeys.map(k => {
            const files = filterFiles(hotspots[k] || []);
            const isExpanded = expandedCategories.has(k);
            
            return (
              <div key={k} className="category-card">
                <div 
                  className="category-header"
                  onClick={() => toggleCategory(k)}
                  style={{ borderLeftColor: getCategoryColor(k) }}
                >
                  <div className="category-info">
                    <span className="category-name">{getCategoryLabel(k)}</span>
                    <span className="file-count">
                      {searchTerm ? `${files.length}/${hotspots[k]?.length || 0}` : files.length} files
                    </span>
                    {files.length > 50 && (
                      <span className="severity-badge high">HIGH</span>
                    )}
                  </div>
                  <span className="expand-icon">
                    {isExpanded ? "▼" : "▶"}
                  </span>
                </div>
                
                {isExpanded && (
                  <div className="file-list">
                    <VirtualizedList
                      items={files}
                      renderItem={(file, index) => {
                        const fileExt = file.split('.').pop() || 'unknown';
                        return (
                          <div className="file-item">
                            <span className={`file-ext ext-${fileExt}`}>{fileExt}</span>
                            <code className="file-path">{file}</code>
                          </div>
                        );
                      }}
                      itemHeight={32}
                      maxHeight={400}
                      pageSize={100}
                      searchable={files.length > 50}
                      searchKey={undefined}
                      emptyMessage="No files found matching search criteria"
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
      
      <div className="hotspot-footer">
        <p className="help-text">
          💡 <strong>Tip:</strong> Use the search box to quickly find specific files. Click category headers to expand/collapse sections.
        </p>
      </div>
    </div>
  );
}
import React, { useState, useEffect } from "react";

interface DirectoryHealthProps {
  scanId: number;
}

interface DirectoryInfo {
  path: string;
  fileCount: number;
  totalSize: number;
  avgFileSize: number;
  extensions: string[];
  healthScore: number;
}

export default function DirectoryHealth({ scanId }: DirectoryHealthProps) {
  const [directories, setDirectories] = useState<DirectoryInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortBy, setSortBy] = useState<"health" | "size" | "files">("health");
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadDirectoryData();
  }, [scanId]);

  const loadDirectoryData = async () => {
    setLoading(true);
    try {
      const mockDirectories: DirectoryInfo[] = [
        {
          path: "src/",
          fileCount: 150,
          totalSize: 2500000,
          avgFileSize: 16667,
          extensions: ["ts", "tsx", "js"],
          healthScore: 85
        },
        {
          path: "tests/",
          fileCount: 45,
          totalSize: 890000,
          avgFileSize: 19778,
          extensions: ["test.ts", "spec.tsx"],
          healthScore: 92
        },
        {
          path: "tools/codebase_monitor/",
          fileCount: 25,
          totalSize: 1200000,
          avgFileSize: 48000,
          extensions: ["py", "json"],
          healthScore: 65
        },
        {
          path: "web/dashboard_modules/",
          fileCount: 58,
          totalSize: 3200000,
          avgFileSize: 55172,
          extensions: ["py", "html", "css", "js"],
          healthScore: 45
        },
        {
          path: "PRODUCTION_PACKAGES/",
          fileCount: 1200,
          totalSize: 45000000,
          avgFileSize: 37500,
          extensions: ["min.js", "gz", "map"],
          healthScore: 25
        },
        {
          path: "organized_codebase/",
          fileCount: 800,
          totalSize: 15000000,
          avgFileSize: 18750,
          extensions: ["py", "js", "md", "json"],
          healthScore: 70
        }
      ];
      
      setDirectories(mockDirectories);
    } catch (error) {
      console.error("Failed to load directory data:", error);
    } finally {
      setLoading(false);
    }
  };

  const formatBytes = (bytes: number): string => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const getHealthColor = (score: number): string => {
    if (score >= 80) return '#28a745';
    if (score >= 60) return '#ffc107';
    return '#dc3545';
  };

  const getHealthLabel = (score: number): string => {
    if (score >= 80) return 'Good';
    if (score >= 60) return 'Fair';
    return 'Poor';
  };

  const sortedDirectories = [...directories].sort((a, b) => {
    switch (sortBy) {
      case "health":
        return b.healthScore - a.healthScore;
      case "size":
        return b.totalSize - a.totalSize;
      case "files":
        return b.fileCount - a.fileCount;
      default:
        return 0;
    }
  });

  const toggleDirectory = (path: string) => {
    const newExpanded = new Set(expandedDirs);
    if (newExpanded.has(path)) {
      newExpanded.delete(path);
    } else {
      newExpanded.add(path);
    }
    setExpandedDirs(newExpanded);
  };

  if (loading) {
    return (
      <div className="directory-health">
        <div className="loading">Loading directory analysis...</div>
      </div>
    );
  }

  return (
    <div className="directory-health">
      <div className="directory-header">
        <h2>Directory Health Analysis</h2>
        <div className="controls">
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value as "health" | "size" | "files")}
            className="sort-select"
          >
            <option value="health">Sort by Health Score</option>
            <option value="size">Sort by Size</option>
            <option value="files">Sort by File Count</option>
          </select>
        </div>
      </div>

      <div className="directory-list">
        {sortedDirectories.map((dir) => {
          const isExpanded = expandedDirs.has(dir.path);
          const healthColor = getHealthColor(dir.healthScore);
          
          return (
            <div key={dir.path} className="directory-card">
              <div 
                className="directory-summary"
                onClick={() => toggleDirectory(dir.path)}
              >
                <div className="directory-info">
                  <div className="directory-path">
                    <code>{dir.path}</code>
                  </div>
                  <div className="directory-metrics">
                    <span className="metric">
                      📁 {dir.fileCount} files
                    </span>
                    <span className="metric">
                      📊 {formatBytes(dir.totalSize)}
                    </span>
                    <span 
                      className="health-score"
                      style={{ backgroundColor: healthColor }}
                    >
                      {dir.healthScore} • {getHealthLabel(dir.healthScore)}
                    </span>
                  </div>
                </div>
                <span className="expand-icon">
                  {isExpanded ? "▼" : "▶"}
                </span>
              </div>

              {isExpanded && (
                <div className="directory-details">
                  <div className="detail-grid">
                    <div className="detail-item">
                      <span className="detail-label">Average File Size:</span>
                      <span className="detail-value">{formatBytes(dir.avgFileSize)}</span>
                    </div>
                    <div className="detail-item">
                      <span className="detail-label">File Types:</span>
                      <span className="detail-value">
                        {dir.extensions.map((ext, i) => (
                          <span key={ext} className={`extension-tag ext-${ext.replace('.', '')}`}>
                            {ext}
                          </span>
                        ))}
                      </span>
                    </div>
                  </div>

                  <div className="health-factors">
                    <h4>Health Factors</h4>
                    <div className="factor-list">
                      <div className="factor">
                        <span className="factor-name">File Size Distribution</span>
                        <span className={`factor-status ${dir.avgFileSize < 30000 ? 'good' : 'poor'}`}>
                          {dir.avgFileSize < 30000 ? '✓ Good' : '⚠ Large files'}
                        </span>
                      </div>
                      <div className="factor">
                        <span className="factor-name">Directory Size</span>
                        <span className={`factor-status ${dir.totalSize < 5000000 ? 'good' : 'poor'}`}>
                          {dir.totalSize < 5000000 ? '✓ Manageable' : '⚠ Very large'}
                        </span>
                      </div>
                      <div className="factor">
                        <span className="factor-name">File Count</span>
                        <span className={`factor-status ${dir.fileCount < 100 ? 'good' : 'warning'}`}>
                          {dir.fileCount < 100 ? '✓ Well organized' : '⚠ Many files'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {dir.healthScore < 60 && (
                    <div className="recommendations">
                      <h4>Recommendations</h4>
                      <ul>
                        {dir.avgFileSize > 50000 && (
                          <li>Consider breaking down large files (&gt; 50KB)</li>
                        )}
                        {dir.fileCount > 200 && (
                          <li>Directory has many files - consider reorganization</li>
                        )}
                        {dir.path.includes('PRODUCTION') && (
                          <li>Move production artifacts outside of main repository</li>
                        )}
                        {dir.extensions.includes('gz') && (
                          <li>Archive compressed files to reduce repository bloat</li>
                        )}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="summary-actions">
        <h3>Directory Actions</h3>
        <div className="action-grid">
          <div className="action-card">
            <h4>🧹 Cleanup Large Directories</h4>
            <p>Identify and relocate directories consuming excessive space</p>
            <button className="action-btn">Generate Cleanup Script</button>
          </div>
          <div className="action-card">
            <h4>📁 Reorganize Structure</h4>
            <p>Suggest better directory organization for improved maintainability</p>
            <button className="action-btn">Analyze Structure</button>
          </div>
          <div className="action-card">
            <h4>🎯 Focus Areas</h4>
            <p>Prioritize directories needing immediate attention</p>
            <button className="action-btn">Create Action Plan</button>
          </div>
        </div>
      </div>
    </div>
  );
}
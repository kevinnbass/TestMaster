import React from "react";
import { ScanSummary } from "../api/ApiClient";

interface OverviewProps {
  summary: ScanSummary | null;
  scanId: number;
}

export default function Overview({ summary, scanId }: OverviewProps) {
  const formatBytes = (bytes: number): string => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDate = (epochSeconds: number): string => {
    return new Date(epochSeconds * 1000).toLocaleString();
  };

  const formatNumber = (num: number): string => {
    return num.toLocaleString();
  };

  const getHealthScore = (summary: ScanSummary): { score: number; grade: string; color: string } => {
    const avgFileSize = summary.total_size_bytes / summary.total_files;
    const avgLinesPerFile = summary.total_code_lines / summary.total_files;
    
    let score = 100;
    
    if (avgFileSize > 50000) score -= 20;
    else if (avgFileSize > 20000) score -= 10;
    
    if (avgLinesPerFile > 500) score -= 20;
    else if (avgLinesPerFile > 200) score -= 10;
    
    if (summary.total_files > 10000) score -= 15;
    else if (summary.total_files > 5000) score -= 5;
    
    if (score >= 90) return { score, grade: 'A', color: '#28a745' };
    if (score >= 80) return { score, grade: 'B', color: '#6f9c3c' };
    if (score >= 70) return { score, grade: 'C', color: '#ffc107' };
    if (score >= 60) return { score, grade: 'D', color: '#fd7e14' };
    return { score, grade: 'F', color: '#dc3545' };
  };

  if (!summary) {
    return (
      <div className="overview">
        <div className="loading-summary">
          Loading summary data...
        </div>
      </div>
    );
  }

  const health = getHealthScore(summary);

  return (
    <div className="overview">
      <div className="overview-header">
        <h2>Codebase Overview</h2>
        <div className="scan-info">
          <span>Scan ID: {scanId}</span>
          <span>Generated: {formatDate(summary.generated_at_epoch)}</span>
        </div>
      </div>

      <div className="metrics-grid">
        <div className="metric-card primary">
          <div className="metric-icon">📁</div>
          <div className="metric-content">
            <div className="metric-value">{formatNumber(summary.total_files)}</div>
            <div className="metric-label">Total Files</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">📊</div>
          <div className="metric-content">
            <div className="metric-value">{formatBytes(summary.total_size_bytes)}</div>
            <div className="metric-label">Total Size</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">📝</div>
          <div className="metric-content">
            <div className="metric-value">{formatNumber(summary.total_code_lines)}</div>
            <div className="metric-label">Lines of Code</div>
          </div>
        </div>

        <div className="metric-card health" style={{ borderColor: health.color }}>
          <div className="metric-icon">🎯</div>
          <div className="metric-content">
            <div className="metric-value" style={{ color: health.color }}>
              {health.grade} ({health.score})
            </div>
            <div className="metric-label">Health Score</div>
          </div>
        </div>
      </div>

      <div className="summary-stats">
        <div className="stat-row">
          <div className="stat-item">
            <span className="stat-label">Average File Size:</span>
            <span className="stat-value">{formatBytes(Math.round(summary.total_size_bytes / summary.total_files))}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Average Lines per File:</span>
            <span className="stat-value">{Math.round(summary.total_code_lines / summary.total_files)}</span>
          </div>
        </div>
        
        <div className="stat-row">
          <div className="stat-item">
            <span className="stat-label">Root Directory:</span>
            <span className="stat-value code">{summary.root}</span>
          </div>
        </div>
      </div>

      <div className="health-breakdown">
        <h3>Health Score Breakdown</h3>
        <div className="health-factors">
          <div className="health-factor">
            <span className="factor-name">File Size Distribution</span>
            <span className={`factor-status ${summary.total_size_bytes / summary.total_files < 20000 ? 'good' : 'warning'}`}>
              {summary.total_size_bytes / summary.total_files < 20000 ? '✓ Good' : '⚠ Large files detected'}
            </span>
          </div>
          <div className="health-factor">
            <span className="factor-name">Code Complexity</span>
            <span className={`factor-status ${summary.total_code_lines / summary.total_files < 200 ? 'good' : 'warning'}`}>
              {summary.total_code_lines / summary.total_files < 200 ? '✓ Manageable' : '⚠ High complexity'}
            </span>
          </div>
          <div className="health-factor">
            <span className="factor-name">Repository Size</span>
            <span className={`factor-status ${summary.total_files < 5000 ? 'good' : 'warning'}`}>
              {summary.total_files < 5000 ? '✓ Well-sized' : '⚠ Large repository'}
            </span>
          </div>
        </div>
      </div>

      <div className="quick-actions">
        <h3>Quick Actions</h3>
        <div className="action-buttons">
          <button className="action-btn">
            📋 Download Report
          </button>
          <button className="action-btn">
            🧹 Generate Cleanup Scripts
          </button>
          <button className="action-btn">
            📈 View Trends
          </button>
        </div>
      </div>
    </div>
  );
}
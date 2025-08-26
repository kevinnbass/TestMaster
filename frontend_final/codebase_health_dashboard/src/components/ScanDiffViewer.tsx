import React, { useState, useEffect } from 'react';
import { getHotspots, getDuplicates, getScanSummary } from '../api/ApiClient';

interface ScanData {
  scanId: number;
  hotspots: Record<string, string[]>;
  duplicates: string[][];
  summary: any;
}

interface ScanDiffViewerProps {
  scanId1: number;
  scanId2: number;
  onClose: () => void;
}

interface DiffResult {
  added: string[];
  removed: string[];
  unchanged: string[];
}

export default function ScanDiffViewer({ scanId1, scanId2, onClose }: ScanDiffViewerProps) {
  const [scan1, setScan1] = useState<ScanData | null>(null);
  const [scan2, setScan2] = useState<ScanData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'hotspots' | 'duplicates' | 'summary'>('hotspots');

  useEffect(() => {
    loadScanData();
  }, [scanId1, scanId2]);

  const loadScanData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [
        [hotspots1, duplicates1, summary1],
        [hotspots2, duplicates2, summary2]
      ] = await Promise.all([
        Promise.all([
          getHotspots(scanId1),
          getDuplicates(scanId1, 1000),
          getScanSummary(scanId1)
        ]),
        Promise.all([
          getHotspots(scanId2),
          getDuplicates(scanId2, 1000),
          getScanSummary(scanId2)
        ])
      ]);

      setScan1({ scanId: scanId1, hotspots: hotspots1, duplicates: duplicates1, summary: summary1 });
      setScan2({ scanId: scanId2, hotspots: hotspots2, duplicates: duplicates2, summary: summary2 });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load scan data');
    } finally {
      setLoading(false);
    }
  };

  const computeHotspotDiff = (type: string): DiffResult => {
    if (!scan1 || !scan2) return { added: [], removed: [], unchanged: [] };
    
    const files1 = new Set(scan1.hotspots[type] || []);
    const files2 = new Set(scan2.hotspots[type] || []);
    
    const added = Array.from(files2).filter(file => !files1.has(file));
    const removed = Array.from(files1).filter(file => !files2.has(file));
    const unchanged = Array.from(files1).filter(file => files2.has(file));
    
    return { added, removed, unchanged };
  };

  const computeDuplicateDiff = (): DiffResult => {
    if (!scan1 || !scan2) return { added: [], removed: [], unchanged: [] };
    
    // Convert duplicate groups to sets of files for comparison
    const flatFiles1 = new Set(scan1.duplicates.flat());
    const flatFiles2 = new Set(scan2.duplicates.flat());
    
    const added = Array.from(flatFiles2).filter(file => !flatFiles1.has(file));
    const removed = Array.from(flatFiles1).filter(file => !flatFiles2.has(file));
    const unchanged = Array.from(flatFiles1).filter(file => flatFiles2.has(file));
    
    return { added, removed, unchanged };
  };

  const renderDiffSection = (title: string, diff: DiffResult) => (
    <div className="diff-section">
      <h4>{title}</h4>
      {diff.added.length > 0 && (
        <div className="diff-group added">
          <h5>Added ({diff.added.length})</h5>
          <ul>
            {diff.added.slice(0, 50).map(item => (
              <li key={item} className="added-item">+ {item}</li>
            ))}
            {diff.added.length > 50 && (
              <li className="more-items">... and {diff.added.length - 50} more</li>
            )}
          </ul>
        </div>
      )}
      {diff.removed.length > 0 && (
        <div className="diff-group removed">
          <h5>Removed ({diff.removed.length})</h5>
          <ul>
            {diff.removed.slice(0, 50).map(item => (
              <li key={item} className="removed-item">- {item}</li>
            ))}
            {diff.removed.length > 50 && (
              <li className="more-items">... and {diff.removed.length - 50} more</li>
            )}
          </ul>
        </div>
      )}
      {diff.unchanged.length > 0 && (
        <details className="diff-group unchanged">
          <summary>Unchanged ({diff.unchanged.length})</summary>
          <ul>
            {diff.unchanged.slice(0, 50).map(item => (
              <li key={item} className="unchanged-item">= {item}</li>
            ))}
            {diff.unchanged.length > 50 && (
              <li className="more-items">... and {diff.unchanged.length - 50} more</li>
            )}
          </ul>
        </details>
      )}
    </div>
  );

  if (loading) {
    return (
      <div className="modal-overlay">
        <div className="modal-content">
          <div className="loading">Loading scan comparison...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="modal-overlay">
        <div className="modal-content">
          <div className="error">Error: {error}</div>
          <button onClick={onClose}>Close</button>
        </div>
      </div>
    );
  }

  return (
    <div className="modal-overlay">
      <div className="modal-content scan-diff-viewer">
        <div className="modal-header">
          <h3>Scan Comparison: #{scanId1} vs #{scanId2}</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        <div className="diff-nav">
          <button 
            className={viewMode === 'summary' ? 'active' : ''}
            onClick={() => setViewMode('summary')}
          >
            Summary
          </button>
          <button 
            className={viewMode === 'hotspots' ? 'active' : ''}
            onClick={() => setViewMode('hotspots')}
          >
            Hotspots
          </button>
          <button 
            className={viewMode === 'duplicates' ? 'active' : ''}
            onClick={() => setViewMode('duplicates')}
          >
            Duplicates
          </button>
        </div>

        <div className="diff-content">
          {viewMode === 'summary' && scan1 && scan2 && (
            <div className="summary-diff">
              <div className="summary-comparison">
                <div className="scan-summary">
                  <h4>Scan #{scanId1}</h4>
                  <p>Files: {scan1.summary?.total_files || 'N/A'}</p>
                  <p>Code Lines: {scan1.summary?.total_code_lines || 'N/A'}</p>
                  <p>Size: {scan1.summary?.total_size_bytes ? Math.round(scan1.summary.total_size_bytes / 1024 / 1024) + 'MB' : 'N/A'}</p>
                </div>
                <div className="scan-summary">
                  <h4>Scan #{scanId2}</h4>
                  <p>Files: {scan2.summary?.total_files || 'N/A'}</p>
                  <p>Code Lines: {scan2.summary?.total_code_lines || 'N/A'}</p>
                  <p>Size: {scan2.summary?.total_size_bytes ? Math.round(scan2.summary.total_size_bytes / 1024 / 1024) + 'MB' : 'N/A'}</p>
                </div>
              </div>
              
              {scan1.summary && scan2.summary && (
                <div className="summary-changes">
                  <h4>Changes</h4>
                  <p>Files: {(scan2.summary.total_files - scan1.summary.total_files) > 0 ? '+' : ''}{scan2.summary.total_files - scan1.summary.total_files}</p>
                  <p>Code Lines: {(scan2.summary.total_code_lines - scan1.summary.total_code_lines) > 0 ? '+' : ''}{scan2.summary.total_code_lines - scan1.summary.total_code_lines}</p>
                  <p>Size: {Math.round((scan2.summary.total_size_bytes - scan1.summary.total_size_bytes) / 1024 / 1024) > 0 ? '+' : ''}{Math.round((scan2.summary.total_size_bytes - scan1.summary.total_size_bytes) / 1024 / 1024)}MB</p>
                </div>
              )}
            </div>
          )}

          {viewMode === 'hotspots' && scan1 && scan2 && (
            <div className="hotspots-diff">
              {Object.keys({...scan1.hotspots, ...scan2.hotspots}).sort().map(type => {
                const diff = computeHotspotDiff(type);
                if (diff.added.length === 0 && diff.removed.length === 0) return null;
                
                return renderDiffSection(`${type} Hotspots`, diff);
              })}
            </div>
          )}

          {viewMode === 'duplicates' && scan1 && scan2 && (
            <div className="duplicates-diff">
              {renderDiffSection('Duplicate Files', computeDuplicateDiff())}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
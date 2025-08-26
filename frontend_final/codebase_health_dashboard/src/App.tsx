import React, { useEffect, useState, useCallback } from "react";
import { 
  latestScanId, runScan, getHotspots, getDuplicates, getScanSummary, 
  getJobStatus, checkHealth, checkReadiness, formatError, isRetryableError 
} from "./api/ApiClient";
import HotspotTable from "./components/HotspotTable";
import DuplicateExplorer from "./components/DuplicateExplorer";
import Overview from "./components/Overview";
import DirectoryHealth from "./components/DirectoryHealth";
import Trends from "./components/Trends";
import ScanDiffViewer from "./components/ScanDiffViewer";
import { exportToJson, prepareScanDataForExport } from "./utils/export";
import "./styles/app.css";

export default function App() {
  const [scanId, setScanId] = useState<number | null>(null);
  const [hotspots, setHotspots] = useState<Record<string, string[]>>({});
  const [dupes, setDupes] = useState<string[][]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<string>('');
  const [healthStatus, setHealthStatus] = useState<any>(null);
  const [readinessStatus, setReadinessStatus] = useState<any>(null);
  const [activeTab, setActiveTab] = useState<"overview" | "hotspots" | "duplicates" | "directory" | "trends">("overview");
  const [retryCount, setRetryCount] = useState(0);
  const [showDiffViewer, setShowDiffViewer] = useState(false);
  const [diffScanId1, setDiffScanId1] = useState<number | null>(null);
  const [diffScanId2, setDiffScanId2] = useState<number | null>(null);
  const [compareWithScan, setCompareWithScan] = useState<string>('');

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Check health and readiness first
      const [health, readiness] = await Promise.all([
        checkHealth().catch(() => ({ status: 'error' })),
        checkReadiness().catch(() => ({ ready: false, missing: ['connection'] }))
      ]);
      setHealthStatus(health);
      setReadinessStatus(readiness);
      
      const id = await latestScanId();
      setScanId(id);
      
      if (id >= 0) {
        const [hotspotsData, dupesData, summaryData] = await Promise.all([
          getHotspots(id),
          getDuplicates(id, 50),
          getScanSummary(id)
        ]);
        setHotspots(hotspotsData);
        setDupes(dupesData);
        setSummary(summaryData);
      }
      setRetryCount(0); // Reset retry count on success
    } catch (err) {
      const formattedError = formatError(err);
      setError(formattedError);
      
      // Show retry option for retryable errors
      if (isRetryableError(err)) {
        setRetryCount(prev => prev + 1);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const handleRunScan = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      setJobId(null);
      setJobStatus('');
      
      const response = await runScan();
      
      if (response.job_id) {
        // Job queue mode - poll for status
        setJobId(response.job_id);
        setJobStatus('queued');
        
        const pollInterval = setInterval(async () => {
          try {
            const status = await getJobStatus(response.job_id);
            setJobStatus(status.status);
            
            if (status.status === 'done') {
              clearInterval(pollInterval);
              setJobId(null);
              setJobStatus('');
              await refresh();
              setLoading(false);
            } else if (status.status === 'failed') {
              clearInterval(pollInterval);
              setError(status.error || 'Scan failed');
              setJobId(null);
              setJobStatus('');
              setLoading(false);
            }
          } catch (err) {
            clearInterval(pollInterval);
            setError(err instanceof Error ? err.message : 'Failed to check job status');
            setJobId(null);
            setJobStatus('');
            setLoading(false);
          }
        }, 3000); // Poll every 3 seconds
        
      } else {
        // Direct scan mode
        await refresh();
        setLoading(false);
      }
    } catch (err) {
      const formattedError = formatError(err);
      setError(formattedError);
      setLoading(false);
      
      if (isRetryableError(err)) {
        setRetryCount(prev => prev + 1);
      }
    }
  }, [refresh]);

  const handleExportAll = useCallback(() => {
    if (!scanId || !summary) return;
    
    const exportData = prepareScanDataForExport({
      scanId,
      summary,
      hotspots,
      duplicates: dupes
    });
    
    exportToJson(exportData, { filename: `scan-${scanId}-complete` });
  }, [scanId, summary, hotspots, dupes]);

  const handleCompareScans = useCallback(() => {
    const compareId = parseInt(compareWithScan);
    if (!scanId || !compareId || compareId === scanId) {
      setError('Please enter a valid scan ID different from the current one');
      return;
    }
    
    setDiffScanId1(scanId);
    setDiffScanId2(compareId);
    setShowDiffViewer(true);
  }, [scanId, compareWithScan]);

  useEffect(() => {
    refresh();
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Codebase Health Dashboard</h1>
        <div className="header-info">
          {healthStatus && (
            <div className="health-status">
              <span className={`status-indicator ${healthStatus.status === 'ok' ? 'healthy' : 'error'}`}>
                ●
              </span>
              <span>{healthStatus.status === 'ok' ? 'Healthy' : 'Error'}</span>
            </div>
          )}
          {readinessStatus && (
            <div className="readiness-status">
              <span className={`status-indicator ${readinessStatus.ready ? 'ready' : 'not-ready'}`}>
                ●
              </span>
              <span>{readinessStatus.ready ? 'Ready' : 'Not Ready'}</span>
              {!readinessStatus.ready && readinessStatus.missing && (
                <span className="missing-deps">
                  (Missing: {readinessStatus.missing.join(', ')})
                </span>
              )}
            </div>
          )}
        </div>
        <div className="controls">
          <button 
            onClick={handleRunScan} 
            disabled={loading}
            className="btn btn-primary"
          >
            {loading ? (
              jobStatus ? `${jobStatus.charAt(0).toUpperCase() + jobStatus.slice(1)}...` : "Running..."
            ) : "Run Scan"}
          </button>
          <button 
            onClick={refresh} 
            disabled={loading}
            className="btn btn-secondary"
          >
            Refresh
          </button>
          {scanId && (
            <>
              <button 
                onClick={handleExportAll}
                className="btn btn-small"
                title="Export complete scan data"
              >
                📥 Export
              </button>
              <div className="compare-controls">
                <input
                  type="number"
                  placeholder="Scan ID"
                  value={compareWithScan}
                  onChange={(e) => setCompareWithScan(e.target.value)}
                  className="compare-input"
                />
                <button 
                  onClick={handleCompareScans}
                  className="btn btn-small"
                  disabled={!compareWithScan}
                  title="Compare with another scan"
                >
                  🔍 Compare
                </button>
              </div>
            </>
          )}
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <span className="error-icon">⚠</span>
          <span className="error-text">{error}</span>
          <div className="error-actions">
            {retryCount > 0 && retryCount < 3 && (
              <button className="btn btn-small btn-retry" onClick={refresh}>
                Retry ({3 - retryCount} left)
              </button>
            )}
            <button className="error-dismiss" onClick={() => setError(null)}>×</button>
          </div>
        </div>
      )}
      
      {jobId && jobStatus && (
        <div className="job-status-banner">
          <span className="job-icon">⏳</span>
          <span className="job-text">
            Scan job {jobId.substring(0, 8)}... is {jobStatus}
            {jobStatus === 'running' && <span className="loading-dots">...</span>}
          </span>
        </div>
      )}

      <div className="tabs">
        <button 
          className={`tab ${activeTab === "overview" ? "active" : ""}`}
          onClick={() => setActiveTab("overview")}
        >
          Overview
        </button>
        <button 
          className={`tab ${activeTab === "hotspots" ? "active" : ""}`}
          onClick={() => setActiveTab("hotspots")}
        >
          Hotspots
        </button>
        <button 
          className={`tab ${activeTab === "duplicates" ? "active" : ""}`}
          onClick={() => setActiveTab("duplicates")}
        >
          Duplicates
        </button>
        <button 
          className={`tab ${activeTab === "directory" ? "active" : ""}`}
          onClick={() => setActiveTab("directory")}
        >
          Directory Health
        </button>
        <button 
          className={`tab ${activeTab === "trends" ? "active" : ""}`}
          onClick={() => setActiveTab("trends")}
        >
          Trends
        </button>
      </div>

      <main className="content">
        {loading && <div className="loading">Loading...</div>}
        
        {!loading && scanId !== null && scanId >= 0 && (
          <>
            {activeTab === "overview" && <Overview summary={summary} scanId={scanId} />}
            {activeTab === "hotspots" && <HotspotTable hotspots={hotspots} />}
            {activeTab === "duplicates" && <DuplicateExplorer groups={dupes} />}
            {activeTab === "directory" && <DirectoryHealth scanId={scanId} />}
            {activeTab === "trends" && <Trends scanId={scanId} />}
          </>
        )}

        {!loading && (scanId === null || scanId < 0) && (
          <div className="no-data">
            No scan data available. Click "Run Scan" to generate a report.
          </div>
        )}
      </main>

      {showDiffViewer && diffScanId1 && diffScanId2 && (
        <ScanDiffViewer
          scanId1={diffScanId1}
          scanId2={diffScanId2}
          onClose={() => {
            setShowDiffViewer(false);
            setDiffScanId1(null);
            setDiffScanId2(null);
            setCompareWithScan('');
          }}
        />
      )}
    </div>
  );
}
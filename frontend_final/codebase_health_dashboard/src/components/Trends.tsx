import React, { useState, useEffect } from "react";

interface TrendsProps {
  scanId: number;
}

interface TrendData {
  date: string;
  totalFiles: number;
  totalSize: number;
  codeLines: number;
  duplicateGroups: number;
  healthScore: number;
}

export default function Trends({ scanId }: TrendsProps) {
  const [trendData, setTrendData] = useState<TrendData[]>([]);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState<"7d" | "30d" | "90d">("30d");
  const [selectedMetric, setSelectedMetric] = useState<keyof TrendData>("healthScore");

  useEffect(() => {
    loadTrendData();
  }, [scanId, timeRange]);

  const loadTrendData = async () => {
    setLoading(true);
    try {
      const mockTrends: TrendData[] = [
        {
          date: "2025-08-18",
          totalFiles: 2450,
          totalSize: 45600000,
          codeLines: 180000,
          duplicateGroups: 125,
          healthScore: 68
        },
        {
          date: "2025-08-19",
          totalFiles: 2455,
          totalSize: 45800000,
          codeLines: 181200,
          duplicateGroups: 128,
          healthScore: 67
        },
        {
          date: "2025-08-20",
          totalFiles: 2460,
          totalSize: 46000000,
          codeLines: 182000,
          duplicateGroups: 130,
          healthScore: 66
        },
        {
          date: "2025-08-21",
          totalFiles: 2440,
          totalSize: 44200000,
          codeLines: 179800,
          duplicateGroups: 115,
          healthScore: 72
        },
        {
          date: "2025-08-22",
          totalFiles: 2430,
          totalSize: 43800000,
          codeLines: 178500,
          duplicateGroups: 108,
          healthScore: 75
        },
        {
          date: "2025-08-23",
          totalFiles: 2425,
          totalSize: 43500000,
          codeLines: 177900,
          duplicateGroups: 102,
          healthScore: 78
        },
        {
          date: "2025-08-24",
          totalFiles: 2420,
          totalSize: 43200000,
          codeLines: 177200,
          duplicateGroups: 95,
          healthScore: 82
        },
        {
          date: "2025-08-25",
          totalFiles: 2415,
          totalSize: 42900000,
          codeLines: 176800,
          duplicateGroups: 88,
          healthScore: 85
        }
      ];
      
      setTrendData(mockTrends.slice(-getDaysForRange(timeRange)));
    } catch (error) {
      console.error("Failed to load trend data:", error);
    } finally {
      setLoading(false);
    }
  };

  const getDaysForRange = (range: string): number => {
    switch (range) {
      case "7d": return 7;
      case "30d": return 30;
      case "90d": return 90;
      default: return 30;
    }
  };

  const formatValue = (key: keyof TrendData, value: number): string => {
    switch (key) {
      case "totalSize":
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        if (value === 0) return '0 Bytes';
        const i = Math.floor(Math.log(value) / Math.log(1024));
        return Math.round(value / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
      case "totalFiles":
      case "codeLines":
      case "duplicateGroups":
        return value.toLocaleString();
      case "healthScore":
        return value.toString();
      default:
        return value.toString();
    }
  };

  const getMetricLabel = (key: keyof TrendData): string => {
    const labels = {
      totalFiles: "Total Files",
      totalSize: "Repository Size",
      codeLines: "Lines of Code",
      duplicateGroups: "Duplicate Groups",
      healthScore: "Health Score"
    };
    return labels[key as keyof typeof labels] || key;
  };

  const getTrendDirection = (data: TrendData[], key: keyof TrendData): "up" | "down" | "stable" => {
    if (data.length < 2) return "stable";
    const recent = data[data.length - 1][key];
    const previous = data[data.length - 2][key];
    
    if (recent > previous) return "up";
    if (recent < previous) return "down";
    return "stable";
  };

  const getTrendIcon = (direction: "up" | "down" | "stable", isGood: boolean): string => {
    if (direction === "stable") return "→";
    if (direction === "up") return isGood ? "📈" : "📉";
    return isGood ? "📈" : "📉";
  };

  const isPositiveTrend = (key: keyof TrendData, direction: "up" | "down" | "stable"): boolean => {
    if (direction === "stable") return true;
    if (key === "healthScore") return direction === "up";
    if (key === "duplicateGroups") return direction === "down";
    return direction === "up";
  };

  if (loading) {
    return (
      <div className="trends">
        <div className="loading">Loading trend analysis...</div>
      </div>
    );
  }

  const currentData = trendData[trendData.length - 1];
  const previousData = trendData.length > 1 ? trendData[trendData.length - 2] : currentData;

  return (
    <div className="trends">
      <div className="trends-header">
        <h2>Codebase Health Trends</h2>
        <div className="controls">
          <select 
            value={timeRange} 
            onChange={(e) => setTimeRange(e.target.value as "7d" | "30d" | "90d")}
            className="time-range-select"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
          </select>
        </div>
      </div>

      <div className="trend-summary">
        <div className="summary-cards">
          {(Object.keys(currentData) as Array<keyof TrendData>)
            .filter(key => key !== "date")
            .map((key) => {
              const direction = getTrendDirection(trendData, key);
              const isGood = isPositiveTrend(key, direction);
              const change = currentData[key] - previousData[key];
              const percentChange = previousData[key] !== 0 ? 
                Math.abs((change / previousData[key]) * 100) : 0;
              
              return (
                <div key={key} className="trend-card">
                  <div className="trend-icon">
                    {getTrendIcon(direction, isGood)}
                  </div>
                  <div className="trend-content">
                    <div className="trend-value">
                      {formatValue(key, currentData[key])}
                    </div>
                    <div className="trend-label">
                      {getMetricLabel(key)}
                    </div>
                    <div className={`trend-change ${isGood ? 'positive' : 'negative'}`}>
                      {direction !== "stable" && (
                        <span>
                          {change > 0 ? '+' : ''}{formatValue(key, Math.abs(change))} 
                          ({percentChange.toFixed(1)}%)
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      <div className="chart-section">
        <div className="chart-controls">
          <h3>Trend Visualization</h3>
          <select 
            value={selectedMetric} 
            onChange={(e) => setSelectedMetric(e.target.value as keyof TrendData)}
            className="metric-select"
          >
            <option value="healthScore">Health Score</option>
            <option value="totalFiles">Total Files</option>
            <option value="totalSize">Repository Size</option>
            <option value="codeLines">Lines of Code</option>
            <option value="duplicateGroups">Duplicate Groups</option>
          </select>
        </div>

        <div className="simple-chart">
          <div className="chart-title">{getMetricLabel(selectedMetric)} Over Time</div>
          <div className="chart-data">
            {trendData.map((point, index) => {
              const value = point[selectedMetric];
              const maxValue = Math.max(...trendData.map(d => d[selectedMetric]));
              const height = (value / maxValue) * 100;
              
              return (
                <div key={point.date} className="chart-bar">
                  <div 
                    className="bar"
                    style={{ height: `${height}%` }}
                    title={`${point.date}: ${formatValue(selectedMetric, value)}`}
                  />
                  <div className="bar-label">
                    {new Date(point.date).getMonth() + 1}/{new Date(point.date).getDate()}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="insights">
        <h3>Trend Insights</h3>
        <div className="insight-list">
          {getTrendDirection(trendData, "healthScore") === "up" && (
            <div className="insight positive">
              ✓ Health score is improving over time
            </div>
          )}
          {getTrendDirection(trendData, "duplicateGroups") === "down" && (
            <div className="insight positive">
              ✓ Number of duplicate files is decreasing
            </div>
          )}
          {getTrendDirection(trendData, "totalSize") === "down" && (
            <div className="insight positive">
              ✓ Repository size is being reduced
            </div>
          )}
          {getTrendDirection(trendData, "healthScore") === "down" && (
            <div className="insight negative">
              ⚠ Health score is declining - review recent changes
            </div>
          )}
          {getTrendDirection(trendData, "duplicateGroups") === "up" && (
            <div className="insight negative">
              ⚠ Duplicate files are increasing - run cleanup scripts
            </div>
          )}
          {getTrendDirection(trendData, "totalFiles") === "up" && 
           getTrendDirection(trendData, "codeLines") === "up" && (
            <div className="insight neutral">
              → Codebase is growing - monitor complexity
            </div>
          )}
        </div>
      </div>

      <div className="trend-actions">
        <h3>Recommended Actions</h3>
        <div className="action-list">
          <div className="action-item">
            <span className="action-icon">📊</span>
            <div className="action-content">
              <div className="action-title">Schedule Regular Monitoring</div>
              <div className="action-desc">Set up automated daily scans to track progress</div>
            </div>
          </div>
          <div className="action-item">
            <span className="action-icon">🎯</span>
            <div className="action-content">
              <div className="action-title">Set Health Goals</div>
              <div className="action-desc">Define target health score and monitor progress</div>
            </div>
          </div>
          <div className="action-item">
            <span className="action-icon">🧹</span>
            <div className="action-content">
              <div className="action-title">Regular Cleanup</div>
              <div className="action-desc">Run cleanup scripts weekly to maintain health</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
export interface ExportOptions {
  filename?: string;
  includeTimestamp?: boolean;
}

export function exportToJson<T>(data: T, options: ExportOptions = {}): void {
  const { filename = 'data', includeTimestamp = true } = options;
  
  const timestamp = includeTimestamp 
    ? new Date().toISOString().replace(/[:.]/g, '-') 
    : '';
  
  const finalFilename = `${filename}${timestamp ? '-' + timestamp : ''}.json`;
  
  const jsonData = JSON.stringify(data, null, 2);
  downloadFile(jsonData, finalFilename, 'application/json');
}

export function exportToCsv<T extends Record<string, any>>(
  data: T[], 
  options: ExportOptions = {}
): void {
  if (data.length === 0) return;
  
  const { filename = 'data', includeTimestamp = true } = options;
  
  const timestamp = includeTimestamp 
    ? new Date().toISOString().replace(/[:.]/g, '-') 
    : '';
  
  const finalFilename = `${filename}${timestamp ? '-' + timestamp : ''}.csv`;
  
  // Get all unique keys from all objects
  const allKeys = new Set<string>();
  data.forEach(item => {
    Object.keys(item).forEach(key => allKeys.add(key));
  });
  
  const headers = Array.from(allKeys);
  const csvRows = [
    headers.join(','),
    ...data.map(item => 
      headers.map(key => {
        const value = item[key];
        // Handle nested objects, arrays, and special characters
        if (value == null) return '';
        if (typeof value === 'object') return `"${JSON.stringify(value).replace(/"/g, '""')}"`;
        const stringValue = String(value);
        // Escape quotes and wrap in quotes if contains comma, quote, or newline
        if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
          return `"${stringValue.replace(/"/g, '""')}"`;
        }
        return stringValue;
      }).join(',')
    )
  ];
  
  const csvData = csvRows.join('\n');
  downloadFile(csvData, finalFilename, 'text/csv');
}

export function exportHotspotsToCsv(
  hotspots: Record<string, string[]>, 
  options: ExportOptions = {}
): void {
  const flatData = Object.entries(hotspots).flatMap(([type, files]) =>
    files.map(file => ({ type, file }))
  );
  
  exportToCsv(flatData, { ...options, filename: options.filename || 'hotspots' });
}

export function exportDuplicatesToCsv(
  duplicates: string[][], 
  options: ExportOptions = {}
): void {
  const flatData = duplicates.flatMap((group, groupIndex) =>
    group.map(file => ({ group_id: groupIndex + 1, file }))
  );
  
  exportToCsv(flatData, { ...options, filename: options.filename || 'duplicates' });
}

function downloadFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.style.display = 'none';
  
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  // Clean up the object URL
  setTimeout(() => URL.revokeObjectURL(url), 100);
}

// Utility to format scan data for export
export function prepareScanDataForExport(data: {
  scanId: number;
  summary?: any;
  hotspots: Record<string, string[]>;
  duplicates: string[][];
}) {
  return {
    export_info: {
      scan_id: data.scanId,
      exported_at: new Date().toISOString(),
      dashboard_version: '1.0.0'
    },
    scan_summary: data.summary,
    hotspots: data.hotspots,
    duplicate_groups: data.duplicates.map((group, index) => ({
      group_id: index + 1,
      files: group
    })),
    statistics: {
      total_hotspot_types: Object.keys(data.hotspots).length,
      total_hotspot_files: Object.values(data.hotspots).reduce((sum, files) => sum + files.length, 0),
      total_duplicate_groups: data.duplicates.length,
      total_duplicate_files: data.duplicates.reduce((sum, group) => sum + group.length, 0)
    }
  };
}
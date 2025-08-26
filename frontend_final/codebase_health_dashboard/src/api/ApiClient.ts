import { config } from '../config';

export const API_BASE = config.api.baseUrl;
export const API_KEY = config.auth.apiKey;

export interface JobStatus {
  job_id: string;
  status: 'queued' | 'running' | 'done' | 'failed';
  scan_id?: number;
  error?: string;
  report?: {
    json_report?: string;
    summary_report?: string;
  };
}

export interface ScanReport {
  scan_id: number;
  status: string;
  message: string;
  summary?: ScanSummary;
  json_report?: string;
  markdown_report?: string;
  duration_seconds?: number;
}

export interface LatestScan {
  scan_id: number;
  hotspots?: Record<string, string[]>;
  duplicates?: string[][];
}

export interface ScanSummary {
  total_files: number;
  total_size_bytes: number;
  total_code_lines: number;
  generated_at_epoch: number;
  root: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public statusText?: string,
    public originalError?: Error
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function apiRequest(path: string, options: RequestInit = {}) {
  const headers = new Headers(options.headers || {});
  if (API_KEY) {
    headers.set('X-API-Key', API_KEY);
  }
  headers.set('Content-Type', 'application/json');

  let lastError: Error;
  
  for (let attempt = 1; attempt <= config.api.retryAttempts; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), config.api.timeout);
      
      const response = await fetch(`${API_BASE}${path}`, {
        ...options,
        headers,
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        const error = new ApiError(
          `API request failed: ${errorText}`,
          response.status,
          response.statusText
        );
        
        // Don't retry client errors (4xx)
        if (response.status >= 400 && response.status < 500) {
          throw error;
        }
        
        lastError = error;
        if (attempt < config.api.retryAttempts) {
          await sleep(Math.pow(2, attempt - 1) * 1000); // Exponential backoff
          continue;
        }
        throw error;
      }

      return response.json();
    } catch (error) {
      lastError = error as Error;
      
      // Handle timeout or network errors
      if (error instanceof TypeError || (error as any).name === 'AbortError') {
        const apiError = new ApiError(
          `Network error: ${error.message}`,
          undefined,
          undefined,
          error as Error
        );
        
        if (attempt < config.api.retryAttempts) {
          await sleep(Math.pow(2, attempt - 1) * 1000);
          continue;
        }
        throw apiError;
      }
      
      // Re-throw other errors immediately
      throw error;
    }
  }
  
  throw lastError;
}

export async function runScan(): Promise<{ job_id: string; status: string }> {
  return apiRequest('/scan/run', { method: 'POST' });
}

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  return apiRequest(`/scan/job/${jobId}`);
}

export async function latestScanId(): Promise<number> {
  const data = await apiRequest('/scan/latest');
  return data.scan_id ?? -1;
}

export async function getLatestScan(): Promise<LatestScan> {
  return apiRequest('/scan/latest');
}

export async function getHotspots(scanId: number): Promise<Record<string, string[]>> {
  const response = await apiRequest(`/scan/${scanId}/hotspots`);
  return response.hotspots || {};
}

export async function getDuplicates(scanId: number, limit = 50): Promise<string[][]> {
  const response = await apiRequest(`/scan/${scanId}/duplicates?limit=${limit}`);
  return response.duplicate_groups || [];
}

export async function getScanSummary(scanId: number): Promise<ScanSummary | null> {
  try {
    return await apiRequest(`/scan/${scanId}`);
  } catch {
    return null;
  }
}

export async function checkHealth(): Promise<{ status: string }> {
  return apiRequest('/health');
}

export async function checkReadiness(): Promise<{ ready: boolean; missing: string[] }> {
  return apiRequest('/ready');
}

// Export utility functions for UI use
export function formatError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return 'Authentication failed. Please check your API key.';
    if (error.status === 429) return 'Too many requests. Please wait before trying again.';
    if (error.status === 503) return 'Service temporarily unavailable. Please try again later.';
    if (error.status && error.status >= 500) return 'Server error. Please try again later.';
    return error.message;
  }
  if (error instanceof Error) {
    if (error.message.includes('NetworkError') || error.message.includes('fetch')) {
      return 'Network error. Please check your connection and try again.';
    }
    return error.message;
  }
  return 'An unexpected error occurred.';
}

export function isRetryableError(error: unknown): boolean {
  if (error instanceof ApiError) {
    return error.status ? error.status >= 500 : true;
  }
  return error instanceof TypeError || (error as any).name === 'AbortError';
}
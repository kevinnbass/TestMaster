export const config = {
  api: {
    baseUrl: 'http://127.0.0.1:8090',
    timeout: 30000,
    retryAttempts: 3
  },
  auth: {
    apiKey: 'gamma-hardened-api-key-2025-08-25-secure'
  },
  dashboard: {
    refreshInterval: 60000, // 1 minute
    maxHotspotsPerCategory: 100,
    maxDuplicateGroups: 50,
    maxFilesPerGroup: 20
  },
  theme: {
    primaryColor: '#007bff',
    successColor: '#28a745',
    warningColor: '#ffc107',
    dangerColor: '#dc3545',
    infoColor: '#17a2b8'
  }
};
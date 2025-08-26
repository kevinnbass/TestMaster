# Codebase Health Dashboard

A React-based dashboard for monitoring and analyzing codebase health metrics, hotspots, duplicates, and trends.

## Features

- **📊 Overview**: Comprehensive metrics and health scoring
- **🔥 Hotspots**: Identify code areas needing attention
- **📋 Duplicates**: Find and manage duplicate files
- **📁 Directory Health**: Analyze directory-level health
- **📈 Trends**: Track codebase health over time

## Quick Start

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- Backend codebase monitor service running on port 8088

### Installation & Setup

1. **Install dependencies**:
   ```powershell
   npm install
   ```

2. **Start the dashboard**:
   ```powershell
   # Option 1: Use the startup script
   .\start_dashboard.ps1
   
   # Option 2: Manual start
   npm run dev
   ```

3. **Access the dashboard**:
   Open http://localhost:3000 in your browser

### Backend Service

The dashboard requires the backend codebase monitor service to be running:

```powershell
# From the project root
cd scripts
.\scan_server.ps1

# Or run directly:
python -m backend.codebase_monitor.service
```

The backend should be accessible at http://127.0.0.1:8088

## API Endpoints Used

The dashboard connects to these backend endpoints:

- `GET /health` - Service health check
- `POST /scan/run` - Trigger a new codebase scan
- `GET /scan/latest` - Get latest scan ID
- `GET /scan/{id}/hotspots` - Get hotspots for a scan
- `GET /scan/{id}/duplicates` - Get duplicate files
- `GET /scan/{id}/summary` - Get scan summary (optional)

## Project Structure

```
src/
├── api/
│   └── ApiClient.ts          # API client for backend communication
├── components/
│   ├── Overview.tsx          # Main metrics overview
│   ├── HotspotTable.tsx      # Code hotspots analysis
│   ├── DuplicateExplorer.tsx # Duplicate file management
│   ├── DirectoryHealth.tsx   # Directory-level health analysis
│   └── Trends.tsx            # Health trends over time
├── styles/
│   └── app.css               # Main stylesheet
├── config.ts                 # Configuration constants
├── App.tsx                   # Main application component
└── index.tsx                 # Application entry point
```

## Configuration

Edit `src/config.ts` to modify:

- API endpoint URLs
- Refresh intervals
- Display limits
- Theme colors

## Development

```powershell
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Features Overview

### Overview Tab
- Total files, size, and lines of code
- Health score calculation
- Quick action buttons
- System recommendations

### Hotspots Tab
- Categorized problem areas
- Searchable file lists
- Color-coded severity levels
- Expandable category details

### Duplicates Tab
- Grouped duplicate files
- File extension indicators
- Cleanup recommendations
- Primary file designation

### Directory Health Tab
- Directory-level metrics
- Health scoring per directory
- Recommendations for improvement
- Action plan generation

### Trends Tab
- Historical health metrics
- Trend visualization
- Insights and recommendations
- Configurable time ranges

## Troubleshooting

### Backend Connection Issues
- Ensure backend service is running on port 8088
- Check firewall settings
- Verify API endpoints are accessible

### Performance Issues
- Limit the number of displayed items in config
- Use browser dev tools to monitor network requests
- Check backend response times

### Build Issues
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`
- Check Node.js version compatibility
- Verify all dependencies are properly installed

## Contributing

1. Follow the existing code style and patterns
2. Add proper TypeScript types for new components
3. Update this README for any new features
4. Test with the backend service integration

## License

Part of the TestMaster codebase analysis system.
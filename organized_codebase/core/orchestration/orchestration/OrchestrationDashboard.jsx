/**
 * Orchestration Dashboard Component
 * =================================
 * 
 * React component for managing and monitoring TestMaster's multi-agent orchestration.
 * Provides Phase 1 UI capabilities for agent coordination and observability.
 * 
 * Author: TestMaster Team
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Button,
  Card,
  CardContent,
  CardActions,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  PlayArrow,
  Stop,
  Refresh,
  Add,
  Visibility,
  Settings,
  Timeline,
  Dashboard,
  Memory,
  Speed,
  Security,
  BugReport,
  Analytics
} from '@mui/icons-material';

// Custom hook for API calls
const useOrchestrationAPI = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const apiCall = useCallback(async (endpoint, options = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/orchestration${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { apiCall, loading, error };
};

// Agent Status Chip Component
const AgentStatusChip = ({ status }) => {
  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'idle': return 'default';
      case 'running': return 'primary';
      case 'completed': return 'success';
      case 'failed': return 'error';
      case 'paused': return 'warning';
      default: return 'default';
    }
  };

  return (
    <Chip 
      label={status.toUpperCase()} 
      color={getStatusColor(status)}
      size="small"
    />
  );
};

// System Health Indicator
const SystemHealthIndicator = ({ health, metrics }) => {
  const getHealthColor = (health) => {
    switch (health) {
      case 'excellent': return '#4caf50';
      case 'good': return '#8bc34a';
      case 'degraded': return '#ff9800';
      case 'critical': return '#f44336';
      default: return '#9e9e9e';
    }
  };

  return (
    <Box display="flex" alignItems="center" gap={1}>
      <Box
        width={12}
        height={12}
        borderRadius="50%"
        bgcolor={getHealthColor(health)}
      />
      <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
        {health}
      </Typography>
      {metrics && (
        <Typography variant="caption" color="text.secondary">
          ({metrics.active_agents} agents)
        </Typography>
      )}
    </Box>
  );
};

// Main Orchestration Dashboard Component
const OrchestrationDashboard = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [agents, setAgents] = useState({});
  const [sessions, setSessions] = useState({});
  const [tools, setTools] = useState({});
  const [systemStatus, setSystemStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [createAgentDialog, setCreateAgentDialog] = useState(false);
  const [createSessionDialog, setCreateSessionDialog] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(null);

  const { apiCall, loading, error } = useOrchestrationAPI();

  // Auto-refresh data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statusData, agentsData, metricsData, toolsData] = await Promise.all([
          apiCall('/status'),
          apiCall('/agents'),
          apiCall('/metrics'),
          apiCall('/tools')
        ]);

        setSystemStatus(statusData.data);
        setAgents(agentsData.data.agents || {});
        setMetrics(metricsData.data);
        setTools(toolsData.data.tools || {});
      } catch (err) {
        console.error('Failed to fetch data:', err);
      }
    };

    fetchData();
    
    // Set up auto-refresh
    const interval = setInterval(fetchData, 5000); // Refresh every 5 seconds
    setRefreshInterval(interval);

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [apiCall]);

  // Create Agent Dialog Component
  const CreateAgentDialog = () => {
    const [agentData, setAgentData] = useState({
      name: '',
      role: '',
      capabilities: [],
      dependencies: []
    });

    const handleCreate = async () => {
      try {
        await apiCall('/agents', {
          method: 'POST',
          body: JSON.stringify(agentData)
        });
        setCreateAgentDialog(false);
        setAgentData({ name: '', role: '', capabilities: [], dependencies: [] });
        // Refresh agents list
        const agentsData = await apiCall('/agents');
        setAgents(agentsData.data.agents || {});
      } catch (err) {
        console.error('Failed to create agent:', err);
      }
    };

    return (
      <Dialog open={createAgentDialog} onClose={() => setCreateAgentDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Agent</DialogTitle>
        <DialogContent>
          <Box display="flex" flexDirection="column" gap={2} mt={1}>
            <TextField
              label="Agent Name"
              value={agentData.name}
              onChange={(e) => setAgentData({ ...agentData, name: e.target.value })}
              fullWidth
            />
            <TextField
              label="Role"
              value={agentData.role}
              onChange={(e) => setAgentData({ ...agentData, role: e.target.value })}
              fullWidth
            />
            <TextField
              label="Capabilities (comma-separated)"
              value={agentData.capabilities.join(', ')}
              onChange={(e) => setAgentData({ 
                ...agentData, 
                capabilities: e.target.value.split(',').map(s => s.trim()).filter(s => s)
              })}
              multiline
              rows={3}
              fullWidth
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateAgentDialog(false)}>Cancel</Button>
          <Button onClick={handleCreate} variant="contained" disabled={!agentData.name || !agentData.role}>
            Create Agent
          </Button>
        </DialogActions>
      </Dialog>
    );
  };

  // Agents Overview Tab
  const AgentsTab = () => (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6">Agent Management</Typography>
        <Box display="flex" gap={1}>
          <Button
            variant="outlined"
            startIcon={<Add />}
            onClick={() => setCreateAgentDialog(true)}
          >
            Create Agent
          </Button>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={async () => {
              const agentsData = await apiCall('/agents');
              setAgents(agentsData.data.agents || {});
            }}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      <Grid container spacing={3}>
        {Object.entries(agents).map(([agentId, agent]) => (
          <Grid item xs={12} md={6} lg={4} key={agentId}>
            <Card>
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="start" mb={2}>
                  <Typography variant="h6" component="h3">
                    {agent.name}
                  </Typography>
                  <AgentStatusChip status={agent.status} />
                </Box>
                
                <Typography color="text.secondary" gutterBottom>
                  Role: {agent.role}
                </Typography>
                
                <Typography variant="body2" mb={2}>
                  Capabilities: {agent.capabilities?.join(', ') || 'None'}
                </Typography>
                
                {agent.current_task && (
                  <Typography variant="body2" color="primary">
                    Current Task: {agent.current_task}
                  </Typography>
                )}
                
                <Box mt={2}>
                  <Typography variant="caption" display="block">
                    Tasks Completed: {agent.performance_metrics?.tasks_completed || 0}
                  </Typography>
                  <Typography variant="caption" display="block">
                    Success Rate: {(agent.performance_metrics?.success_rate || 0).toFixed(1)}%
                  </Typography>
                </Box>
              </CardContent>
              
              <CardActions>
                <Button 
                  size="small" 
                  startIcon={<Visibility />}
                  onClick={() => setSelectedAgent({ agentId, ...agent })}
                >
                  View Details
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {Object.keys(agents).length === 0 && (
        <Box textAlign="center" py={4}>
          <Typography color="text.secondary">
            No agents registered. Create your first agent to get started.
          </Typography>
        </Box>
      )}
    </Box>
  );

  // System Status Tab
  const StatusTab = () => (
    <Box>
      <Typography variant="h6" mb={3}>System Status</Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" mb={2}>Orchestration Health</Typography>
            {systemStatus?.orchestration && (
              <Box>
                <SystemHealthIndicator 
                  health={systemStatus.orchestration.system_health}
                  metrics={systemStatus.orchestration.metrics}
                />
                <Box mt={2}>
                  <Typography variant="body2">
                    Active Sessions: {systemStatus.orchestration.active_sessions}
                  </Typography>
                  <Typography variant="body2">
                    Total Sessions: {systemStatus.orchestration.metrics?.total_sessions || 0}
                  </Typography>
                  <Typography variant="body2">
                    Success Rate: {((systemStatus.orchestration.metrics?.successful_sessions || 0) / Math.max(1, systemStatus.orchestration.metrics?.total_sessions || 1) * 100).toFixed(1)}%
                  </Typography>
                </Box>
              </Box>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" mb={2}>Observability Health</Typography>
            {systemStatus?.observability && (
              <Box>
                <SystemHealthIndicator 
                  health={systemStatus.observability.system_health}
                />
                <Box mt={2}>
                  <Typography variant="body2">
                    Active Sessions: {systemStatus.observability.active_sessions}
                  </Typography>
                  <Typography variant="body2">
                    Total Cost: ${(systemStatus.observability.analytics?.total_cost || 0).toFixed(4)}
                  </Typography>
                  <Typography variant="body2">
                    Total Tokens: {(systemStatus.observability.analytics?.total_tokens || 0).toLocaleString()}
                  </Typography>
                </Box>
              </Box>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" mb={2}>Performance Metrics</Typography>
            {metrics && (
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Metric</TableCell>
                      <TableCell>Orchestration</TableCell>
                      <TableCell>Observability</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>Total Sessions</TableCell>
                      <TableCell>{metrics.orchestration_metrics?.total_sessions || 0}</TableCell>
                      <TableCell>{metrics.observability_metrics?.total_sessions || 0}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Successful Sessions</TableCell>
                      <TableCell>{metrics.orchestration_metrics?.successful_sessions || 0}</TableCell>
                      <TableCell>{metrics.observability_metrics?.completed_sessions || 0}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Average Duration</TableCell>
                      <TableCell>{(metrics.orchestration_metrics?.average_session_time || 0).toFixed(2)}s</TableCell>
                      <TableCell>{(metrics.observability_metrics?.average_duration || 0).toFixed(2)}s</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );

  // Tools Tab
  const ToolsTab = () => (
    <Box>
      <Typography variant="h6" mb={3}>Available Tools</Typography>
      
      <Grid container spacing={3}>
        {Object.entries(tools).map(([toolName, tool]) => (
          <Grid item xs={12} md={6} lg={4} key={toolName}>
            <Card>
              <CardContent>
                <Typography variant="h6" mb={1}>
                  {tool.metadata?.name || toolName}
                </Typography>
                <Typography color="text.secondary" mb={2}>
                  {tool.metadata?.description}
                </Typography>
                
                <Chip 
                  label={tool.metadata?.category} 
                  color="primary" 
                  size="small"
                  sx={{ mb: 2 }}
                />
                
                <Box>
                  <Typography variant="body2">
                    Version: {tool.metadata?.version}
                  </Typography>
                  <Typography variant="body2">
                    Executions: {tool.performance?.total_executions || 0}
                  </Typography>
                  <Typography variant="body2">
                    Success Rate: {(tool.performance?.success_rate || 0).toFixed(1)}%
                  </Typography>
                </Box>
              </CardContent>
              
              <CardActions>
                <Button size="small" startIcon={<PlayArrow />}>
                  Execute
                </Button>
                <Button size="small" startIcon={<Settings />}>
                  Configure
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Typography variant="h4" component="h1">
          Orchestration Dashboard
        </Typography>
        <Box display="flex" alignItems="center" gap={2}>
          {systemStatus && (
            <SystemHealthIndicator 
              health={systemStatus.orchestration?.system_health || 'unknown'}
            />
          )}
          <Chip 
            icon={<Dashboard />} 
            label="Phase 1 Active" 
            color="success" 
            variant="outlined"
          />
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Paper sx={{ width: '100%', mb: 2 }}>
        <Tabs
          value={activeTab}
          onChange={(e, newValue) => setActiveTab(newValue)}
          indicatorColor="primary"
          textColor="primary"
        >
          <Tab icon={<Memory />} label="Agents" />
          <Tab icon={<Dashboard />} label="Status" />
          <Tab icon={<BugReport />} label="Tools" />
          <Tab icon={<Analytics />} label="Sessions" />
        </Tabs>
      </Paper>

      <Box role="tabpanel" hidden={activeTab !== 0}>
        {activeTab === 0 && <AgentsTab />}
      </Box>
      
      <Box role="tabpanel" hidden={activeTab !== 1}>
        {activeTab === 1 && <StatusTab />}
      </Box>
      
      <Box role="tabpanel" hidden={activeTab !== 2}>
        {activeTab === 2 && <ToolsTab />}
      </Box>
      
      <Box role="tabpanel" hidden={activeTab !== 3}>
        {activeTab === 3 && (
          <Typography>Session management coming in next update...</Typography>
        )}
      </Box>

      <CreateAgentDialog />

      {loading && (
        <Box display="flex" justifyContent="center" mt={4}>
          <CircularProgress />
        </Box>
      )}
    </Container>
  );
};

export default OrchestrationDashboard;
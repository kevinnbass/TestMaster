/**
 * Phase 2 Multi-Agent Dashboard
 * =============================
 * 
 * React component for managing Phase 2 multi-agent testing framework
 * and enhanced monitoring capabilities.
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
  Tooltip,
  LinearProgress,
  Avatar,
  Accordion,
  AccordionSummary,
  AccordionDetails
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
  Analytics,
  Groups,
  Monitor,
  Chat,
  ExpandMore,
  CheckCircle,
  Error,
  Warning,
  Info
} from '@mui/icons-material';

// Custom hook for Phase 2 API calls
const usePhase2API = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const apiCall = useCallback(async (endpoint, options = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/phase2${endpoint}`, {
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

// Team Status Component
const TeamStatusCard = ({ team, teamId, onExecuteWorkflow, onStopTeam }) => {
  const getStatusColor = (active) => active ? 'success' : 'default';
  
  const roleIcons = {
    'architect': <Dashboard />,
    'engineer': <BugReport />,
    'qa_agent': <Security />,
    'executor': <PlayArrow />,
    'coordinator': <Groups />
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="start" mb={2}>
          <Typography variant="h6" component="h3">
            Team {teamId.substring(0, 8)}
          </Typography>
          <Chip 
            label={team.active ? 'Active' : 'Inactive'} 
            color={getStatusColor(team.active)}
            size="small"
          />
        </Box>
        
        <Typography color="text.secondary" gutterBottom>
          Mode: {team.configuration.supervisor_mode}
        </Typography>
        
        <Typography variant="body2" mb={2}>
          Workflow: {team.configuration.workflow_type}
        </Typography>
        
        <Box mb={2}>
          <Typography variant="subtitle2" gutterBottom>
            Team Roles:
          </Typography>
          <Box display="flex" flexWrap="wrap" gap={0.5}>
            {team.configuration.roles.map((role, index) => (
              <Chip
                key={index}
                icon={roleIcons[role] || <Memory />}
                label={role.replace('_', ' ')}
                size="small"
                variant="outlined"
              />
            ))}
          </Box>
        </Box>
        
        <Box>
          <Typography variant="caption" display="block">
            Workflows Completed: {team.team_metrics.workflows_completed}
          </Typography>
          <Typography variant="caption" display="block">
            Success Rate: {team.team_metrics.success_rate.toFixed(1)}%
          </Typography>
          <Typography variant="caption" display="block">
            Quality Score: {team.team_metrics.quality_score.toFixed(1)}
          </Typography>
        </Box>
      </CardContent>
      
      <CardActions>
        <Button 
          size="small" 
          startIcon={<PlayArrow />}
          onClick={() => onExecuteWorkflow(teamId)}
          disabled={!team.active}
        >
          Execute
        </Button>
        <Button 
          size="small" 
          startIcon={<Stop />}
          onClick={() => onStopTeam(teamId)}
          disabled={!team.active}
          color="error"
        >
          Stop
        </Button>
      </CardActions>
    </Card>
  );
};

// Monitor Status Component
const MonitorStatusCard = ({ monitor, monitorId, onChat, onStopMonitor }) => {
  const getAlertIcon = (level) => {
    switch (level) {
      case 'critical': return <Error color="error" />;
      case 'error': return <Error color="error" />;
      case 'warning': return <Warning color="warning" />;
      default: return <Info color="info" />;
    }
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="start" mb={2}>
          <Typography variant="h6" component="h3">
            Monitor {monitorId.substring(0, 8)}
          </Typography>
          <Chip 
            label={monitor.mode.toUpperCase()} 
            color="primary"
            size="small"
          />
        </Box>
        
        <Typography color="text.secondary" gutterBottom>
          Uptime: {(monitor.uptime / 3600).toFixed(1)}h
        </Typography>
        
        <Box mb={2}>
          <Typography variant="subtitle2" gutterBottom>
            Monitoring Agents:
          </Typography>
          <Box display="flex" flexWrap="wrap" gap={0.5}>
            {Object.entries(monitor.agent_statuses).map(([agentName, status]) => (
              <Chip
                key={agentName}
                label={agentName.replace('_monitor', '')}
                size="small"
                color={status.active ? 'success' : 'default'}
                variant="outlined"
              />
            ))}
          </Box>
        </Box>
        
        <Box mb={2}>
          <Typography variant="subtitle2" gutterBottom>
            Recent Events:
          </Typography>
          <List dense>
            {monitor.recent_events.slice(0, 3).map((event, index) => (
              <ListItem key={index} sx={{ py: 0 }}>
                <ListItemIcon sx={{ minWidth: 30 }}>
                  {getAlertIcon(event.level)}
                </ListItemIcon>
                <ListItemText 
                  primary={event.message}
                  secondary={new Date(event.timestamp).toLocaleTimeString()}
                  primaryTypographyProps={{ variant: 'caption' }}
                  secondaryTypographyProps={{ variant: 'caption' }}
                />
              </ListItem>
            ))}
          </List>
        </Box>
        
        <Box>
          <Typography variant="caption" display="block">
            Events Processed: {monitor.metrics.events_processed}
          </Typography>
          <Typography variant="caption" display="block">
            Conversations: {monitor.metrics.conversations_handled}
          </Typography>
        </Box>
      </CardContent>
      
      <CardActions>
        <Button 
          size="small" 
          startIcon={<Chat />}
          onClick={() => onChat(monitorId)}
        >
          Chat
        </Button>
        <Button 
          size="small" 
          startIcon={<Stop />}
          onClick={() => onStopMonitor(monitorId)}
          color="error"
        >
          Stop
        </Button>
      </CardActions>
    </Card>
  );
};

// Chat Dialog Component
const ChatDialog = ({ open, onClose, monitorId, apiCall }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [chatLoading, setChatLoading] = useState(false);

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;
    
    const userMessage = { sender: 'user', message: inputMessage, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    
    setChatLoading(true);
    try {
      const response = await apiCall(`/monitoring/${monitorId}/chat`, {
        method: 'POST',
        body: JSON.stringify({ message: inputMessage })
      });
      
      const botMessage = { 
        sender: 'assistant', 
        message: response.data.response, 
        timestamp: new Date(response.data.timestamp) 
      };
      setMessages(prev => [...prev, botMessage]);
      setInputMessage('');
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setChatLoading(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Chat with Monitor {monitorId.substring(0, 8)}</DialogTitle>
      <DialogContent>
        <Box height={400} overflow="auto" mb={2}>
          {messages.map((msg, index) => (
            <Box key={index} mb={1} display="flex" justifyContent={msg.sender === 'user' ? 'flex-end' : 'flex-start'}>
              <Paper 
                sx={{ 
                  p: 1, 
                  maxWidth: '70%',
                  backgroundColor: msg.sender === 'user' ? 'primary.light' : 'grey.100'
                }}
              >
                <Typography variant="body2">{msg.message}</Typography>
                <Typography variant="caption" color="text.secondary">
                  {msg.timestamp.toLocaleTimeString()}
                </Typography>
              </Paper>
            </Box>
          ))}
          {chatLoading && (
            <Box display="flex" justifyContent="flex-start" mb={1}>
              <Paper sx={{ p: 1 }}>
                <CircularProgress size={16} />
              </Paper>
            </Box>
          )}
        </Box>
        <Box display="flex" gap={1}>
          <TextField
            fullWidth
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Ask about performance, quality, security..."
            onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
          />
          <Button onClick={sendMessage} disabled={chatLoading}>
            Send
          </Button>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

// Create Team Dialog Component
const CreateTeamDialog = ({ open, onClose, onCreate }) => {
  const [teamConfig, setTeamConfig] = useState({
    type: 'standard',
    roles: [],
    supervisor_mode: 'guided',
    workflow_type: 'standard'
  });

  const handleCreate = () => {
    onCreate(teamConfig);
    onClose();
    setTeamConfig({
      type: 'standard',
      roles: [],
      supervisor_mode: 'guided',
      workflow_type: 'standard'
    });
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Create Testing Team</DialogTitle>
      <DialogContent>
        <Box display="flex" flexDirection="column" gap={2} mt={1}>
          <FormControl fullWidth>
            <InputLabel>Team Type</InputLabel>
            <Select
              value={teamConfig.type}
              onChange={(e) => setTeamConfig({ ...teamConfig, type: e.target.value })}
            >
              <MenuItem value="standard">Standard Team</MenuItem>
              <MenuItem value="minimal">Minimal Team</MenuItem>
              <MenuItem value="custom">Custom Team</MenuItem>
            </Select>
          </FormControl>
          
          <FormControl fullWidth>
            <InputLabel>Supervisor Mode</InputLabel>
            <Select
              value={teamConfig.supervisor_mode}
              onChange={(e) => setTeamConfig({ ...teamConfig, supervisor_mode: e.target.value })}
            >
              <MenuItem value="autonomous">Autonomous</MenuItem>
              <MenuItem value="guided">Guided</MenuItem>
              <MenuItem value="directed">Directed</MenuItem>
              <MenuItem value="collaborative">Collaborative</MenuItem>
              <MenuItem value="hierarchical">Hierarchical</MenuItem>
            </Select>
          </FormControl>
          
          <FormControl fullWidth>
            <InputLabel>Workflow Type</InputLabel>
            <Select
              value={teamConfig.workflow_type}
              onChange={(e) => setTeamConfig({ ...teamConfig, workflow_type: e.target.value })}
            >
              <MenuItem value="standard">Standard</MenuItem>
              <MenuItem value="minimal">Minimal</MenuItem>
              <MenuItem value="quality_focused">Quality Focused</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleCreate} variant="contained">
          Create Team
        </Button>
      </DialogActions>
    </Dialog>
  );
};

// Main Phase 2 Dashboard Component
const Phase2Dashboard = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [teams, setTeams] = useState({});
  const [monitors, setMonitors] = useState({});
  const [phase2Status, setPhase2Status] = useState(null);
  const [createTeamDialog, setCreateTeamDialog] = useState(false);
  const [createMonitorDialog, setCreateMonitorDialog] = useState(false);
  const [chatDialog, setChatDialog] = useState({ open: false, monitorId: null });

  const { apiCall, loading, error } = usePhase2API();

  // Auto-refresh data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statusData, teamsData, monitorsData] = await Promise.all([
          apiCall('/status'),
          apiCall('/teams'),
          apiCall('/monitoring')
        ]);

        setPhase2Status(statusData.data);
        setTeams(teamsData.data.teams || {});
        setMonitors(monitorsData.data.monitors || {});
      } catch (err) {
        console.error('Failed to fetch Phase 2 data:', err);
      }
    };

    fetchData();
    
    // Set up auto-refresh
    const interval = setInterval(fetchData, 10000); // Refresh every 10 seconds

    return () => clearInterval(interval);
  }, [apiCall]);

  const handleCreateTeam = async (teamConfig) => {
    try {
      await apiCall('/teams', {
        method: 'POST',
        body: JSON.stringify(teamConfig)
      });
      
      // Refresh teams list
      const teamsData = await apiCall('/teams');
      setTeams(teamsData.data.teams || {});
    } catch (err) {
      console.error('Failed to create team:', err);
    }
  };

  const handleCreateMonitor = async () => {
    try {
      const monitorConfig = {
        mode: 'interactive',
        agents: ['performance', 'quality', 'security', 'collaboration']
      };
      
      await apiCall('/monitoring', {
        method: 'POST',
        body: JSON.stringify(monitorConfig)
      });
      
      // Refresh monitors list
      const monitorsData = await apiCall('/monitoring');
      setMonitors(monitorsData.data.monitors || {});
      setCreateMonitorDialog(false);
    } catch (err) {
      console.error('Failed to create monitor:', err);
    }
  };

  const handleExecuteWorkflow = async (teamId) => {
    try {
      const workflowConfig = {
        target_path: '.',
        workflow: 'standard'
      };
      
      await apiCall(`/teams/${teamId}/execute`, {
        method: 'POST',
        body: JSON.stringify(workflowConfig)
      });
      
      // Show success message or update UI
      console.log('Workflow execution started');
    } catch (err) {
      console.error('Failed to execute workflow:', err);
    }
  };

  const handleStopTeam = async (teamId) => {
    try {
      await apiCall(`/teams/${teamId}`, { method: 'DELETE' });
      
      // Refresh teams list
      const teamsData = await apiCall('/teams');
      setTeams(teamsData.data.teams || {});
    } catch (err) {
      console.error('Failed to stop team:', err);
    }
  };

  const handleStopMonitor = async (monitorId) => {
    try {
      await apiCall(`/monitoring/${monitorId}`, { method: 'DELETE' });
      
      // Refresh monitors list
      const monitorsData = await apiCall('/monitoring');
      setMonitors(monitorsData.data.monitors || {});
    } catch (err) {
      console.error('Failed to stop monitor:', err);
    }
  };

  const handleOpenChat = (monitorId) => {
    setChatDialog({ open: true, monitorId });
  };

  // Teams Tab
  const TeamsTab = () => (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6">Multi-Agent Testing Teams</Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setCreateTeamDialog(true)}
        >
          Create Team
        </Button>
      </Box>

      <Grid container spacing={3}>
        {Object.entries(teams).map(([teamId, team]) => (
          <Grid item xs={12} md={6} lg={4} key={teamId}>
            <TeamStatusCard
              team={team}
              teamId={teamId}
              onExecuteWorkflow={handleExecuteWorkflow}
              onStopTeam={handleStopTeam}
            />
          </Grid>
        ))}
      </Grid>

      {Object.keys(teams).length === 0 && (
        <Box textAlign="center" py={4}>
          <Typography color="text.secondary">
            No active teams. Create your first multi-agent testing team to get started.
          </Typography>
        </Box>
      )}
    </Box>
  );

  // Monitoring Tab
  const MonitoringTab = () => (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h6">Enhanced Monitoring</Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={handleCreateMonitor}
        >
          Create Monitor
        </Button>
      </Box>

      <Grid container spacing={3}>
        {Object.entries(monitors).map(([monitorId, monitor]) => (
          <Grid item xs={12} md={6} lg={4} key={monitorId}>
            <MonitorStatusCard
              monitor={monitor}
              monitorId={monitorId}
              onChat={handleOpenChat}
              onStopMonitor={handleStopMonitor}
            />
          </Grid>
        ))}
      </Grid>

      {Object.keys(monitors).length === 0 && (
        <Box textAlign="center" py={4}>
          <Typography color="text.secondary">
            No active monitors. Create your first enhanced monitor to get started.
          </Typography>
        </Box>
      )}
    </Box>
  );

  // Status Tab
  const StatusTab = () => (
    <Box>
      <Typography variant="h6" mb={3}>Phase 2 System Status</Typography>
      
      {phase2Status && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" mb={2}>System Overview</Typography>
              <Box>
                <Typography variant="body2">
                  Phase 2 Available: {phase2Status.phase2_available ? '✅ Yes' : '❌ No'}
                </Typography>
                <Typography variant="body2">
                  Active Teams: {phase2Status.active_teams}
                </Typography>
                <Typography variant="body2">
                  Active Monitors: {phase2Status.active_monitors}
                </Typography>
              </Box>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" mb={2}>Available Components</Typography>
              <Box display="flex" flexDirection="column" gap={1}>
                {Object.entries(phase2Status.components).map(([component, available]) => (
                  <Chip
                    key={component}
                    label={component.replace('_', ' ')}
                    color={available ? 'success' : 'error'}
                    size="small"
                    variant="outlined"
                  />
                ))}
              </Box>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  );

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Typography variant="h4" component="h1">
          Phase 2: Advanced Coordination
        </Typography>
        <Box display="flex" alignItems="center" gap={2}>
          {phase2Status && (
            <Chip 
              icon={phase2Status.phase2_available ? <CheckCircle /> : <Error />} 
              label={phase2Status.phase2_available ? "Available" : "Unavailable"}
              color={phase2Status.phase2_available ? "success" : "error"}
              variant="outlined"
            />
          )}
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
          <Tab icon={<Groups />} label="Teams" />
          <Tab icon={<Monitor />} label="Monitoring" />
          <Tab icon={<Dashboard />} label="Status" />
        </Tabs>
      </Paper>

      <Box role="tabpanel" hidden={activeTab !== 0}>
        {activeTab === 0 && <TeamsTab />}
      </Box>
      
      <Box role="tabpanel" hidden={activeTab !== 1}>
        {activeTab === 1 && <MonitoringTab />}
      </Box>
      
      <Box role="tabpanel" hidden={activeTab !== 2}>
        {activeTab === 2 && <StatusTab />}
      </Box>

      <CreateTeamDialog
        open={createTeamDialog}
        onClose={() => setCreateTeamDialog(false)}
        onCreate={handleCreateTeam}
      />

      <ChatDialog
        open={chatDialog.open}
        onClose={() => setChatDialog({ open: false, monitorId: null })}
        monitorId={chatDialog.monitorId}
        apiCall={apiCall}
      />

      {loading && (
        <Box display="flex" justifyContent="center" mt={4}>
          <CircularProgress />
        </Box>
      )}
    </Container>
  );
};

export default Phase2Dashboard;
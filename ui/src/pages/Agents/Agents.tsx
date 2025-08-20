/**
 * Agents page component for the Ruckus UI.
 * 
 * This component displays and manages agents including
 * listing, monitoring, and viewing agent status.
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  CircularProgress,
  LinearProgress,
} from '@mui/material';

import { orchestratorApi } from '../../services/api';

interface Agent {
  id: string;
  name: string;
  host: string;
  port: number;
  status: string;
  capabilities: {
    runtime: string;
    platform: string;
    max_memory?: number;
    gpu_count?: number;
    gpu_type?: string;
  };
  created_at: string;
  last_heartbeat?: string;
}

const Agents: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAgents = async () => {
      try {
        setLoading(true);
        const response = await orchestratorApi.get('/agents');
        setAgents(response.data);
      } catch (err) {
        setError('Failed to fetch agents');
        console.error('Agents fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchAgents();

    // Set up polling for agent status updates
    const interval = setInterval(fetchAgents, 10000);
    return () => clearInterval(interval);
  }, []);

  const formatDate = (dateString?: string) => {
    if (!dateString) return '-';
    return new Date(dateString).toLocaleString();
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'online':
        return 'success';
      case 'offline':
        return 'default';
      case 'busy':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const getLastSeenText = (lastHeartbeat?: string) => {
    if (!lastHeartbeat) return 'Never';
    
    const now = new Date();
    const heartbeat = new Date(lastHeartbeat);
    const diffMs = now.getTime() - heartbeat.getTime();
    const diffMinutes = Math.floor(diffMs / (1000 * 60));
    
    if (diffMinutes < 1) return 'Just now';
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    
    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  };

  if (loading && agents.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <Typography color="error" variant="h6">
          {error}
        </Typography>
      </Box>
    );
  }

  const onlineAgents = agents.filter(agent => agent.status === 'online').length;
  const busyAgents = agents.filter(agent => agent.status === 'busy').length;

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Agents
      </Typography>

      {loading && (
        <LinearProgress sx={{ mb: 2 }} />
      )}

      <Box mb={3}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Agent Summary
            </Typography>
            <Box display="flex" gap={4}>
              <Box>
                <Typography variant="h4" color="primary">
                  {agents.length}
                </Typography>
                <Typography color="textSecondary">Total Agents</Typography>
              </Box>
              <Box>
                <Typography variant="h4" color="success.main">
                  {onlineAgents}
                </Typography>
                <Typography color="textSecondary">Online</Typography>
              </Box>
              <Box>
                <Typography variant="h4" color="warning.main">
                  {busyAgents}
                </Typography>
                <Typography color="textSecondary">Busy</Typography>
              </Box>
              <Box>
                <Typography variant="h4" color="error.main">
                  {agents.filter(agent => agent.status === 'offline').length}
                </Typography>
                <Typography color="textSecondary">Offline</Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Box>

      {agents.length === 0 ? (
        <Card>
          <CardContent>
            <Typography color="textSecondary" align="center">
              No agents registered. Agents will appear here when they connect to the orchestrator.
            </Typography>
          </CardContent>
        </Card>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Address</TableCell>
                <TableCell>Capabilities</TableCell>
                <TableCell>Last Seen</TableCell>
                <TableCell>Registered</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {agents.map((agent) => (
                <TableRow key={agent.id} hover>
                  <TableCell>
                    <Box>
                      <Typography variant="subtitle2">
                        {agent.name}
                      </Typography>
                      <Typography variant="body2" color="textSecondary" fontFamily="monospace">
                        {agent.id.substring(0, 8)}...
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={agent.status}
                      color={getStatusColor(agent.status) as any}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {agent.host}:{agent.port}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Box>
                      <Box display="flex" gap={1} mb={1}>
                        <Chip
                          label={agent.capabilities.runtime}
                          size="small"
                          variant="outlined"
                        />
                        <Chip
                          label={agent.capabilities.platform}
                          size="small"
                          variant="outlined"
                        />
                      </Box>
                      {agent.capabilities.gpu_type && (
                        <Typography variant="body2" color="textSecondary">
                          {agent.capabilities.gpu_count}x {agent.capabilities.gpu_type}
                        </Typography>
                      )}
                      {agent.capabilities.max_memory && (
                        <Typography variant="body2" color="textSecondary">
                          {agent.capabilities.max_memory}GB RAM
                        </Typography>
                      )}
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {getLastSeenText(agent.last_heartbeat)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {formatDate(agent.created_at)}
                    </Typography>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
};

export default Agents;

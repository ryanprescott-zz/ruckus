/**
 * Jobs page component for the Ruckus UI.
 * 
 * This component displays and manages jobs including
 * listing, monitoring, and controlling job execution.
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

interface Job {
  id: string;
  experiment_id: string;
  agent_id?: string;
  status: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  results?: any;
}

const Jobs: React.FC = () => {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        setLoading(true);
        const response = await orchestratorApi.get('/jobs');
        setJobs(response.data);
      } catch (err) {
        setError('Failed to fetch jobs');
        console.error('Jobs fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchJobs();

    // Set up polling for job status updates
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  const formatDate = (dateString?: string) => {
    if (!dateString) return '-';
    return new Date(dateString).toLocaleString();
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'pending':
        return 'default';
      case 'running':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'cancelled':
        return 'warning';
      default:
        return 'default';
    }
  };

  const calculateDuration = (startTime?: string, endTime?: string) => {
    if (!startTime) return '-';
    
    const start = new Date(startTime);
    const end = endTime ? new Date(endTime) : new Date();
    const durationMs = end.getTime() - start.getTime();
    const durationSeconds = Math.floor(durationMs / 1000);
    
    if (durationSeconds < 60) {
      return `${durationSeconds}s`;
    } else if (durationSeconds < 3600) {
      return `${Math.floor(durationSeconds / 60)}m ${durationSeconds % 60}s`;
    } else {
      const hours = Math.floor(durationSeconds / 3600);
      const minutes = Math.floor((durationSeconds % 3600) / 60);
      return `${hours}h ${minutes}m`;
    }
  };

  if (loading && jobs.length === 0) {
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

  const runningJobs = jobs.filter(job => job.status === 'running').length;

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Jobs
      </Typography>

      {loading && (
        <LinearProgress sx={{ mb: 2 }} />
      )}

      <Box mb={3}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Job Summary
            </Typography>
            <Box display="flex" gap={4}>
              <Box>
                <Typography variant="h4" color="primary">
                  {jobs.length}
                </Typography>
                <Typography color="textSecondary">Total Jobs</Typography>
              </Box>
              <Box>
                <Typography variant="h4" color="warning.main">
                  {runningJobs}
                </Typography>
                <Typography color="textSecondary">Running</Typography>
              </Box>
              <Box>
                <Typography variant="h4" color="success.main">
                  {jobs.filter(job => job.status === 'completed').length}
                </Typography>
                <Typography color="textSecondary">Completed</Typography>
              </Box>
              <Box>
                <Typography variant="h4" color="error.main">
                  {jobs.filter(job => job.status === 'failed').length}
                </Typography>
                <Typography color="textSecondary">Failed</Typography>
              </Box>
            </Box>
          </CardContent>
        </Card>
      </Box>

      {jobs.length === 0 ? (
        <Card>
          <CardContent>
            <Typography color="textSecondary" align="center">
              No jobs found. Jobs will appear here when experiments are executed.
            </Typography>
          </CardContent>
        </Card>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Job ID</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Agent</TableCell>
                <TableCell>Created</TableCell>
                <TableCell>Duration</TableCell>
                <TableCell>Results</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {jobs.map((job) => (
                <TableRow key={job.id} hover>
                  <TableCell>
                    <Typography variant="body2" fontFamily="monospace">
                      {job.id.substring(0, 8)}...
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={job.status}
                      color={getStatusColor(job.status) as any}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {job.agent_id ? `${job.agent_id.substring(0, 8)}...` : 'Unassigned'}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {formatDate(job.created_at)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {calculateDuration(job.started_at, job.completed_at)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    {job.results ? (
                      <Typography variant="body2" color="success.main">
                        Available
                      </Typography>
                    ) : (
                      <Typography variant="body2" color="textSecondary">
                        -
                      </Typography>
                    )}
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

export default Jobs;

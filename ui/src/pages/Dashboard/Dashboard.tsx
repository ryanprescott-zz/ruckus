/**
 * Dashboard page component for the Ruckus UI.
 * 
 * This component displays the main dashboard with overview
 * information about experiments, jobs, and agents.
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  CircularProgress,
} from '@mui/material';
import {
  Science as ExperimentsIcon,
  Work as JobsIcon,
  Computer as AgentsIcon,
} from '@mui/icons-material';

import { orchestratorApi } from '../../services/api';

interface DashboardStats {
  totalExperiments: number;
  totalJobs: number;
  runningJobs: number;
  totalAgents: number;
  onlineAgents: number;
}

interface StatCardProps {
  title: string;
  value: number;
  subtitle?: string;
  icon: React.ReactElement;
  color: string;
}

const StatCard: React.FC<StatCardProps> = ({ title, value, subtitle, icon, color }) => (
  <Card>
    <CardContent>
      <Box display="flex" alignItems="center" justifyContent="space-between">
        <Box>
          <Typography color="textSecondary" gutterBottom variant="overline">
            {title}
          </Typography>
          <Typography variant="h4" component="h2">
            {value}
          </Typography>
          {subtitle && (
            <Typography color="textSecondary" variant="body2">
              {subtitle}
            </Typography>
          )}
        </Box>
        <Box sx={{ color, fontSize: 40 }}>
          {icon}
        </Box>
      </Box>
    </CardContent>
  </Card>
);

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        
        // Fetch data from multiple endpoints
        const [experimentsResponse, jobsResponse, agentsResponse] = await Promise.all([
          orchestratorApi.get('/experiments'),
          orchestratorApi.get('/jobs'),
          orchestratorApi.get('/agents'),
        ]);

        const experiments = experimentsResponse.data;
        const jobs = jobsResponse.data;
        const agents = agentsResponse.data;

        const dashboardStats: DashboardStats = {
          totalExperiments: experiments.length,
          totalJobs: jobs.length,
          runningJobs: jobs.filter((job: any) => job.status === 'running').length,
          totalAgents: agents.length,
          onlineAgents: agents.filter((agent: any) => agent.status === 'online').length,
        };

        setStats(dashboardStats);
      } catch (err) {
        setError('Failed to fetch dashboard data');
        console.error('Dashboard data fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);

  if (loading) {
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

  if (!stats) {
    return null;
  }

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={4}>
          <StatCard
            title="Total Experiments"
            value={stats.totalExperiments}
            icon={<ExperimentsIcon />}
            color="#1976d2"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={4}>
          <StatCard
            title="Jobs"
            value={stats.totalJobs}
            subtitle={`${stats.runningJobs} running`}
            icon={<JobsIcon />}
            color="#2e7d32"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={4}>
          <StatCard
            title="Agents"
            value={stats.totalAgents}
            subtitle={`${stats.onlineAgents} online`}
            icon={<AgentsIcon />}
            color="#ed6c02"
          />
        </Grid>
      </Grid>

      <Box mt={4}>
        <Typography variant="h5" component="h2" gutterBottom>
          Recent Activity
        </Typography>
        <Card>
          <CardContent>
            <Typography color="textSecondary">
              Recent activity will be displayed here in future versions.
            </Typography>
          </CardContent>
        </Card>
      </Box>
    </Box>
  );
};

export default Dashboard;

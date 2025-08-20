/**
 * Experiments page component for the Ruckus UI.
 * 
 * This component displays and manages experiments including
 * listing, creating, and editing experiments.
 */

import React, { useEffect, useState } from 'react';
import {
  Box,
  Button,
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
} from '@mui/material';
import { Add as AddIcon } from '@mui/icons-material';

import { orchestratorApi } from '../../services/api';

interface Experiment {
  id: string;
  name: string;
  description?: string;
  model_name: string;
  runtime: string;
  platform: string;
  created_at: string;
  updated_at: string;
}

const Experiments: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchExperiments = async () => {
      try {
        setLoading(true);
        const response = await orchestratorApi.get('/experiments');
        setExperiments(response.data);
      } catch (err) {
        setError('Failed to fetch experiments');
        console.error('Experiments fetch error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchExperiments();
  }, []);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const getRuntimeColor = (runtime: string) => {
    switch (runtime.toLowerCase()) {
      case 'transformers':
        return 'primary';
      case 'vllm':
        return 'secondary';
      case 'pytorch':
        return 'success';
      default:
        return 'default';
    }
  };

  const getPlatformColor = (platform: string) => {
    switch (platform.toLowerCase()) {
      case 'cuda':
        return 'success';
      case 'cpu':
        return 'warning';
      default:
        return 'default';
    }
  };

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

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Experiments
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {
            // TODO: Implement create experiment dialog
            console.log('Create experiment clicked');
          }}
        >
          Create Experiment
        </Button>
      </Box>

      {experiments.length === 0 ? (
        <Card>
          <CardContent>
            <Typography color="textSecondary" align="center">
              No experiments found. Create your first experiment to get started.
            </Typography>
          </CardContent>
        </Card>
      ) : (
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Name</TableCell>
                <TableCell>Model</TableCell>
                <TableCell>Runtime</TableCell>
                <TableCell>Platform</TableCell>
                <TableCell>Created</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {experiments.map((experiment) => (
                <TableRow key={experiment.id} hover>
                  <TableCell>
                    <Box>
                      <Typography variant="subtitle2">
                        {experiment.name}
                      </Typography>
                      {experiment.description && (
                        <Typography variant="body2" color="textSecondary">
                          {experiment.description}
                        </Typography>
                      )}
                    </Box>
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {experiment.model_name}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={experiment.runtime}
                      color={getRuntimeColor(experiment.runtime) as any}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={experiment.platform}
                      color={getPlatformColor(experiment.platform) as any}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>
                    <Typography variant="body2">
                      {formatDate(experiment.created_at)}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Button
                      size="small"
                      onClick={() => {
                        // TODO: Implement view/edit experiment
                        console.log('View experiment:', experiment.id);
                      }}
                    >
                      View
                    </Button>
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

export default Experiments;

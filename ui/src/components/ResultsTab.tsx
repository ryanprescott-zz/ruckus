import React, { useState, useEffect, useCallback } from 'react';
import type { 
  ExperimentResult,
  ExperimentResultTableRow,
} from '../types/api';
import { apiClient } from '../services/api';
import { formatTimestamp, formatAgentDetails } from '../utils/format';
import './ResultsTab.css';

// Configuration
const DEFAULT_POLLING_INTERVAL = 5000; // 5 seconds
const getPollingInterval = (): number => {
  const envInterval = import.meta.env.VITE_RESULTS_POLLING_INTERVAL;
  return envInterval ? parseInt(envInterval, 10) * 1000 : DEFAULT_POLLING_INTERVAL;
};

const ResultsTab: React.FC = () => {
  // State
  const [results, setResults] = useState<ExperimentResultTableRow[]>([]);
  const [selectedResult, setSelectedResult] = useState<ExperimentResult | null>(null);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortColumn, setSortColumn] = useState<keyof ExperimentResultTableRow>('experiment_id');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [toast, setToast] = useState<{message: string, type: 'success' | 'error'} | null>(null);

  // Polling interval
  const pollingInterval = getPollingInterval();

  // Show toast message
  const showToast = useCallback((message: string, type: 'success' | 'error') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  }, []);

  // Fetch experiment results data
  const fetchExperimentResults = useCallback(async () => {
    try {
      const response = await apiClient.listExperimentResults();
      
      // Convert to table rows
      const tableRows: ExperimentResultTableRow[] = response.results.map(result => ({
        experiment_id: result.experiment_id,
        job_id: result.job_id,
        agent_id: result.agent_id,
        status: result.status.toUpperCase(),
        export: '', // For export button column
        experimentResult: result
      }));

      setResults(tableRows);
      setError(null);
    } catch (err) {
      console.error('Error fetching experiment results:', err);
      setError('Failed to fetch experiment results');
    } finally {
      setLoading(false);
    }
  }, []);

  // Handle row selection
  const handleRowClick = (result: ExperimentResultTableRow) => {
    setSelectedResult(result.experimentResult);
  };

  // Handle sort
  const handleSort = (column: keyof ExperimentResultTableRow) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc'); // Default to ascending for experiment_id
    }
  };

  // Sort results
  const sortedResults = [...results].sort((a, b) => {
    const aVal = a[sortColumn];
    const bVal = b[sortColumn];
    
    let comparison = 0;
    if (aVal < bVal) comparison = -1;
    if (aVal > bVal) comparison = 1;
    
    return sortDirection === 'asc' ? comparison : -comparison;
  });

  // Handle export
  const handleExport = async (jobId: string, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent row selection
    
    try {
      const response = await apiClient.getExperimentResult(jobId);
      
      // Create and download JSON file
      const dataStr = JSON.stringify(response.result, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      
      const link = document.createElement('a');
      link.href = URL.createObjectURL(dataBlob);
      link.download = `results_${response.result.experiment_id}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
      
      showToast('Results exported successfully', 'success');
    } catch (err) {
      console.error('Error exporting results:', err);
      showToast('Failed to export results', 'error');
    }
  };

  // Initial data fetch and polling setup
  useEffect(() => {
    fetchExperimentResults();

    const interval = setInterval(fetchExperimentResults, pollingInterval);
    return () => clearInterval(interval);
  }, [fetchExperimentResults, pollingInterval]);

  return (
    <div className="results-tab">
      {/* Toast notifications */}
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          {toast.message}
        </div>
      )}

      {/* Results table section */}
      <div className="results-section">
        <div className="section-header">
          <h2>Experiment Results</h2>
        </div>

        {loading && <div className="loading">Loading experiment results...</div>}
        {error && <div className="error">{error}</div>}

        {!loading && !error && (
          <div className="table-container">
            <table className="results-table">
              <thead>
                <tr>
                  <th onClick={() => handleSort('experiment_id')} className="sortable">
                    Experiment ID {sortColumn === 'experiment_id' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th onClick={() => handleSort('job_id')} className="sortable">
                    Job ID {sortColumn === 'job_id' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th onClick={() => handleSort('agent_id')} className="sortable">
                    Agent ID {sortColumn === 'agent_id' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th onClick={() => handleSort('status')} className="sortable">
                    Status {sortColumn === 'status' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </th>
                  <th>Export</th>
                </tr>
              </thead>
              <tbody>
                {sortedResults.map((result) => (
                  <tr
                    key={result.job_id}
                    onClick={() => handleRowClick(result)}
                    className={selectedResult?.job_id === result.job_id ? 'selected' : ''}
                  >
                    <td>{result.experiment_id}</td>
                    <td>{result.job_id}</td>
                    <td>{result.agent_id}</td>
                    <td>
                      <span className={`status status-${result.status.toLowerCase()}`}>
                        {result.status}
                      </span>
                    </td>
                    <td>
                      <button
                        className="btn btn-secondary btn-sm"
                        onClick={(e) => handleExport(result.job_id, e)}
                      >
                        Export
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Details panel */}
      <div className="details-panel">
        {selectedResult && (
          <div className="result-details">
            <div className="details-header">
              <h3>Experiment Result Details</h3>
              <span className="details-id">{selectedResult.job_id}</span>
            </div>
            <div className="details-content">
              <textarea
                className="details-textarea"
                value={formatAgentDetails('Experiment Result', selectedResult)}
                readOnly
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsTab;
import React, { useState, useEffect, useCallback, useRef } from 'react';
import type {
  ExperimentResult,
  ExperimentResultTableRow,
  LLMConversationResult,
} from '../types/api';
import { apiClient } from '../services/api';
import { formatAgentDetails } from '../utils/format';
import VisualizationTab from './VisualizationTab';
import ConversationDisplay from './ConversationDisplay';
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
  const [showVisualization, setShowVisualization] = useState(false);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortColumn, setSortColumn] = useState<keyof ExperimentResultTableRow>('experiment_name');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [toast, setToast] = useState<{message: string, type: 'success' | 'error'} | null>(null);

  // Resizable splitter state
  const [splitPercentage, setSplitPercentage] = useState(60); // Default 60% for table, 40% for details
  const [isDragging, setIsDragging] = useState(false);

  // Ref for click-outside detection
  const tableRef = useRef<HTMLDivElement>(null);

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

      // Fetch experiment details for each result to get name and type
      const tableRows: ExperimentResultTableRow[] = await Promise.all(
        response.results.map(async (result) => {
          try {
            const experimentResponse = await apiClient.getExperiment(result.experiment_id);
            const experiment = experimentResponse.experiment;

            return {
              experiment_id: result.experiment_id,
              experiment_name: experiment.name,
              experiment_type: experiment.task.type,
              job_id: result.job_id,
              agent_id: result.agent_id,
              status: result.status.toUpperCase(),
              export: '', // For export button column
              experimentResult: result
            };
          } catch (err) {
            console.warn(`Failed to fetch experiment details for ${result.experiment_id}:`, err);
            return {
              experiment_id: result.experiment_id,
              experiment_name: 'Unknown',
              experiment_type: 'Unknown',
              job_id: result.job_id,
              agent_id: result.agent_id,
              status: result.status.toUpperCase(),
              export: '', // For export button column
              experimentResult: result
            };
          }
        })
      );

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
    setShowVisualization(false); // Reset to details view when selecting a new result
  };

  // Handle click outside to deselect
  const handleClickOutside = useCallback((event: MouseEvent) => {
    if (tableRef.current && !tableRef.current.contains(event.target as Node)) {
      // Only deselect if clicking outside the entire table area
      if (!event.target || !(event.target as Element).closest('.details-panel')) {
        setSelectedResult(null);
        setShowVisualization(false);
      }
    }
  }, []);

  // Add click-outside listener
  useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [handleClickOutside]);

  // Handle visualization toggle
  const handleShowVisualization = (result: ExperimentResultTableRow, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent row selection
    setSelectedResult(result.experimentResult);
    setShowVisualization(true);
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

  // Resizable splitter handlers
  const handleSplitterMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return;

    const container = document.querySelector('.results-tab');
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const mouseY = e.clientY - containerRect.top;
    const containerHeight = containerRect.height;

    // Calculate new split percentage, with bounds checking
    let newPercentage = (mouseY / containerHeight) * 100;
    newPercentage = Math.max(20, Math.min(80, newPercentage)); // Limit between 20% and 80%

    setSplitPercentage(newPercentage);
  }, [isDragging]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Add global mouse event listeners when dragging
  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'row-resize';
      document.body.style.userSelect = 'none';

      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

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

      {/* Main content area with proper flex layout */}
      <div style={{
        display: 'flex',
        flexDirection: selectedResult ? 'column' : 'column',
        height: '100%',
        minHeight: 0,
        position: 'relative'
      }}>
        {/* Results table section */}
        <div className="results-section" ref={tableRef} style={{
          flex: selectedResult ? `0 0 ${splitPercentage}%` : '1',
          minHeight: 0,
          display: 'flex',
          flexDirection: 'column'
        }}>
          <div className="section-header">
            <h2>Experiment Results</h2>
            {selectedResult && (
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => {
                  setSelectedResult(null);
                  setShowVisualization(false);
                }}
                title="Clear selection"
              >
                ✕ Clear Selection
              </button>
            )}
          </div>

          {loading && <div className="loading">Loading experiment results...</div>}
          {error && <div className="error">{error}</div>}

          {!loading && !error && (
            <div className="table-container">
              <table className="results-table">
                <thead>
                  <tr>
                    <th onClick={() => handleSort('experiment_name')} className="sortable">
                      Experiment {sortColumn === 'experiment_name' && (sortDirection === 'asc' ? '↑' : '↓')}
                    </th>
                    <th onClick={() => handleSort('experiment_type')} className="sortable">
                      Type {sortColumn === 'experiment_type' && (sortDirection === 'asc' ? '↑' : '↓')}
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
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedResults.map((result) => (
                    <tr
                      key={result.job_id}
                      onClick={() => handleRowClick(result)}
                      className={selectedResult?.job_id === result.job_id ? 'selected' : ''}
                    >
                      <td>{result.experiment_name}</td>
                      <td>
                        <span className="experiment-type">
                          {result.experiment_type.replace('_', ' ').toUpperCase()}
                        </span>
                      </td>
                      <td>{result.job_id}</td>
                      <td>{result.agent_id}</td>
                      <td>
                        <span className={`status status-${result.status.toLowerCase()}`}>
                          {result.status === 'FAILED' ? '❌ FAILED' : result.status}
                        </span>
                      </td>
                      <td>
                        <div className="action-buttons">
                          <button
                            className="btn btn-primary btn-sm"
                            onClick={(e) => handleShowVisualization(result, e)}
                            title="View Charts"
                          >
                            📊 Charts
                          </button>
                          <button
                            className="btn btn-secondary btn-sm"
                            onClick={(e) => handleExport(result.job_id, e)}
                            title="Export JSON"
                          >
                            📥 Export
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Resizable splitter */}
        {selectedResult && (
          <div
            className="splitter"
            onMouseDown={handleSplitterMouseDown}
            style={{
              height: '4px',
              backgroundColor: isDragging ? '#007bff' : '#ddd',
              cursor: 'row-resize',
              flexShrink: 0,
              transition: isDragging ? 'none' : 'background-color 0.2s ease',
              borderRadius: '2px',
              margin: '2px 0',
              position: 'relative',
              zIndex: 10
            }}
            onMouseEnter={(e) => {
              if (!isDragging) {
                e.currentTarget.style.backgroundColor = '#007bff';
              }
            }}
            onMouseLeave={(e) => {
              if (!isDragging) {
                e.currentTarget.style.backgroundColor = '#ddd';
              }
            }}
            title="Drag to resize panels"
          />
        )}

        {/* Details panel */}
        {selectedResult && (
          <div className="details-panel" style={{
            flex: `0 0 ${100 - splitPercentage}%`,
            minHeight: 0,
            overflowY: 'auto'
          }}>
            {!showVisualization && (
              <div className="result-details">
                <div className="details-header">
                  <h3>Experiment Result Details</h3>
                  <span className="details-id">{selectedResult.job_id}</span>
                  <div className="details-actions">
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => setShowVisualization(true)}
                    >
                      📊 View Charts
                    </button>
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => {
                        setSelectedResult(null);
                        setShowVisualization(false);
                      }}
                      title="Close details"
                    >
                      ✕ Close
                    </button>
                  </div>
                </div>

                {/* Show conversation for LLM experiments */}
                {selectedResult.output && typeof selectedResult.output === 'object' && selectedResult.output.conversation && (
                  <ConversationDisplay
                    conversationResult={selectedResult.output as LLMConversationResult}
                    title="LLM Conversation"
                  />
                )}

                <div className="details-content">
                  <textarea
                    className="details-textarea"
                    value={formatAgentDetails('Experiment Result', selectedResult)}
                    readOnly
                  />
                </div>
              </div>
            )}

            {showVisualization && (
              <div className="visualization-container">
                <VisualizationTab
                  experimentResult={selectedResult}
                  onClose={() => setShowVisualization(false)}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsTab;
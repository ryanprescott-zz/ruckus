import React, { useState, useEffect, useCallback } from 'react';
import type { 
  JobInfo,
  JobTableRow,
  RegisteredAgentInfo,
  ExperimentSpec
} from '../types/api';
import { apiClient } from '../services/api';
import { formatTimestamp, formatAgentDetails } from '../utils/format';
import './JobsTab.css';

// Configuration
const DEFAULT_POLLING_INTERVAL = 5000; // 5 seconds
const MIN_POLLING_INTERVAL = 1000; // 1 second minimum to avoid flooding logs
const getPollingInterval = (): number => {
  const envInterval = import.meta.env.VITE_JOBS_POLLING_INTERVAL;
  const interval = envInterval ? parseInt(envInterval, 10) * 1000 : DEFAULT_POLLING_INTERVAL;
  // Ensure minimum 1 second interval to prevent log flooding
  return Math.max(interval, MIN_POLLING_INTERVAL);
};

const JobsTab: React.FC = () => {
  // State
  const [jobs, setJobs] = useState<JobTableRow[]>([]);
  const [selectedJob, setSelectedJob] = useState<JobInfo | null>(null);
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentSpec | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<RegisteredAgentInfo | null>(null);

  // Loading states for job details
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [experimentError, setExperimentError] = useState<string | null>(null);
  const [agentError, setAgentError] = useState<string | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortColumn, setSortColumn] = useState<keyof JobTableRow>('updated');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [toast, setToast] = useState<{message: string, type: 'success' | 'error'} | null>(null);

  // New Job state
  const [isCreatingJob, setIsCreatingJob] = useState(false);
  const [experiments, setExperiments] = useState<ExperimentSpec[]>([]);
  const [agents, setAgents] = useState<RegisteredAgentInfo[]>([]);
  const [newJobExperimentId, setNewJobExperimentId] = useState<string>('');
  const [newJobAgentId, setNewJobAgentId] = useState<string>('');

  // Resizable splitter state
  const [splitPercentage, setSplitPercentage] = useState(60); // Default 60% for jobs, 40% for details
  const [isDragging, setIsDragging] = useState(false);

  // Polling interval
  const pollingInterval = getPollingInterval();

  // Show toast message
  const showToast = useCallback((message: string, type: 'success' | 'error') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  }, []);

  // Fetch jobs data and flatten into table rows
  const fetchJobsData = useCallback(async () => {
    try {
      const response = await apiClient.listJobs();

      // Flatten jobs from all agents into a single array
      const allJobs: JobInfo[] = [];
      Object.values(response.jobs).forEach(agentJobs => {
        allJobs.push(...agentJobs);
      });

      // Convert to table rows
      const tableRows: JobTableRow[] = allJobs.map(job => ({
        job_id: job.job_id,
        experiment_id: job.experiment_id,
        agent_id: job.agent_id,
        status: job.status.status.toUpperCase(),
        updated: formatTimestamp(job.status.timestamp),
        cancel: '', // For cancel button column
        jobInfo: job
      }));

      setJobs(tableRows);

      // Note: Selected job update logic moved to separate useEffect to avoid polling restart

      setError(null);
    } catch (err) {
      console.error('Error fetching jobs:', err);
      setError('Failed to fetch jobs');
    } finally {
      setLoading(false);
    }
  }, []); // Remove selectedJob dependency to prevent polling restart on job selection

  // Fetch experiments for New Job dropdown
  const fetchExperiments = useCallback(async () => {
    try {
      const response = await apiClient.listExperiments();
      setExperiments(response.experiments);
    } catch (err) {
      console.error('Error fetching experiments:', err);
    }
  }, []);

  // Fetch agents for New Job dropdown
  const fetchAgents = useCallback(async () => {
    try {
      const response = await apiClient.listAgents();
      setAgents(response.agents);
    } catch (err) {
      console.error('Error fetching agents:', err);
    }
  }, []);

  // Fetch job details (experiment and agent info)
  const fetchJobDetails = useCallback(async (job: JobInfo) => {
    setDetailsLoading(true);
    setExperimentError(null);
    setAgentError(null);
    setSelectedExperiment(null);
    setSelectedAgent(null);

    try {
      // Fetch experiment and agent details separately to handle individual failures
      const experimentPromise = apiClient.getExperiment(job.experiment_id).catch(err => {
        console.error('Error fetching experiment details:', err);
        const errorMsg = `Failed to load experiment details: ${err.message || err}`;
        setExperimentError(errorMsg);
        return null;
      });

      const agentPromise = apiClient.getAgent(job.agent_id).catch(err => {
        console.error('Error fetching agent details:', err);
        let errorMsg;
        if (err.message && err.message.includes('404')) {
          errorMsg = `Agent ${job.agent_id} was deregistered after this job was created`;
        } else {
          errorMsg = `Unable to load agent details: ${err.message || err}`;
        }
        setAgentError(errorMsg);
        return null;
      });

      const [experimentResponse, agentResponse] = await Promise.all([
        experimentPromise,
        agentPromise
      ]);

      if (experimentResponse) {
        setSelectedExperiment(experimentResponse.experiment);
      }
      if (agentResponse) {
        setSelectedAgent(agentResponse.agent);
      }
    } catch (err) {
      console.error('Unexpected error fetching job details:', err);
      // This catch is for any unexpected errors not handled above
    } finally {
      setDetailsLoading(false);
    }
  }, []);

  // Handle row selection
  const handleRowClick = (job: JobTableRow) => {
    if (isCreatingJob) return; // Don't change selection when creating job

    // Clear previous details states
    setExperimentError(null);
    setAgentError(null);

    setSelectedJob(job.jobInfo);
    fetchJobDetails(job.jobInfo);
  };

  // Handle sort
  const handleSort = (column: keyof JobTableRow) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('desc');
    }
  };

  // Sort jobs
  const sortedJobs = [...jobs].sort((a, b) => {
    const aVal = a[sortColumn];
    const bVal = b[sortColumn];
    
    let comparison = 0;
    if (aVal < bVal) comparison = -1;
    if (aVal > bVal) comparison = 1;
    
    return sortDirection === 'asc' ? comparison : -comparison;
  });

  // Cancel job with confirmation
  const handleCancelJob = async (jobId: string, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent row selection
    
    if (!confirm(`Are you sure you want to cancel job ${jobId}?`)) {
      return;
    }

    try {
      await apiClient.cancelJob(jobId);
      showToast('Job cancelled successfully', 'success');
      
      // Refresh jobs list
      await fetchJobsData();
      
      // If this was the selected job, clear details
      if (selectedJob && selectedJob.job_id === jobId) {
        setSelectedJob(null);
        setSelectedExperiment(null);
        setSelectedAgent(null);
      }
    } catch (err) {
      console.error('Error cancelling job:', err);
      showToast('Failed to cancel job', 'error');
    }
  };

  // Check if cancel button should be enabled
  const canCancelJob = (status: string): boolean => {
    const cancelableStatuses = ['RUNNING', 'ASSIGNED', 'QUEUED'];
    return cancelableStatuses.includes(status);
  };

  // Handle New Job button
  const handleNewJob = () => {
    setIsCreatingJob(true);
    setSelectedJob(null);
    setSelectedExperiment(null);
    setSelectedAgent(null);
    setNewJobExperimentId('');
    setNewJobAgentId('');
  };

  // Handle experiment selection in New Job mode
  const handleExperimentSelect = (experimentId: string) => {
    setNewJobExperimentId(experimentId);
    const experiment = experiments.find(exp => exp.id === experimentId);
    setSelectedExperiment(experiment || null);
    setNewJobAgentId(''); // Reset agent selection
  };

  // Get filtered agents for the selected experiment
  const getFilteredAgents = (): RegisteredAgentInfo[] => {
    if (!selectedExperiment) return agents;
    
    // Debug logging
    console.log('=== Agent Filtering Debug ===');
    console.log('Selected experiment:', selectedExperiment);
    console.log('Selected experiment model (raw):', `"${selectedExperiment.model}"`);
    console.log('Selected experiment framework (raw):', selectedExperiment.framework);
    console.log('Selected experiment framework name (raw):', `"${selectedExperiment.framework.name}"`);
    console.log('Available agents count:', agents.length);
    
    const expectedModel = selectedExperiment.model.toLowerCase().trim();
    const expectedFramework = selectedExperiment.framework.name.toLowerCase().trim();
    console.log('Expected model (processed):', `"${expectedModel}"`);
    console.log('Expected framework (processed):', `"${expectedFramework}"`);
    
    const filteredAgents = agents.filter(agent => {
      console.log('\n--- Checking agent:', agent.agent_id);
      console.log('Full agent object:', JSON.stringify(agent, null, 2));
      
      // Check multiple possible structures based on description
      let modelMatches = false;
      let frameworkMatches = false;
      
      // Try structure 1: agent.system_info.models and agent.system_info.frameworks
      console.log('Checking structure 1: agent.system_info.models and agent.system_info.frameworks');
      console.log('agent.system_info exists:', !!agent.system_info);
      console.log('agent.system_info.models exists:', !!agent.system_info?.models);
      console.log('agent.system_info.frameworks exists:', !!agent.system_info?.frameworks);
      
      // Check models in system_info.models (object with model names as keys)
      if (agent.system_info?.models && typeof agent.system_info.models === 'object') {
        console.log('Agent models (system_info):', Object.keys(agent.system_info.models));
        
        // Check if the expected model exists as a key in the models object
        const modelKeys = Object.keys(agent.system_info.models);
        for (const modelKey of modelKeys) {
          const modelKeyProcessed = modelKey.toLowerCase().trim();
          console.log(`  Checking model key: "${modelKey}" -> "${modelKeyProcessed}" vs "${expectedModel}"`);
          if (modelKeyProcessed === expectedModel) {
            modelMatches = true;
            console.log(`🔍 MODEL MATCH FOUND: "${modelKey}" matches "${selectedExperiment.model}"`);
            break;
          }
        }
        
        if (!modelMatches) {
          console.log('❌ No model key matches in system_info.models');
        }
      } else {
        console.log('❌ agent.system_info.models is not available or not an object');
      }

      // Special case: Hardware benchmarks don't require specific models
      if (!modelMatches && (expectedModel === 'hardware-test' || expectedModel === '' || !expectedModel)) {
        console.log('🔧 HARDWARE BENCHMARK detected - skipping model requirement');
        modelMatches = true; // Hardware benchmarks can run on any agent with the right framework
      }
      
      // Check frameworks in system_info.frameworks (array of framework objects)
      if (agent.system_info?.frameworks && Array.isArray(agent.system_info.frameworks)) {
        console.log('Agent frameworks (system_info):', agent.system_info.frameworks);
        
        for (let i = 0; i < agent.system_info.frameworks.length; i++) {
          const framework = agent.system_info.frameworks[i];
          if (framework && framework.name) {
            const frameworkName = framework.name.toLowerCase().trim();
            console.log(`  Framework ${i}: "${framework.name}" -> "${frameworkName}" vs "${expectedFramework}"`);
            if (frameworkName === expectedFramework) {
              frameworkMatches = true;
              console.log(`🔍 FRAMEWORK MATCH FOUND: "${framework.name}" matches "${selectedExperiment.framework.name}"`);
              break;
            }
          }
        }
        
        if (!frameworkMatches) {
          console.log('❌ No framework name matches in system_info.frameworks');
        }
      } else {
        console.log('❌ agent.system_info.frameworks is not available or not an array');
      }
      
      // Try structure 2: agent.capabilities.models[] and agent.capabilities.frameworks[]
      if (!modelMatches && agent.capabilities) {
        console.log('\nTrying structure 2: capabilities');
        console.log('agent.capabilities:', agent.capabilities);
        console.log('agent.capabilities.models exists:', !!agent.capabilities.models);
        console.log('agent.capabilities.frameworks exists:', !!agent.capabilities.frameworks);
        
        if (agent.capabilities.models && Array.isArray(agent.capabilities.models)) {
          console.log('Agent models (structure 2):', agent.capabilities.models);
          agent.capabilities.models.forEach((model: string, index: number) => {
            const processedModel = model.toLowerCase().trim();
            const matches = processedModel === expectedModel;
            console.log(`  Model ${index}: "${model}" -> "${processedModel}" matches "${expectedModel}": ${matches}`);
            if (matches) modelMatches = true;
          });
          console.log('Model matches (structure 2):', modelMatches);
        }
        
        if (!frameworkMatches && agent.capabilities.frameworks && Array.isArray(agent.capabilities.frameworks)) {
          console.log('Agent frameworks (structure 2):', agent.capabilities.frameworks);
          agent.capabilities.frameworks.forEach((framework: string, index: number) => {
            const processedFramework = framework.toLowerCase().trim();
            const matches = processedFramework === expectedFramework;
            console.log(`  Framework ${index}: "${framework}" -> "${processedFramework}" matches "${expectedFramework}": ${matches}`);
            if (matches) frameworkMatches = true;
          });
          console.log('Framework matches (structure 2):', frameworkMatches);
        }
      }
      
      console.log('\n🏁 FINAL RESULT for agent', agent.agent_id);
      console.log('  modelMatches:', modelMatches);
      console.log('  frameworkMatches:', frameworkMatches);
      console.log('  Agent included:', modelMatches && frameworkMatches);
      console.log('='.repeat(50));
      
      return modelMatches && frameworkMatches;
    });
    
    console.log('\n📊 FILTERING SUMMARY');
    console.log('Total agents:', agents.length);
    console.log('Filtered agents:', filteredAgents.length);
    console.log('Filtered agent IDs:', filteredAgents.map(a => a.agent_id));
    
    return filteredAgents;
  };

  // Handle agent selection in New Job mode
  const handleAgentSelect = (agentId: string) => {
    setNewJobAgentId(agentId);
    const agent = agents.find(a => a.agent_id === agentId);
    setSelectedAgent(agent || null);
  };

  // Submit new job
  const handleSubmitJob = async () => {
    if (!newJobExperimentId || !newJobAgentId) {
      showToast('Please select both experiment and agent', 'error');
      return;
    }

    try {
      await apiClient.createJob({
        experiment_id: newJobExperimentId,
        agent_id: newJobAgentId
      });
      
      showToast('Job created successfully', 'success');
      setIsCreatingJob(false);
      await fetchJobsData();
    } catch (err) {
      console.error('Error creating job:', err);
      showToast('Failed to create job', 'error');
    }
  };

  // Cancel new job creation
  const handleCancelNewJob = () => {
    setIsCreatingJob(false);
    setNewJobExperimentId('');
    setNewJobAgentId('');
    setSelectedExperiment(null);
    setSelectedAgent(null);
  };

  // Resizable splitter handlers
  const handleSplitterMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return;

    const container = document.querySelector('.jobs-tab');
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const mouseX = e.clientX - containerRect.left;
    const containerWidth = containerRect.width;

    // Calculate new split percentage, with bounds checking
    let newPercentage = (mouseX / containerWidth) * 100;
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
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';

      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  // Update selected job when jobs list changes (separate from polling to avoid restart)
  useEffect(() => {
    if (selectedJob && jobs.length > 0) {
      const updatedSelectedJobRow = jobs.find(row => row.job_id === selectedJob.job_id);
      if (updatedSelectedJobRow) {
        const updatedSelectedJob = updatedSelectedJobRow.jobInfo;
        // Only refresh details if the status actually changed
        if (updatedSelectedJob.status.status !== selectedJob.status.status) {
          console.log(`Job ${selectedJob.job_id} status changed from ${selectedJob.status.status} to ${updatedSelectedJob.status.status}`);
          // Refresh job details only when status changes, not on every poll
          setTimeout(() => fetchJobDetails(updatedSelectedJob), 100);
        }
        setSelectedJob(updatedSelectedJob);
      } else {
        // Selected job no longer exists, clear selection
        console.log(`Selected job ${selectedJob.job_id} no longer exists`);
        setSelectedJob(null);
        setSelectedExperiment(null);
        setSelectedAgent(null);
      }
    }
  }, [jobs, selectedJob, fetchJobDetails]);

  // Note: Job details will be refreshed automatically when job status changes
  // This provides a good balance between responsiveness and not flooding the logs

  // Initial data fetch (experiments and agents don't need frequent polling)
  useEffect(() => {
    fetchExperiments();
    fetchAgents();
  }, []); // Only fetch once on component mount

  // Jobs polling setup (only jobs need frequent updates)
  useEffect(() => {
    fetchJobsData();

    const interval = setInterval(fetchJobsData, pollingInterval);
    return () => clearInterval(interval);
  }, [fetchJobsData, pollingInterval]);

  return (
    <div className="jobs-tab">
      {/* Toast notifications */}
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          {toast.message}
        </div>
      )}

      {/* Main content area with proper flex layout */}
      <div style={{
        display: 'flex',
        flexDirection: selectedJob || isCreatingJob ? 'row' : 'column',
        height: '100%',
        minHeight: 0,
        position: 'relative'
      }}>
        {/* Jobs table section */}
        <div className="jobs-section" style={{
          flex: (selectedJob || isCreatingJob) ? `0 0 ${splitPercentage}%` : '1',
          minHeight: 0,
          display: 'flex',
          flexDirection: 'column',
          marginRight: (selectedJob || isCreatingJob) ? '0' : '0'
        }}>
          <div className="section-header">
            <h2>Jobs</h2>
            <button
              className="new-job-button"
              onClick={handleNewJob}
              disabled={isCreatingJob}
            >
              New Job
            </button>
          </div>

          {loading && <div className="loading">Loading jobs...</div>}
          {error && <div className="error">{error}</div>}

          {!loading && !error && (
            <div className="table-container" style={{
              flex: 1,
              minHeight: 0,
              overflowY: 'auto'
            }}>
              <table className="jobs-table">
                <thead>
                  <tr>
                    <th onClick={() => handleSort('job_id')} className="sortable">
                      Job ID {sortColumn === 'job_id' && (sortDirection === 'asc' ? '↑' : '↓')}
                    </th>
                    <th onClick={() => handleSort('experiment_id')} className="sortable">
                      Experiment ID {sortColumn === 'experiment_id' && (sortDirection === 'asc' ? '↑' : '↓')}
                    </th>
                    <th onClick={() => handleSort('agent_id')} className="sortable">
                      Agent ID {sortColumn === 'agent_id' && (sortDirection === 'asc' ? '↑' : '↓')}
                    </th>
                    <th onClick={() => handleSort('status')} className="sortable">
                      Status {sortColumn === 'status' && (sortDirection === 'asc' ? '↑' : '↓')}
                    </th>
                    <th onClick={() => handleSort('updated')} className="sortable">
                      Updated {sortColumn === 'updated' && (sortDirection === 'asc' ? '↑' : '↓')}
                    </th>
                    <th>Cancel</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedJobs.map((job) => (
                    <tr
                      key={job.job_id}
                      onClick={() => handleRowClick(job)}
                      className={selectedJob?.job_id === job.job_id ? 'selected' : ''}
                    >
                      <td>{job.job_id}</td>
                      <td>{job.experiment_id}</td>
                      <td>{job.agent_id}</td>
                      <td>
                        <span className={`status status-${job.status.toLowerCase()}`}>
                          {job.status}
                        </span>
                      </td>
                      <td>{job.updated}</td>
                      <td>
                        <button
                          className="btn btn-danger btn-sm"
                          onClick={(e) => handleCancelJob(job.job_id, e)}
                          disabled={!canCancelJob(job.status)}
                        >
                          Cancel
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Resizable splitter */}
        {(selectedJob || isCreatingJob) && (
          <div
            className="splitter"
            onMouseDown={handleSplitterMouseDown}
            style={{
              width: '4px',
              backgroundColor: isDragging ? '#007bff' : '#ddd',
              cursor: 'col-resize',
              flexShrink: 0,
              transition: isDragging ? 'none' : 'background-color 0.2s ease',
              borderRadius: '2px',
              margin: '0 2px',
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
        {isCreatingJob && (
          <div className="details-panel" style={{
            flex: `0 0 ${100 - splitPercentage}%`,
            minHeight: 0,
            overflowY: 'auto'
          }}>
            <div className="new-job-form">
              <div className="form-header">
                <h3>Create New Job</h3>
                <div className="form-actions">
                  <button className="btn btn-secondary" onClick={handleCancelNewJob}>
                    Cancel
                  </button>
                  <button
                    className="btn btn-primary"
                    onClick={handleSubmitJob}
                    style={{
                      backgroundColor: '#28a745',
                      borderColor: '#28a745',
                      fontWeight: 'bold',
                      fontSize: '16px',
                      padding: '12px 24px',
                      boxShadow: '0 4px 8px rgba(40, 167, 69, 0.3)',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = '#218838';
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 6px 12px rgba(40, 167, 69, 0.4)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = '#28a745';
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = '0 4px 8px rgba(40, 167, 69, 0.3)';
                    }}
                  >
                    🚀 Submit Job
                  </button>
                </div>
              </div>

              <div className="form-content">
                <div className="form-section">
                  <div className="form-field">
                    <label>Experiment:</label>
                    <select
                      value={newJobExperimentId}
                      onChange={(e) => handleExperimentSelect(e.target.value)}
                    >
                      <option value="">Select an experiment...</option>
                      {experiments.map(exp => (
                        <option key={exp.id} value={exp.id}>{exp.name}</option>
                      ))}
                    </select>
                  </div>

                  <div className="form-field">
                    <label>Agent:</label>
                    <select
                      value={newJobAgentId}
                      onChange={(e) => handleAgentSelect(e.target.value)}
                      disabled={!newJobExperimentId}
                    >
                      <option value="">Select an agent...</option>
                      {getFilteredAgents().map(agent => (
                        <option key={agent.agent_id} value={agent.agent_id}>
                          {agent.agent_id}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Show selected experiment/agent details in New Job mode */}
              {(selectedExperiment || selectedAgent) && (
                <div className="job-details">
                  <div className="details-grid">
                    {selectedExperiment && (
                      <div className="experiment-details">
                        <div className="details-header">
                          <h3>Experiment Details</h3>
                          <span className="details-id">{selectedExperiment.id}</span>
                        </div>
                        <div className="details-content">
                          <textarea
                            className="details-textarea"
                            value={formatAgentDetails('Experiment', selectedExperiment)}
                            readOnly
                          />
                        </div>
                      </div>
                    )}

                    {selectedAgent && (
                      <div className="agent-details">
                        <div className="details-header">
                          <h3>Agent Details</h3>
                          <span className="details-id">{selectedAgent.agent_id}</span>
                        </div>
                        <div className="details-content">
                          <textarea
                            className="details-textarea"
                            value={formatAgentDetails('Agent', selectedAgent)}
                            readOnly
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Job details panel - shown alongside jobs table when selected */}
        {selectedJob && (
          <div className="details-panel" style={{
            flex: `0 0 ${100 - splitPercentage}%`,
            minHeight: 0,
            overflowY: 'auto'
          }}>
            <div className="job-details">
              {/* Job Status Summary */}
              <div className="job-status-summary" style={{
                marginBottom: '20px',
                padding: '15px',
                backgroundColor: 'var(--bg-secondary, #f9f9f9)',
                borderRadius: '8px',
                border: selectedJob.status.status === 'failed'
                  ? '2px solid var(--error-color, #dc3545)'
                  : '1px solid var(--border-color, #ddd)',
                position: 'relative'
              }}>
                <button
                  onClick={() => setSelectedJob(null)}
                  style={{
                    position: 'absolute',
                    top: '10px',
                    right: '10px',
                    background: 'none',
                    border: 'none',
                    fontSize: '20px',
                    cursor: 'pointer',
                    color: 'var(--text-secondary)',
                    padding: '5px',
                    borderRadius: '4px',
                    transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = 'var(--bg-tertiary, #e5e7eb)';
                    e.currentTarget.style.color = 'var(--text-primary)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                    e.currentTarget.style.color = 'var(--text-secondary)';
                  }}
                  title="Close job details"
                >
                  ✕
                </button>
                <h3>Job Status: {selectedJob.status.status.toUpperCase()}</h3>
                <p><strong>Job ID:</strong> {selectedJob.job_id}</p>
                <p><strong>Updated:</strong> {formatTimestamp(selectedJob.status.timestamp)}</p>
                {selectedJob.status.message && (
                  <p><strong>Status Message:</strong> {selectedJob.status.message}</p>
                )}
                {selectedJob.status.status === 'failed' && (
                  <div style={{
                    marginTop: '15px',
                    padding: '12px',
                    backgroundColor: 'var(--error-bg, #f8d7da)',
                    color: 'var(--error-text, #721c24)',
                    borderRadius: '6px',
                    fontWeight: 'bold'
                  }}>
                    ⚠️ This job failed. Check the <strong>Results tab</strong> for detailed error information.
                  </div>
                )}
              </div>
              <div className="details-grid">
                <div className="experiment-details">
                  <div className="details-header">
                    <h3>Experiment Details</h3>
                    <span className="details-id">{selectedJob.experiment_id}</span>
                  </div>
                  <div className="details-content">
                    <textarea
                      className="details-textarea"
                      value={
                        experimentError ? `Unable to load experiment details: ${experimentError}` :
                        selectedExperiment ? formatAgentDetails('Experiment', selectedExperiment) :
                        detailsLoading ? 'Loading experiment details...' :
                        'No experiment data available'
                      }
                      readOnly
                    />
                  </div>
                </div>

                <div className="agent-details">
                  <div className="details-header">
                    <h3>Agent Details</h3>
                    <span className="details-id">{selectedJob.agent_id}</span>
                  </div>
                  <div className="details-content">
                    <textarea
                      className="details-textarea"
                      value={
                        agentError ?
                          `${agentError}\n\nThis is normal if the agent was removed from the cluster after the job completed.\n\nJob Information:\n• Agent ID: ${selectedJob.agent_id}\n• Job Status: ${selectedJob.status.status.toUpperCase()}\n• Last Updated: ${formatTimestamp(selectedJob.status.timestamp)}\n• Experiment: ${selectedJob.experiment_id}` :
                        selectedAgent ? formatAgentDetails('Agent', selectedAgent) :
                        detailsLoading ? 'Loading agent details...' :
                        'No agent data available'
                      }
                      readOnly
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default JobsTab;
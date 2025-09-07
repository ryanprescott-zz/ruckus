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
const getPollingInterval = (): number => {
  const envInterval = import.meta.env.VITE_JOBS_POLLING_INTERVAL;
  return envInterval ? parseInt(envInterval, 10) * 1000 : DEFAULT_POLLING_INTERVAL;
};

const JobsTab: React.FC = () => {
  // State
  const [jobs, setJobs] = useState<JobTableRow[]>([]);
  const [selectedJob, setSelectedJob] = useState<JobInfo | null>(null);
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentSpec | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<RegisteredAgentInfo | null>(null);
  
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
      setError(null);
    } catch (err) {
      console.error('Error fetching jobs:', err);
      setError('Failed to fetch jobs');
    } finally {
      setLoading(false);
    }
  }, []);

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
    try {
      const [experimentResponse, agentResponse] = await Promise.all([
        apiClient.getExperiment(job.experiment_id),
        apiClient.getAgent(job.agent_id)
      ]);
      
      setSelectedExperiment(experimentResponse.experiment);
      setSelectedAgent(agentResponse.agent);
    } catch (err) {
      console.error('Error fetching job details:', err);
      setSelectedExperiment(null);
      setSelectedAgent(null);
    }
  }, []);

  // Handle row selection
  const handleRowClick = (job: JobTableRow) => {
    if (isCreatingJob) return; // Don't change selection when creating job
    
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

  // Initial data fetch and polling setup
  useEffect(() => {
    fetchJobsData();
    fetchExperiments();
    fetchAgents();

    const interval = setInterval(fetchJobsData, pollingInterval);
    return () => clearInterval(interval);
  }, [fetchJobsData, fetchExperiments, fetchAgents, pollingInterval]);

  return (
    <div className="jobs-tab">
      {/* Toast notifications */}
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          {toast.message}
        </div>
      )}

      {/* Jobs table section */}
      <div className="jobs-section">
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
          <div className="table-container">
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

      {/* Details panel */}
      <div className="details-panel">
        {isCreatingJob ? (
          // New Job form
          <div className="new-job-form">
            <div className="form-header">
              <h3>Create New Job</h3>
              <div className="form-actions">
                <button className="btn btn-secondary" onClick={handleCancelNewJob}>
                  Cancel
                </button>
                <button className="btn btn-primary" onClick={handleSubmitJob}>
                  Submit
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
          </div>
        ) : (
          // Job details view
          selectedJob && (
            <div className="job-details">
              <div className="details-grid">
                <div className="experiment-details">
                  <div className="details-header">
                    <h3>Experiment Details</h3>
                    <span className="details-id">{selectedJob.experiment_id}</span>
                  </div>
                  <div className="details-content">
                    <textarea
                      className="details-textarea"
                      value={selectedExperiment ? formatAgentDetails('Experiment', selectedExperiment) : 'Loading...'}
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
                      value={selectedAgent ? formatAgentDetails('Agent', selectedAgent) : 'Loading...'}
                      readOnly
                    />
                  </div>
                </div>
              </div>
            </div>
          )
        )}
        
        {/* Show selected experiment/agent details in New Job mode */}
        {isCreatingJob && (selectedExperiment || selectedAgent) && (
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
  );
};

export default JobsTab;
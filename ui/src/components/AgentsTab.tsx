import React, { useState, useEffect, useCallback } from 'react';
import type { 
  RegisteredAgentInfo, 
  AgentStatus, 
  AgentTableRow,
  AgentCompatibilityMatrixResponse
} from '../types/api';
import { AgentStatusEnum } from '../types/api';
import { apiClient } from '../services/api';
import { formatUptime, formatTimestamp, formatAgentDetails } from '../utils/format';
import './AgentsTab.css';

// Configuration
const POLLING_INTERVAL_MS = 3000; // 3 seconds


const AgentsTab: React.FC = () => {
  // State
  const [agents, setAgents] = useState<AgentTableRow[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<RegisteredAgentInfo | null>(null);
  const [compatibilityMatrix, setCompatibilityMatrix] = useState<AgentCompatibilityMatrixResponse | null>(null);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [registerUrl, setRegisterUrl] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);
  const [toast, setToast] = useState<{message: string, type: 'success' | 'error'} | null>(null);
  const [wasDisconnected, setWasDisconnected] = useState(false);


  // Fetch agent data and merge with status
  const fetchAgentData = useCallback(async () => {
    try {
      const [agentInfoResponse, agentStatusResponse] = await Promise.all([
        apiClient.listAgents(),
        apiClient.listAgentStatus()
      ]);
      
      // Fetch compatibility matrix to show capabilities
      try {
        const compatibilityResponse = await apiClient.getAgentCompatibilityMatrix();
        setCompatibilityMatrix(compatibilityResponse);
      } catch (compatError) {
        console.warn('Failed to fetch compatibility matrix:', compatError);
        // Don't fail the whole request if compatibility matrix fails
        setCompatibilityMatrix(null);
      }

      const agentInfo = agentInfoResponse.agents;
      const agentStatus = agentStatusResponse.agents;

      // Create a map of status by agent ID
      const statusMap = new Map<string, AgentStatus>();
      agentStatus.forEach(status => {
        statusMap.set(status.agent_id, status);
      });

      // Merge agent info with status
      const tableRows: AgentTableRow[] = agentInfo.map((agent, idx) => {
        const status = statusMap.get(agent.agent_id);
        
        // Get jobs display - show first running job ID or empty
        let jobsDisplay = '';
        if (status?.running_jobs && status.running_jobs.length > 0) {
          jobsDisplay = status.running_jobs[0];
        }
        
        const row = {
          id: agent.agent_id,
          name: agent.agent_name || agent.agent_id,
          status: status?.status || AgentStatusEnum.UNAVAILABLE,
          jobs: jobsDisplay,
          uptime: status ? formatUptime(status.uptime_seconds) : 'UNKNOWN',
          lastStatusChange: formatTimestamp(agent.last_updated),
          unregister: agent.agent_id, // For the unregister action column
          agent
        };
        
        // Debug to check actual data
        if (idx === 0) {
          console.log('Raw agent from API:', agent);
          console.log('Raw status from API:', status);
          console.log('Created table row:', row);
          console.log('Expected data mapping:');
          console.log('  id =', row.id);
          console.log('  name =', row.name);
          console.log('  status =', row.status);
          console.log('  jobs =', row.jobs);
          console.log('  uptime =', row.uptime);
          console.log('  lastStatusChange =', row.lastStatusChange);
        }
        
        return row;
      });

      setAgents(tableRows);
      setError(null);
      
      // Check if connection was restored after being disconnected
      if (wasDisconnected) {
        setWasDisconnected(false);
        showToast('Server connection restored', 'success');
      }
    } catch (err) {
      console.error('Failed to fetch agent data:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      
      // Check if it's a connection error
      if (errorMessage.includes('CONNECTION_ERROR')) {
        setWasDisconnected(true);
        showToast('Cannot connect to server. Please check if the server is running.', 'error');
        setError('Cannot connect to server');
      } else {
        setError(`Failed to fetch agent data: ${errorMessage}`);
      }
    } finally {
      setLoading(false);
    }
  }, [wasDisconnected]);

  // Initial load and polling setup
  useEffect(() => {
    fetchAgentData();
    
    const interval = setInterval(fetchAgentData, POLLING_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchAgentData]);

  // Clear selected agent when agents list becomes empty
  useEffect(() => {
    if (agents.length === 0 && selectedAgent) {
      setSelectedAgent(null);
    }
  }, [agents.length, selectedAgent]);

  // Handle agent row selection
  const handleAgentRowClick = (agent: RegisteredAgentInfo) => {
    setSelectedAgent(agent);
  };

  // Handle agent unregistration
  const handleUnregisterAgent = async (agentId: string, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent row selection
    
    try {
      await apiClient.unregisterAgent({ agent_id: agentId });
      // Data will be updated on next poll
      showToast('Agent unregistered successfully', 'success');
    } catch (err) {
      console.error('Failed to unregister agent:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      showToast(`Failed to unregister agent: ${errorMessage}`, 'error');
    }
  };

  // Handle agent registration
  const handleRegisterAgent = async () => {
    if (!registerUrl.trim()) return;

    setIsRegistering(true);
    try {
      await apiClient.registerAgent({ agent_url: registerUrl.trim() });
      setRegisterUrl('');
      // Data will be updated on next poll
      setError(null);
      
      // Check if connection was restored after being disconnected
      if (wasDisconnected) {
        setWasDisconnected(false);
        showToast('Server connection restored', 'success');
      }
    } catch (err) {
      console.error('Failed to register agent:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      
      // Check if it's a connection error
      if (errorMessage.includes('CONNECTION_ERROR')) {
        setWasDisconnected(true);
        showToast('Cannot connect to server. Please check if the server is running.', 'error');
        setError('Cannot connect to server');
      } else {
        const fullErrorMessage = `Failed to register agent: ${errorMessage}`;
        setError(fullErrorMessage);
        showToast(fullErrorMessage, 'error');
      }
    } finally {
      setIsRegistering(false);
    }
  };

  // Toast notification function
  const showToast = (message: string, type: 'success' | 'error') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000); // Hide after 3 seconds
  };

  // Helper functions for displaying hardware info
  const getAgentHardwareSummary = (agent: RegisteredAgentInfo): string => {
    const gpus = agent.system_info?.gpus || [];
    if (gpus.length === 0) {
      return 'CPU Only';
    }
    
    const primaryGpu = gpus[0];
    const memoryGB = primaryGpu.memory_total_mb ? (primaryGpu.memory_total_mb / 1024).toFixed(0) + 'GB' : '';
    return `${primaryGpu.name}${memoryGB ? ` (${memoryGB})` : ''}`;
  };

  const getAgentFrameworks = (agent: RegisteredAgentInfo): string => {
    const frameworks = agent.system_info?.frameworks || [];
    if (frameworks.length === 0) {
      return 'None';
    }
    return frameworks.map(fw => fw.name).join(', ');
  };

  const getAgentModels = (agent: RegisteredAgentInfo): string => {
    const models = agent.system_info?.models || {};
    const loadedModels = Object.entries(models)
      .filter(([, modelInfo]) => modelInfo.loaded)
      .map(([modelKey]) => modelKey);
    
    if (loadedModels.length === 0) {
      return 'None loaded';
    }
    return loadedModels.join(', ');
  };

  const getAgentCapabilities = (agent: RegisteredAgentInfo): string => {
    const capabilities = agent.capabilities || {};
    const enabledCapabilities = Object.entries(capabilities)
      .filter(([, enabled]) => enabled)
      .map(([capability]) => capability.replace(/_/g, ' '));
    
    if (enabledCapabilities.length === 0) {
      return 'None';
    }
    return enabledCapabilities.join(', ');
  };


  // Handle key press in register input
  const handleRegisterKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleRegisterAgent();
    }
  };


  if (loading && agents.length === 0) {
    return (
      <div className="agents-tab">
        <div className="loading">Loading agents...</div>
      </div>
    );
  }

  return (
    <div className="agents-tab">
      {/* Toast Notification */}
      {toast && (
        <div className={`toast toast-${toast.type}`}>
          <div className="toast-content">
            <span className="toast-icon">
              {toast.type === 'success' ? '✓' : '⚠'}
            </span>
            <span className="toast-message">{toast.message}</span>
            <button 
              className="toast-close"
              onClick={() => setToast(null)}
              aria-label="Close notification"
            >
              ×
            </button>
          </div>
        </div>
      )}
      
      {/* Agent Registration Section */}
      <div className="agent-registration">
        <label htmlFor="agent-url">URL</label>
        <input
          id="agent-url"
          type="text"
          value={registerUrl}
          onChange={(e) => setRegisterUrl(e.target.value)}
          onKeyPress={handleRegisterKeyPress}
          placeholder="Enter agent URL..."
          disabled={isRegistering}
        />
        <button
          className="register-button"
          onClick={handleRegisterAgent}
          disabled={isRegistering || !registerUrl.trim()}
        >
          {isRegistering ? 'Registering...' : 'Register Agent'}
        </button>
      </div>

      {/* Error Display */}
      {error && <div className="error-message">{error}</div>}

      {/* Agents Table */}
      <div className="agents-table-container">
        <table className="agents-table">
          <thead>
            <tr>
              <th>Id</th>
              <th>Name</th>
              <th>Hardware</th>
              <th>Models</th>
              <th>Status</th>
              <th>Jobs</th>
              <th>Uptime</th>
              <th>Last Status Change</th>
              <th>Unregister</th>
            </tr>
          </thead>
          <tbody>
            {agents.length === 0 ? (
              <tr>
                <td colSpan={9} className="empty-cell">
                  <div className="empty-content">
                    No agents registered
                  </div>
                </td>
              </tr>
            ) : (
              agents.map((agent) => (
                <tr
                  key={agent.id}
                  className={`agent-row ${selectedAgent?.agent_id === agent.id ? 'selected' : ''} ${agent.status.toLowerCase()}`}
                  onClick={() => handleAgentRowClick(agent.agent)}
                >
                  <td className="agent-id">{agent.id}</td>
                  <td className="agent-name">{agent.name}</td>
                  <td className="agent-hardware" title={getAgentHardwareSummary(agent.agent)}>
                    <span className="hardware-summary">{getAgentHardwareSummary(agent.agent)}</span>
                  </td>
                  <td className="agent-models" title={getAgentModels(agent.agent)}>
                    <span className="models-summary">{getAgentModels(agent.agent)}</span>
                  </td>
                  <td className="agent-status">
                    <span className={`status-badge status-${agent.status.toLowerCase()}`}>
                      {agent.status}
                    </span>
                  </td>
                  <td className="agent-jobs">
                    {agent.jobs ? (
                      <span className="job-link">{agent.jobs}</span>
                    ) : (
                      <span className="no-jobs">—</span>
                    )}
                  </td>
                  <td className="agent-uptime">{agent.uptime}</td>
                  <td className="agent-last-change">{agent.lastStatusChange}</td>
                  <td className="agent-unregister">
                    <button
                      className="unregister-button"
                      onClick={(e) => handleUnregisterAgent(agent.id, e)}
                      title="Unregister agent"
                    >
                      <span className="unregister-icon">⊖</span>
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Agent Details Panel - 2x3 Grid - Only show when agent is selected */}
      {selectedAgent && (
        <div className="agent-details-grid">
          <div className="agent-details-section">
            <h3>Agent Info</h3>
            <textarea
              readOnly
              value={[
                `Agent Id: ${selectedAgent.agent_id}`,
                `Agent Name: ${selectedAgent.agent_name || 'N/A'}`,
                `Agent Type: ${selectedAgent.agent_type}`,
                `Last Updated: ${formatTimestamp(selectedAgent.last_updated)}`,
                `Agent Url: ${selectedAgent.agent_url}`,
                `Registered At: ${formatTimestamp(selectedAgent.registered_at)}`
              ].join('\n')}
            />
          </div>

          <div className="agent-details-section">
            <h3>Hardware</h3>
            <textarea
              readOnly
              value={formatAgentDetails('Hardware Information', {
                system: selectedAgent.system_info?.system,
                cpu: selectedAgent.system_info?.cpu,
                gpus: selectedAgent.system_info?.gpus
              })}
            />
          </div>

          <div className="agent-details-section">
            <h3>Capabilities</h3>
            <textarea
              readOnly
              value={[
                `Enabled Capabilities:`,
                getAgentCapabilities(selectedAgent) || 'None',
                ``,
                `Available Frameworks:`,
                getAgentFrameworks(selectedAgent),
                ``,
                `Hardware Features:`,
                selectedAgent.system_info?.gpus?.length > 0 ? 'GPU Support' : 'CPU Only',
                selectedAgent.system_info?.metrics?.map((m: any) => m.name).join(', ') || 'No metrics available'
              ].join('\n')}
            />
          </div>

          <div className="agent-details-section">
            <h3>Models</h3>
            <textarea
              readOnly
              value={formatAgentDetails('Available Models', selectedAgent.system_info?.models || {})}
            />
          </div>

          <div className="agent-details-section">
            <h3>Frameworks</h3>
            <textarea
              readOnly
              value={formatAgentDetails('Supported Frameworks', selectedAgent.system_info?.frameworks || {})}
            />
          </div>

          <div className="agent-details-section">
            <h3>Compatibility</h3>
            <textarea
              readOnly
              value={compatibilityMatrix && compatibilityMatrix.agents[selectedAgent.agent_id] ? 
                [
                  `Experiment Compatibility:`,
                  ...Object.entries(compatibilityMatrix.agents[selectedAgent.agent_id].experiment_compatibility).map(([expType, compat]) => 
                    `• ${expType.replace(/_/g, ' ')}: ${compat.compatible ? '✅ Compatible' : '❌ Incompatible'}`
                  ),
                  ``,
                  `Total Experiment Types: ${compatibilityMatrix.experiment_types.length}`,
                  `Compatible Types: ${Object.values(compatibilityMatrix.agents[selectedAgent.agent_id].experiment_compatibility).filter(c => c.compatible).length}`,
                  `Last Checked: ${formatTimestamp(compatibilityMatrix.checked_at)}`
                ].join('\n')
                : 'Compatibility information not available'
              }
            />
          </div>
        </div>
      )}

    </div>
  );
};

export default AgentsTab;
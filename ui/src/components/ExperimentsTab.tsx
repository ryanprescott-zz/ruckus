import React, { useState, useEffect, useCallback } from 'react';
import type {
  ExperimentSpec,
  CreateExperimentSpec,
  ExperimentTableRow,
  LLMGenerationParams,
  GPUBenchmarkParams,
  MemoryBenchmarkParams,
  ComputeBenchmarkParams,
  PromptMessage,
  RegisteredAgentInfo
} from '../types/api';
import { TaskType, FrameworkName, PromptRole } from '../types/api';
import { apiClient } from '../services/api';
import { formatTimestamp } from '../utils/format';
import './ExperimentsTab.css';

// Configuration
const POLLING_INTERVAL_MS = 5000; // 5 seconds

type ParameterSection = 'task' | 'model' | 'runtime' | 'metrics';

const ExperimentsTab: React.FC = () => {
  // State
  const [experiments, setExperiments] = useState<ExperimentTableRow[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<ExperimentSpec | null>(null);
  const [agents, setAgents] = useState<RegisteredAgentInfo[]>([]);
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<{message: string, type: 'success' | 'error'} | null>(null);
  
  // New experiment mode
  const [isCreatingNew, setIsCreatingNew] = useState(false);
  const [newExperiment, setNewExperiment] = useState<CreateExperimentSpec | null>(null);
  const [selectedParameterSection, setSelectedParameterSection] = useState<ParameterSection>('task');

  // Fetch experiments and agents data
  const fetchData = useCallback(async () => {
    try {
      const [experimentsResponse, agentsResponse] = await Promise.all([
        apiClient.listExperiments(),
        apiClient.listAgents()
      ]);

      // Process experiments
      const experimentRows: ExperimentTableRow[] = experimentsResponse.experiments.map((exp) => ({
        id: exp.id,
        name: exp.name,
        jobs: '—', // TODO: Will be populated later with jobs data
        created: exp.created_at ? formatTimestamp(exp.created_at) : 'Unknown',
        remove: exp.id,
        experiment: exp
      }));

      setExperiments(experimentRows);
      setAgents(agentsResponse.agents);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch data:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      
      if (errorMessage.includes('CONNECTION_ERROR')) {
        showToast('Cannot connect to server. Please check if the server is running.', 'error');
        setError('Cannot connect to server');
      } else {
        setError(`Failed to fetch data: ${errorMessage}`);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load and polling setup
  useEffect(() => {
    fetchData();
    
    const interval = setInterval(fetchData, POLLING_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Handle experiment row selection
  const handleExperimentRowClick = async (experimentId: string) => {
    if (isCreatingNew) return; // Don't allow selection during creation
    
    try {
      const response = await apiClient.getExperiment(experimentId);
      setSelectedExperiment(response.experiment);
      setIsCreatingNew(false);
      setNewExperiment(null);
    } catch (err) {
      console.error('Failed to fetch experiment:', err);
      showToast('Failed to fetch experiment details', 'error');
    }
  };

  // Handle experiment removal
  const handleRemoveExperiment = async (experimentId: string, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent row selection
    
    try {
      await apiClient.deleteExperiment(experimentId);
      showToast('Experiment removed successfully', 'success');
      
      // Clear selection if the deleted experiment was selected
      if (selectedExperiment?.id === experimentId) {
        setSelectedExperiment(null);
      }
      
      // Refresh data
      fetchData();
    } catch (err) {
      console.error('Failed to remove experiment:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      showToast(`Failed to remove experiment: ${errorMessage}`, 'error');
    }
  };

  // Helper function to create default parameters based on task type
  const createDefaultParams = (taskType: TaskType) => {
    switch (taskType) {
      case TaskType.LLM_GENERATION:
        return {
          prompt_template: {
            messages: []
          }
        } as LLMGenerationParams;

      case TaskType.GPU_BENCHMARK:
        return {
          test_memory_bandwidth: true,
          test_compute_flops: true,
          test_tensor_cores: false,
          max_memory_usage_percent: 80.0,
          benchmark_duration_seconds: 30
        } as GPUBenchmarkParams;

      case TaskType.MEMORY_BENCHMARK:
        return {
          test_sizes_mb: [64, 256, 1024],
          test_patterns: ["sequential", "random"],
          iterations_per_size: 10
        } as MemoryBenchmarkParams;

      case TaskType.COMPUTE_BENCHMARK:
        return {
          precision_types: ["fp32", "fp16"],
          matrix_sizes: [1024, 2048, 4096],
          include_tensor_ops: true
        } as ComputeBenchmarkParams;

      default:
        return {} as any;
    }
  };

  // Helper function to determine appropriate model and framework defaults based on task type
  const getDefaultsForTaskType = (taskType: TaskType) => {
    switch (taskType) {
      case TaskType.LLM_GENERATION:
        return { model: '', framework: FrameworkName.TRANSFORMERS };

      case TaskType.GPU_BENCHMARK:
      case TaskType.MEMORY_BENCHMARK:
      case TaskType.COMPUTE_BENCHMARK:
        return { model: 'hardware-test', framework: FrameworkName.PYTORCH };

      default:
        return { model: '', framework: FrameworkName.PYTORCH };
    }
  };

  // Handle new experiment creation
  const handleNewExperiment = () => {
    const taskType = TaskType.LLM_GENERATION;
    const defaults = getDefaultsForTaskType(taskType);

    const emptyExperiment: CreateExperimentSpec = {
      name: '',
      description: '',
      model: defaults.model,
      task: {
        name: '',
        type: taskType,
        description: '',
        params: createDefaultParams(taskType)
      },
      framework: {
        name: defaults.framework,
        params: {}
      },
      metrics: {
        metrics: {}
      }
    };
    
    setIsCreatingNew(true);
    setNewExperiment(emptyExperiment);
    setSelectedExperiment(null);
    setSelectedParameterSection('task');
  };

  // Handle create experiment submission
  const handleCreateExperiment = async () => {
    if (!newExperiment || !newExperiment.name.trim()) {
      showToast('Please provide an experiment name', 'error');
      return;
    }

    try {
      await apiClient.createExperiment({ experiment_spec: newExperiment });
      showToast('Experiment created successfully', 'success');
      
      setIsCreatingNew(false);
      setNewExperiment(null);
      
      // Refresh data
      fetchData();
    } catch (err) {
      console.error('Failed to create experiment:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      showToast(`Failed to create experiment: ${errorMessage}`, 'error');
    }
  };

  // Handle cancel creation
  const handleCancelCreation = () => {
    setIsCreatingNew(false);
    setNewExperiment(null);
  };

  // Toast notification function
  const showToast = (message: string, type: 'success' | 'error') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000); // Hide after 3 seconds
  };

  // Get available models from all agents
  const getAvailableModels = (): string[] => {
    const modelSet = new Set<string>();
    agents.forEach(agent => {
      if (agent.system_info?.models) {
        Object.keys(agent.system_info.models).forEach(modelName => {
          modelSet.add(modelName);
        });
      }
    });
    return Array.from(modelSet).sort();
  };

  // Get available frameworks from all agents
  const getAvailableFrameworks = (): string[] => {
    const frameworkSet = new Set<string>();
    agents.forEach(agent => {
      if (agent.system_info?.frameworks && Array.isArray(agent.system_info.frameworks)) {
        agent.system_info.frameworks.forEach((framework: any) => {
          if (framework && framework.name) {
            frameworkSet.add(framework.name);
          }
        });
      }
    });
    return Array.from(frameworkSet).sort();
  };

  if (loading && experiments.length === 0) {
    return (
      <div className="experiments-tab">
        <div className="loading">Loading experiments...</div>
      </div>
    );
  }

  return (
    <div className="experiments-tab">
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
      
      {/* New Experiment Button */}
      <div className="experiment-actions">
        <button
          className="new-experiment-button"
          onClick={handleNewExperiment}
          disabled={isCreatingNew}
        >
          New Experiment
        </button>
      </div>

      {/* Error Display */}
      {error && <div className="error-message">{error}</div>}

      {/* Experiments Table */}
      <div className="experiments-table-container">
        <table className="experiments-table">
          <thead>
            <tr>
              <th>Id</th>
              <th>Name</th>
              <th>Jobs</th>
              <th>Created</th>
              <th>Remove</th>
            </tr>
          </thead>
          <tbody>
            {experiments.length === 0 ? (
              <tr>
                <td colSpan={5} className="empty-cell">
                  <div className="empty-content">
                    No experiments found
                  </div>
                </td>
              </tr>
            ) : (
              experiments.map((experiment) => (
                <tr
                  key={experiment.id}
                  className={`experiment-row ${selectedExperiment?.id === experiment.id ? 'selected' : ''}`}
                  onClick={() => handleExperimentRowClick(experiment.id)}
                >
                  <td className="experiment-id">{experiment.id}</td>
                  <td className="experiment-name">{experiment.name}</td>
                  <td className="experiment-jobs">
                    {experiment.jobs ? (
                      <span className="job-link">{experiment.jobs}</span>
                    ) : (
                      <span className="no-jobs">—</span>
                    )}
                  </td>
                  <td className="experiment-created">{experiment.created}</td>
                  <td className="experiment-remove">
                    <button
                      className="remove-button"
                      onClick={(e) => handleRemoveExperiment(experiment.id, e)}
                      title="Remove experiment"
                    >
                      <span className="remove-icon">⊖</span>
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Experiment Details/Creation Panel */}
      {(selectedExperiment || isCreatingNew) && (
        <div className="experiment-details-panel">
          <div className="experiment-basic-info">
            <div className="form-field">
              <label htmlFor="experiment-name">Name</label>
              <input
                id="experiment-name"
                type="text"
                value={isCreatingNew ? newExperiment?.name || '' : selectedExperiment?.name || ''}
                onChange={(e) => {
                  if (isCreatingNew && newExperiment) {
                    setNewExperiment({
                      ...newExperiment,
                      name: e.target.value,
                      // Auto-sync task name with experiment name
                      task: {
                        ...newExperiment.task,
                        name: e.target.value
                      }
                    });
                  }
                }}
                readOnly={!isCreatingNew}
              />
            </div>
            
            <div className="form-field">
              <label htmlFor="experiment-description">Description</label>
              <textarea
                id="experiment-description"
                value={isCreatingNew ? newExperiment?.description || '' : selectedExperiment?.description || ''}
                onChange={(e) => {
                  if (isCreatingNew && newExperiment) {
                    setNewExperiment({ ...newExperiment, description: e.target.value });
                  }
                }}
                readOnly={!isCreatingNew}
                rows={4}
              />
            </div>
          </div>

          <div className="experiment-parameters">
            <div className="parameters-sidebar">
              <h3>Parameters</h3>
              <div className="parameter-sections">
                {(['task', 'model', 'runtime', 'metrics'] as ParameterSection[]).map((section) => (
                  <button
                    key={section}
                    className={`parameter-section-button ${selectedParameterSection === section ? 'active' : ''}`}
                    onClick={() => setSelectedParameterSection(section)}
                  >
                    {section.charAt(0).toUpperCase() + section.slice(1)}
                  </button>
                ))}
              </div>
              
              {isCreatingNew && (
                <div className="creation-actions">
                  <button className="create-button" onClick={handleCreateExperiment}>
                    Create
                  </button>
                  <button className="cancel-button" onClick={handleCancelCreation}>
                    Cancel
                  </button>
                </div>
              )}
            </div>

            <div className="parameters-content">
              {selectedParameterSection === 'task' && (
                <div className="task-parameters">
                  <div className="form-field">
                    <label htmlFor="task-type">Type</label>
                    <select
                      id="task-type"
                      value={isCreatingNew ? newExperiment?.task.type || TaskType.LLM_GENERATION : selectedExperiment?.task.type || TaskType.LLM_GENERATION}
                      onChange={(e) => {
                        if (isCreatingNew && newExperiment) {
                          const newTaskType = e.target.value as TaskType;
                          const defaults = getDefaultsForTaskType(newTaskType);

                          setNewExperiment({
                            ...newExperiment,
                            model: defaults.model,
                            task: {
                              ...newExperiment.task,
                              type: newTaskType,
                              params: createDefaultParams(newTaskType)
                            },
                            framework: {
                              ...newExperiment.framework,
                              name: defaults.framework
                            }
                          });
                        }
                      }}
                      disabled={!isCreatingNew}
                    >
                      {Object.values(TaskType).map(type => (
                        <option key={type} value={type}>
                          {type.replace('_', ' ').toUpperCase()}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Dynamic Parameters Based on Task Type */}
                  <div className="task-specific-parameters">
                    {/* LLM Generation Parameters */}
                    {(isCreatingNew ? newExperiment?.task.type : selectedExperiment?.task.type) === TaskType.LLM_GENERATION && (
                      <div className="llm-parameters">
                        <h4>Prompt Messages</h4>
                        <div className="prompt-messages-table">
                          <table>
                            <thead>
                              <tr>
                                <th>Role</th>
                                <th>Content</th>
                                {isCreatingNew && <th>Action</th>}
                              </tr>
                            </thead>
                            <tbody>
                              {(isCreatingNew
                                ? (newExperiment?.task.params as LLMGenerationParams)?.prompt_template?.messages || []
                                : (selectedExperiment?.task.params as LLMGenerationParams)?.prompt_template?.messages || []
                              ).map((message, index) => (
                                <tr key={index}>
                                  <td>{message.role}</td>
                                  <td className="message-content">{message.content}</td>
                                  {isCreatingNew && (
                                    <td>
                                      <button
                                        className="delete-message-button"
                                        onClick={() => {
                                          if (newExperiment) {
                                            const currentMessages = (newExperiment.task.params as LLMGenerationParams).prompt_template.messages;
                                            const updatedMessages = currentMessages.filter((_, i) => i !== index);
                                            setNewExperiment({
                                              ...newExperiment,
                                              task: {
                                                ...newExperiment.task,
                                                params: {
                                                  prompt_template: {
                                                    ...((newExperiment.task.params as LLMGenerationParams).prompt_template),
                                                    messages: updatedMessages
                                                  }
                                                } as LLMGenerationParams
                                              }
                                            });
                                          }
                                        }}
                                        title="Remove message"
                                      >
                                        ⊖
                                      </button>
                                    </td>
                                  )}
                                </tr>
                              ))}
                            </tbody>
                          </table>

                          {isCreatingNew && (
                            <div className="add-message-section">
                              <select id="new-message-role">
                                {Object.values(PromptRole).map(role => (
                                  <option key={role} value={role}>
                                    {role.charAt(0).toUpperCase() + role.slice(1)}
                                  </option>
                                ))}
                              </select>
                              <input
                                type="text"
                                id="new-message-content"
                                placeholder="Message content..."
                              />
                              <button
                                className="add-message-button"
                                onClick={() => {
                                  const roleSelect = document.getElementById('new-message-role') as HTMLSelectElement;
                                  const contentInput = document.getElementById('new-message-content') as HTMLInputElement;

                                  if (newExperiment && contentInput.value.trim()) {
                                    const currentMessages = (newExperiment.task.params as LLMGenerationParams).prompt_template.messages;
                                    const newMessage: PromptMessage = {
                                      role: roleSelect.value as PromptRole,
                                      content: contentInput.value.trim()
                                    };

                                    setNewExperiment({
                                      ...newExperiment,
                                      task: {
                                        ...newExperiment.task,
                                        params: {
                                          prompt_template: {
                                            ...((newExperiment.task.params as LLMGenerationParams).prompt_template),
                                            messages: [...currentMessages, newMessage]
                                          }
                                        } as LLMGenerationParams
                                      }
                                    });

                                    contentInput.value = '';
                                  }
                                }}
                              >
                                Add
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* GPU Benchmark Parameters */}
                    {(isCreatingNew ? newExperiment?.task.type : selectedExperiment?.task.type) === TaskType.GPU_BENCHMARK && (
                      <div className="gpu-benchmark-parameters">
                        <h4>GPU Benchmark Configuration</h4>
                        <div className="benchmark-description">
                          <p><strong>What this measures:</strong> Comprehensive GPU performance analysis using real PyTorch operations</p>
                          <ul>
                            <li><strong>Memory Bandwidth:</strong> CPU→GPU transfer, GPU read/write speeds using tensor operations</li>
                            <li><strong>Compute FLOPS:</strong> Matrix multiplication performance in FP32, FP16, BF16 precisions</li>
                            <li><strong>Precision Scaling:</strong> Performance comparison across different data types</li>
                          </ul>
                          <p><strong>Use case:</strong> Overall GPU health check and performance characterization. Tests scale automatically based on available GPU memory (512×512 to ~50% of VRAM).</p>
                        </div>

                        <div className="form-field">
                          <label>
                            <input
                              type="checkbox"
                              checked={isCreatingNew
                                ? (newExperiment?.task.params as GPUBenchmarkParams)?.test_memory_bandwidth || false
                                : (selectedExperiment?.task.params as GPUBenchmarkParams)?.test_memory_bandwidth || false
                              }
                              onChange={(e) => {
                                if (isCreatingNew && newExperiment) {
                                  setNewExperiment({
                                    ...newExperiment,
                                    task: {
                                      ...newExperiment.task,
                                      params: {
                                        ...(newExperiment.task.params as GPUBenchmarkParams),
                                        test_memory_bandwidth: e.target.checked
                                      } as GPUBenchmarkParams
                                    }
                                  });
                                }
                              }}
                              disabled={!isCreatingNew}
                            />
                            Test Memory Bandwidth
                          </label>
                        </div>

                        <div className="form-field">
                          <label>
                            <input
                              type="checkbox"
                              checked={isCreatingNew
                                ? (newExperiment?.task.params as GPUBenchmarkParams)?.test_compute_flops || false
                                : (selectedExperiment?.task.params as GPUBenchmarkParams)?.test_compute_flops || false
                              }
                              onChange={(e) => {
                                if (isCreatingNew && newExperiment) {
                                  setNewExperiment({
                                    ...newExperiment,
                                    task: {
                                      ...newExperiment.task,
                                      params: {
                                        ...(newExperiment.task.params as GPUBenchmarkParams),
                                        test_compute_flops: e.target.checked
                                      } as GPUBenchmarkParams
                                    }
                                  });
                                }
                              }}
                              disabled={!isCreatingNew}
                            />
                            Test Compute FLOPS
                          </label>
                        </div>

                        <div className="form-field">
                          <label>
                            <input
                              type="checkbox"
                              checked={isCreatingNew
                                ? (newExperiment?.task.params as GPUBenchmarkParams)?.test_tensor_cores || false
                                : (selectedExperiment?.task.params as GPUBenchmarkParams)?.test_tensor_cores || false
                              }
                              onChange={(e) => {
                                if (isCreatingNew && newExperiment) {
                                  setNewExperiment({
                                    ...newExperiment,
                                    task: {
                                      ...newExperiment.task,
                                      params: {
                                        ...(newExperiment.task.params as GPUBenchmarkParams),
                                        test_tensor_cores: e.target.checked
                                      } as GPUBenchmarkParams
                                    }
                                  });
                                }
                              }}
                              disabled={!isCreatingNew}
                            />
                            Test Tensor Cores
                          </label>
                        </div>

                        <div className="form-field">
                          <label htmlFor="max-memory-usage">Max Memory Usage (%)</label>
                          <input
                            id="max-memory-usage"
                            type="number"
                            min="10"
                            max="95"
                            value={isCreatingNew
                              ? (newExperiment?.task.params as GPUBenchmarkParams)?.max_memory_usage_percent || 80
                              : (selectedExperiment?.task.params as GPUBenchmarkParams)?.max_memory_usage_percent || 80
                            }
                            onChange={(e) => {
                              if (isCreatingNew && newExperiment) {
                                setNewExperiment({
                                  ...newExperiment,
                                  task: {
                                    ...newExperiment.task,
                                    params: {
                                      ...(newExperiment.task.params as GPUBenchmarkParams),
                                      max_memory_usage_percent: parseFloat(e.target.value)
                                    } as GPUBenchmarkParams
                                  }
                                });
                              }
                            }}
                            readOnly={!isCreatingNew}
                          />
                        </div>

                        <div className="form-field">
                          <label htmlFor="benchmark-duration">Benchmark Duration (seconds)</label>
                          <input
                            id="benchmark-duration"
                            type="number"
                            min="5"
                            max="300"
                            value={isCreatingNew
                              ? (newExperiment?.task.params as GPUBenchmarkParams)?.benchmark_duration_seconds || 30
                              : (selectedExperiment?.task.params as GPUBenchmarkParams)?.benchmark_duration_seconds || 30
                            }
                            onChange={(e) => {
                              if (isCreatingNew && newExperiment) {
                                setNewExperiment({
                                  ...newExperiment,
                                  task: {
                                    ...newExperiment.task,
                                    params: {
                                      ...(newExperiment.task.params as GPUBenchmarkParams),
                                      benchmark_duration_seconds: parseInt(e.target.value)
                                    } as GPUBenchmarkParams
                                  }
                                });
                              }
                            }}
                            readOnly={!isCreatingNew}
                          />
                        </div>
                      </div>
                    )}

                    {/* Memory Benchmark Parameters */}
                    {(isCreatingNew ? newExperiment?.task.type : selectedExperiment?.task.type) === TaskType.MEMORY_BENCHMARK && (
                      <div className="memory-benchmark-parameters">
                        <h4>Memory Benchmark Configuration</h4>
                        <div className="benchmark-description">
                          <p><strong>What this measures:</strong> Deep analysis of memory subsystem performance patterns</p>
                          <ul>
                            <li><strong>Bandwidth Scaling:</strong> How performance changes with allocation size (64MB to 1GB+)</li>
                            <li><strong>Access Patterns:</strong> Sequential vs random memory access performance</li>
                            <li><strong>Peak vs Sustained:</strong> Maximum achievable vs average bandwidth over time</li>
                          </ul>
                          <p><strong>Use case:</strong> Memory bottleneck diagnosis and capacity planning. Useful for ML workloads with large datasets.</p>
                        </div>

                        <div className="form-field">
                          <label htmlFor="test-sizes">Test Sizes (MB)</label>
                          <input
                            id="test-sizes"
                            type="text"
                            placeholder="64, 256, 1024"
                            value={isCreatingNew
                              ? (newExperiment?.task.params as MemoryBenchmarkParams)?.test_sizes_mb?.join(', ') || ''
                              : (selectedExperiment?.task.params as MemoryBenchmarkParams)?.test_sizes_mb?.join(', ') || ''
                            }
                            onChange={(e) => {
                              if (isCreatingNew && newExperiment) {
                                const sizes = e.target.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
                                setNewExperiment({
                                  ...newExperiment,
                                  task: {
                                    ...newExperiment.task,
                                    params: {
                                      ...(newExperiment.task.params as MemoryBenchmarkParams),
                                      test_sizes_mb: sizes
                                    } as MemoryBenchmarkParams
                                  }
                                });
                              }
                            }}
                            readOnly={!isCreatingNew}
                          />
                        </div>

                        <div className="form-field">
                          <label htmlFor="test-patterns">Test Patterns</label>
                          <input
                            id="test-patterns"
                            type="text"
                            placeholder="sequential, random"
                            value={isCreatingNew
                              ? (newExperiment?.task.params as MemoryBenchmarkParams)?.test_patterns?.join(', ') || ''
                              : (selectedExperiment?.task.params as MemoryBenchmarkParams)?.test_patterns?.join(', ') || ''
                            }
                            onChange={(e) => {
                              if (isCreatingNew && newExperiment) {
                                const patterns = e.target.value.split(',').map(s => s.trim()).filter(s => s);
                                setNewExperiment({
                                  ...newExperiment,
                                  task: {
                                    ...newExperiment.task,
                                    params: {
                                      ...(newExperiment.task.params as MemoryBenchmarkParams),
                                      test_patterns: patterns
                                    } as MemoryBenchmarkParams
                                  }
                                });
                              }
                            }}
                            readOnly={!isCreatingNew}
                          />
                        </div>

                        <div className="form-field">
                          <label htmlFor="iterations-per-size">Iterations per Size</label>
                          <input
                            id="iterations-per-size"
                            type="number"
                            min="1"
                            max="100"
                            value={isCreatingNew
                              ? (newExperiment?.task.params as MemoryBenchmarkParams)?.iterations_per_size || 10
                              : (selectedExperiment?.task.params as MemoryBenchmarkParams)?.iterations_per_size || 10
                            }
                            onChange={(e) => {
                              if (isCreatingNew && newExperiment) {
                                setNewExperiment({
                                  ...newExperiment,
                                  task: {
                                    ...newExperiment.task,
                                    params: {
                                      ...(newExperiment.task.params as MemoryBenchmarkParams),
                                      iterations_per_size: parseInt(e.target.value)
                                    } as MemoryBenchmarkParams
                                  }
                                });
                              }
                            }}
                            readOnly={!isCreatingNew}
                          />
                        </div>
                      </div>
                    )}

                    {/* Compute Benchmark Parameters */}
                    {(isCreatingNew ? newExperiment?.task.type : selectedExperiment?.task.type) === TaskType.COMPUTE_BENCHMARK && (
                      <div className="compute-benchmark-parameters">
                        <h4>Compute Benchmark Configuration</h4>
                        <div className="benchmark-description">
                          <p><strong>What this measures:</strong> Raw computational performance for ML/AI workloads</p>
                          <ul>
                            <li><strong>FLOPS by Precision:</strong> Matrix multiplication performance in FP32, FP16, BF16</li>
                            <li><strong>Matrix Size Scaling:</strong> Performance across different problem sizes (1K×1K to 4K×4K)</li>
                            <li><strong>Peak vs Actual:</strong> How close you get to theoretical hardware peak performance</li>
                          </ul>
                          <p><strong>Use case:</strong> AI/ML performance optimization and hardware capability assessment for training/inference workloads.</p>
                        </div>

                        <div className="form-field">
                          <label htmlFor="precision-types">Precision Types</label>
                          <input
                            id="precision-types"
                            type="text"
                            placeholder="fp32, fp16"
                            value={isCreatingNew
                              ? (newExperiment?.task.params as ComputeBenchmarkParams)?.precision_types?.join(', ') || ''
                              : (selectedExperiment?.task.params as ComputeBenchmarkParams)?.precision_types?.join(', ') || ''
                            }
                            onChange={(e) => {
                              if (isCreatingNew && newExperiment) {
                                const types = e.target.value.split(',').map(s => s.trim()).filter(s => s);
                                setNewExperiment({
                                  ...newExperiment,
                                  task: {
                                    ...newExperiment.task,
                                    params: {
                                      ...(newExperiment.task.params as ComputeBenchmarkParams),
                                      precision_types: types
                                    } as ComputeBenchmarkParams
                                  }
                                });
                              }
                            }}
                            readOnly={!isCreatingNew}
                          />
                        </div>

                        <div className="form-field">
                          <label htmlFor="matrix-sizes">Matrix Sizes</label>
                          <input
                            id="matrix-sizes"
                            type="text"
                            placeholder="1024, 2048, 4096"
                            value={isCreatingNew
                              ? (newExperiment?.task.params as ComputeBenchmarkParams)?.matrix_sizes?.join(', ') || ''
                              : (selectedExperiment?.task.params as ComputeBenchmarkParams)?.matrix_sizes?.join(', ') || ''
                            }
                            onChange={(e) => {
                              if (isCreatingNew && newExperiment) {
                                const sizes = e.target.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
                                setNewExperiment({
                                  ...newExperiment,
                                  task: {
                                    ...newExperiment.task,
                                    params: {
                                      ...(newExperiment.task.params as ComputeBenchmarkParams),
                                      matrix_sizes: sizes
                                    } as ComputeBenchmarkParams
                                  }
                                });
                              }
                            }}
                            readOnly={!isCreatingNew}
                          />
                        </div>

                        <div className="form-field">
                          <label>
                            <input
                              type="checkbox"
                              checked={isCreatingNew
                                ? (newExperiment?.task.params as ComputeBenchmarkParams)?.include_tensor_ops || false
                                : (selectedExperiment?.task.params as ComputeBenchmarkParams)?.include_tensor_ops || false
                              }
                              onChange={(e) => {
                                if (isCreatingNew && newExperiment) {
                                  setNewExperiment({
                                    ...newExperiment,
                                    task: {
                                      ...newExperiment.task,
                                      params: {
                                        ...(newExperiment.task.params as ComputeBenchmarkParams),
                                        include_tensor_ops: e.target.checked
                                      } as ComputeBenchmarkParams
                                    }
                                  });
                                }
                              }}
                              disabled={!isCreatingNew}
                            />
                            Include Tensor Operations
                          </label>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {selectedParameterSection === 'model' && (
                <div className="model-parameters">
                  {/* Different model UI for different task types */}
                  {(isCreatingNew ? newExperiment?.task.type : selectedExperiment?.task.type) === TaskType.LLM_GENERATION ? (
                    <div className="llm-model-selection">
                      <div className="form-field">
                        <label htmlFor="model-selection">Model</label>
                        <select
                          id="model-selection"
                          value={isCreatingNew ? newExperiment?.model || '' : selectedExperiment?.model || ''}
                          onChange={(e) => {
                            if (isCreatingNew && newExperiment) {
                              setNewExperiment({ ...newExperiment, model: e.target.value });
                            }
                          }}
                          disabled={!isCreatingNew}
                        >
                          <option value="">Select a model...</option>
                          {getAvailableModels().map(model => (
                            <option key={model} value={model}>
                              {model}
                            </option>
                          ))}
                        </select>
                      </div>

                      {selectedExperiment && (
                        <div className="model-details">
                          <h4>Details</h4>
                          <div className="details-placeholder">
                            Model details would be shown here for existing experiments
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="hardware-model-info">
                      <div className="form-field">
                        <label htmlFor="hardware-model">Model</label>
                        <input
                          id="hardware-model"
                          type="text"
                          value="hardware-test"
                          readOnly
                          disabled
                          className="readonly-field"
                        />
                        <div className="field-help">
                          Hardware benchmarks don't require specific models - they test the underlying hardware capabilities.
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {selectedParameterSection === 'runtime' && (
                <div className="runtime-parameters">
                  <div className="form-field">
                    <label htmlFor="runtime-type">Framework</label>
                    <select
                      id="runtime-type"
                      value={isCreatingNew ? newExperiment?.framework.name || '' : selectedExperiment?.framework.name || ''}
                      onChange={(e) => {
                        if (isCreatingNew && newExperiment) {
                          setNewExperiment({
                            ...newExperiment,
                            framework: { ...newExperiment.framework, name: e.target.value as FrameworkName }
                          });
                        }
                      }}
                      disabled={!isCreatingNew}
                    >
                      <option value="">Select a framework...</option>
                      {getAvailableFrameworks().map(framework => (
                        <option key={framework} value={framework}>
                          {framework}
                        </option>
                      ))}
                    </select>
                    {/* Show framework help based on task type */}
                    {(isCreatingNew ? newExperiment?.task.type : selectedExperiment?.task.type) !== TaskType.LLM_GENERATION && (
                      <div className="field-help">
                        Hardware benchmarks use PyTorch for GPU operations and tensor computations.
                      </div>
                    )}
                  </div>

                  <div className="runtime-info">
                    {(isCreatingNew ? newExperiment?.task.type : selectedExperiment?.task.type) === TaskType.LLM_GENERATION ? (
                      <div className="llm-runtime-info">
                        <h4>Runtime Configuration</h4>
                        <div className="params-placeholder">LLM runtime parameters coming soon</div>
                      </div>
                    ) : (
                      <div className="hardware-runtime-info">
                        <h4>Hardware Testing Framework</h4>
                        <div className="framework-details">
                          <p>Hardware benchmarks automatically configure the optimal settings for:</p>
                          <ul>
                            <li><strong>GPU Detection</strong> - Automatic CUDA/device detection</li>
                            <li><strong>Memory Management</strong> - Safe memory allocation based on available VRAM</li>
                            <li><strong>Tensor Operations</strong> - Optimized operations for benchmarking</li>
                            <li><strong>Precision Testing</strong> - FP32, FP16, and other precision types</li>
                          </ul>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {selectedParameterSection === 'metrics' && (
                <div className="metrics-parameters">
                  {(isCreatingNew ? newExperiment?.task.type : selectedExperiment?.task.type) === TaskType.LLM_GENERATION ? (
                    <div className="llm-metrics-info">
                      <h4>LLM Performance Metrics</h4>
                      <div className="metrics-list">
                        <p>The following metrics will be automatically collected:</p>
                        <ul>
                          <li><strong>Inference Time</strong> - Time per generation</li>
                          <li><strong>Throughput</strong> - Tokens per second</li>
                          <li><strong>Memory Usage</strong> - GPU memory consumption</li>
                          <li><strong>GPU Utilization</strong> - GPU usage percentage</li>
                          <li><strong>Model Load Time</strong> - Cold start metrics</li>
                        </ul>
                      </div>
                    </div>
                  ) : (isCreatingNew ? newExperiment?.task.type : selectedExperiment?.task.type) === TaskType.GPU_BENCHMARK ? (
                    <div className="gpu-metrics-info">
                      <h4>GPU Benchmark Metrics</h4>
                      <div className="metrics-list">
                        <p>The following metrics will be automatically collected:</p>
                        <ul>
                          <li><strong>Memory Bandwidth</strong> - Copy, write, read bandwidth (GB/s)</li>
                          <li><strong>Compute FLOPS</strong> - Floating point operations per second</li>
                          <li><strong>Precision Performance</strong> - FP32, FP16 throughput</li>
                          <li><strong>Tensor Core Performance</strong> - If enabled and supported</li>
                          <li><strong>GPU Temperature</strong> - Thermal monitoring</li>
                          <li><strong>Power Usage</strong> - Energy consumption</li>
                        </ul>
                      </div>
                    </div>
                  ) : (isCreatingNew ? newExperiment?.task.type : selectedExperiment?.task.type) === TaskType.MEMORY_BENCHMARK ? (
                    <div className="memory-metrics-info">
                      <h4>Memory Benchmark Metrics</h4>
                      <div className="metrics-list">
                        <p>The following metrics will be automatically collected:</p>
                        <ul>
                          <li><strong>Bandwidth by Size</strong> - Performance across different memory sizes</li>
                          <li><strong>Access Patterns</strong> - Sequential vs random access performance</li>
                          <li><strong>Peak Bandwidth</strong> - Maximum achievable bandwidth</li>
                          <li><strong>Average Bandwidth</strong> - Sustained performance metrics</li>
                          <li><strong>Latency Metrics</strong> - Memory access timing</li>
                        </ul>
                      </div>
                    </div>
                  ) : (isCreatingNew ? newExperiment?.task.type : selectedExperiment?.task.type) === TaskType.COMPUTE_BENCHMARK ? (
                    <div className="compute-metrics-info">
                      <h4>Compute Benchmark Metrics</h4>
                      <div className="metrics-list">
                        <p>The following metrics will be automatically collected:</p>
                        <ul>
                          <li><strong>FLOPS by Precision</strong> - FP32, FP16, etc. performance</li>
                          <li><strong>Matrix Operation Performance</strong> - Different matrix sizes</li>
                          <li><strong>Peak Compute</strong> - Maximum theoretical performance</li>
                          <li><strong>Tensor Operations</strong> - If enabled, specialized tensor ops</li>
                          <li><strong>Compute Efficiency</strong> - Relative to peak theoretical</li>
                        </ul>
                      </div>
                    </div>
                  ) : (
                    <div className="generic-metrics-info">
                      <h4>Metrics Collection</h4>
                      <div className="params-placeholder">
                        Select a task type to see available metrics
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ExperimentsTab;
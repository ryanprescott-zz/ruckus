"""Experiment orchestration service for RUCKUS.

This service handles the core orchestration logic:
1. Breaking experiments into individual jobs based on parameter grids
2. Matching jobs to capable agents based on requirements
3. Dispatching jobs and tracking progress
4. Aggregating results when experiments complete
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from ruckus_common.models import (
    ExperimentSpec, ExperimentExecution, ExperimentSubmission, ExperimentSubmissionResponse,
    JobRequest, JobSpec, JobStatus, AgentRequirements, AgentScore, JobAssignment,
    AgentMatchingRequest, AgentMatchingResponse, RegisteredAgentInfo, AgentStatus, AgentStatusEnum,
    MultiRunJobResult, SingleRunResult, MetricStatistics
)
from .storage.base import StorageBackend


class OrchestrationError(Exception):
    """Base exception for orchestration errors."""
    pass


class NoCompatibleAgentsError(OrchestrationError):
    """Raised when no agents are compatible with job requirements."""
    pass


class ExperimentOrchestrator:
    """Manages experiment execution workflow."""
    
    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self.logger = logging.getLogger(__name__)
        
        # In-memory tracking of active experiments
        self.active_experiments: Dict[str, ExperimentExecution] = {}
        
    async def submit_experiment(self, submission: ExperimentSubmission) -> ExperimentSubmissionResponse:
        """Submit an experiment for execution.
        
        Args:
            submission: Experiment submission with spec and options
            
        Returns:
            Response with experiment ID and planning results
            
        Raises:
            OrchestrationError: If experiment planning fails
        """
        spec = submission.spec
        self.logger.info(f"Submitting experiment {spec.experiment_id}: {spec.name}")
        
        try:
            # Create experiment execution record
            execution = ExperimentExecution(
                experiment_id=spec.experiment_id,
                spec=spec,
                status="planning"
            )
            
            # Plan the experiment (break into jobs)
            planned_jobs = await self._plan_experiment(spec)
            execution.total_jobs = len(planned_jobs)
            execution.planned_jobs = [job.job_id for job in planned_jobs]
            execution.queued_jobs = execution.planned_jobs.copy()
            
            # Store in memory and persistence
            self.active_experiments[spec.experiment_id] = execution
            await self.storage.store_experiment_execution(execution)
            
            # Store individual job specs
            for job_spec in planned_jobs:
                await self.storage.create_job_spec(job_spec)
            
            self.logger.info(f"Planned {len(planned_jobs)} jobs for experiment {spec.experiment_id}")
            
            if submission.dry_run:
                # Don't actually start execution for dry runs
                execution.status = "planned_only"
                response_status = "planned"
            elif submission.submit_immediately:
                # Start execution immediately
                await self._start_experiment_execution(execution)
                response_status = "running"
            else:
                # Queue for later execution
                execution.status = "queued"
                response_status = "queued"
            
            # Calculate estimates
            estimated_duration = await self._estimate_experiment_duration(spec, planned_jobs)
            
            return ExperimentSubmissionResponse(
                experiment_id=spec.experiment_id,
                status=response_status,
                message=f"Experiment planned with {len(planned_jobs)} jobs",
                estimated_jobs=len(planned_jobs),
                estimated_duration_hours=estimated_duration,
                tracking_url=f"/api/v1/experiments/{spec.experiment_id}/status"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to submit experiment {spec.experiment_id}: {e}")
            raise OrchestrationError(f"Experiment submission failed: {str(e)}")
    
    async def _plan_experiment(self, spec: ExperimentSpec) -> List[JobSpec]:
        """Break an experiment into individual job specifications.
        
        Args:
            spec: Experiment specification
            
        Returns:
            List of job specifications to execute
        """
        self.logger.info(f"Planning jobs for experiment {spec.experiment_id}")
        
        jobs = []
        
        # Generate parameter configurations
        parameter_configs = spec.parameter_grid.generate_configurations()
        
        # Create a job for each model × parameter configuration combination
        for model in spec.models:
            for param_config in parameter_configs:
                # Merge base parameters with specific config
                merged_params = {**spec.base_parameters, **param_config}
                
                # Generate unique job ID
                job_id = f"{spec.experiment_id}-{model}-{uuid.uuid4().hex[:8]}"
                
                # Create job specification
                job_spec = JobSpec(
                    job_id=job_id,
                    experiment_id=spec.experiment_id,
                    model=model,
                    framework="auto",  # Let agent selection determine best framework
                    hardware_target="auto",  # Let agent selection determine
                    task_type=spec.task_type,
                    task_config=spec.task_config,
                    parameters=merged_params,
                    timeout_seconds=spec.timeout_seconds,
                    max_retries=spec.max_retries,
                    priority=spec.priority,
                    status=JobStatus.QUEUED,
                    runs_per_job=getattr(spec, 'runs_per_job', 1)  # Support multi-run jobs
                )
                
                jobs.append(job_spec)
        
        self.logger.info(f"Generated {len(jobs)} job specifications")
        return jobs
    
    async def _start_experiment_execution(self, execution: ExperimentExecution):
        """Start executing an experiment by dispatching jobs to agents.
        
        Args:
            execution: Experiment execution to start
        """
        execution.status = "running"
        execution.started_at = datetime.utcnow()
        
        self.logger.info(f"Starting execution of experiment {execution.experiment_id}")
        
        # Start background task to dispatch jobs
        asyncio.create_task(self._execute_experiment_jobs(execution))
    
    async def _execute_experiment_jobs(self, execution: ExperimentExecution):
        """Execute jobs for an experiment with agent matching and dispatch.
        
        Args:
            execution: Experiment execution to process
        """
        try:
            while execution.queued_jobs and execution.status == "running":
                # Get next batch of jobs to dispatch
                batch_size = min(10, len(execution.queued_jobs))  # Process in batches
                job_batch = execution.queued_jobs[:batch_size]
                
                # Find agents for this batch
                agent_assignments = await self._match_jobs_to_agents(
                    job_batch, execution.spec.agent_requirements
                )
                
                # Dispatch jobs that have compatible agents
                dispatched_jobs = []
                for job_id, assignment in agent_assignments.items():
                    if assignment:  # Has at least one compatible agent
                        best_agent = assignment[0]  # Use best scoring agent
                        success = await self._dispatch_job_to_agent(job_id, best_agent)
                        if success:
                            dispatched_jobs.append(job_id)
                            execution.running_jobs[job_id] = best_agent.agent_id
                
                # Remove dispatched jobs from queue
                for job_id in dispatched_jobs:
                    if job_id in execution.queued_jobs:
                        execution.queued_jobs.remove(job_id)
                
                # Update progress
                execution.progress_percent = execution.calculate_progress()
                
                # Persist state
                await self.storage.update_experiment_execution(execution)
                
                # Wait before next batch (avoid overwhelming agents)
                if execution.queued_jobs:
                    await asyncio.sleep(2.0)
                
            self.logger.info(f"Completed job dispatch for experiment {execution.experiment_id}")
            
        except Exception as e:
            self.logger.error(f"Error executing experiment {execution.experiment_id}: {e}")
            execution.status = "failed"
            execution.error_summary = str(e)
    
    async def _match_jobs_to_agents(
        self, job_ids: List[str], requirements: AgentRequirements
    ) -> Dict[str, List[AgentScore]]:
        """Match a batch of jobs to compatible agents.
        
        Args:
            job_ids: List of job IDs to match
            requirements: Agent requirements from experiment spec
            
        Returns:
            Dict mapping job_id to list of compatible agent scores
        """
        # Get all registered agents
        registered_agents = await self.storage.list_registered_agent_info()
        
        # Score each agent against the requirements
        agent_scores = {}
        for agent in registered_agents:
            score = await self._score_agent_compatibility(agent, requirements)
            if score.score > 0:  # Only include compatible agents
                agent_scores[agent.agent_id] = score
        
        # For simplicity, return the same agent list for all jobs in this batch
        # In a more sophisticated implementation, we'd do per-job matching
        compatible_agents = sorted(agent_scores.values(), key=lambda x: x.score, reverse=True)
        
        return {job_id: compatible_agents for job_id in job_ids}
    
    async def _score_agent_compatibility(
        self, agent: RegisteredAgentInfo, requirements: AgentRequirements
    ) -> AgentScore:
        """Score an agent's compatibility with job requirements.
        
        Args:
            agent: Registered agent information
            requirements: Job requirements
            
        Returns:
            Agent compatibility score
        """
        score = AgentScore(
            agent_id=agent.agent_id,
            agent_name=agent.agent_name or agent.agent_id,
            score=0.0
        )
        
        # Check required models (from system_info)
        system_info = agent.system_info or {}
        agent_models = system_info.get('models', [])
        if agent_models:
            model_names = [model.get('name', '') if isinstance(model, dict) else str(model) for model in agent_models]
            
            for required_model in requirements.required_models:
                if required_model in model_names:
                    score.model_compatibility += 20.0
                    score.compatible_models.append(required_model)
                else:
                    score.missing_requirements.append(f"model:{required_model}")
            
            # Bonus for preferred models
            for preferred_model in requirements.preferred_models:
                if preferred_model in model_names:
                    score.model_compatibility += 5.0
        else:
            # If no models specified, add missing requirements
            for required_model in requirements.required_models:
                score.missing_requirements.append(f"model:{required_model}")
        
        # Check frameworks (from system_info and capabilities)
        system_frameworks = system_info.get('frameworks', [])
        capabilities_frameworks = agent.capabilities.get('frameworks', []) if agent.capabilities else []
        
        # Combine framework sources
        agent_frameworks = []
        for fw_list in [system_frameworks, capabilities_frameworks]:
            for fw in fw_list:
                fw_name = fw.get('name', fw) if isinstance(fw, dict) else str(fw)
                if fw_name not in agent_frameworks:
                    agent_frameworks.append(fw_name)
        
        if agent_frameworks:
            for required_fw in requirements.required_frameworks:
                if required_fw in agent_frameworks:
                    score.framework_compatibility += 15.0
                else:
                    score.missing_requirements.append(f"framework:{required_fw}")
            
            # Bonus for preferred frameworks
            for preferred_fw in requirements.preferred_frameworks:
                if preferred_fw in agent_frameworks:
                    score.framework_compatibility += 3.0
        else:
            # If no frameworks specified, add missing requirements
            for required_fw in requirements.required_frameworks:
                score.missing_requirements.append(f"framework:{required_fw}")
        
        # Check GPU requirements (from system_info)
        agent_gpus = system_info.get('gpus', [])
        if agent_gpus:
            gpu_count = len(agent_gpus)
            if gpu_count >= requirements.min_gpu_count:
                score.hardware_suitability += 10.0
            else:
                score.missing_requirements.append(f"min_gpu_count:{requirements.min_gpu_count}")
                
            # Check GPU memory if specified
            if requirements.min_gpu_memory_mb:
                max_gpu_memory = max((gpu.get('memory_mb', 0) if isinstance(gpu, dict) else 0 for gpu in agent_gpus), default=0)
                if max_gpu_memory >= requirements.min_gpu_memory_mb:
                    score.hardware_suitability += 5.0
                else:
                    score.missing_requirements.append(f"min_gpu_memory:{requirements.min_gpu_memory_mb}MB")
        else:
            # If no GPUs but GPU required, add missing requirement
            if requirements.min_gpu_count > 0:
                score.missing_requirements.append(f"min_gpu_count:{requirements.min_gpu_count}")
        
        # Check agent type compatibility
        if requirements.agent_types and agent.agent_type:
            if agent.agent_type in requirements.agent_types:
                score.capability_match += 10.0
            else:
                score.missing_requirements.append(f"agent_type:{requirements.agent_types}")
        
        # Apply exclusions
        if agent.agent_id in requirements.excluded_agents:
            score.score = 0.0
            score.warnings.append("Agent explicitly excluded")
            return score
        
        # Calculate final score
        total_score = (
            score.model_compatibility + 
            score.framework_compatibility + 
            score.hardware_suitability + 
            score.capability_match + 
            score.availability_bonus
        )
        
        # Penalize for missing requirements
        if score.missing_requirements:
            total_score *= 0.1  # Heavy penalty for missing requirements
        
        score.score = min(100.0, max(0.0, total_score))
        
        return score
    
    async def _dispatch_job_to_agent(self, job_id: str, agent_score: AgentScore) -> bool:
        """Dispatch a single job to an agent.
        
        Args:
            job_id: Job to dispatch
            agent_score: Selected agent with compatibility info
            
        Returns:
            True if job was successfully dispatched
        """
        try:
            # Get job spec
            job_spec = await self.storage.get_job_spec(job_id)
            if not job_spec:
                self.logger.error(f"Job spec not found for {job_id}")
                return False
            
            # Get agent info
            agent = await self.storage.get_registered_agent_info(agent_score.agent_id)
            if not agent:
                self.logger.error(f"Agent not found: {agent_score.agent_id}")
                return False
            
            # Create job request for the agent
            job_request = JobRequest(
                job_id=job_spec.job_id,
                experiment_id=job_spec.experiment_id,
                model=job_spec.model,
                framework=job_spec.framework,
                task_type=job_spec.task_type,
                task_config=job_spec.task_config,
                parameters=job_spec.parameters,
                timeout_seconds=job_spec.timeout_seconds,
                runs_per_job=job_spec.runs_per_job
            )
            
            # TODO: Actually send HTTP request to agent
            # For now, simulate successful dispatch
            self.logger.info(f"Dispatched job {job_id} to agent {agent_score.agent_id}")
            
            # Update job status
            job_spec.status = JobStatus.ASSIGNED
            job_spec.agent_id = agent_score.agent_id
            await self.storage.update_job_spec(job_spec)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to dispatch job {job_id} to agent {agent_score.agent_id}: {e}")
            return False
    
    async def _estimate_experiment_duration(
        self, spec: ExperimentSpec, jobs: List[JobSpec]
    ) -> Optional[float]:
        """Estimate experiment completion time in hours.
        
        Args:
            spec: Experiment specification
            jobs: List of planned jobs
            
        Returns:
            Estimated duration in hours, or None if cannot estimate
        """
        if not jobs:
            return 0.0
        
        # Simple estimation: assume average job takes 10 minutes
        # and we can run max_parallel_jobs concurrently
        avg_job_minutes = 10.0
        max_parallel = spec.max_parallel_jobs or 5
        
        total_job_hours = (len(jobs) * avg_job_minutes) / 60.0
        estimated_hours = total_job_hours / max_parallel
        
        return estimated_hours
    
    async def get_experiment_execution(self, experiment_id: str) -> Optional[ExperimentExecution]:
        """Get current execution status of an experiment.
        
        Args:
            experiment_id: ID of experiment to check
            
        Returns:
            Experiment execution status, or None if not found
        """
        # Check in-memory cache first
        if experiment_id in self.active_experiments:
            return self.active_experiments[experiment_id]
        
        # Fall back to storage
        return await self.storage.get_experiment_execution(experiment_id)
    
    async def handle_job_update(self, job_id: str, status: JobStatus, result_data: Optional[dict] = None, multi_run_result: Optional[MultiRunJobResult] = None):
        """Handle a job status update from an agent.
        
        Args:
            job_id: ID of job being updated
            status: New job status
            result_data: Job result data if completed
        """
        try:
            # Find which experiment this job belongs to
            job_spec = await self.storage.get_job_spec(job_id)
            if not job_spec:
                self.logger.warning(f"Received update for unknown job {job_id}")
                return
            
            experiment_id = job_spec.experiment_id
            execution = await self.get_experiment_execution(experiment_id)
            if not execution:
                self.logger.warning(f"Received job update for unknown experiment {experiment_id}")
                return
            
            # Update job status and results
            job_spec.status = status
            if multi_run_result:
                # Store the enhanced multi-run result structure
                job_spec.results = multi_run_result.model_dump()
            elif result_data:
                # Fallback for single-run results
                job_spec.results = result_data
            await self.storage.update_job_spec(job_spec)
            
            # Update experiment execution state
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                # Job finished - remove from running, add to appropriate completed list
                if job_id in execution.running_jobs:
                    del execution.running_jobs[job_id]
                
                if status == JobStatus.COMPLETED:
                    execution.completed_jobs.append(job_id)
                else:
                    execution.failed_jobs.append(job_id)
                
                # Check if experiment is complete
                if not execution.running_jobs and not execution.queued_jobs:
                    execution.status = "completed"
                    execution.completed_at = datetime.utcnow()
                    
                    # Aggregate results for completed experiment
                    await self._aggregate_experiment_results(execution)
                    self.logger.info(f"Experiment {experiment_id} completed with result aggregation")
            
            # Update progress and persist
            execution.progress_percent = execution.calculate_progress()
            await self.storage.update_experiment_execution(execution)
            
        except Exception as e:
            self.logger.error(f"Error handling job update for {job_id}: {e}")
    
    async def _aggregate_experiment_results(self, execution: ExperimentExecution):
        """Aggregate results from all completed jobs in an experiment.
        
        Args:
            execution: Experiment execution to aggregate results for
        """
        try:
            self.logger.info(f"Aggregating results for experiment {execution.experiment_id}")
            
            # Collect results from all completed jobs
            job_results = []
            for job_id in execution.completed_jobs:
                job_spec = await self.storage.get_job_spec(job_id)
                if job_spec and hasattr(job_spec, 'results') and job_spec.results:
                    # Check if this is a multi-run result
                    result_entry = {
                        'job_id': job_id,
                        'model': job_spec.model,
                        'parameters': job_spec.parameters,
                        'results': job_spec.results
                    }
                    
                    # Parse multi-run structure if present
                    if isinstance(job_spec.results, dict) and 'total_runs' in job_spec.results:
                        try:
                            multi_run_result = MultiRunJobResult.model_validate(job_spec.results)
                            result_entry['multi_run_data'] = multi_run_result
                            result_entry['is_multi_run'] = True
                        except Exception as e:
                            self.logger.warning(f"Failed to parse multi-run result for job {job_id}: {e}")
                            result_entry['is_multi_run'] = False
                    else:
                        result_entry['is_multi_run'] = False
                    
                    job_results.append(result_entry)
            
            # Generate aggregated summary
            summary = await self._generate_experiment_summary(execution, job_results)
            execution.aggregated_results = {
                'job_results': job_results,
                'summary': summary,
                'aggregated_at': datetime.utcnow().isoformat(),
                'total_jobs': execution.total_jobs,
                'completed_jobs': len(execution.completed_jobs),
                'failed_jobs': len(execution.failed_jobs)
            }
            
            self.logger.info(f"Aggregated {len(job_results)} job results for experiment {execution.experiment_id}")
            
        except Exception as e:
            self.logger.error(f"Error aggregating results for experiment {execution.experiment_id}: {e}")
    
    async def _generate_experiment_summary(self, execution: ExperimentExecution, job_results: list) -> dict:
        """Generate summary statistics and insights from job results.
        
        Args:
            execution: Experiment execution
            job_results: List of job results with model, parameters, and metrics
            
        Returns:
            Summary dictionary with aggregated metrics and insights
        """
        if not job_results:
            return {
                'status': 'no_results',
                'message': 'No completed jobs with results'
            }
        
        try:
            # Extract metrics from job results with multi-run support
            all_metrics = {}
            model_performance = {}
            parameter_impact = {}
            multi_run_summary = {'jobs_with_multi_runs': 0, 'total_individual_runs': 0}
            
            for job_result in job_results:
                model = job_result['model']
                parameters = job_result['parameters']
                results = job_result['results']
                is_multi_run = job_result.get('is_multi_run', False)
                
                # Track model performance
                if model not in model_performance:
                    model_performance[model] = []
                
                if is_multi_run and 'multi_run_data' in job_result:
                    # Handle multi-run results
                    multi_run_data = job_result['multi_run_data']
                    model_performance[model].append({
                        'type': 'multi_run',
                        'data': multi_run_data,
                        'summary_stats': multi_run_data.summary_stats
                    })
                    
                    multi_run_summary['jobs_with_multi_runs'] += 1
                    multi_run_summary['total_individual_runs'] += multi_run_data.total_runs
                    
                    # Extract summary statistics as primary metrics
                    for metric_name, metric_stats in multi_run_data.summary_stats.items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append({
                            'value': metric_stats.mean,
                            'std': metric_stats.std,
                            'min': metric_stats.min_value,
                            'max': metric_stats.max_value,
                            'outliers': len(metric_stats.outliers) if metric_stats.outliers else 0,
                            'model': model,
                            'parameters': parameters,
                            'type': 'multi_run_summary',
                            'sample_count': metric_stats.count
                        })
                else:
                    # Handle single-run results
                    model_performance[model].append({
                        'type': 'single_run',
                        'data': results
                    })
                    
                    # Collect all metrics from single runs
                    if isinstance(results, dict):
                        for metric_name, metric_value in results.items():
                            if isinstance(metric_value, (int, float)):
                                if metric_name not in all_metrics:
                                    all_metrics[metric_name] = []
                                all_metrics[metric_name].append({
                                    'value': metric_value,
                                    'model': model,
                                    'parameters': parameters,
                                    'type': 'single_run'
                                })
            
            # Calculate aggregate statistics with multi-run awareness
            metric_summaries = {}
            for metric_name, values in all_metrics.items():
                numeric_values = [v['value'] for v in values]
                if numeric_values:
                    # Basic statistics
                    mean_val = sum(numeric_values) / len(numeric_values)
                    min_val = min(numeric_values)
                    max_val = max(numeric_values)
                    
                    # Calculate standard deviation if we have enough data points
                    std_val = None
                    if len(numeric_values) > 1:
                        variance = sum((x - mean_val) ** 2 for x in numeric_values) / (len(numeric_values) - 1)
                        std_val = variance ** 0.5
                    
                    metric_summaries[metric_name] = {
                        'mean': mean_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val,
                        'count': len(numeric_values),
                        'multi_run_jobs': len([v for v in values if v.get('type') == 'multi_run_summary']),
                        'single_run_jobs': len([v for v in values if v.get('type') == 'single_run'])
                    }
            
            # Model comparison
            model_comparison = {}
            for model, results_list in model_performance.items():
                model_comparison[model] = {
                    'job_count': len(results_list),
                    'avg_metrics': self._calculate_avg_metrics(results_list)
                }
            
            return {
                'status': 'success',
                'metric_summaries': metric_summaries,
                'model_comparison': model_comparison,
                'multi_run_summary': multi_run_summary,
                'total_successful_jobs': len(job_results),
                'unique_models_tested': len(model_performance),
                'parameter_combinations': len(set(str(jr['parameters']) for jr in job_results))
            }
            
        except Exception as e:
            self.logger.error(f"Error generating experiment summary: {e}")
            return {
                'status': 'error',
                'message': f"Failed to generate summary: {str(e)}"
            }
    
    def _calculate_avg_metrics(self, results_list: list) -> dict:
        """Calculate average metrics across multiple job results with multi-run support.
        
        Args:
            results_list: List of result dictionaries (may contain multi-run data)
            
        Returns:
            Dictionary with averaged metrics
        """
        if not results_list:
            return {}
        
        # Collect all metric values with multi-run awareness
        metric_totals = {}
        metric_counts = {}
        
        for results in results_list:
            if isinstance(results, dict):
                if results.get('type') == 'multi_run':
                    # Handle multi-run results - use summary stats
                    summary_stats = results.get('summary_stats', {})
                    for metric_name, metric_stats in summary_stats.items():
                        if hasattr(metric_stats, 'mean'):
                            if metric_name not in metric_totals:
                                metric_totals[metric_name] = 0
                                metric_counts[metric_name] = 0
                            metric_totals[metric_name] += metric_stats.mean
                            metric_counts[metric_name] += 1
                elif results.get('type') == 'single_run':
                    # Handle single-run results
                    data = results.get('data', {})
                    for metric_name, metric_value in data.items():
                        if isinstance(metric_value, (int, float)):
                            if metric_name not in metric_totals:
                                metric_totals[metric_name] = 0
                                metric_counts[metric_name] = 0
                            metric_totals[metric_name] += metric_value
                            metric_counts[metric_name] += 1
                else:
                    # Legacy format - treat as single run
                    for metric_name, metric_value in results.items():
                        if isinstance(metric_value, (int, float)):
                            if metric_name not in metric_totals:
                                metric_totals[metric_name] = 0
                                metric_counts[metric_name] = 0
                            metric_totals[metric_name] += metric_value
                            metric_counts[metric_name] += 1
        
        # Calculate averages
        avg_metrics = {}
        for metric_name in metric_totals:
            avg_metrics[metric_name] = metric_totals[metric_name] / metric_counts[metric_name]
        
        return avg_metrics
    
    async def get_experiment_results(self, experiment_id: str) -> Optional[dict]:
        """Get aggregated results for a completed experiment.
        
        Args:
            experiment_id: ID of experiment to get results for
            
        Returns:
            Aggregated results dictionary or None if not found/not completed
        """
        execution = await self.get_experiment_execution(experiment_id)
        if not execution:
            return None
        
        if execution.status != "completed":
            return {
                'status': 'not_completed',
                'current_status': execution.status,
                'message': 'Experiment has not completed yet'
            }
        
        return execution.aggregated_results or {
            'status': 'no_results',
            'message': 'Results not yet aggregated'
        }
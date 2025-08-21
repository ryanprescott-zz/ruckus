# Overview
RUCKUS is a distributed benchmarking and evaluation system designed to assess the performance of machine learning models across heterogeneous hardware configurations. The system addresses a critical challenge in ML deployment: understanding how different models perform when deployed on vastly different infrastructure, from edge devices like Raspberry Pis to high-end datacenter GPUs like H100 clusters. RUCKUS provides a unified framework for running consistent benchmarks across this spectrum, automatically adapting to the capabilities available on each platform - whether that's full white-box access with detailed GPU metrics, gray-box API access through services like vLLM, or black-box access to external model endpoints.

The ruckus project includes four subprojects: 1. The *ruckus server* manages experiments, schedules and monitors experiment jobs, collects experiment job results, runs experiment post-processing (such as results scoring), and provides persistent access to experiments, jobs and results. 2. The *ruckus agent* executes experiments on target hardware with target runtimes (e.g. VLLM, transformers) with target models for target tasks with target data, and collects target metrics. Ruckus agents expose an API that the ruckus server uses to register them. Agents respond to registration requests from the server with their complete hardware profile (CPU, GPU, memory), available software frameworks (transformers, PyTorch, vLLM), monitoring capabilities (nvidia-smi, memory profiling), and cached models. The ruckus server uses this information to intelligently distribute benchmark jobs, ensuring each agent only receives workloads it can execute. As jobs run, agents collect whatever metrics their access level permits - from basic latency measurements to detailed token throughput and memory usage - and report results back to the ruckus server for aggregation. This architecture enables organizations to make data-driven decisions about model deployment, comparing performance per dollar, performance per watt, and quality metrics across their entire infrastructure portfolio. 3. The *ruckus common* subproject provides common software shared by the other two subprojects, including the Agent Protocol shared data model used to communicate between the server and agent. 4. The *ruckus ui* provides a user interface for users to define, run, monitor status and visualize experiment results via APIs to the server.

# Core System Components
## Ruckus server
The ruckus server will be the central brain of RUCKUS, responsible for managing experiments, coordinating agents, scheduling jobs, and aggregating results. It will expose a REST API for experiment creation and management, provide polling endpoints for status updates, and maintain all system state in a SQLite database. The ruckus server will handle ruckus agent registration and health monitoring, job distribution based on ruckus agent capabilities, and result normalization across different ruckus agent types.

The ruckus server will store experiment definitions, jobs & job status, and experiment results. The server will store data using a postgres database. The database Schema will include tables for experiments (storing configuration and status), jobs (tracking individual benchmark runs), agents (registration and capabilities), and results (metrics and outputs). We'll use SQLite with SQLAlchemy for the ORM, with indexes optimized for polling queries.

## Ruckus agent
The ruckus agent will be the worker component that actually runs benchmarks. We'll create a flexible agent that can operate in three modes: white-box (full control over model loading and metrics), gray-box (working with existing model APIs like vLLM), and black-box (minimal API-only access). Each ruckus agent will advertise its capabilities to the ruckus server, execute jobs asynchronously, report progress through HTTP callbacks, and collect whatever metrics its access level allows.

The Agent Protocol will define how the ruckus server and ruckus agents communicate, including job request/response formats, capability negotiation protocols, progress reporting stages, and metric definitions. This shared protocol ensures compatibility across different agent implementations.

The Agent will persist data for experiments in progress to the filesystem to provide durability.


# API Structure
Experiment Management Endpoints will allow users to create new experiments with model/framework/hardware combinations, retrieve experiment status and progress, list all experiments, and delete or modify existing experiments.

Agent Management Endpoints will handle agent registration with capability advertisement, health status checking, capability updates, and agent removal or deactivation.

Job Management Endpoints will provide job status polling with efficient filtering, individual job detail retrieval, batch job updates from agents, and job cancellation or retry functionality.

Results Endpoints will offer aggregated experiment results, individual job metrics, comparison matrices across configurations, and raw data export capabilities.

# Tech stack

The Ruckus server and agent will be implemented in Python 3.13 using FastAPI for web services, Pydantic models for data and data validation, pydantic settings for configuration settings. All files will include docstrings using Google style for all files, classes and methods. Pytests will be developed for all methods of all classes.

The Ruckus ui will be implemneted in typescript using react.


# Task Implementation
Wikipedia Summarization Task will be our initial benchmark, including document loading and preprocessing, prompt construction for summarization, output collection, and quality metric calculation (ROUGE scores). This task will demonstrate the system's ability to handle real workloads across different hardware configurations.

# Metrics System
Universal Metrics that work everywhere will include total execution time, success/failure status, and timestamp information.

Capability-Dependent Metrics will include token counts and throughput (requiring tokenization capability), memory usage (requiring system monitoring), GPU utilization (requiring GPU access), and model loading time (requiring model control).

Post-Process Metrics computed by the ruckus server will include ROUGE scores for summarization quality, custom evaluations for specific tasks, and comparative analysis across runs.

# Deployment Infrastructure
Docker Containers will package the ruckus server with all dependencies, create agent containers for different hardware configurations, and use docker-compose for local development orchestration.

Configuration Files will include Docker compose for multi-container setup, environment files for different deployment modes, and volume mounts for models and data persistence.

# Dashboard and Monitoring
Simple Web Dashboard will use HTML/JavaScript with polling for updates, display experiment progress and job status grid, show real-time metrics as jobs complete, and provide result comparison visualizations.

Logging and Monitoring will implement structured logging throughout the system, progress tracking for long-running jobs, error reporting and failure analysis, and system resource monitoring.

# Testing and Demo Infrastructure
Test Data will include sample Wikipedia articles of varying lengths, pre-computed reference summaries, and configuration templates for common scenarios.

Demo Scripts will automate experiment creation, agent deployment across different hardware, benchmark execution, and result presentation.

# Development Tooling
CLI Tools for experiment management, agent control, result inspection, and system health monitoring.

Development Utilities including database migration scripts, data seeding tools, log analysis utilities, and performance profiling helpers.

# Error Handling and Resilience
Failure Management will handle agent disconnections gracefully, retry failed jobs automatically, timeout long-running operations, and provide graceful degradation when capabilities are unavailable.

Validation will ensure configuration compatibility with agent capabilities, input data format verification, metric data validation, and result completeness checking.

# Documentation
API Documentation using FastAPI's automatic OpenAPI/Swagger generation, with README files for setup and usage, architecture diagrams, and inline code documentation.

Deployment Guides for different scenarios (local, cloud, airgapped), configuration examples, and troubleshooting guides.

This initial implementation will create a fully functional benchmarking system that can coordinate distributed model evaluation across heterogeneous hardware, with automatic adaptation to available capabilities and comprehensive result collection. The modular design ensures easy extension for additional models, tasks, and metrics in future iterations.
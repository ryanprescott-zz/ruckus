"""SQLAlchemy database models."""

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Boolean, Text, Enum
from sqlalchemy.sql import func
from datetime import datetime

from .database import Base
from ruckus_common.models import JobStatus, AgentType


class Agent(Base):
    """Agent registration in database."""
    __tablename__ = "agents"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(Enum(AgentType), nullable=False)
    hostname = Column(String)
    ip_address = Column(String)

    # Capabilities
    capabilities = Column(JSON)
    frameworks = Column(JSON)
    hardware_info = Column(JSON)

    # Status
    status = Column(String, default="offline")
    last_heartbeat = Column(DateTime)

    # Metadata
    registered_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Experiment(Base):
    """Experiment definition."""
    __tablename__ = "experiments"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)

    # Configuration
    models = Column(JSON)
    frameworks = Column(JSON)
    task_type = Column(String)
    task_config = Column(JSON)
    parameters = Column(JSON)

    # Status
    status = Column(String, default="created")
    progress = Column(Float, default=0.0)

    # Metadata
    owner = Column(String)
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)


class Job(Base):
    """Job definition."""
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)
    experiment_id = Column(String, nullable=False)
    agent_id = Column(String)

    # Configuration
    model = Column(String)
    framework = Column(String)
    config = Column(JSON)

    # Status
    status = Column(Enum(JobStatus), default=JobStatus.QUEUED)
    stage = Column(String)
    progress = Column(Integer, default=0)

    # Timing
    queued_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)s
    duration_seconds = Column(Float)

    # Results
    results = Column(JSON)
    metrics = Column(JSON)
    error = Column(Text)


class Result(Base):
    """Benchmark results."""
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, nullable=False)
    experiment_id = Column(String, nullable=False)

    # Metrics
    metrics = Column(JSON)
    output = Column(JSON)

    # Metadata
    created_at = Column(DateTime, default=func.now())
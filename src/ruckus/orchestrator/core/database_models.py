"""
SQLAlchemy database models for the Ruckus Orchestrator.

This module defines the database table models using SQLAlchemy ORM
for experiments, jobs, agents, and related entities.
"""

from datetime import datetime
from uuid import uuid4
from typing import Dict, Any

from sqlalchemy import Column, String, DateTime, Text, ForeignKey, Integer, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from .database import Base


class ExperimentDB(Base):
    """Database model for experiments."""
    
    __tablename__ = "experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    model_name = Column(String(255), nullable=False)
    runtime = Column(String(100), nullable=False)
    platform = Column(String(100), nullable=False)
    task_config = Column(JSON, nullable=False)
    data_config = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    jobs = relationship("JobDB", back_populates="experiment", cascade="all, delete-orphan")


class JobDB(Base):
    """Database model for jobs."""
    
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    experiment_id = Column(UUID(as_uuid=True), ForeignKey("experiments.id"), nullable=False)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=True)
    status = Column(String(50), nullable=False, default="pending", index=True)
    config = Column(JSON, nullable=False)
    results = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    experiment = relationship("ExperimentDB", back_populates="jobs")
    agent = relationship("AgentDB", back_populates="jobs")


class AgentDB(Base):
    """Database model for agents."""
    
    __tablename__ = "agents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, index=True)
    host = Column(String(255), nullable=False)
    port = Column(Integer, nullable=False)
    capabilities = Column(JSON, nullable=False)
    status = Column(String(50), nullable=False, default="offline", index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_heartbeat = Column(DateTime, nullable=True)

    # Relationships
    jobs = relationship("JobDB", back_populates="agent")

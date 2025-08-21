"""Database connection and session management."""

import os
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from .config import settings

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for models
Base = declarative_base()


async def get_db() -> AsyncSession:
    """Dependency to get database session."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """Initialize database tables."""
    # Ensure database directory exists
    if settings.database_url.startswith("sqlite"):
        # Extract path from sqlite URL (format: sqlite+aiosqlite:///./data/ruckus.db)
        db_path = settings.database_url.split("///")[-1]
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    async with engine.begin() as conn:
        # TODO: Create tables
        # await conn.run_sync(Base.metadata.create_all)
        pass

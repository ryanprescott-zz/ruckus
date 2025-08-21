"""Benchmark tasks."""

from .base import BaseTask
from .wikipedia_summarization import WikipediaSummarizationTask

__all__ = ["BaseTask", "WikipediaSummarizationTask"]
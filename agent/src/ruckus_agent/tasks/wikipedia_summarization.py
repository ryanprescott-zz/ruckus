"""Wikipedia article summarization task."""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from .base import BaseTask, TaskResult

logger = logging.getLogger(__name__)


class WikipediaSummarizationTask(BaseTask):
    """Summarize Wikipedia articles."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.articles = []
        self.references = []
        logger.info(f"WikipediaSummarizationTask initialized with config: {config}")

    async def prepare(self) -> None:
        """Load Wikipedia articles."""
        logger.info("WikipediaSummarizationTask preparing data")
        try:
            # TODO: Load articles from configured path
            data_path = Path(self.config.get("data_path", "data/wikipedia"))
            logger.debug(f"Loading articles from: {data_path}")

            # Load articles based on categories
            categories = self.config.get("categories", ["short", "medium", "long"])
            logger.debug(f"Processing categories: {categories}")
            
            for category in categories:
                category_path = data_path / category
                if category_path.exists():
                    logger.debug(f"Found category path: {category_path}")
                    # TODO: Load articles
                    pass
                else:
                    logger.warning(f"Category path not found: {category_path}")

            logger.info(f"WikipediaSummarizationTask loaded {len(self.articles)} articles for summarization")
        except Exception as e:
            logger.error(f"WikipediaSummarizationTask preparation failed: {e}")
            raise

    async def run(self, model_adapter, parameters: Dict[str, Any]) -> TaskResult:
        """Run summarization task."""
        logger.info(f"WikipediaSummarizationTask starting run with {len(self.articles)} articles")
        try:
            outputs = []
            metrics = {
                "articles_processed": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }

            # Process each article
            for i, article in enumerate(self.articles):
                logger.debug(f"Processing article {i+1}/{len(self.articles)}")
                
                # Create prompt
                prompt = self._create_prompt(article)

                # Generate summary
                summary = await model_adapter.generate(prompt, parameters)
                outputs.append(summary)

                # Update metrics
                metrics["articles_processed"] += 1

                # Count tokens if available
                try:
                    input_tokens = await model_adapter.count_tokens(prompt)
                    output_tokens = await model_adapter.count_tokens(summary)
                    metrics["total_input_tokens"] += input_tokens
                    metrics["total_output_tokens"] += output_tokens
                    logger.debug(f"Article {i+1} tokens - input: {input_tokens}, output: {output_tokens}")
                except Exception as e:
                    logger.warning(f"Token counting failed for article {i+1}: {e}")

            result = TaskResult(
                output=outputs,
                metrics=metrics,
                metadata={
                    "task": "wikipedia_summarization",
                    "num_articles": len(self.articles),
                }
            )
            logger.info(f"WikipediaSummarizationTask completed successfully: {metrics['articles_processed']} articles processed")
            return result
        except Exception as e:
            logger.error(f"WikipediaSummarizationTask run failed: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up task resources."""
        logger.info("WikipediaSummarizationTask cleaning up resources")
        self.articles = []
        self.references = []
        logger.debug("WikipediaSummarizationTask cleanup completed")

    def _create_prompt(self, article: str) -> str:
        """Create summarization prompt."""
        template = self.config.get(
            "prompt_template",
            "Summarize the following article in a concise paragraph:\n\n{article}\n\nSummary:"
        )
        return template.format(article=article)

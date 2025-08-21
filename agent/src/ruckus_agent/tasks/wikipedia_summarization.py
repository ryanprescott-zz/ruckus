"""Wikipedia article summarization task."""

from typing import Dict, Any, List, Optional
from pathlib import Path
from .base import BaseTask, TaskResult


class WikipediaSummarizationTask(BaseTask):
    """Summarize Wikipedia articles."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.articles = []
        self.references = []

    async def prepare(self) -> None:
        """Load Wikipedia articles."""
        # TODO: Load articles from configured path
        data_path = Path(self.config.get("data_path", "data/wikipedia"))

        # Load articles based on categories
        categories = self.config.get("categories", ["short", "medium", "long"])
        for category in categories:
            category_path = data_path / category
            if category_path.exists():
                # TODO: Load articles
                pass

        print(f"Loaded {len(self.articles)} articles for summarization")

    async def run(self, model_adapter, parameters: Dict[str, Any]) -> TaskResult:
        """Run summarization task."""
        outputs = []
        metrics = {
            "articles_processed": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

        # Process each article
        for article in self.articles:
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
            except Exception:
                pass

        return TaskResult(
            output=outputs,
            metrics=metrics,
            metadata={
                "task": "wikipedia_summarization",
                "num_articles": len(self.articles),
            }
        )

    async def cleanup(self) -> None:
        """Clean up task resources."""
        self.articles = []
        self.references = []

    def _create_prompt(self, article: str) -> str:
        """Create summarization prompt."""
        template = self.config.get(
            "prompt_template",
            "Summarize the following article in a concise paragraph:\n\n{article}\n\nSummary:"
        )
        return template.format(article=article)
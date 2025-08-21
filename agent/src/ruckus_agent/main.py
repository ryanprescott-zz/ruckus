"""RUCKUS agent main entry point."""

import asyncio
from .core.agent import Agent


async def main():
    """Main entry point."""
    agent = Agent()
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())

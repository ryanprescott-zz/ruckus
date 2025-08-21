"""RUCKUS agent core implementation."""

import asyncio
from .detector import SystemDetector
from .models import AgentConfig


class Agent:
    """RUCKUS agent implementation."""
    
    def __init__(self, config: AgentConfig = None):
        """Initialize agent."""
        self.config = config or AgentConfig()
        self.detector = SystemDetector()
        
    async def start(self):
        """Start the agent."""
        print(f"Starting RUCKUS agent {self.config.agent_id}")
        capabilities = self.detector.detect_capabilities()
        print(f"Detected capabilities: {capabilities}")
        
        # Main agent loop would go here
        while True:
            await asyncio.sleep(self.config.poll_interval)

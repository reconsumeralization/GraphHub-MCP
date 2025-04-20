"""
AgentNode module for FastMCP builder.

Defines the AgentNode class, representing an agent node in the builder graph.
"""

from typing import Dict, Any, Optional

class AgentNode:
    """
    Represents an agent node in the FastMCP builder graph.

    Attributes:
        node_id (str): Unique identifier for the node.
        config (Dict[str, Any]): Configuration dictionary for the agent.
        state (Optional[Dict[str, Any]]): Optional runtime state for the agent.
    """

    def __init__(self, node_id: str, config: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize an AgentNode.

        Args:
            node_id (str): Unique identifier for the node.
            config (Dict[str, Any]): Configuration dictionary for the agent.
            state (Optional[Dict[str, Any]]): Optional runtime state for the agent.
        """
        self.node_id: str = node_id
        self.config: Dict[str, Any] = config
        self.state: Dict[str, Any] = state.copy() if state else {}

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the agent node logic.

        Args:
            input_data (Dict[str, Any]): Input data for the agent.

        Returns:
            Dict[str, Any]: Output data after agent processing.
        """
        # TODO: Implement agent-specific logic here.
        # For now, just echo input_data for scaffolding.
        output: Dict[str, Any] = {
            "node_id": self.node_id,
            "input": input_data,
            "config": self.config,
            "state": self.state,
            "result": "Agent logic not yet implemented."
        }
        return output

    def update_state(self, updates: Dict[str, Any]) -> None:
        """
        Updates the agent's internal state.

        Args:
            updates (Dict[str, Any]): State updates to apply.
        """
        if not isinstance(updates, dict):
            raise ValueError("State updates must be a dictionary.")
        self.state.update(updates)

    def reset_state(self) -> None:
        """
        Resets the agent's internal state to an empty dictionary.
        """
        self.state.clear()

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a copy of the agent's internal state.

        Returns:
            Dict[str, Any]: The current state of the agent.
        """
        return self.state.copy()

# TODO: Add unit tests for AgentNode covering initialization, run, state updates, reset_state, and get_state.
# TODO: Consider edge cases for state mutation and thread safety if AgentNode is used concurrently.
from typing import Any, Dict, Callable, Protocol
from .custom_nodes import BaseNode

class RunnableSubgraph(Protocol):
    def run(self, inputs: Dict[str, Any]) -> Any:
        ...

class AIGraphRunnerNode(BaseNode):
    """
    Node that runs a subgraph or graph runner (must have a .run(inputs) method).
    """
    subgraph: RunnableSubgraph

    def __init__(self, node_id: str, label: str, subgraph: RunnableSubgraph) -> None:
        super().__init__(node_id, label)
        self.subgraph = subgraph

    def process(self, inputs: Dict[str, Any]) -> Any:
        # TODO: Consider supporting input validation schemas for subgraphs
        if not hasattr(self.subgraph, "run") or not callable(getattr(self.subgraph, "run")):
            raise AttributeError(
                f"Subgraph in node {self.node_id} does not have a callable 'run' method"
            )
        if not isinstance(inputs, dict):
            raise TypeError(
                f"Inputs to subgraph in node {self.node_id} must be a dict"
            )
        # TODO: Add logging of subgraph execution for audit trail
        return self.subgraph.run(inputs)

    def to_dict(self) -> Dict[str, Any]:
        # TODO: Consider serializing subgraph metadata if available
        return {
            "type": "AIGraphRunnerNode",
            "node_id": self.node_id,
            "label": self.label,
            "subgraph": repr(self.subgraph),
        }


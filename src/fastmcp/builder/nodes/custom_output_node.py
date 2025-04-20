from typing import Any, Dict, List, Optional, Callable
from .custom_nodes import BaseNode

class CustomOutputNode(BaseNode):
    """
    Node that receives and stores a value as output.
    """
    input_value: Any

    def __init__(self, node_id: str, label: str) -> None:
        super().__init__(node_id, label)
        self.input_value: Any = None

    def process(self, input_value: Any) -> None:
        if input_value is None:
            raise ValueError(f"Output node {self.node_id} received None as input_value")
        self.input_value = input_value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "CustomOutputNode",
            "node_id": self.node_id,
            "label": self.label,
            "input_value": self.input_value,
        }



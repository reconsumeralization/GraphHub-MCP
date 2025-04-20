from typing import Any, Dict, List, Optional, Callable
from .custom_nodes import BaseNode

class CustomProcessingNode(BaseNode):
    """
    Node that applies a user-supplied operation to its input.
    """
    operation: Callable[[Any], Any]

    def __init__(self, node_id: str, label: str, operation: Callable[[Any], Any]) -> None:
        super().__init__(node_id, label)
        self.operation = operation

    def process(self, input_value: Any) -> Any:
        if not callable(self.operation):
            raise TypeError(f"Operation for node {self.node_id} is not callable")
        try:
            return self.operation(input_value)
        except Exception as e:
            import logging
            logging.error(f"Processing failed in node {self.node_id}: {e}", exc_info=True)
            raise RuntimeError(f"Processing failed in node {self.node_id}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        op_name = getattr(self.operation, "__name__", str(self.operation))
        return {
            "type": "CustomProcessingNode",
            "node_id": self.node_id,
            "label": self.label,
            "operation": op_name,
        }



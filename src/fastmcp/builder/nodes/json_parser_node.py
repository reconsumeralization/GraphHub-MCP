from typing import Any, Dict, List, Optional, Callable
from .custom_nodes import BaseNode

class JSONParserNode(BaseNode):
    """
    Node that parses a JSON string into Python objects.
    """
    def process(self, json_string: str) -> Any:
        import json
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input for node {self.node_id}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "JSONParserNode",
            "node_id": self.node_id,
            "label": self.label,
        }



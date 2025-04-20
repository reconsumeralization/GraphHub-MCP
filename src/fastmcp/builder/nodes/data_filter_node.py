from typing import Any, Dict, List, Optional, Callable
from .custom_nodes import BaseNode

class DataFilterNode(BaseNode):
    """
    Node that filters a list of items based on a condition function.
    """
    condition: Callable[[Any], bool]

    def __init__(self, node_id: str, label: str, condition: Callable[[Any], bool]) -> None:
        super().__init__(node_id, label)
        self.condition = condition

    def process(self, items: List[Any]) -> List[Any]:
        if not isinstance(items, list):
            raise TypeError(f"DataFilterNode expects list input, got {type(items)}")
        return [item for item in items if self.condition(item)]

    def to_dict(self) -> Dict[str, Any]:
        cond_name = getattr(self.condition, "__name__", str(self.condition))
        return {
            "type": "DataFilterNode",
            "node_id": self.node_id,
            "label": self.label,
            "condition": cond_name,
        }



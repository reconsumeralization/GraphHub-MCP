from typing import Any, Dict, List, Optional, Callable
from .custom_nodes import BaseNode

class DataMergeNode(BaseNode):
    """
    Node that merges multiple dict inputs into one according to a strategy.
    """
    strategy: str

    def __init__(self, node_id: str, label: str, strategy: str = "overwrite") -> None:
        super().__init__(node_id, label)
        self.strategy = strategy.lower()

    def process(self, dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(dicts, list):
            raise TypeError(f"DataMergeNode expects list of dicts, got {type(dicts)}")
        result: Dict[str, Any] = {}
        for d in dicts:
            if not isinstance(d, dict):
                raise TypeError(f"DataMergeNode list items must be dicts, got {type(d)}")
            if self.strategy == "overwrite":
                result.update(d)
            else:
                # TODO: implement deep merge strategy
                raise NotImplementedError(f"Merge strategy {self.strategy} not implemented.")
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "DataMergeNode",
            "node_id": self.node_id,
            "label": self.label,
            "strategy": self.strategy,
        }



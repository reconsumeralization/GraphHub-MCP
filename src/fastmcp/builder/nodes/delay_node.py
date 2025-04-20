from typing import Any, Dict, List, Optional, Callable
from .custom_nodes import BaseNode

class DelayNode(BaseNode):
    """
    Node that delays execution for a specified duration.
    """
    duration: float

    def __init__(self, node_id: str, label: str, duration: float) -> None:
        super().__init__(node_id, label)
        self.duration = duration

    def process(self, _: Any = None) -> None:
        import time
        time.sleep(self.duration)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "DelayNode",
            "node_id": self.node_id,
            "label": self.label,
            "duration": self.duration,
        }



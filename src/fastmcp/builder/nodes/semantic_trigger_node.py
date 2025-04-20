from typing import Any, Dict, Callable, Optional
from .custom_nodes import BaseNode

class SemanticTriggerNode(BaseNode):
    """
    Node that waits for a semantic event or UI change specified in natural language.
    Properties:
      - trigger_description: str (e.g., "When the invoice table appears")
      - check_interval: float seconds between polls
      - timeout: Optional[float] overall wait time
    """
    trigger_description: str
    check_interval: float
    timeout: Optional[float]

    def __init__(
        self,
        node_id: str,
        label: str,
        trigger_description: str,
        check_interval: float = 1.0,
        timeout: Optional[float] = None
    ) -> None:
        super().__init__(node_id, label)
        self.trigger_description = trigger_description
        self.check_interval = check_interval
        self.timeout = timeout

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: Poll semantic targeting engine until trigger condition is met
        raise NotImplementedError(
            f"SemanticTriggerNode [{self.node_id}] not implemented"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "SemanticTriggerNode",
            "node_id": self.node_id,
            "label": self.label,
            "trigger_description": self.trigger_description,
            "check_interval": self.check_interval,
            "timeout": self.timeout,
        } 
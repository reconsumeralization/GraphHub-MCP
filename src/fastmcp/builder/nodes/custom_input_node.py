from typing import Any, Dict, Optional
from .custom_nodes import BaseNode

class CustomInputNode(BaseNode):
    """
    Node that provides a static or externally-injected value.

    - Enforces explicit typing and validation.
    - Raises clear errors for missing input.
    - Designed for extensibility and robust integration.
    """

    value: Any

    def __init__(self, node_id: str, label: str, value: Optional[Any] = None) -> None:
        super().__init__(node_id, label)
        self.value = value

    def process(self) -> Any:
        """
        Returns the stored value after validation.

        Raises:
            ValueError: If the value is None.
        """
        if self.value is None:
            # TODO: Consider supporting default values or type-based validation in future.
            raise ValueError(f"Input value for node '{self.node_id}' is None")
        return self.value

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the node to a dictionary for graph export or debugging.

        Returns:
            Dict[str, Any]: Node properties as a dictionary.
        """
        return {
            "type": self.__class__.__name__,
            "node_id": self.node_id,
            "label": self.label,
            "value": self.value,
        }

# TODO: Add unit tests for edge cases (e.g., value is 0, empty string, empty list, etc.)
# TODO: Consider supporting type hints for value and runtime type checking.

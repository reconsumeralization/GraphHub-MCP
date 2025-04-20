from typing import Any, Dict, List, Union
from .custom_nodes import BaseNode

class AINormalizeNode(BaseNode):
    """
    Node that normalizes numeric input(s) using mean and std.
    """
    mean: float
    std: float

    def __init__(self, node_id: str, label: str, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__(node_id, label)
        self.mean = mean
        # Avoid division by zero, but log a warning for clarity
        if std == 0:
            # TODO: Consider raising a ValueError or logging a warning instead of silently setting std to 1.0
            self.std = 1.0
        else:
            self.std = std

    def process(self, input_value: Union[float, List[float]]) -> Union[float, List[float]]:
        """
        Normalize a single float or a list of floats using the node's mean and std.
        """
        if isinstance(input_value, list):
            # Defensive: ensure all elements are numeric
            if not all(isinstance(x, (int, float)) for x in input_value):
                raise TypeError("All elements in input_value list must be int or float")
            return [(x - self.mean) / self.std for x in input_value]
        elif isinstance(input_value, (int, float)):
            return (input_value - self.mean) / self.std
        else:
            raise TypeError(f"AINormalizeNode expects float or List[float], got {type(input_value)}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the node to a dictionary.
        """
        return {
            "type": "AINormalizeNode",
            "node_id": self.node_id,
            "label": self.label,
            "mean": self.mean,
            "std": self.std,
        }

# TODO: Add unit tests for edge cases (e.g., std=0, input_value with non-numeric types)

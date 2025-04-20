from typing import Any, Dict, Protocol, TypeVar, Generic, Optional
from .custom_nodes import BaseNode

# --- Type Definitions ---

class VisionModelProtocol(Protocol):
    """
    Protocol for AI vision models that require a .predict method.
    """
    def predict(self, image: Any) -> Any:
        ...

TModel = TypeVar("TModel", bound=VisionModelProtocol)

class AIVisionModelNode(BaseNode, Generic[TModel]):
    """
    Node that wraps an AI vision model (must have a .predict method).
    """
    model_name: str
    model: Optional[TModel]

    def __init__(
        self,
        node_id: str,
        label: str,
        model_name: str,
        model: Optional[TModel]
    ) -> None:
        super().__init__(node_id, label)
        self.model_name = model_name
        self.model = model

    def process(self, image: Any) -> Any:
        """
        Process an image using the vision model's predict method.

        Args:
            image: The input image to process.

        Returns:
            The prediction result from the model.

        Raises:
            ValueError: If the model or image is not provided.
            AttributeError: If the model does not have a callable 'predict' method.
        """
        if self.model is None:
            raise ValueError(f"Model not loaded in node {self.node_id}")
        # Enforce the protocol at runtime for extra safety
        predict = getattr(self.model, "predict", None)
        if not callable(predict):
            raise AttributeError(
                f"Model in node {self.node_id} does not have a callable 'predict' method"
            )
        if image is None:
            raise ValueError(f"Image input to node {self.node_id} is None")
        # TODO: Add input validation for image type/format if possible
        return predict(image)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the node to a dictionary (excluding the model object).

        Returns:
            A dictionary representation of the node.
        """
        return {
            "type": "AIVisionModelNode",
            "node_id": self.node_id,
            "label": self.label,
            "model_name": self.model_name,
            # NOTE: Model object is intentionally not serialized for security and size reasons
        }

# TODO: Add unit tests for AIVisionModelNode covering:
#   - Model missing
#   - Model without predict
#   - Image missing
#   - Successful prediction
#   - Edge cases (e.g., invalid image type)

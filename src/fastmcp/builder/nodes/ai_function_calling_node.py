from typing import Any, Dict, List, TypedDict
from .custom_nodes import BaseNode

class ToolSchema(TypedDict):
    name: str
    description: str
    parameters: Dict[str, Any]

class AIFunctionCallingNode(BaseNode):
    """
    Node that lets an LLM decide which function/tool to call with given schemas.
    """
    model_id: str
    api_key_ref: str
    system_prompt: str
    tool_schemas: List[ToolSchema]
    temperature: float
    max_tokens: int

    def __init__(
        self,
        node_id: str,
        label: str,
        model_id: str,
        api_key_ref: str,
        system_prompt: str,
        tool_schemas: List[ToolSchema],
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> None:
        super().__init__(node_id, label)
        self.model_id = model_id
        self.api_key_ref = api_key_ref
        self.system_prompt = system_prompt
        self.tool_schemas = tool_schemas
        self.temperature = temperature
        self.max_tokens = max_tokens

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data by calling the LLM to select a function/tool and parse arguments.

        Args:
            input_data (Dict[str, Any]): The input data for the node.

        Returns:
            Dict[str, Any]: The output data after function selection and argument parsing.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        # TODO: Integrate with LLM API to select function/tool and parse arguments.
        # TODO: Validate input_data and sanitize before sending to LLM.
        # TODO: Handle LLM errors gracefully and provide actionable error messages.
        # TODO: Add unit tests for edge cases and error handling.
        raise NotImplementedError(
            f"AIFunctionCallingNode [{self.node_id}] logic not implemented."
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the node to a dictionary.

        Returns:
            Dict[str, Any]: The serialized node.
        """
        return {
            "type": "AIFunctionCallingNode",
            "node_id": self.node_id,
            "label": self.label,
            "model_id": self.model_id,
            "system_prompt": self.system_prompt,
            "tool_schemas": self.tool_schemas,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

# TODO: Consider moving ToolSchema to a shared types module if used elsewhere.
# TODO: Add input/output validation and sanitization for all LLM interactions.
# TODO: Add logging for all LLM calls and error paths.

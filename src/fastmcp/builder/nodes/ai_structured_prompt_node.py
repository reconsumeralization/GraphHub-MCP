from typing import Any, Dict, List, Optional, Callable
from .custom_nodes import BaseNode

class AIStructuredPromptNode(BaseNode):
    """
    Node that interacts with a Large Language Model (LLM) via structured prompts.
    """
    model_id: str
    api_key_ref: str
    system_prompt: str
    user_prompt_template: str
    output_schema: Any
    temperature: float
    max_tokens: int
    stop_sequences: Optional[List[str]]

    def __init__(self, node_id: str, label: str, model_id: str, api_key_ref: str,
                 system_prompt: str, user_prompt_template: str, output_schema: Any,
                 temperature: float = 0.7, max_tokens: int = 256,
                 stop_sequences: Optional[List[str]] = None) -> None:
        super().__init__(node_id, label)
        self.model_id = model_id
        self.api_key_ref = api_key_ref
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.output_schema = output_schema
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop_sequences = stop_sequences

    def process(self, input_data: Any) -> Any:
        # TODO: Implement actual LLM API call and output parsing.
        raise NotImplementedError(f"AIStructuredPromptNode [{self.node_id}] logic not implemented.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "AIStructuredPromptNode",
            "node_id": self.node_id,
            "label": self.label,
            "model_id": self.model_id,
            "system_prompt": self.system_prompt,
            "user_prompt_template": self.user_prompt_template,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop_sequences": self.stop_sequences,
        }



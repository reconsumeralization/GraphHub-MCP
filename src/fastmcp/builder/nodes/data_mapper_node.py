from typing import Any, Dict, List, Optional, Callable
from .custom_nodes import BaseNode

class DataMapperNode(BaseNode):
    """
    Node that maps and renames fields from input data based on mapping rules.
    """
    mapping_rules: Dict[str, str]

    def __init__(self, node_id: str, label: str, mapping_rules: Dict[str, str]) -> None:
        super().__init__(node_id, label)
        self.mapping_rules = mapping_rules

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(input_data, dict):
            raise TypeError(f"DataMapperNode expects dict input, got {type(input_data)}")
        output: Dict[str, Any] = {}
        for source_field, target_field in self.mapping_rules.items():
            output[target_field] = input_data.get(source_field)
        return output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "DataMapperNode",
            "node_id": self.node_id,
            "label": self.label,
            "mapping_rules": self.mapping_rules,
        }



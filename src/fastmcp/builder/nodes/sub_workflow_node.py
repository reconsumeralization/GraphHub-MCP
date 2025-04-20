from typing import Any, Dict, Optional, Mapping, cast
from .custom_nodes import BaseNode

class SubWorkflowNode(BaseNode):
    """
    Node that encapsulates a reusable sub-workflow (subgraph).
    Designed for agentic and GUI-driven workflow composition.
    """

    subflow_spec: Dict[str, Any]
    parameters: Dict[str, Any]
    version: Optional[str]
    description: Optional[str]

    def __init__(
        self,
        node_id: str,
        label: str,
        subflow_spec: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Args:
            node_id: Unique identifier for this node.
            label: Human-readable label for the node.
            subflow_spec: The subgraph specification (MCP YAML/JSON dict).
            parameters: Default parameters for the subflow.
            version: Optional version string for the subflow.
            description: Optional description for GUI/agent UX.
        """
        super().__init__(node_id, label)
        self.subflow_spec = subflow_spec
        self.parameters = parameters or {}
        self.version = version
        self.description = description

    def process(self, inputs: Dict[str, Any]) -> Any:
        """
        Executes the sub-workflow with merged parameters and inputs.
        In a real system, this would delegate to the workflow engine's subflow runner.

        Args:
            inputs: Input values for the subflow.

        Returns:
            Dict with execution metadata and merged inputs.
        """
        if not isinstance(inputs, dict):
            raise TypeError(f"Inputs to subflow in node {self.node_id} must be a dict")
        merged_inputs = {**self.parameters, **inputs}
        # TODO: Integrate with WorkflowManager/engine for actual subflow execution.
        # TODO: Add agent context propagation if available.
        return {
            "subflow_executed": True,
            "subflow_spec": self.subflow_spec,
            "inputs": merged_inputs,
            "version": self.version,
            "node_id": self.node_id,
            "label": self.label,
            "description": self.description,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the node for GUI, registry, or export.
        """
        return {
            "type": "SubWorkflowNode",
            "node_id": self.node_id,
            "label": self.label,
            "subflow_spec": self.subflow_spec,
            "parameters": self.parameters,
            "version": self.version,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubWorkflowNode":
        """
        Factory for deserialization from dict (for GUI/agent/registry use).
        """
        return cls(
            node_id=cast(str, data.get("node_id")),
            label=cast(str, data.get("label")),
            subflow_spec=cast(Dict[str, Any], data.get("subflow_spec", {})),
            parameters=cast(Optional[Dict[str, Any]], data.get("parameters")),
            version=cast(Optional[str], data.get("version")),
            description=cast(Optional[str], data.get("description")),
        )

    def get_agent_metadata(self) -> Dict[str, Any]:
        """
        Returns metadata for agentic UX and GUI display.
        """
        return {
            "node_id": self.node_id,
            "label": self.label,
            "type": "SubWorkflowNode",
            "description": self.description or "Reusable sub-workflow node.",
            "parameters": list(self.parameters.keys()),
            "subflow_spec_summary": self.subflow_spec.get("name", None) or str(self.subflow_spec)[:80],
            "version": self.version,
        }

# TODO: Register SubWorkflowNode in node_registry for dynamic discovery.
# TODO: Add validation for subflow_spec structure and parameter types.
# TODO: Support subflow spec import/export via GUI (see MasterPlan Sprint 3).

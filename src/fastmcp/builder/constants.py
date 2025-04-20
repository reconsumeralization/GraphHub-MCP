"""
FastMCP Shared Constants & Type-Safe Enumerations

This module centralizes all node category and registry identifiers to ensure
consistency and deep synergy across the builder, agent, and execution engine.

Feature-Driven Extensions:
- Agent-assisted suggestions (GUI/agent)
- Adaptive execution control (engine/agent)
- Workflow optimization & analysis (builder/agent)
- Execution history learning (engine/agent)
- Composite & hierarchical sub-workflows (builder/agent)

All enums and types in this file are single-source-of-truth for cross-package use.
"""

from enum import Enum, unique
from typing import Dict, List, TypedDict, Optional, Literal, Union

# --- Node Categories ---

@unique
class NodeCategory(str, Enum):
    """Type-safe node categories for workflow nodes."""
    MCP = "MCP Tasks"
    CONTROL = "Workflow Control"
    AI = "AI Tasks"
    NATURAL_LANGUAGE = "Natural Language"
    VISION = "AI Vision Tasks"
    SUBWORKFLOW = "Sub-Workflow Macro"  # For composite/hierarchical sub-workflows
    AGENT = "Agent Nodes"
    SEMANTIC = "Semantic Nodes"
    IO = "Input/Output"
    DATA = "Data Processing"
    UTILITY = "Utility"

NODE_CATEGORY_LABELS: Dict[NodeCategory, str] = {
    NodeCategory.MCP: "MCP Tasks",
    NodeCategory.CONTROL: "Workflow Control",
    NodeCategory.AI: "AI Tasks",
    NodeCategory.NATURAL_LANGUAGE: "Natural Language",
    NodeCategory.VISION: "AI Vision Tasks",
    NodeCategory.SUBWORKFLOW: "Sub-Workflow Macro",
    NodeCategory.AGENT: "Agent Nodes",
    NodeCategory.SEMANTIC: "Semantic Nodes",
    NodeCategory.IO: "Input/Output",
    NodeCategory.DATA: "Data Processing",
    NodeCategory.UTILITY: "Utility",
}

# --- Node Registry Identifiers ---

@unique
class NodeRegistryIdentifier(str, Enum):
    """Type-safe registry identifiers for node registration and lookup."""
    MCP_SERVER = "mcp.server.nodes"
    AI = "mcp.ai.nodes"
    NL = "mcp.nl.nodes"
    VISION = "mcp.vision.nodes"
    SUBWORKFLOW = "mcp.subflow.nodes"  # For sub-workflow macro nodes
    AGENT = "mcp.agent.nodes"
    SEMANTIC = "mcp.semantic.nodes"
    IO = "mcp.io.nodes"
    DATA = "mcp.data.nodes"
    UTILITY = "mcp.utility.nodes"

NODE_REGISTRY_LABELS: Dict[NodeRegistryIdentifier, str] = {
    NodeRegistryIdentifier.MCP_SERVER: "MCP Server Nodes",
    NodeRegistryIdentifier.AI: "AI Nodes",
    NodeRegistryIdentifier.NL: "Natural Language Nodes",
    NodeRegistryIdentifier.VISION: "Vision Nodes",
    NodeRegistryIdentifier.SUBWORKFLOW: "Sub-Workflow Macro Nodes",
    NodeRegistryIdentifier.AGENT: "Agent Nodes",
    NodeRegistryIdentifier.SEMANTIC: "Semantic Nodes",
    NodeRegistryIdentifier.IO: "Input/Output Nodes",
    NodeRegistryIdentifier.DATA: "Data Processing Nodes",
    NodeRegistryIdentifier.UTILITY: "Utility Nodes",
}

# --- Optimization Hints (for agent/builder synergy) ---

class OptimizationHint(TypedDict):
    description: str
    affected_nodes: List[str]
    suggestion: str
    priority: Optional[Literal["low", "medium", "high"]]  # For agent prioritization
    rationale: Optional[str]  # Why this suggestion is made

# --- Agent Suggestion Types ---

class AgentSuggestion(TypedDict):
    next_node_category: NodeCategory
    default_properties: Dict[str, str]
    auto_wire: Optional[List[str]]  # Node IDs to auto-connect
    rationale: Optional[str]  # Why this suggestion is made
    confidence: Optional[float]  # 0.0-1.0, agent's confidence in suggestion

# --- Agent-Builder Communication Types ---

class AgentAssistedEvent(TypedDict):
    """Event structure for agent-assisted suggestions in the builder GUI."""
    event_type: Literal["suggestion", "optimization", "auto_wire", "info"]
    payload: Union[AgentSuggestion, OptimizationHint, Dict[str, str]]
    timestamp: Optional[str]

# --- Legacy Constants (DEPRECATED: use enums above) ---

NODE_CATEGORY_MCP: str = NodeCategory.MCP.value
NODE_CATEGORY_CONTROL: str = NodeCategory.CONTROL.value
NODE_CATEGORY_AI: str = NodeCategory.AI.value
NODE_CATEGORY_NL: str = NodeCategory.NATURAL_LANGUAGE.value
NODE_CATEGORY_VISION: str = NodeCategory.VISION.value
NODE_CATEGORY_SUBWORKFLOW: str = NodeCategory.SUBWORKFLOW.value
NODE_CATEGORY_AGENT: str = NodeCategory.AGENT.value
NODE_CATEGORY_SEMANTIC: str = NodeCategory.SEMANTIC.value
NODE_CATEGORY_IO: str = NodeCategory.IO.value
NODE_CATEGORY_DATA: str = NodeCategory.DATA.value
NODE_CATEGORY_UTILITY: str = NodeCategory.UTILITY.value

MCP_SERVER_IDENTIFIER: str = NodeRegistryIdentifier.MCP_SERVER.value
AI_NODE_IDENTIFIER: str = NodeRegistryIdentifier.AI.value
NL_NODE_IDENTIFIER: str = NodeRegistryIdentifier.NL.value
VISION_NODE_IDENTIFIER: str = NodeRegistryIdentifier.VISION.value
SUBWORKFLOW_NODE_IDENTIFIER: str = NodeRegistryIdentifier.SUBWORKFLOW.value
AGENT_NODE_IDENTIFIER: str = NodeRegistryIdentifier.AGENT.value
SEMANTIC_NODE_IDENTIFIER: str = NodeRegistryIdentifier.SEMANTIC.value
IO_NODE_IDENTIFIER: str = NodeRegistryIdentifier.IO.value
DATA_NODE_IDENTIFIER: str = NodeRegistryIdentifier.DATA.value
UTILITY_NODE_IDENTIFIER: str = NodeRegistryIdentifier.UTILITY.value

# --- Agent/Builder Synergy Integration ---

# These hooks are to be called by WorkflowManager and GUI for agent-assisted UX.
def emit_agent_suggestion_event(event: AgentAssistedEvent) -> None:
    """
    Signal/hook for agent-assisted suggestions in the builder GUI.
    GUI and WorkflowManager should call this to surface agent suggestions.
    """
    # TODO: Connect this to the GUI event bus or observer pattern.
    # TODO: Log all agent events for audit and learning.
    pass  # Implementation to be provided in WorkflowManager/GUI layer

# --- Documentation for Developers ---
"""
Developer Notes:
- All new node categories and registry identifiers must be added to the enums above.
- Use NodeCategory.SUBWORKFLOW and NodeRegistryIdentifier.SUBWORKFLOW for subflow macro nodes.
- Use AgentSuggestion and OptimizationHint types for agent/builder communication.
- Use emit_agent_suggestion_event to surface agent suggestions in the GUI.
- All usages in builder, agent, and execution_engine must use enums for type safety.
- See developer docs for integration patterns and extension guidelines.
"""

# TODO: Remove all legacy string constants in the codebase and use enums exclusively.
# TODO: Ensure all agent/builder communication uses AgentSuggestion, OptimizationHint, and AgentAssistedEvent.
# TODO: Connect emit_agent_suggestion_event to the GUI event bus and WorkflowManager.
# TODO: Add audit logging for all agent-assisted events.
# TODO: Add/expand unit tests for agent suggestion and optimization hint types.
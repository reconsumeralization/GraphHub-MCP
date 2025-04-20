"""
FastMCP Builder Index

This module exposes the public API surface for the builder package, with a focus on agent usability and seamless agent integration within the GUI workflow builder.

Key Features:
- GUI code (`gui_launcher`) is separated from headless APIs (`workflow_manager`) and the execution engine.
- Agent logic is in the `fastmcp.agent` package, but agent tools and planners are exposed for builder/GUI use.
- All imports are explicit for static analysis and type checking.
- Types are consolidated in `types.py`; all public APIs are fully type-annotated.
- Agentic node types (SemanticTargetNode, SemanticTriggerNode, ActionRecorderNode) are registered and visible in the GUI.
- NodeRegistry is the single source of truth for node discovery, including agentic and custom nodes.
- TODO: Remove any remaining # type: ignore suppressions by refining signatures and implementing missing methods.
- See README for usage and CLI docs.

Agent-first, GUI-powered, and ready for next-gen automation. ☕️🚀
"""

# --- Builder Core Components ---
from .nodes.custom_nodes import BaseNode as CustomNode
from .nodes.node_registry import NodeRegistry
from .workflow_manager import WorkflowManager
from .gui_launcher import launch_gui_builder as launch_gui

# --- Agentic Node Types (ensure these are registered and visible in the GUI) ---
from .nodes.semantic_target_node import SemanticTargetNode
from .nodes.semantic_trigger_node import SemanticTriggerNode
from .nodes.action_recorder_node import ActionRecorderNode

# --- Execution Engine ---
from ..execution_engine.executor import ExecutionEngine

# --- Agent Components (for agent usability in builder and GUI) ---
from ..agent.graph_model import AgentGraph
from ..agent.planner import MCPGraphAgentPlanner
from ..agent.tools import AgenticWorkflowTools, GraphBuilderService

# --- Utilities ---
from .graph_cli import build_graph as graph_cli_main
from .graph_logger import GraphLogger
from .graph_validator import GraphValidator
from . import yamle_graph as YamleGraph

# --- Agent Usability Enhancements ---
# Register agentic node types in the NodeRegistry for GUI discoverability
NodeRegistry.register_node_type(SemanticTargetNode)
NodeRegistry.register_node_type(SemanticTriggerNode)
NodeRegistry.register_node_type(ActionRecorderNode)
# TODO: Consider auto-discovering all agentic nodes in nodes/ for future extensibility

# --- Public API ---
__all__ = [
    "AgentGraph",
    "AgenticWorkflowTools",
    "CustomNode",
    "ExecutionEngine",
    "GraphBuilderService",
    "GraphLogger",
    "GraphValidator",
    "MCPGraphAgentPlanner",
    "NodeRegistry",
    "WorkflowManager",
    "YamleGraph",
    "graph_cli_main",
    "launch_gui",
    # Agentic node types for direct use and GUI visibility
    "SemanticTargetNode",
    "SemanticTriggerNode",
    "ActionRecorderNode",
]

# TODO: Add type-annotated stubs for all public API functions and classes.
# TODO: Add agent usability tests for builder/GUI integration.
# TODO: Add accessibility (a11y) hooks for agentic nodes in the GUI.

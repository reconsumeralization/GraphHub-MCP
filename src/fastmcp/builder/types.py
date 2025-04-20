# pyright: reportMissingImports=false

"""
Centralized type definitions for FastMCP core modules, with agent- and GUI-enhanced metadata.

This module provides shared, strongly-typed structures for:
- Node/connection/graph specifications
- Patch/mutation operations
- Agent/GUI/Executor protocol interfaces
- Execution records and undo/rollback
- (Optional) Pydantic models for runtime validation

This file is intentionally UI-agnostic and agent-agnostic: it is imported by
the GUI, WorkflowManager (programmatic API), Execution Engine, and Agent packages.

Enhancements for agent usability and agent/GUI integration:
- NodeSpec and PropertySpec support agent/GUI hints, provenance, and semantic tags
- GraphSpec and PatchOperation support agent traceability and rationale
- SuggestionResult and WorkflowExecutionRecord support agent feedback, preview, and trace
"""

from typing import (
    TypedDict,
    List,
    Dict,
    Tuple,
    Optional,
    Any,
    runtime_checkable,
    Protocol,
    Literal,
)
from typing_extensions import NotRequired

# Fallback stub for BaseModel if pydantic is unavailable
try:
    from pydantic import BaseModel  # type: ignore
except ImportError:
    BaseModel = object

# --- Core Type Aliases ---
NodeID = str
PortName = str
WorkflowID = str
ExecutionID = str
GraphHandle = WorkflowID  # Alias for agent-facing graph handle

# --- Property Specification ---
class PropertySpec(TypedDict, total=False):
    value: Any
    type: str  # e.g. 'string', 'int', 'float', 'bool', 'json', 'combo', etc.
    items: List[Any]
    min: float
    max: float
    options: List[Any]
    description: str
    required: bool
    editable: bool
    visible: bool
    advanced: bool
    group: str
    placeholder: str
    choices: List[Any]
    validation_pattern: NotRequired[str]  # regex or similar
    ui_widget: NotRequired[str]  # e.g. 'slider', 'dropdown', 'colorpicker'
    agent_hint: NotRequired[str]  # hint for agent about property semantics
    provenance: NotRequired[str]  # e.g. 'user', 'agent', 'default'
    semantic_tags: NotRequired[List[str]]  # e.g. ['input', 'output', 'parameter']
    # For GUI/agent: show/hide, highlight, etc.

# --- Node and Connection Specifications ---
class NodeSpec(TypedDict):
    type: str
    name: str
    properties: Dict[str, PropertySpec]
    position: Optional[Tuple[float, float]]
    icon: NotRequired[str]
    color: NotRequired[str]
    description: NotRequired[str]
    category: NotRequired[str]
    agent_managed: NotRequired[bool]
    tags: NotRequired[List[str]]
    agent_provenance: NotRequired[str]  # e.g. 'user', 'agent', 'imported'
    agent_rationale: NotRequired[str]   # why the agent added/modified this node
    semantic_type: NotRequired[str]     # e.g. 'SemanticTrigger', 'SemanticTarget'
    gui_highlight: NotRequired[bool]    # for GUI to visually highlight agent nodes
    agent_metadata: NotRequired[Dict[str, Any]]  # arbitrary agent info

class ConnectionSpec(TypedDict):
    from_node: NodeID
    from_port: PortName
    to_node: NodeID
    to_port: PortName
    label: NotRequired[str]
    style: NotRequired[str]
    agent_provenance: NotRequired[str]  # e.g. 'user', 'agent'
    agent_confidence: NotRequired[float]  # agent's confidence in this connection
    agent_metadata: NotRequired[Dict[str, Any]]

# --- Graph Specification ---
class GraphSpec(TypedDict):
    workflow_id: WorkflowID
    description: NotRequired[Optional[str]]
    version: NotRequired[Optional[str]]
    nodes: Dict[NodeID, NodeSpec]
    connections: List[ConnectionSpec]
    metadata: NotRequired[Dict[str, Any]]
    created_by: NotRequired[str]
    created_at: NotRequired[str]
    updated_at: NotRequired[str]
    agent_context: NotRequired[Dict[str, Any]]
    agent_trace: NotRequired[List[Dict[str, Any]]]  # agent actions, for GUI/undo
    agent_recommendations: NotRequired[List[Dict[str, Any]]]  # for GUI to show agent suggestions

# --- Patch Specification (for mutations) ---
class PatchOperation(TypedDict, total=False):
    op: Literal['add', 'remove', 'replace', 'move', 'copy', 'test']
    path: str
    value: Any
    from_: str
    agent_generated: NotRequired[bool]
    rationale: NotRequired[str]
    agent_id: NotRequired[str]
    agent_name: NotRequired[str]
    agent_confidence: NotRequired[float]
    agent_metadata: NotRequired[Dict[str, Any]]

PatchSpec = List[PatchOperation]

# --- Suggestion Result (for agent recommendations) ---
class AutoWireSpec(TypedDict):
    from_node: NodeID
    from_port: PortName
    to_node: NodeID
    to_port: PortName
    confidence: NotRequired[float]
    agent_metadata: NotRequired[Dict[str, Any]]

class SuggestionResult(TypedDict):
    node_type: str
    default_properties: Dict[str, PropertySpec]
    auto_wire: Optional[List[AutoWireSpec]]
    rationale: str
    agent_name: NotRequired[str]
    confidence: NotRequired[float]
    preview: NotRequired[Dict[str, Any]]
    agent_trace: NotRequired[List[Dict[str, Any]]]  # for GUI to show agent's reasoning
    gui_highlight: NotRequired[bool]  # GUI can highlight agent suggestions
    agent_metadata: NotRequired[Dict[str, Any]]

# --- Workflow Execution Record ---
class WorkflowExecutionRecord(TypedDict, total=False):
    execution_id: ExecutionID
    graph_id: WorkflowID
    start_time: str
    end_time: Optional[str]
    status: str
    result: Any
    error: NotRequired[str]
    logs: NotRequired[List[str]]
    triggered_by: NotRequired[str]
    agent_id: NotRequired[Optional[str]]
    agent_details: NotRequired[Optional[Dict[str, Any]]]
    agent_trace: NotRequired[List[Dict[str, Any]]]
    agent_rationale: NotRequired[str]
    agent_metadata: NotRequired[Dict[str, Any]]

# --- Workflow Access Control ---
class WorkflowAccessControl(TypedDict, total=False):
    """Defines access permissions for a workflow."""
    owner_user: NotRequired[str]
    owner_group: NotRequired[str]
    read_users: NotRequired[List[str]]
    read_groups: NotRequired[List[str]]
    write_users: NotRequired[List[str]]
    write_groups: NotRequired[List[str]]
    execute_users: NotRequired[List[str]]
    execute_groups: NotRequired[List[str]]
    public_read: NotRequired[bool]
    public_execute: NotRequired[bool]

# --- Undo/Rollback Support for Mutations ---
class UndoMutationRecord(TypedDict):
    graph_id: WorkflowID
    patch_spec: PatchSpec
    timestamp: str
    user: NotRequired[str]
    reason: NotRequired[str]
    agent_generated: NotRequired[bool]
    context: NotRequired[Dict[str, Any]]
    agent_id: NotRequired[str]
    agent_rationale: NotRequired[str]
    agent_metadata: NotRequired[Dict[str, Any]]

# --- Error/Exception Types for Protocol Methods ---
class SuggestionError(Exception):
    """Raised when agent suggestion fails."""
    pass

class MutationConflictError(Exception):
    """Raised when a graph mutation cannot be applied due to a conflict."""
    pass

class OptimizationError(Exception):
    """Raised when graph optimization fails."""
    pass

class HistoryRetrievalError(Exception):
    """Raised when execution history cannot be retrieved."""
    pass

class SubflowExtractionError(Exception):
    """Raised when subflow extraction fails."""
    pass

# --- Protocols for Agent/Builder/Executor Interactions ---
@runtime_checkable
class AgentSuggestionProvider(Protocol):
    async def recommend_next_node(
        self,
        partial_graph_spec: GraphSpec,
        context: Optional[Dict[str, Any]] = None,
    ) -> SuggestionResult:
        """
        Agent recommends the next node to add, with rationale and preview.
        GUI can display rationale, preview, and highlight agent suggestions.
        """
        ...

@runtime_checkable
class AgentGraphMutationTool(Protocol):
    async def trigger_graph_mutation(
        self,
        graph_spec: GraphSpec,
        patch_spec: PatchSpec,
        context: Optional[Dict[str, Any]] = None,
    ) -> GraphSpec:
        """
        Agent applies a patch/mutation to the graph.
        GUI can show agent-generated changes and rationale.
        """
        ...

@runtime_checkable
class AgentGraphOptimizer(Protocol):
    async def optimize_graph(
        self,
        graph_spec: GraphSpec,
        context: Optional[Dict[str, Any]] = None,
    ) -> GraphSpec:
        """
        Agent optimizes the graph (e.g., refactoring, auto-wiring).
        GUI can show before/after and agent's reasoning.
        """
        ...

@runtime_checkable
class AgentHistoryProvider(Protocol):
    async def get_execution_history(
        self,
        graph_id: WorkflowID,
        limit: int = 10,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[WorkflowExecutionRecord]:
        """
        Agent provides execution history, including agent trace/rationale.
        GUI can display agent involvement in past runs.
        """
        ...

@runtime_checkable
class AgentSubflowManager(Protocol):
    async def extract_subflow(
        self,
        graph_spec: GraphSpec,
        node_ids: List[NodeID],
        subflow_name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GraphSpec, NodeID]:
        """
        Agent extracts a subflow from selected nodes.
        GUI can show agent's subflow extraction and rationale.
        """
        ...

# --- Pydantic Models for Runtime Validation (Optional) ---
try:
    from pydantic import BaseModel  # type: ignore

    class PropertySpecModel(BaseModel):
        value: Any
        type: str
        items: Optional[List[Any]] = None
        min: Optional[float] = None
        max: Optional[float] = None
        options: Optional[List[Any]] = None
        description: Optional[str] = None
        required: Optional[bool] = None
        editable: Optional[bool] = None
        visible: Optional[bool] = None
        advanced: Optional[bool] = None
        group: Optional[str] = None
        placeholder: Optional[str] = None
        choices: Optional[List[Any]] = None
        validation_pattern: Optional[str] = None
        ui_widget: Optional[str] = None
        agent_hint: Optional[str] = None
        provenance: Optional[str] = None
        semantic_tags: Optional[List[str]] = None

    class NodeSpecModel(BaseModel):
        type: str
        name: str
        properties: Dict[str, PropertySpec]
        position: Optional[Tuple[float, float]] = None
        icon: Optional[str] = None
        color: Optional[str] = None
        description: Optional[str] = None
        category: Optional[str] = None
        agent_managed: Optional[bool] = None
        tags: Optional[List[str]] = None
        agent_provenance: Optional[str] = None
        agent_rationale: Optional[str] = None
        semantic_type: Optional[str] = None
        gui_highlight: Optional[bool] = None
        agent_metadata: Optional[Dict[str, Any]] = None

    class ConnectionSpecModel(BaseModel):
        from_node: NodeID
        from_port: PortName
        to_node: NodeID
        to_port: PortName
        label: Optional[str] = None
        style: Optional[str] = None
        agent_provenance: Optional[str] = None
        agent_confidence: Optional[float] = None
        agent_metadata: Optional[Dict[str, Any]] = None

    class GraphSpecModel(BaseModel):
        workflow_id: WorkflowID
        description: Optional[str] = None
        version: Optional[str] = None
        nodes: Dict[NodeID, NodeSpec]
        connections: List[ConnectionSpec]
        metadata: Optional[Dict[str, Any]] = None
        created_by: Optional[str] = None
        created_at: Optional[str] = None
        updated_at: Optional[str] = None
        agent_context: Optional[Dict[str, Any]] = None
        agent_trace: Optional[List[Dict[str, Any]]] = None
        agent_recommendations: Optional[List[Dict[str, Any]]] = None

    class PatchOperationModel(BaseModel):
        op: Literal['add', 'remove', 'replace', 'move', 'copy', 'test']
        path: str
        value: Optional[Any] = None
        from_: Optional[str] = None
        agent_generated: Optional[bool] = None
        rationale: Optional[str] = None
        agent_id: Optional[str] = None
        agent_name: Optional[str] = None
        agent_confidence: Optional[float] = None
        agent_metadata: Optional[Dict[str, Any]] = None

    class AutoWireSpecModel(BaseModel):
        from_node: NodeID
        from_port: PortName
        to_node: NodeID
        to_port: PortName
        confidence: Optional[float] = None
        agent_metadata: Optional[Dict[str, Any]] = None

    class SuggestionResultModel(BaseModel):
        node_type: str
        default_properties: Dict[str, PropertySpec]
        auto_wire: Optional[List[AutoWireSpec]] = None
        rationale: str
        agent_name: Optional[str] = None
        confidence: Optional[float] = None
        preview: Optional[Dict[str, Any]] = None
        agent_trace: Optional[List[Dict[str, Any]]] = None
        gui_highlight: Optional[bool] = None
        agent_metadata: Optional[Dict[str, Any]] = None

    class WorkflowExecutionRecordModel(BaseModel):
        execution_id: ExecutionID
        graph_id: WorkflowID
        start_time: str
        end_time: Optional[str] = None
        status: str
        result: Any
        error: Optional[str] = None
        logs: Optional[List[str]] = None
        triggered_by: Optional[str] = None
        agent_id: Optional[str] = None
        agent_details: Optional[Dict[str, Any]] = None
        agent_trace: Optional[List[Dict[str, Any]]] = None
        agent_rationale: Optional[str] = None
        agent_metadata: Optional[Dict[str, Any]] = None

except ImportError:
    pass
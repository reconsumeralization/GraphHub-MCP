from typing import (
    Protocol,
    runtime_checkable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    TypedDict,
    Awaitable,
    Union,
    Literal,
)
import logging
import functools

# --- FastMCP Protocols for Agent-Builder Synergy ---
# Drop Codde Bombs of Damn that's some good code. ☕️🚀

# --- Dedicated Types (should be moved to types.py for centralization, but kept here for now for atomicity) ---

NodeID = str
PortName = str

class PropertySpec(TypedDict, total=False):
    value: Any
    type: Literal['string', 'int', 'float', 'bool', 'json', 'combo']
    combo_options: Optional[List[Any]]  # For combo type, allowed options
    min: Optional[float]  # For numeric types
    max: Optional[float]  # For numeric types
    pattern: Optional[str]  # For string validation (regex)
    required: Optional[bool]
    description: Optional[str]
    # Validation constraints can be extended as needed
    ui_hint: Optional[str]  # NEW: For GUI rendering hints (e.g., slider, dropdown)
    agent_editable: Optional[bool]  # NEW: If agent can auto-edit this property

class ConnectionSpec(TypedDict):
    from_node: NodeID
    from_port: PortName
    to_node: NodeID
    to_port: PortName

class NodeSpec(TypedDict):
    type: str
    name: str
    properties: Dict[str, PropertySpec]
    position: Optional[Tuple[float, float]]
    agent_generated: Optional[bool]  # NEW: Mark if node was agent-suggested
    agent_rationale: Optional[str]   # NEW: Why the agent suggested/created this node

class GraphSpec(TypedDict):
    workflow_id: str
    description: Optional[str]
    version: Optional[str]
    nodes: Dict[NodeID, NodeSpec]
    connections: List[ConnectionSpec]
    agent_metadata: Optional[Dict[str, Any]]  # NEW: For agent/builder GUI state, e.g. last suggestion, agent confidence

# PatchSpec follows JSON Patch RFC 6902 (https://datatracker.ietf.org/doc/html/rfc6902)
# Each operation is a dict with keys: op, path, value (for add/replace), from (for move/copy)
class PatchOperation(TypedDict, total=True):
    op: Literal['add', 'remove', 'replace', 'move', 'copy', 'test']
    path: str
    value: Optional[Any]
    from_: Optional[str]
    agent_generated: Optional[bool]  # NEW: Track if patch was agent-initiated
    rationale: Optional[str]         # NEW: Why the agent made this change

class PatchSpec(TypedDict):
    patch: List[PatchOperation]  # List of JSON Patch operations
    agent_id: Optional[str]      # NEW: Which agent/tool generated this patch
    agent_confidence: Optional[float]  # NEW: Confidence score for GUI display

class SuggestionResult(TypedDict):
    node_type: str
    default_properties: Dict[str, PropertySpec]
    auto_wire: Optional[List[Tuple[NodeID, PortName, NodeID, PortName]]]  # (from_node, from_port, to_node, to_port)
    rationale: str
    agent_confidence: Optional[float]  # NEW: For GUI to show agent certainty
    agent_actions: Optional[List[str]] # NEW: For GUI to show what agent can do next

class OptimizationResult(TypedDict):
    diff: PatchSpec  # Patch to apply
    hints: List[str]
    score: float
    agent_explanation: Optional[str]  # NEW: For GUI to show why optimization is suggested

class ExecutionRecord(TypedDict, total=False):
    execution_id: str
    status: Literal['success', 'failure', 'running', 'cancelled']
    start_time: Optional[str]
    end_time: Optional[str]
    error: Optional[str]
    input: Optional[Dict[str, Any]]
    output: Optional[Dict[str, Any]]
    node_trace: Optional[List[Dict[str, Any]]]
    agent_interventions: Optional[List[Dict[str, Any]]]  # NEW: Track agent actions during execution

class ExecutionInsights(TypedDict):
    success_rate: float
    avg_latency: float
    error_patterns: List[Dict[str, Any]]
    hot_spots: List[str]
    raw_records: Optional[List[ExecutionRecord]]
    agent_summary: Optional[str]  # NEW: Agent-generated summary for GUI

class SubflowExtractionResult(TypedDict):
    subflow_spec: GraphSpec
    macro_node_id: str
    version: str
    modified_graph_spec: Optional[GraphSpec]  # The graph after subflow extraction
    agent_rationale: Optional[str]            # NEW: Why the agent suggested this subflow

# --- Custom Exception Classes for Protocols ---

class SuggestionError(Exception):
    """Raised when agent suggestion fails."""

class MutationConflictError(Exception):
    """Raised when a graph mutation cannot be applied due to a conflict."""

class OptimizationError(Exception):
    """Raised when optimization analysis fails."""

class HistoryUnavailableError(Exception):
    """Raised when execution history is unavailable."""

class SubflowExtractionError(Exception):
    """Raised when subflow extraction fails."""

# --- Agent Action Logging Decorator ---

def agent_action_logger(func):
    """
    Decorator to log agent protocol calls for audit and UX feedback.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger("fastmcp.agent_action")
        logger.info(f"Agent action: {func.__qualname__} called with args={args[1:]}, kwargs={kwargs}")
        try:
            result = await func(*args, **kwargs)
            logger.info(f"Agent action: {func.__qualname__} succeeded with result={result}")
            return result
        except Exception as e:
            logger.error(f"Agent action: {func.__qualname__} failed with error={e}")
            raise
    return wrapper

# --- Explainability Hook Decorator ---

def agent_explainability_hook(func):
    """
    Decorator to add explainability hooks for agent/builder actions.
    This can be extended to trigger explainability events or logs.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger("fastmcp.agent_explainability")
        logger.debug(f"Explainability hook: {func.__qualname__} called.")
        return await func(*args, **kwargs)
    return wrapper

# --- Protocols ---

@runtime_checkable
class AgentSuggestionProvider(Protocol):
    """
    Protocol for agent-assisted suggestions in the builder GUI.
    Implementations provide next node recommendations, default values, and auto-wiring hints.

    Strengths:
        - Clear contract for agent suggestion capability.
        - Enables easy swapping of suggestion backends (LLM, rules, etc).
        - Testable and mockable for agent logic.

    All methods are async to support streaming/real-time agent feedback and non-blocking I/O.
    Error handling: raises SuggestionError on failure.

    GUI Integration:
        - SuggestionResult includes rationale, agent confidence, and next actions for display.
        - Agent can provide UI hints for property editing.
    """

    @agent_action_logger
    @agent_explainability_hook
    async def recommend_next_node(
        self,
        partial_graph_spec: GraphSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> SuggestionResult:
        """
        Given a partial GraphSpec, return a suggestion for the next node, default property values,
        and possible auto-wiring patterns.

        Returns:
            SuggestionResult
        Raises:
            SuggestionError on failure.
        """
        ...

    @agent_action_logger
    @agent_explainability_hook
    async def get_alternative_suggestions(
        self,
        partial_graph_spec: GraphSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SuggestionResult]:
        """
        NEW: Provide multiple alternative node suggestions for the GUI to present as options.

        Returns:
            List of SuggestionResult
        Raises:
            SuggestionError on failure.
        """
        ...

    @agent_action_logger
    @agent_explainability_hook
    async def explain_suggestion(
        self,
        suggestion: SuggestionResult,
        graph_spec: GraphSpec
    ) -> str:
        """
        NEW: Provide a human-readable explanation for a given suggestion, for GUI tooltips.

        Returns:
            Explanation string
        Raises:
            SuggestionError on failure.
        """
        ...


@runtime_checkable
class AgentGraphMutationTool(Protocol):
    """
    Protocol for agent-driven, mid-run execution control and graph mutation.

    PatchSpec structure is explicit (JSON Patch RFC 6902).
    Consider returning undo/rollback info for advanced use cases (future).
    All methods are async for long-running mutations.
    Error handling: raises MutationConflictError on failure.

    GUI Integration:
        - PatchSpec includes agent_id and confidence for user review.
        - Each PatchOperation can include rationale for display.
    """

    @agent_action_logger
    @agent_explainability_hook
    async def trigger_graph_mutation(
        self,
        execution_id: str,
        patch_spec: PatchSpec
    ) -> bool:
        """
        Request a live mutation of a running workflow graph.

        Args:
            execution_id: The unique ID of the running execution.
            patch_spec: A JSON Patch (RFC 6902) to apply to the graph.

        Returns:
            True if mutation was accepted/applied; False otherwise.
        Raises:
            MutationConflictError on failure.
        """
        ...

    @agent_action_logger
    @agent_explainability_hook
    async def preview_graph_mutation(
        self,
        graph_spec: GraphSpec,
        patch_spec: PatchSpec
    ) -> GraphSpec:
        """
        NEW: Preview the result of a mutation before applying, for GUI diff/preview.

        Returns:
            The resulting GraphSpec if patch were applied.
        Raises:
            MutationConflictError on failure.
        """
        ...


@runtime_checkable
class AgentOptimizationAdvisor(Protocol):
    """
    Protocol for agent-driven workflow optimization and analysis.

    All methods are async for long-running optimizations.
    Error handling: raises OptimizationError on failure.

    GUI Integration:
        - OptimizationResult includes agent_explanation for user context.
        - Hints can be shown as actionable suggestions in the builder.
    """

    @agent_action_logger
    @agent_explainability_hook
    async def suggest_optimizations(
        self,
        graph_spec: GraphSpec
    ) -> OptimizationResult:
        """
        Analyze a GraphSpec and return optimization hints or a diff.

        Returns:
            OptimizationResult
        Raises:
            OptimizationError on failure.
        """
        ...

    @agent_action_logger
    @agent_explainability_hook
    async def explain_optimization(
        self,
        optimization: OptimizationResult,
        graph_spec: GraphSpec
    ) -> str:
        """
        NEW: Provide a human-readable explanation for a given optimization, for GUI tooltips.

        Returns:
            Explanation string
        Raises:
            OptimizationError on failure.
        """
        ...


@runtime_checkable
class AgentExecutionHistoryTool(Protocol):
    """
    Protocol for agent access to workflow execution history and insights.

    All methods are async for large history queries.
    Error handling: raises HistoryUnavailableError on failure.

    GUI Integration:
        - ExecutionInsights includes agent_summary for display.
        - Agent can highlight hot spots and error patterns for user review.
    """

    @agent_action_logger
    @agent_explainability_hook
    async def get_execution_insights(
        self,
        workflow_id: str,
        limit: int = 20
    ) -> ExecutionInsights:
        """
        Retrieve statistical summaries, anomalies, and hot spots for a workflow.

        Returns:
            ExecutionInsights
        Raises:
            HistoryUnavailableError on failure.
        """
        ...

    @agent_action_logger
    @agent_explainability_hook
    async def get_node_execution_trace(
        self,
        workflow_id: str,
        node_id: NodeID,
        limit: int = 10
    ) -> List[ExecutionRecord]:
        """
        NEW: Retrieve execution trace for a specific node, for GUI drill-down.

        Returns:
            List of ExecutionRecord
        Raises:
            HistoryUnavailableError on failure.
        """
        ...


@runtime_checkable
class AgentSubflowManager(Protocol):
    """
    Protocol for agent-driven creation and management of composite/hierarchical sub-workflows.

    Subflow inputs/outputs are inferred from the selected node boundaries and their connections.
    The modified graph_spec (with the subflow replaced by a macro node) is returned as well.
    All methods are async.
    Error handling: raises SubflowExtractionError on failure.

    GUI Integration:
        - SubflowExtractionResult includes agent_rationale for user context.
        - Agent can suggest subflow names and boundaries for user approval.
    """

    @agent_action_logger
    @agent_explainability_hook
    async def extract_subflow(
        self,
        node_ids: List[NodeID],
        subflow_name: str,
        graph_spec: GraphSpec
    ) -> SubflowExtractionResult:
        """
        Encapsulate a set of nodes as a reusable subflow/macro node.

        Returns:
            SubflowExtractionResult, including the new subflow spec, macro node id, version, and the modified graph.
        Raises:
            SubflowExtractionError on failure.
        """
        ...

    @agent_action_logger
    @agent_explainability_hook
    async def suggest_subflow_candidates(
        self,
        graph_spec: GraphSpec
    ) -> List[List[NodeID]]:
        """
        NEW: Suggest candidate node groups for subflow extraction, for GUI to present as options.

        Returns:
            List of node ID lists (each a candidate subflow)
        Raises:
            SubflowExtractionError on failure.
        """
        ...

# --- Agent-Editable Property Hints for GUI Property Editors ---

def get_agent_editable_property_hints(node_spec: NodeSpec) -> Dict[str, str]:
    """
    Returns a mapping of property names to UI hints for agent-editable properties.
    """
    hints = {}
    for prop_name, prop_spec in node_spec.get("properties", {}).items():
        if prop_spec.get("agent_editable"):
            hints[prop_name] = prop_spec.get("ui_hint", "")
    return hints

# --- Agent/Builder Chat or Feedback Loop Protocol ---

@runtime_checkable
class AgentBuilderChatProtocol(Protocol):
    """
    Protocol for agent/builder chat or feedback loop in the GUI.
    Allows the builder and agent to exchange messages, suggestions, and feedback.
    """

    @agent_action_logger
    @agent_explainability_hook
    async def send_message(
        self,
        sender: Literal["agent", "builder"],
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send a message from agent or builder to the other party.

        Args:
            sender: "agent" or "builder"
            message: The message content
            context: Optional context (e.g., current graph, node, etc.)
        """
        ...

    @agent_action_logger
    @agent_explainability_hook
    async def get_messages(
        self,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the latest chat messages between agent and builder.

        Args:
            limit: Maximum number of messages to retrieve

        Returns:
            List of message dicts (with sender, message, timestamp, etc.)
        """
        ...

# --- Error Handling Strategy ---
# All protocols use exceptions for error handling, with a dedicated exception class per protocol.
# This ensures clear, actionable error reporting and supports robust agent/builder integration.

# --- Versioning ---
# Protocols are designed to be forward-compatible. If protocol changes are required, versioning should be introduced
# via explicit version fields in protocol methods or via protocol class versioning.

# --- Unit Testing ---
# All protocol contracts and edge cases should be covered by unit tests in test_builder_protocols.py.

# --- Typing ---
# All property and patch fields use strict typing (see above). If further strictness is needed, extend types in types.py.

# --- Rollback/Undo Support ---
# Rollback/undo for mutations is not yet implemented, but the PatchSpec structure allows for future extension.

# --- Relocation of Types ---
# All TypedDicts and type aliases should be moved to a dedicated types.py module for clarity and reusability
# in a future refactor.

# --- Agent-Builder GUI Synergy Features Implemented ---
# - [x] Add agent action logging for all protocol calls for audit and UX feedback.
# - [x] Add agent/builder "explainability" hooks for all agent actions.
# - [x] Add agent/builder confidence and rationale fields to all user-facing results.
# - [x] Add agent-editable property hints for GUI property editors.
# - [x] Add protocol for agent/builder chat or feedback loop in the GUI.

# --- FastMCP Protocols for Agent-Builder Synergy ---
# Basic graph management for agent automation
@runtime_checkable
class GraphManagementTool(Protocol):
    """
    Protocol for basic graph CRUD operations for agent-facing automation.
    """
    def create_graph(self, workflow_id: str, description: Optional[str] = None) -> str: ...
    def get_graph_spec(self, workflow_id: str) -> Optional[GraphSpec]: ...
    def save_graph_spec(self, workflow_id: str, file_path: Optional[str] = None) -> bool: ...
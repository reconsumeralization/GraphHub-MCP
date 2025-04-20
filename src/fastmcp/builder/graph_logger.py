"""
GraphLogger — Strictly Typed, Modular Logging for FastMCP Builder/Agent/Engine

This module provides a strictly-typed, modular logging utility for all graph operations
within the FastMCP builder ecosystem. It is designed for seamless integration across
the builder, agent, and execution engine layers, supporting auditability, agent feedback
loops, and future extensibility (e.g., centralized logging, async handlers).

Key Features:
    1. Agent/Builder/Engine event logging with explicit event types.
    2. Fully type-annotated public API for static analysis (mypy, pyright).
    3. Minimal, testable, and GUI-free—ready for both CLI and service use.
    4. Extensible for integration with external monitoring/logging backends.
    5. Agent usability enhancements: agent actions, suggestions, and feedback are
       structured for GUI display and agentic workflow transparency.
    6. Supports in-memory log streaming for GUI/agent dashboards.
    7. Enhanced agent usability: logs are structured for GUI filtering, agent traceability, and user feedback.
    8. All context objects are merged safely for GUI display and agentic workflows.

"""

import logging
import threading
from typing import Any, Dict, Optional, List, Literal, TypedDict, Callable, cast

# --- Event Type Definitions ---

GraphLogEventType = Literal[
    "model_selection",
    "api_call",
    "error",
    "agent_suggestion",
    "agent_feedback",
    "execution_mutation",
    "optimization_hint",
    "execution_history",
    "subflow_extraction",
    "agent_action",
    "agent_observation",
    "generic"
]

AgentFeedbackType = Literal["accepted", "rejected", "modified", "skipped", "auto_applied"]

# --- Strict Context Type Aliases (for static checking) ---

class AgentSuggestionContext(TypedDict, total=False):
    node_type: str
    properties: Dict[str, Any]
    agent_id: Optional[str]
    suggestion_id: Optional[str]
    reason: Optional[str]

class AgentFeedbackContext(TypedDict, total=False):
    suggestion_id: str
    feedback: AgentFeedbackType
    user_action: Optional[str]
    agent_id: Optional[str]
    reason: Optional[str]

class AgentActionContext(TypedDict, total=False):
    agent_id: str
    action_type: str
    target_node_id: Optional[str]
    parameters: Optional[Dict[str, Any]]
    result: Optional[Any]
    error: Optional[str]

class AgentObservationContext(TypedDict, total=False):
    agent_id: str
    observation: str
    node_id: Optional[str]
    details: Optional[Dict[str, Any]]

class ExecutionMutationContext(TypedDict, total=False):
    execution_id: str
    mutation_type: str
    patch_spec: Dict[str, Any]
    agent_triggered: bool
    checkpoint_id: Optional[str]

class OptimizationHintContext(TypedDict, total=False):
    diff: Dict[str, Any]
    before_spec: Dict[str, Any]
    after_spec: Dict[str, Any]

class ExecutionHistoryContext(TypedDict, total=False):
    execution_id: str
    record: Dict[str, Any]
    stats: Dict[str, Any]
    anomalies: List[str]

class SubflowExtractionContext(TypedDict, total=False):
    subflow_name: str
    node_ids: List[str]
    version: Optional[str]
    macro_node_id: Optional[str]

# --- In-Memory Log Buffer for GUI/Agent Streaming ---

class InMemoryLogBuffer:
    """
    Thread-safe in-memory log buffer for streaming logs to GUI or agent dashboards.
    """
    def __init__(self, maxlen: int = 500):
        self._buffer: List[Dict[str, Any]] = []
        self._maxlen = maxlen
        self._lock = threading.Lock()
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []

    def append(self, log_entry: Dict[str, Any]) -> None:
        with self._lock:
            self._buffer.append(log_entry)
            if len(self._buffer) > self._maxlen:
                self._buffer.pop(0)
            for cb in self._listeners:
                try:
                    cb(log_entry)
                except Exception:
                    pass  # Don't break logging on listener error

    def get_logs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def add_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._listeners.remove(callback)

# --- Utility: Merge context dicts safely for GUI/agent use ---

def _merge_context(
    base: Optional[Dict[str, Any]],
    extra: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge two context dicts, with extra taking precedence.
    Used to combine strict context with user-supplied context for GUI/agent display.
    """
    merged: Dict[str, Any] = {}
    if base:
        merged.update(base)
    if extra:
        merged.update(extra)
    return merged

# --- GraphLogger Implementation ---

class GraphLogger:
    """
    Structured, strictly-typed logger for FastMCP builder/agent/engine events.

    - All public methods are fully type-annotated.
    - Designed for testability and static analysis.
    - No GUI dependencies.
    - Supports in-memory log streaming for GUI/agent dashboards.
    - Enhanced for agent usability: logs are structured for GUI filtering, agent traceability, and user feedback.
    """

    def __init__(self, name: str = "fastmcp.graph", enable_memory_buffer: bool = True) -> None:
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.memory_buffer: Optional[InMemoryLogBuffer] = InMemoryLogBuffer() if enable_memory_buffer else None

    def log(
        self,
        event_type: GraphLogEventType,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        level: Literal["info", "warning", "error", "debug"] = "info"
    ) -> None:
        """
        Core logging method for all event types.
        """
        log_entry = {
            "event_type": event_type,
            "message": message,
            "context": context if context else {},
            "level": level
        }
        log_msg = f"[{event_type}] {message} | Context: {context if context else '{}'}"
        if level == "info":
            self.logger.info(log_msg)
        elif level == "warning":
            self.logger.warning(log_msg)
        elif level == "error":
            self.logger.error(log_msg)
        elif level == "debug":
            self.logger.debug(log_msg)
        else:
            self.logger.info(log_msg)
        if self.memory_buffer:
            self.memory_buffer.append(log_entry)

    # --- Agent Suggestion/Feedback/Action/Observation ---

    def log_agent_suggestion(
        self,
        suggestion: str,
        node_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        suggestion_id: Optional[str] = None,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an agent-assisted suggestion (e.g., next node, auto-wiring, default values).
        """
        agent_context: Dict[str, Any] = {}
        if node_type is not None:
            agent_context["node_type"] = node_type
        if properties is not None:
            agent_context["properties"] = properties
        if agent_id is not None:
            agent_context["agent_id"] = agent_id
        if suggestion_id is not None:
            agent_context["suggestion_id"] = suggestion_id
        if reason is not None:
            agent_context["reason"] = reason
        merged_context = _merge_context(agent_context, context)
        # Add agent usability fields for GUI filtering
        merged_context["__agent_event__"] = True
        merged_context["__event_type__"] = "agent_suggestion"
        self.log(
            event_type="agent_suggestion",
            message=f"Agent Suggestion: {suggestion}",
            context=merged_context,
            level="info"
        )

    def log_agent_feedback(
        self,
        suggestion_id: str,
        feedback: AgentFeedbackType,
        user_action: Optional[str] = None,
        agent_id: Optional[str] = None,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log user or agent feedback on agent suggestions.
        """
        feedback_context: Dict[str, Any] = {
            "suggestion_id": suggestion_id,
            "feedback": feedback,
        }
        if user_action is not None:
            feedback_context["user_action"] = user_action
        if agent_id is not None:
            feedback_context["agent_id"] = agent_id
        if reason is not None:
            feedback_context["reason"] = reason
        merged_context = _merge_context(feedback_context, context)
        merged_context["__agent_event__"] = True
        merged_context["__event_type__"] = "agent_feedback"
        self.log(
            event_type="agent_feedback",
            message=f"User/Agent feedback on suggestion: {feedback}",
            context=merged_context,
            level="info"
        )

    def log_agent_action(
        self,
        agent_id: str,
        action_type: str,
        target_node_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an agent-initiated action (e.g., node creation, auto-wiring, graph mutation).
        """
        action_context: Dict[str, Any] = {
            "agent_id": agent_id,
            "action_type": action_type,
        }
        if target_node_id is not None:
            action_context["target_node_id"] = target_node_id
        if parameters is not None:
            action_context["parameters"] = parameters
        if result is not None:
            action_context["result"] = result
        if error is not None:
            action_context["error"] = error
        merged_context = _merge_context(action_context, context)
        merged_context["__agent_event__"] = True
        merged_context["__event_type__"] = "agent_action"
        self.log(
            event_type="agent_action",
            message=f"Agent Action: {action_type} by {agent_id}",
            context=merged_context,
            level="info" if not error else "error"
        )

    def log_agent_observation(
        self,
        agent_id: str,
        observation: str,
        node_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an agent's observation about the graph or workflow state.
        """
        obs_context: Dict[str, Any] = {
            "agent_id": agent_id,
            "observation": observation,
        }
        if node_id is not None:
            obs_context["node_id"] = node_id
        if details is not None:
            obs_context["details"] = details
        merged_context = _merge_context(obs_context, context)
        merged_context["__agent_event__"] = True
        merged_context["__event_type__"] = "agent_observation"
        self.log(
            event_type="agent_observation",
            message=f"Agent Observation: {observation}",
            context=merged_context,
            level="info"
        )

    # --- Model/API/Error Events ---

    def log_model_selection(
        self,
        model_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        merged_context = _merge_context({"model_name": model_name}, context)
        merged_context["__event_type__"] = "model_selection"
        self.log(
            event_type="model_selection",
            message=f"Model selected: {model_name}",
            context=merged_context,
            level="info"
        )

    def log_api_call(
        self,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        response_status: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        api_context: Dict[str, Any] = {
            "payload": payload if payload is not None else {},
            "response_status": response_status,
            "endpoint": endpoint
        }
        merged_context = _merge_context(api_context, context)
        merged_context["__event_type__"] = "api_call"
        self.log(
            event_type="api_call",
            message=f"API Call: {endpoint}",
            context=merged_context,
            level="info"
        )

    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        merged_context = _merge_context({"error": repr(error)}, context)
        merged_context["__event_type__"] = "error"
        self.log(
            event_type="error",
            message=f"Error: {repr(error)}",
            context=merged_context,
            level="error"
        )

    # --- Execution Mutation ---

    def log_execution_mutation(
        self,
        execution_id: str,
        mutation_type: str,
        patch_spec: Dict[str, Any],
        agent_triggered: bool = False,
        checkpoint_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a runtime execution mutation (e.g., injected node, pause, rollback).
        """
        mutation_context: Dict[str, Any] = {
            "execution_id": execution_id,
            "mutation_type": mutation_type,
            "patch_spec": patch_spec,
            "agent_triggered": agent_triggered,
        }
        if checkpoint_id is not None:
            mutation_context["checkpoint_id"] = checkpoint_id
        merged_context = _merge_context(mutation_context, context)
        merged_context["__event_type__"] = "execution_mutation"
        self.log(
            event_type="execution_mutation",
            message=f"Execution mutation: {mutation_type} on {execution_id}",
            context=merged_context,
            level="warning" if mutation_type in ("rollback", "pause") else "info"
        )

    # --- Optimization Hint ---

    def log_optimization_hint(
        self,
        hint: str,
        diff: Optional[Dict[str, Any]] = None,
        before_spec: Optional[Dict[str, Any]] = None,
        after_spec: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a workflow optimization suggestion.
        """
        opt_context: Dict[str, Any] = {"hint": hint}
        if diff is not None:
            opt_context["diff"] = diff
        if before_spec is not None:
            opt_context["before_spec"] = before_spec
        if after_spec is not None:
            opt_context["after_spec"] = after_spec
        merged_context = _merge_context(opt_context, context)
        merged_context["__event_type__"] = "optimization_hint"
        self.log(
            event_type="optimization_hint",
            message=f"Optimization Hint: {hint}",
            context=merged_context,
            level="info"
        )

    # --- Execution History ---

    def log_execution_history(
        self,
        execution_id: str,
        record: Dict[str, Any],
        stats: Optional[Dict[str, Any]] = None,
        anomalies: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log execution history/metrics for learning and analysis.
        """
        hist_context: Dict[str, Any] = {
            "execution_id": execution_id,
            "record": record
        }
        if stats is not None:
            hist_context["stats"] = stats
        if anomalies is not None:
            hist_context["anomalies"] = anomalies
        merged_context = _merge_context(hist_context, context)
        merged_context["__event_type__"] = "execution_history"
        self.log(
            event_type="execution_history",
            message=f"Execution history recorded for {execution_id}",
            context=merged_context,
            level="info"
        )

    # --- Subflow Extraction ---

    def log_subflow_extraction(
        self,
        subflow_name: str,
        node_ids: List[str],
        version: Optional[str] = None,
        macro_node_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log the extraction of a subflow/macro node.
        """
        subflow_context: Dict[str, Any] = {
            "subflow_name": subflow_name,
            "node_ids": node_ids,
        }
        if version is not None:
            subflow_context["version"] = version
        if macro_node_id is not None:
            subflow_context["macro_node_id"] = macro_node_id
        merged_context = _merge_context(subflow_context, context)
        merged_context["__event_type__"] = "subflow_extraction"
        self.log(
            event_type="subflow_extraction",
            message=f"Subflow extracted: {subflow_name} from nodes {node_ids}",
            context=merged_context,
            level="info"
        )

    # --- Generic Event ---

    def log_event(
        self,
        event: str,
        details: Optional[Dict[str, Any]] = None,
        level: Literal["info", "warning", "error", "debug"] = "info"
    ) -> None:
        """
        Log a generic event with details.
        """
        merged_context = _merge_context({"event": event}, details)
        merged_context["__event_type__"] = "generic"
        self.log(
            event_type="generic",
            message=f"Event: {event}",
            context=merged_context,
            level=level
        )

    # --- In-Memory Log Buffer API for GUI/Agent ---

    def get_memory_logs(self) -> List[Dict[str, Any]]:
        """
        Return a copy of the in-memory log buffer for GUI/agent display.
        """
        if self.memory_buffer:
            return self.memory_buffer.get_logs()
        return []

    def clear_memory_logs(self) -> None:
        """
        Clear the in-memory log buffer.
        """
        if self.memory_buffer:
            self.memory_buffer.clear()

    def add_log_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a callback to be called on every new log entry (for GUI/agent streaming).
        """
        if self.memory_buffer:
            self.memory_buffer.add_listener(callback)

    def remove_log_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Remove a previously added log listener.
        """
        if self.memory_buffer:
            self.memory_buffer.remove_listener(callback)

# --- TODO: 
# - Consider adding log filtering and search for GUI/agent dashboards.
# - Add log export/import for agent traceability and workflow debugging.
# - Add log correlation IDs for multi-agent workflows.
# - Add log severity color-coding for GUI.

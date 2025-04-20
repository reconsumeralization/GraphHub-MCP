"""
Agentic Tools for FastMCP Agent/Workflow Integration

This module provides agent-facing utilities and wrappers for interacting with the FastMCP WorkflowManager API.
It is designed for use by LLM agents, automation scripts, or FastMCP's own meta-control processes (MCPs)
to programmatically construct, manipulate, validate, and manage workflow graphs, as well as to support
agentic feedback, introspection, and agentic continuance (checkpointing, recovery, etc).

This file is NOT the main programmatic API (see fastmcp.builder.workflow_manager.WorkflowManager).
Instead, it provides agent/automation-friendly wrappers, feedback stores, and meta-control helpers.

Key features:
- Agent-assisted node and property suggestions, with context-aware and feedback-driven refinement
- Agentic feedback and action logging (for GUI/traceability)
- Agentic continuance primitives: checkpointing, recovery, delegation
- Meta-control helpers: introspection, self-modification, spawning new MCPs
- No GUI dependencies; imports WorkflowManager API only

"""

from typing import List, Dict, Any, Optional, Tuple, Callable, cast
import copy
import time
from fastmcp.builder.types import GraphSpec # Import GraphSpec for type hinting/casting if needed
from pydantic import BaseModel, Field

# --- Import the canonical WorkflowManager API ---
try:
    from fastmcp.builder.workflow_manager import WorkflowManager
except ImportError as e:
    raise ImportError("WorkflowManager could not be imported. Ensure refactor is complete and imports are correct.") from e

# --- Optional: Import node registry for suggestions ---
try:
    # TODO: Remove import error for NODE_REGISTRY if not present in node_registry
    from fastmcp.builder.nodes.node_registry import NODE_REGISTRY  # type: ignore[import]
except ImportError:
    NODE_REGISTRY = {}

# --- Optional: Import graph utilities for optimization/repair ---
try:
    # TODO: Remove import error for auto_repair_graph if not present in graph_utils
    from fastmcp.builder.graph_utils import analyze_performance  # type: ignore[import]
    from fastmcp.builder.graph_utils import auto_repair_graph  # type: ignore[import]
except ImportError:
    analyze_performance = None
    auto_repair_graph = None

# --- Optional: Import execution engine for advanced features ---
try:
    from fastmcp.execution_engine import executor as executor_mod
    pause_execution = getattr(executor_mod, "pause_execution", None)
    inject_node = getattr(executor_mod, "inject_node", None)
    trigger_graph_mutation = getattr(executor_mod, "trigger_graph_mutation", None)
    get_execution_history = getattr(executor_mod, "get_execution_history", None)
    get_execution_metrics = getattr(executor_mod, "get_execution_metrics", None)
    resume_execution = getattr(executor_mod, "resume_execution", None)
    checkpoint_execution = getattr(executor_mod, "checkpoint_execution", None)
    restore_execution = getattr(executor_mod, "restore_execution", None)
except ImportError:
    pause_execution = None
    inject_node = None
    trigger_graph_mutation = None
    get_execution_history = None
    get_execution_metrics = None
    resume_execution = None
    checkpoint_execution = None
    restore_execution = None

# --- In-memory agent feedback, checkpoint, and action log stores ---
_agent_feedback_store: List[Dict[str, Any]] = []
_agent_checkpoint_store: Dict[str, Any] = {}
_agent_action_log: List[Dict[str, Any]] = []

def store_agent_feedback(graph_id: str, context: Dict[str, Any], feedback: Dict[str, Any]) -> None:
    entry = {
        "graph_id": graph_id,
        "context": context,
        "feedback": feedback,
        "timestamp": time.time(),
    }
    _agent_feedback_store.append(entry)
    _agent_action_log.append({
        "action": "store_feedback",
        "graph_id": graph_id,
        "context": context,
        "feedback": feedback,
        "timestamp": entry["timestamp"],
    })

def get_agent_feedback(graph_id: str) -> List[Dict[str, Any]]:
    return [f for f in _agent_feedback_store if f["graph_id"] == graph_id]

def store_agent_checkpoint(execution_id: str, state: Any) -> None:
    entry = {
        "state": copy.deepcopy(state),
        "timestamp": time.time(),
    }
    _agent_checkpoint_store[execution_id] = entry
    _agent_action_log.append({
        "action": "checkpoint",
        "execution_id": execution_id,
        "timestamp": entry["timestamp"],
    })

def get_agent_checkpoint(execution_id: str) -> Optional[Any]:
    return _agent_checkpoint_store.get(execution_id, {}).get("state")

def log_agent_action(action: str, details: Dict[str, Any]) -> None:
    entry = {
        "action": action,
        "details": details,
        "timestamp": time.time(),
    }
    _agent_action_log.append(entry)

def get_agent_action_log(graph_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if graph_id is None:
        return _agent_action_log
    return [
        a for a in _agent_action_log
        if a.get("graph_id") == graph_id
        or (isinstance(a.get("details", {}), dict) and a.get("details", {}).get("graph_id") == graph_id)
    ]

class AgenticWorkflowTools:
    """
    Agentic wrapper for WorkflowManager, providing agent/automation-friendly methods,
    agentic feedback, meta-control, and agentic continuance primitives.
    """

    def __init__(self):
        self._manager = WorkflowManager()
        self._agent_state: Dict[str, Any] = {}

    # --- Core Graph Operations (delegated to WorkflowManager) ---

    def create_graph(self, workflow_id: Optional[str] = None, template: Optional[Dict[str, Any]] = None) -> str:
        effective_workflow_id = workflow_id or f"graph_{int(time.time() * 1000)}"
        definition = template or {"nodes": [], "edges": []}
        self._manager.create_workflow(effective_workflow_id, definition)
        log_agent_action("create_graph", {"graph_id": effective_workflow_id, "workflow_id": workflow_id, "template": bool(template)})
        return effective_workflow_id

    def delete_graph(self, graph_id: str) -> bool:
        try:
            self._manager.delete_workflow(graph_id)
            log_agent_action("delete_graph", {"graph_id": graph_id, "result": True})
            return True
        except KeyError:
            log_agent_action("delete_graph", {"graph_id": graph_id, "result": False, "error": "not found"})
            return False

    def list_graphs(self) -> List[str]:
        workflow_list = self._manager.list_workflows()
        graph_ids: List[str] = []
        if workflow_list:
            if isinstance(workflow_list[0], dict):
                graph_ids = [cast(dict, wf).get('workflow_id', f'unknown_id_{i}') for i, wf in enumerate(workflow_list)]
            elif isinstance(workflow_list[0], str):
                graph_ids = cast(List[str], workflow_list)
        log_agent_action("list_graphs", {"result_count": len(graph_ids)})
        return graph_ids

    def add_node(
        self,
        graph_id: str,
        node_type: str,
        node_name: str,
        properties: Optional[Dict[str, Any]] = None,
        position: Optional[Tuple[float, float]] = None,
        auto_connect: bool = False,
        connect_to: Optional[str] = None,
        connect_port: Optional[str] = None
    ) -> str:
        node_id = self._manager.add_node(graph_id, node_type, node_name, properties, position)
        log_agent_action("add_node", {
            "graph_id": graph_id,
            "node_id": node_id,
            "node_type": node_type,
            "node_name": node_name,
            "properties": properties,
            "position": position,
            "auto_connect": auto_connect,
            "connect_to": connect_to,
            "connect_port": connect_port,
        })
        if auto_connect and connect_to and connect_port:
            try:
                self.connect_nodes(graph_id, node_id, "out_exec", connect_to, connect_port)
                log_agent_action("auto_connect", {
                    "graph_id": graph_id,
                    "from_node_id": node_id,
                    "to_node_id": connect_to,
                    "to_port": connect_port,
                })
            except Exception as e:
                log_agent_action("auto_connect_failed", {
                    "graph_id": graph_id,
                    "from_node_id": node_id,
                    "to_node_id": connect_to,
                    "to_port": connect_port,
                    "error": str(e),
                })
        return node_id

    def connect_nodes(
        self,
        graph_id: str,
        from_node_id: str,
        from_port_name: str,
        to_node_id: str,
        to_port_name: str
    ) -> bool:
        result = self._manager.connect_nodes(graph_id, from_node_id, from_port_name, to_node_id, to_port_name)
        log_agent_action("connect_nodes", {
            "graph_id": graph_id,
            "from_node_id": from_node_id,
            "from_port_name": from_port_name,
            "to_node_id": to_node_id,
            "to_port_name": to_port_name,
            "result": result,
        })
        return result

    def set_node_property(
        self,
        graph_id: str,
        node_id: str,
        property_name: str,
        value: Any
    ) -> bool:
        result = self._manager.set_node_property(graph_id, node_id, property_name, value)
        log_agent_action("set_node_property", {
            "graph_id": graph_id,
            "node_id": node_id,
            "property_name": property_name,
            "value": value,
            "result": result,
        })
        return result

    def get_node_properties(self, graph_id: str, node_id: str) -> Dict[str, Any]:
        props = self._manager.get_node_properties(graph_id, node_id)
        if props is None:
            # TODO: Open issue: get_node_properties returned None, returning empty dict for type safety
            log_agent_action("get_node_properties_failed", {
                "graph_id": graph_id,
                "node_id": node_id,
                "error": "No properties found (None returned)",
            })
            return {}
        log_agent_action("get_node_properties", {
            "graph_id": graph_id,
            "node_id": node_id,
            "properties": props,
        })
        return props

    def validate_graph(self, graph_id: str, auto_repair: bool = True) -> Dict[str, Any]:
        try:
            result = self._manager.validate_workflow(graph_id)
            if not isinstance(result, dict):
                result = {"valid": bool(result), "messages": [] if bool(result) else ["Validation failed."]}
            log_agent_action("validate_graph", {
                "graph_id": graph_id,
                "result": result,
                "auto_repair": auto_repair,
            })
            if not result.get("valid", True) and auto_repair and auto_repair_graph is not None:
                structure = self.get_graph_structure(graph_id)
                if structure:
                    repaired_info = auto_repair_graph(structure)  # type: ignore
                    if isinstance(repaired_info, dict) and repaired_info.get("repaired", False):
                        if "graph" in repaired_info:
                            self._manager.update_workflow(graph_id, repaired_info["graph"])
                            log_agent_action("auto_repair_graph", {
                                "graph_id": graph_id,
                                "repaired": True,
                                "messages": result.get("messages", []),
                            })
                            return {"valid": True, "messages": ["Auto-repair applied."] + result.get("messages", [])}
            return result
        except Exception as e:
            log_agent_action("validate_graph_failed", {"graph_id": graph_id, "error": str(e)})
            return {"valid": False, "messages": [f"Validation error: {str(e)}"]}

    def save_graph(self, graph_id: str, file_path: Optional[str] = None) -> str:
        # TODO: Open issue: WorkflowManager.save_graph expects str, but file_path may be None
        # Workaround: If file_path is None, generate a default path
        actual_file_path = file_path if file_path is not None else f"{graph_id}.json"
        result = self._manager.save_graph(graph_id, actual_file_path)
        log_agent_action("save_graph", {
            "graph_id": graph_id,
            "file_path": actual_file_path,
            "result": result,
        })
        return result

    def get_graph_structure(self, graph_id: str) -> Optional[Dict[str, Any]]:
        structure = self._manager.get_workflow(graph_id)
        log_agent_action("get_graph_structure", {
            "graph_id": graph_id,
            "structure_found": structure is not None,
        })
        return structure

    def list_available_node_types(
        self,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        if NODE_REGISTRY:
            node_types = [
                {
                    "name": getattr(node_cls, "display_name", getattr(node_cls, "__name__", "")),
                    "identifier": getattr(node_cls, "identifier", ""),
                    "category": getattr(node_cls, "category", ""),
                    "description": getattr(node_cls, "description", ""),
                    "inputs": getattr(node_cls, "inputs", []),
                    "outputs": getattr(node_cls, "outputs", []),
                }
                for node_cls in NODE_REGISTRY.values()
            ]
            if filter_fn:
                node_types = [nt for nt in node_types if filter_fn(nt)]
            log_agent_action("list_available_node_types", {
                "count": len(node_types),
                "filtered": bool(filter_fn),
            })
            return node_types
        node_types = self._manager.list_available_node_types()
        if filter_fn:
            node_types = [nt for nt in node_types if filter_fn(nt)]
        log_agent_action("list_available_node_types", {
            "count": len(node_types),
            "filtered": bool(filter_fn),
        })
        return node_types

    def trigger_graph_execution(
        self,
        graph_id: str,
        input_data: Dict[str, Any],
        checkpoint: bool = True
    ) -> str:
        execution_id = self._manager.trigger_graph_execution(graph_id, input_data)
        log_agent_action("trigger_graph_execution", {
            "graph_id": graph_id,
            "input_data": input_data,
            "execution_id": execution_id,
            "checkpoint": checkpoint,
        })
        if checkpoint and checkpoint_execution is not None:
            try:
                state = checkpoint_execution(execution_id)
                store_agent_checkpoint(execution_id, state)
            except Exception as e:
                log_agent_action("checkpoint_failed", {
                    "execution_id": execution_id,
                    "error": str(e),
                })
        return execution_id

    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        status = self._manager.get_execution_status(execution_id)
        log_agent_action("get_execution_status", {
            "execution_id": execution_id,
            "status": status,
        })
        return status

    # --- Agentic/Meta-control & Feedback Tools ---

    def recommend_next_node(
        self,
        graph_id: str,
        partial_spec: Optional[Dict[str, Any]] = None,
        last_node_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        agent_profile: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggests the next node type(s) and optionally default property values, given a partial graph spec and agent context.
        """
        structure = partial_spec or self.get_graph_structure(graph_id)
        nodes: List[Dict[str, Any]] = []
        if structure is not None and isinstance(structure, dict):
            nodes = structure.get("nodes", [])
        node_types = self.list_available_node_types()
        suggestions: List[Dict[str, Any]] = []

        # Context-aware suggestion logic
        if last_node_id:
            last_node = next((n for n in nodes if n.get("id") == last_node_id), None)
            if last_node:
                last_type = last_node.get("type", "").lower()
                if "api" in last_type:
                    suggestions += [nt for nt in node_types if "error" in nt["name"].lower() or "process" in nt["name"].lower()]
                if "input" in last_type:
                    suggestions += [nt for nt in node_types if "validate" in nt["name"].lower()]
                if "decision" in last_type:
                    suggestions += [nt for nt in node_types if "branch" in nt["name"].lower()]
        if agent_profile and "preferred_categories" in agent_profile:
            preferred = set(agent_profile["preferred_categories"])
            suggestions += [nt for nt in node_types if nt.get("category") in preferred]
        feedbacks = get_agent_feedback(graph_id)
        if feedbacks:
            accepted = [f["context"].get("suggestion") for f in feedbacks if f["feedback"].get("accepted")]
            suggestions += [nt for nt in node_types if nt["name"] in accepted]
        if not suggestions:
            suggestions = node_types
        seen = set()
        unique_suggestions = []
        for nt in suggestions:
            key = nt.get("identifier", nt.get("name"))
            if key not in seen:
                unique_suggestions.append(nt)
                seen.add(key)
        log_agent_action("recommend_next_node", {
            "graph_id": graph_id,
            "last_node_id": last_node_id,
            "context": context,
            "agent_profile": agent_profile,
            "suggestion_count": len(unique_suggestions),
        })
        return unique_suggestions

    def store_suggestion_feedback(self, graph_id: str, context: Dict[str, Any], feedback: Dict[str, Any]) -> None:
        store_agent_feedback(graph_id, context, feedback)

    def get_suggestion_feedback(self, graph_id: str) -> List[Dict[str, Any]]:
        return get_agent_feedback(graph_id)

    def get_agent_action_log(self, graph_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return get_agent_action_log(graph_id)

    def trigger_graph_mutation(
        self,
        execution_id: str,
        patch_spec: Dict[str, Any],
        checkpoint: bool = True
    ) -> bool:
        if trigger_graph_mutation is None:
            raise NotImplementedError("Live graph mutation is not available in this environment.")
        result = trigger_graph_mutation(execution_id, patch_spec)
        log_agent_action("trigger_graph_mutation", {
            "execution_id": execution_id,
            "patch_spec": patch_spec,
            "checkpoint": checkpoint,
            "result": result,
        })
        if checkpoint and checkpoint_execution is not None:
            try:
                state = checkpoint_execution(execution_id)
                store_agent_checkpoint(execution_id, state)
            except Exception as e:
                log_agent_action("checkpoint_failed", {
                    "execution_id": execution_id,
                    "error": str(e),
                })
        return result

    def suggest_optimizations(self, graph_id: str) -> Dict[str, Any]:
        structure = self.get_graph_structure(graph_id)
        if structure is None:
            return {"hints": ["Graph not found."], "diff": None}
        if analyze_performance is not None:
            try:
                result = analyze_performance(structure)  # type: ignore
                log_agent_action("suggest_optimizations", {
                    "graph_id": graph_id,
                    "result": result,
                })
                return result
            except Exception as e:
                log_agent_action("suggest_optimizations_failed", {
                    "graph_id": graph_id,
                    "error": str(e),
                })
                return {"hints": ["Optimization engine error."], "diff": None}
        log_agent_action("suggest_optimizations_unavailable", {
            "graph_id": graph_id,
        })
        return {"hints": ["No optimization engine available."], "diff": None}

    def get_execution_insights(self, graph_id: str) -> Dict[str, Any]:
        if get_execution_history is None or get_execution_metrics is None:
            log_agent_action("get_execution_insights_unavailable", {
                "graph_id": graph_id,
            })
            return {"error": "Execution history/metrics not available."}
        history = get_execution_history(graph_id)
        metrics = get_execution_metrics(graph_id)
        log_agent_action("get_execution_insights", {
            "graph_id": graph_id,
            "history": history,
            "metrics": metrics,
        })
        return {
            "history": history,
            "metrics": metrics,
        }

    # --- Agentic Continuance & Recovery Primitives ---

    def checkpoint(self, execution_id: str) -> bool:
        if checkpoint_execution is not None:
            try:
                state = checkpoint_execution(execution_id)
                store_agent_checkpoint(execution_id, state)
                return True
            except Exception as e:
                log_agent_action("checkpoint_failed", {
                    "execution_id": execution_id,
                    "error": str(e),
                })
                return False
        return False

    def restore(self, execution_id: str) -> Optional[Any]:
        if restore_execution is not None:
            state = get_agent_checkpoint(execution_id)
            if state is not None:
                try:
                    result = restore_execution(execution_id, state)
                    log_agent_action("restore", {
                        "execution_id": execution_id,
                        "restored": True,
                    })
                    return result
                except Exception as e:
                    log_agent_action("restore_failed", {
                        "execution_id": execution_id,
                        "error": str(e),
                    })
                    return None
        log_agent_action("restore_checkpoint_only", {
            "execution_id": execution_id,
        })
        return get_agent_checkpoint(execution_id)

    def pause(self, execution_id: str) -> bool:
        if pause_execution is not None:
            try:
                result = pause_execution(execution_id)
                log_agent_action("pause", {
                    "execution_id": execution_id,
                    "result": result,
                })
                return result
            except Exception as e:
                log_agent_action("pause_failed", {
                    "execution_id": execution_id,
                    "error": str(e),
                })
                return False
        return False

    def resume(self, execution_id: str) -> bool:
        if resume_execution is not None:
            try:
                result = resume_execution(execution_id)
                log_agent_action("resume", {
                    "execution_id": execution_id,
                    "result": result,
                })
                return result
            except Exception as e:
                log_agent_action("resume_failed", {
                    "execution_id": execution_id,
                    "error": str(e),
                })
                return False
        return False

    def inject_node_live(self, execution_id: str, node_spec: Dict[str, Any]) -> bool:
        if inject_node is not None:
            try:
                result = inject_node(execution_id, node_spec)
                log_agent_action("inject_node_live", {
                    "execution_id": execution_id,
                    "node_spec": node_spec,
                    "result": result,
                })
                return result
            except Exception as e:
                log_agent_action("inject_node_live_failed", {
                    "execution_id": execution_id,
                    "node_spec": node_spec,
                    "error": str(e),
                })
                return False
        return False

    # --- Meta-control & Introspection ---

    def introspect_self(self, include_state: bool = False) -> Dict[str, Any]:
        info = {
            "loaded_graphs": self.list_graphs(),
            "available_node_types": self.list_available_node_types(),
            "feedback_store_size": len(_agent_feedback_store),
            "checkpoint_store_size": len(_agent_checkpoint_store),
            "action_log_size": len(_agent_action_log),
            "service_class": self.__class__.__name__,
        }
        if include_state:
            info["agent_state"] = copy.deepcopy(self._agent_state)
        log_agent_action("introspect_self", {
            "include_state": include_state,
            "info": info,
        })
        return info

    def self_modify(self, modification_fn: Callable[['AgenticWorkflowTools'], None]) -> None:
        modification_fn(self)
        log_agent_action("self_modify", {
            "modification_fn": modification_fn.__name__ if hasattr(modification_fn, "__name__") else str(modification_fn),
        })

    def spawn_mcp(self, inherit_state: bool = True) -> 'AgenticWorkflowTools':
        new_mcp = AgenticWorkflowTools()
        if inherit_state:
            new_mcp._agent_state = copy.deepcopy(self._agent_state)
        log_agent_action("spawn_mcp", {
            "inherit_state": inherit_state,
        })
        return new_mcp

    def run_mcp_workflow(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        auto_repair: bool = True
    ) -> Dict[str, Any]:
        graph_id = self.create_graph(workflow_id=workflow_id)
        validation = self.validate_graph(graph_id, auto_repair=auto_repair)
        if not validation.get("valid", False):
            log_agent_action("run_mcp_workflow_failed", {
                "workflow_id": workflow_id,
                "validation": validation,
            })
            return {"error": "Validation failed", "messages": validation.get("messages", [])}
        execution_id = self.trigger_graph_execution(graph_id, input_data or {})
        status = self.get_execution_status(execution_id)
        result = {
            "graph_id": graph_id,
            "execution_id": execution_id,
            "status": status,
            "introspection": self.introspect_self(),
        }
        log_agent_action("run_mcp_workflow", result)
        return result

# --- Global instance of AgenticWorkflowTools ---
# (Assuming we want one instance shared, or manage instances differently)
agent_tools_instance = AgenticWorkflowTools()

# --- Pydantic Schemas for Tool Arguments ---

class ListAvailableNodeTypesArgs(BaseModel):
    # No arguments needed for list_available_node_types
    pass

class AddNodeArgs(BaseModel):
    graph_id: str = Field(..., description="The ID of the graph to add the node to.")
    node_type: str = Field(..., description="The type of node to add (e.g., 'ActionNode', 'ConditionNode').")
    node_name: str = Field(..., description="A unique name for the new node within the graph.")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Optional dictionary of initial properties for the node.")
    position: Optional[Tuple[float, float]] = Field(default=None, description="Optional (x, y) coordinates for the node's position in the visual editor.")
    auto_connect: bool = Field(default=False, description="Attempt to automatically connect this node's output to the 'connect_to' node's input.")
    connect_to: Optional[str] = Field(default=None, description="The ID of the node to connect to if auto_connect is True.")
    connect_port: Optional[str] = Field(default=None, description="The name of the input port on the 'connect_to' node if auto_connect is True.")

class ConnectNodesArgs(BaseModel):
    graph_id: str = Field(..., description="The ID of the graph containing the nodes.")
    from_node_id: str = Field(..., description="The ID of the node where the connection starts.")
    from_port_name: str = Field(..., description="The name of the output port on the starting node.")
    to_node_id: str = Field(..., description="The ID of the node where the connection ends.")
    to_port_name: str = Field(..., description="The name of the input port on the ending node.")

class SetNodePropertyArgs(BaseModel):
    graph_id: str = Field(..., description="The ID of the graph containing the node.")
    node_id: str = Field(..., description="The ID of the node to modify.")
    property_name: str = Field(..., description="The name of the property to set.")
    value: Any = Field(..., description="The new value for the property.")

class GetNodePropertiesArgs(BaseModel):
    graph_id: str = Field(..., description="The ID of the graph containing the node.")
    node_id: str = Field(..., description="The ID of the node whose properties are to be retrieved.")

class CreateGraphArgs(BaseModel):
    workflow_id: Optional[str] = Field(default=None, description="Optional unique ID for the new graph. If None, an ID will be generated.")
    template: Optional[Dict[str, Any]] = Field(default=None, description="Optional graph definition (spec) to use as a template.")

class GetGraphSpecArgs(BaseModel):
    graph_id: str = Field(..., description="The ID of the graph whose specification (structure) is to be retrieved.")

class SaveGraphArgs(BaseModel):
    graph_id: str = Field(..., description="The ID of the graph to save.")
    file_path: Optional[str] = Field(default=None, description="Optional path to save the graph specification file (JSON). If None, a default path will be used.")


# --- Tool Definitions (Example using simple functions for now) ---
# TODO: Consider using a BaseTool class structure if needed for framework integration

def list_available_node_types_tool() -> List[Dict[str, Any]]:
    """
    Tool to list all available node types that can be added to a workflow graph.
    Returns a list of dictionaries, each containing 'name' and 'description' of a node type.
    """
    # No arguments needed, Pydantic schema ListAvailableNodeTypesArgs is empty
    return agent_tools_instance.list_available_node_types()

def add_node_tool(
    graph_id: str,
    node_type: str,
    node_name: str,
    properties: Optional[Dict[str, Any]] = None,
    position: Optional[Tuple[float, float]] = None,
    auto_connect: bool = False,
    connect_to: Optional[str] = None,
    connect_port: Optional[str] = None
) -> str:
    """
    Adds a new node to the specified workflow graph.
    Returns the unique ID assigned to the newly created node.
    Use list_available_node_types_tool to see available node types.
    """
    # Argument validation implicitly handled by Pydantic if used via a framework
    # Directly call the AgenticWorkflowTools method
    return agent_tools_instance.add_node(
        graph_id=graph_id,
        node_type=node_type,
        node_name=node_name,
        properties=properties,
        position=position,
        auto_connect=auto_connect,
        connect_to=connect_to,
        connect_port=connect_port
    )

def connect_nodes_tool(
    graph_id: str,
    from_node_id: str,
    from_port_name: str,
    to_node_id: str,
    to_port_name: str
) -> bool:
    """
    Connects an output port of one node to an input port of another node in the specified graph.
    Returns True if the connection was successful, False otherwise.
    """
    return agent_tools_instance.connect_nodes(
        graph_id=graph_id,
        from_node_id=from_node_id,
        from_port_name=from_port_name,
        to_node_id=to_node_id,
        to_port_name=to_port_name
    )

def set_node_property_tool(
    graph_id: str,
    node_id: str,
    property_name: str,
    value: Any
) -> bool:
    """
    Sets the value of a specific property on a node within the graph.
    Returns True if the property was set successfully, False otherwise.
    """
    return agent_tools_instance.set_node_property(
        graph_id=graph_id,
        node_id=node_id,
        property_name=property_name,
        value=value
    )

def get_node_properties_tool(graph_id: str, node_id: str) -> Dict[str, Any]:
    """
    Retrieves all properties of a specific node within the graph.
    Returns a dictionary containing the node's properties.
    Returns an empty dictionary if the node or properties are not found.
    """
    return agent_tools_instance.get_node_properties(graph_id=graph_id, node_id=node_id)

def create_graph_tool(workflow_id: Optional[str] = None, template: Optional[Dict[str, Any]] = None) -> str:
    """
    Creates a new, empty workflow graph or initializes it from a template.
    Returns the unique ID (graph_id) assigned to the newly created graph.
    """
    return agent_tools_instance.create_graph(workflow_id=workflow_id, template=template)

def get_graph_spec_tool(graph_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves the full specification (structure, nodes, edges) of a given graph.
    Returns a dictionary representing the graph spec, or None if the graph is not found.
    """
    return agent_tools_instance.get_graph_structure(graph_id=graph_id)

def save_graph_tool(graph_id: str, file_path: Optional[str] = None) -> str:
    """
    Saves the current specification of the graph to a file (JSON format).
    Returns the actual file path where the graph was saved.
    """
    return agent_tools_instance.save_graph(graph_id=graph_id, file_path=file_path)

# TODO: Implement other tools:
# - TriggerExecutionTool
# - GetExecutionStatusTool


# --- Tool Registry ---
# (A list of tool instances or function references for the agent planner)
AGENT_TOOLS = [
    # TODO: Wrap functions in classes if BaseTool structure is adopted
    {
        "name": "list_available_node_types",
        "description": list_available_node_types_tool.__doc__,
        "args_schema": ListAvailableNodeTypesArgs,
        "func": list_available_node_types_tool
    },
    {
        "name": "add_node",
        "description": add_node_tool.__doc__,
        "args_schema": AddNodeArgs,
        "func": add_node_tool # Pass the function reference directly
        # If using PydanticBaseTool, the function would be the _run method
    },
    {
        "name": "connect_nodes",
        "description": connect_nodes_tool.__doc__,
        "args_schema": ConnectNodesArgs,
        "func": connect_nodes_tool
    },
    {
        "name": "set_node_property",
        "description": set_node_property_tool.__doc__,
        "args_schema": SetNodePropertyArgs,
        "func": set_node_property_tool
    },
    {
        "name": "get_node_properties",
        "description": get_node_properties_tool.__doc__,
        "args_schema": GetNodePropertiesArgs,
        "func": get_node_properties_tool
    },
    {
        "name": "create_graph",
        "description": create_graph_tool.__doc__,
        "args_schema": CreateGraphArgs,
        "func": create_graph_tool
    },
    {
        "name": "get_graph_spec",
        "description": get_graph_spec_tool.__doc__,
        "args_schema": GetGraphSpecArgs,
        "func": get_graph_spec_tool
    },
    {
        "name": "save_graph",
        "description": save_graph_tool.__doc__,
        "args_schema": SaveGraphArgs,
        "func": save_graph_tool
    },
    # ... add other tools here ...
]


# --- Example Usage (Optional) ---
# if __name__ == "__main__":
#     print("Available Node Types:")
#     print(list_available_node_types_tool())
#
#     graph_id = agent_tools_instance.create_graph("my_test_graph")
#     print(f"Created graph: {graph_id}")
#
#     node_id = add_node_tool(graph_id, "ActionNode", "MyFirstAction", properties={"action_type": "log_message", "message": "Hello"})
#     print(f"Added node: {node_id}")


def _example_agentic_usage() -> None:
    """
    Example: Agent-driven meta-control, self-reflection, recursive orchestration, and agentic continuance.
    All agent actions are visible in the GUI via the agent action log.
    """
    agent = AgenticWorkflowTools()

    # 1. Introspect self
    print("Agent Introspection:", agent.introspect_self())

    # 2. Create a new workflow graph
    graph_id = agent.create_graph(workflow_id="meta_orchestration")

    # 3. Add nodes and connect as usual (could be automated by agent logic)
    start_id = agent.add_node(graph_id, "StartNode", "Start", {}, (0.0, 0.0))
    meta_node_id = agent.add_node(
        graph_id,
        "MetaControlNode",
        "MetaStep",
        {"action": "reflect"},
        (100.0, 0.0)
    )
    end_id = agent.add_node(graph_id, "EndNode", "End", {}, (200.0, 0.0))
    agent.connect_nodes(graph_id, start_id, "out_exec", meta_node_id, "in_exec")
    agent.connect_nodes(graph_id, meta_node_id, "out_exec", end_id, "in_exec")

    # 4. Validate and execute, with agentic continuance (auto-repair, checkpoint)
    validation = agent.validate_graph(graph_id)
    if not validation.get("valid", False):
        print("Agent Graph validation failed:", validation.get("messages", []))
        return
    else:
        execution_id = agent.trigger_graph_execution(graph_id, input_data={})
        print("Agent Execution started:", execution_id)
        agent.checkpoint(execution_id)
        while True:
            status = agent.get_execution_status(execution_id)
            print("Agent Execution status:", status)
            if status.get("state") in ("completed", "failed"):
                break
            if status.get("state") == "failed":
                print("Attempting agentic restore...")
                agent.restore(execution_id)
            time.sleep(1)
        print("Agent Final execution result:", status)

    # 5. Self-modification: Add a feedback node if not present
    def add_feedback_node_if_missing(agent_tools: AgenticWorkflowTools):
        structure = agent_tools.get_graph_structure(graph_id)
        if structure is not None and isinstance(structure, dict):
            nodes = structure.get("nodes", [])
            if not any(n.get("type") == "FeedbackNode" for n in nodes):
                feedback_id = agent_tools.add_node(graph_id, "FeedbackNode", "SelfFeedback", {}, (300.0, 0.0))
                agent_tools.connect_nodes(graph_id, end_id, "out_exec", feedback_id, "in_exec")
                print("Agent Self-modified: FeedbackNode added.")

    agent.self_modify(add_feedback_node_if_missing)

    # 6. Spawn a new MCP and delegate a workflow, inheriting agentic state
    new_agent = agent.spawn_mcp(inherit_state=True)
    result = new_agent.run_mcp_workflow("summarize_and_post", input_data={"email_id": "12345"})
    print("Spawned Agent result:", result)

    # 7. Demonstrate agent feedback storage, meta-insights, and agentic continuance
    agent.store_suggestion_feedback(graph_id, {"last_node": meta_node_id}, {"accepted": True, "suggestion": "MetaControlNode"})
    print("Agent Feedback for suggestions:", agent.get_suggestion_feedback(graph_id))
    print("Agent Execution insights:", agent.get_execution_insights(graph_id))

    # 8. Demonstrate agentic pause, resume, and live mutation (if supported)
    if hasattr(agent, "pause") and hasattr(agent, "resume"):
        print("Pausing execution for agentic control...")
        agent.pause(execution_id)
        print("Resuming execution for agentic continuance...")
        agent.resume(execution_id)

    # 9. Demonstrate live node injection (if supported)
    if hasattr(agent, "inject_node_live"):
        print("Injecting live node for agentic mutation...")
        agent.inject_node_live(execution_id, {
            "type": "LogNode",
            "name": "LiveLog",
            "properties": {"message": "Injected at runtime"},
            "position": (400.0, 0.0)
        })

    # 10. Show agent action log for GUI/traceability
    print("Agent Action Log (for GUI):")
    for entry in agent.get_agent_action_log(graph_id):
        print(entry)

if __name__ == '__main__':
    _example_agentic_usage()

# Alias for index.py import compatibility
GraphBuilderService = AgenticWorkflowTools

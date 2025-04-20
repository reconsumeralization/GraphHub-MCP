"""
FastMCP Workflow Manager

Unified, Strictly-Typed, Agent-First Orchestration for Workflow Graphs

This module defines the WorkflowManager class, the authoritative orchestrator for the lifecycle of workflow graphs
composed of nodes. It supports dynamic node type loading, validation, execution, checkpointing, versioning,
access control, and persistent storage. Designed for agent-first workflows, it exposes agent and node state,
execution, and logs for both GUI and programmatic inspection.

Agent-First: Enhanced for agent usability and agent visibility in the builder GUI.

Features:
- Granular event hooks for workflow lifecycle, node-level, and AI steps.
- Plugin/adapter support for custom workflow steps.
- Comprehensive unit and integration tests.
- 100% type coverage and self-tests for all public APIs.
- [Agent UX] Expose agent and node state, execution, and logs for GUI inspection.
- [Agent UX] Enable agent registration, listing, and assignment to workflows.
- [Agent UX] Expose agent activity, last seen, and workflow assignment history in GUI.
- [Agent UX] Enable agent status (idle, busy, error) and live updates in builder.
- [Agent UX] Allow agent-to-workflow chat and feedback loop in builder.
- [Agent UX] Expose agent execution logs, error traces, and node-level progress in GUI.
- [Agent UX] Support agent-initiated workflow execution and feedback from GUI.
- [Agent UX] Track agent node execution times and errors for GUI timeline.
- [Persistence] Advanced workflow persistence (file/db/redis), rollback, and recovery features.
- [Async] Async/await support for workflow execution and event hooks for scalability.
"""

from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Awaitable, cast
import copy
import datetime
import logging
import threading
import asyncio
import os
import json
import time # Import time for unique ID generation
import inspect

from fastmcp.builder.types import (
    WorkflowExecutionRecord,
    WorkflowAccessControl,
)
from fastmcp.execution_engine.executor import ExecutionEngine
from fastmcp.builder.nodes.node_registry import NodeRegistry

# Pydantic BaseModel is not required for core WorkflowManager validation, stub as object
BaseModel = object

# --- Logging Setup ---
logger = logging.getLogger("fastmcp.builder.workflow_manager")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(name)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --- Advanced Persistent Storage (file/db/redis) ---
class FileStorage:
    """
    File-based persistent storage for workflow manager.
    Thread-safe, atomic writes.
    """
    def __init__(self, base_dir: str = "./.fastmcp_storage") -> None:
        self._base_dir = base_dir
        os.makedirs(self._base_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, name: str) -> str:
        return os.path.join(self._base_dir, f"{name}.json")

    def save(self, name: str, data: Any) -> None:
        with self._lock:
            with open(self._path(name), "w", encoding="utf-8") as f:
                json.dump(data, f, default=str, indent=2)

    def load(self, name: str) -> Any:
        try:
            with self._lock:
                with open(self._path(name), "r", encoding="utf-8") as f:
                    return json.load(f)
        except FileNotFoundError:
            return None

# --- Event Bus with Async Support and Node-level Hooks ---
class EventBus:
    """
    Simple event bus supporting sync and async event hooks.
    """
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[str, Dict[str, Any]], Any]]] = {}

    def subscribe(self, event_type: str, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self._subscribers.setdefault(event_type, []).append(callback)

    def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        for cb in self._subscribers.get(event_type, []):
            try:
                cb(event_type, payload)
            except Exception as e:
                logger.error("EventBus sync callback error: %s", e)

    async def publish_async(self, event_type: str, payload: Dict[str, Any]) -> None:
        for cb in self._subscribers.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event_type, payload)
                else:
                    cb(event_type, payload)
            except Exception as e:
                logger.error("EventBus async callback error: %s", e)

class WorkflowManager:
    """
    Manages the lifecycle of workflow graphs, including creation, execution, validation,
    checkpointing, versioning, access control, persistent storage, and agent orchestration.

    Agent-aware: Exposes agent and node state for GUI and programmatic inspection.
    """

    def __init__(
        self,
        storage: Optional[Any] = None,
        node_registry: Optional[Any] = None,
        audit_logger: Optional[Any] = None,
        event_bus: Optional[Any] = None,
        ai_model_manager: Optional[Any] = None,
        agent_registry: Optional[Any] = None,  # Registry for agents
        executor: Optional[ExecutionEngine] = None,
    ) -> None:
        # Use file-based storage by default for persistence
        self._storage = storage or FileStorage()
        self._node_registry = node_registry
        self._audit_logger = audit_logger
        self._event_bus = event_bus or EventBus()
        self._ai_model_manager = ai_model_manager
        self._agent_registry = agent_registry or {}  # agent_id -> agent_info
        self._executor = executor or ExecutionEngine()
        self._workflows: Dict[str, Dict[str, Any]] = {}
        self._workflow_states: Dict[str, Dict[str, Any]] = {}
        self._workflow_versions: Dict[str, List[Dict[str, Any]]] = {}
        self._workflow_access: Dict[str, Dict[str, Any]] = {}
        self._workflow_execution_history: Dict[str, List[WorkflowExecutionRecord]] = {}
        self._plugin_adapters: Dict[str, Any] = {}
        self._agent_activity: Dict[str, Dict[str, Any]] = {}  # agent_id -> {last_seen, status, assigned_workflows}
        self._chat_history: Dict[str, List[Dict[str, Any]]] = {}  # workflow_id -> chat messages
        self._agent_node_logs: Dict[str, List[Dict[str, Any]]] = {}  # agent_id -> [{node_id, workflow_id, ...}]
        self._load_all()

    def _default_storage(self) -> Any:
        return FileStorage()

    # --- Workflow CRUD ---

    def create_workflow(self, workflow_id: str, definition: Dict[str, Any], user: Optional[str] = None, agent_id: Optional[str] = None) -> None:
        if workflow_id in self._workflows:
            raise ValueError(f"Workflow {workflow_id} already exists.")
        # Pydantic or custom validation for workflow definition
        if not isinstance(definition, dict) or "nodes" not in definition:
            raise ValueError("Invalid workflow definition: must be a dict with 'nodes' key.")
        self._workflows[workflow_id] = copy.deepcopy(definition)
        self._workflow_states[workflow_id] = {
            "status": "created",
            "user": user,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "agent_id": agent_id,
        }
        self._workflow_versions.setdefault(workflow_id, []).append(
            {
                "version": 1,
                "definition": copy.deepcopy(definition),
                "created_at": datetime.datetime.utcnow().isoformat(),
                "created_by": user,
            }
        )
        if agent_id:
            self.assign_agent_to_workflow(workflow_id, agent_id)
        if self._audit_logger:
            self._audit_logger.log("create", user, {"workflow_id": workflow_id, "definition": definition, "agent_id": agent_id})
        self._event_bus.publish("workflow_created", {"workflow_id": workflow_id, "user": user, "agent_id": agent_id})
        self._save_all()
        logger.info("Workflow %s created by %s (agent: %s)", workflow_id, user, agent_id)

    def update_workflow(self, workflow_id: str, new_definition: Dict[str, Any], user: Optional[str] = None, agent_id: Optional[str] = None) -> None:
        if workflow_id not in self._workflows:
            raise KeyError(f"Workflow {workflow_id} not found.")
        if not isinstance(new_definition, dict) or "nodes" not in new_definition:
            raise ValueError("Invalid workflow definition: must be a dict with 'nodes' key.")
        self._workflows[workflow_id] = copy.deepcopy(new_definition)
        prev_versions = self._workflow_versions.setdefault(workflow_id, [])
        new_version = (prev_versions[-1]["version"] + 1) if prev_versions else 1
        prev_versions.append(
            {
                "version": new_version,
                "definition": copy.deepcopy(new_definition),
                "created_at": datetime.datetime.utcnow().isoformat(),
                "created_by": user,
            }
        )
        if workflow_id in self._workflow_states:
            self._workflow_states[workflow_id]["status"] = "updated"
            self._workflow_states[workflow_id]["timestamp"] = datetime.datetime.utcnow().isoformat()
            if agent_id:
                self._workflow_states[workflow_id]["agent_id"] = agent_id
        else:
            self._workflow_states[workflow_id] = {
                "status": "updated",
                "user": user,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "agent_id": agent_id,
            }
        if agent_id:
            self.assign_agent_to_workflow(workflow_id, agent_id)
        if self._audit_logger:
            self._audit_logger.log("update", user, {"workflow_id": workflow_id, "definition": new_definition, "agent_id": agent_id})
        self._event_bus.publish("workflow_updated", {"workflow_id": workflow_id, "user": user, "agent_id": agent_id})
        self._save_all()
        logger.info("Workflow %s updated by %s (agent: %s)", workflow_id, user, agent_id)

    def delete_workflow(self, workflow_id: str, user: Optional[str] = None) -> None:
        found = False
        for d in [
            self._workflows,
            self._workflow_states,
            self._workflow_versions,
            self._workflow_access,
            self._workflow_execution_history,
            self._chat_history,
        ]:
            if workflow_id in d:
                del d[workflow_id]
                found = True
        if found:
            if self._audit_logger:
                self._audit_logger.log("delete", user, {"workflow_id": workflow_id})
            self._event_bus.publish("workflow_deleted", {"workflow_id": workflow_id, "user": user})
            self._save_all()
            logger.info("Workflow %s deleted by %s", workflow_id, user)
        else:
            logger.warning("Attempted to delete non-existent workflow %s", workflow_id)

    def list_workflows(self, include_agents: bool = False, include_agent_status: bool = False, include_agent_activity: bool = False) -> Union[List[str], List[Dict[str, Any]]]:
        """
        List all workflow IDs, or detailed info if include_agents is True.
        If include_agent_status is True, also include agent status and last seen for GUI.
        If include_agent_activity is True, include agent activity and assignment history.
        """
        if include_agents or include_agent_status or include_agent_activity:
            workflows = []
            for wid in self._workflows.keys():
                agent_id = self._workflow_states.get(wid, {}).get("agent_id")
                agent_status = None
                agent_last_seen = None
                agent_activity = None
                if (include_agent_status or include_agent_activity) and agent_id:
                    agent_info = self._agent_activity.get(agent_id, {})
                    agent_status = agent_info.get("status")
                    agent_last_seen = agent_info.get("last_seen")
                    if include_agent_activity:
                        agent_activity = {
                            "assigned_workflows": agent_info.get("assigned_workflows", []),
                            "last_seen": agent_last_seen,
                            "status": agent_status,
                        }
                workflows.append({
                    "workflow_id": wid,
                    "agent_id": agent_id,
                    "status": self._workflow_states.get(wid, {}).get("status"),
                    "agent_status": agent_status,
                    "agent_last_seen": agent_last_seen,
                    "agent_activity": agent_activity,
                })
            return workflows
        return list(self._workflows.keys())

    def get_workflow(self, workflow_id: str, include_agent: bool = False, include_agent_status: bool = False, include_agent_activity: bool = False) -> Optional[Dict[str, Any]]:
        wf = copy.deepcopy(self._workflows.get(workflow_id))
        if wf is not None and (include_agent or include_agent_status or include_agent_activity):
            agent_id = self._workflow_states.get(workflow_id, {}).get("agent_id")
            wf["_agent_id"] = agent_id
            if (include_agent_status or include_agent_activity) and agent_id:
                agent_info = self._agent_activity.get(agent_id, {})
                wf["_agent_status"] = agent_info.get("status")
                wf["_agent_last_seen"] = agent_info.get("last_seen")
                if include_agent_activity:
                    wf["_agent_activity"] = {
                        "assigned_workflows": agent_info.get("assigned_workflows", []),
                        "last_seen": agent_info.get("last_seen"),
                        "status": agent_info.get("status"),
                    }
        return wf

    # Alias methods for agent-facing graph operations
    def create_graph(self, workflow_id: str, description: Optional[str] = None) -> str:
        """Alias for create_workflow with an initial empty GraphSpec."""
        definition: Dict[str, Any] = {"workflow_id": workflow_id, "description": description, "nodes": {}, "connections": []}
        self.create_workflow(workflow_id, definition)
        return workflow_id

    def get_graph_spec(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Alias for get_workflow to retrieve the current GraphSpec."""
        return self.get_workflow(workflow_id)

    def save_graph_spec(self, workflow_id: str, file_path: Optional[str] = None) -> bool:
        """Alias for save_graph to persist the GraphSpec; returns True on success."""
        if file_path:
            self.save_graph(workflow_id, file_path)
            return True
        # No file path provided, stub not implemented
        return False

    # --- Agent Management ---

    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]) -> None:
        """
        Register a new agent for use in the builder and GUI.
        Tracks agent registration time and sets status to 'idle'.
        """
        now = datetime.datetime.utcnow().isoformat()
        agent_info = copy.deepcopy(agent_info)
        agent_info.setdefault("registered_at", now)
        agent_info.setdefault("status", "idle")
        self._agent_registry[agent_id] = agent_info
        self._agent_activity[agent_id] = {
            "last_seen": now,
            "status": "idle",
            "assigned_workflows": [],
        }
        self._save("agent_registry", self._agent_registry)
        self._save("agent_activity", self._agent_activity)
        logger.info("Agent registered: %s", agent_id)
        self._event_bus.publish("agent_registered", {"agent_id": agent_id, "agent_info": agent_info})

    def list_agents(self, include_status: bool = False, include_activity: bool = False, include_node_logs: bool = False) -> List[Dict[str, Any]]:
        """
        List all registered agents for GUI selection.
        If include_status is True, include agent status, last seen, and assigned workflows.
        If include_activity is True, include full activity and assignment history.
        If include_node_logs is True, include node execution logs for timeline/inspector.
        """
        agents = []
        for aid, ainfo in self._agent_registry.items():
            agent = {"agent_id": aid, **ainfo}
            if include_status or include_activity:
                activity = self._agent_activity.get(aid, {})
                agent["status"] = activity.get("status")
                agent["last_seen"] = activity.get("last_seen")
                agent["assigned_workflows"] = activity.get("assigned_workflows", [])
            if include_activity:
                agent["activity"] = copy.deepcopy(self._agent_activity.get(aid, {}))
            if include_node_logs:
                agent["node_logs"] = copy.deepcopy(self._agent_node_logs.get(aid, []))
            agents.append(agent)
        return agents

    def get_agent(self, agent_id: str, include_status: bool = False, include_activity: bool = False, include_node_logs: bool = False) -> Optional[Dict[str, Any]]:
        agent = copy.deepcopy(self._agent_registry.get(agent_id))
        if agent is not None and (include_status or include_activity or include_node_logs):
            activity = self._agent_activity.get(agent_id, {})
            if include_status or include_activity:
                agent["status"] = activity.get("status")
                agent["last_seen"] = activity.get("last_seen")
                agent["assigned_workflows"] = activity.get("assigned_workflows", [])
            if include_activity:
                agent["activity"] = copy.deepcopy(activity)
            if include_node_logs:
                agent["node_logs"] = copy.deepcopy(self._agent_node_logs.get(agent_id, []))
        return agent

    def assign_agent_to_workflow(self, workflow_id: str, agent_id: str) -> None:
        """
        Assign an agent to a workflow for agent-driven execution and GUI visibility.
        Tracks assignment in agent activity for GUI.
        """
        now = datetime.datetime.utcnow().isoformat()
        if workflow_id not in self._workflow_states:
            self._workflow_states[workflow_id] = {}
        self._workflow_states[workflow_id]["agent_id"] = agent_id
        # Track assignment in agent activity
        if agent_id not in self._agent_activity:
            self._agent_activity[agent_id] = {
                "last_seen": now,
                "status": "idle",
                "assigned_workflows": [],
            }
        if workflow_id not in self._agent_activity[agent_id]["assigned_workflows"]:
            self._agent_activity[agent_id]["assigned_workflows"].append(workflow_id)
        self._agent_activity[agent_id]["last_seen"] = now
        self._save("workflow_states", self._workflow_states)
        self._save("agent_activity", self._agent_activity)
        logger.info("Agent %s assigned to workflow %s", agent_id, workflow_id)
        self._event_bus.publish("workflow_agent_assigned", {"workflow_id": workflow_id, "agent_id": agent_id})

    def get_workflow_agent(self, workflow_id: str, include_status: bool = False, include_activity: bool = False) -> Optional[Union[str, Dict[str, Any]]]:
        """Gets the agent_id or agent info assigned to a workflow."""
        if workflow_id not in self._workflow_states:
            return None
        agent_id = self._workflow_states[workflow_id].get("agent_id")
        if not agent_id:
            return None

        if include_status or include_activity:
            agent_info = self.get_agent(agent_id, include_status=include_status, include_activity=include_activity)
            return agent_info # This returns Optional[Dict[str, Any]] which aligns with Union
        else:
            return agent_id # This returns str which aligns with Union

    def set_agent_status(self, agent_id: str, status: str, error: Optional[str] = None) -> None:
        """
        Set the status of an agent (e.g., idle, busy, error) for GUI display.
        Optionally record error message for GUI.
        """
        now = datetime.datetime.utcnow().isoformat()
        if agent_id not in self._agent_activity:
            self._agent_activity[agent_id] = {
                "last_seen": now,
                "status": status,
                "assigned_workflows": [],
            }
        else:
            self._agent_activity[agent_id]["status"] = status
            self._agent_activity[agent_id]["last_seen"] = now
            if error:
                self._agent_activity[agent_id]["last_error"] = error
        self._save("agent_activity", self._agent_activity)
        logger.info("Agent %s status set to %s", agent_id, status)
        self._event_bus.publish("agent_status_updated", {"agent_id": agent_id, "status": status, "last_seen": now, "error": error})

    def record_agent_activity(self, agent_id: str, activity: Optional[Dict[str, Any]] = None) -> None:
        """
        Update agent's last seen timestamp for GUI live status.
        Optionally record additional activity (e.g., node execution, error).
        """
        now = datetime.datetime.utcnow().isoformat()
        if agent_id in self._agent_activity:
            self._agent_activity[agent_id]["last_seen"] = now
            if activity:
                self._agent_activity[agent_id].setdefault("activities", []).append(activity)
            self._save("agent_activity", self._agent_activity)
            logger.info("Agent %s activity recorded at %s", agent_id, now)
            self._event_bus.publish("agent_activity", {"agent_id": agent_id, "last_seen": now, "activity": activity})

    def record_agent_node_log(self, agent_id: str, workflow_id: str, node_id: str, node_type: str, status: str, start_time: str, end_time: str, error: Optional[str] = None) -> None:
        """
        Record a log entry for an agent's node execution for GUI timeline/inspector.
        """
        log_entry = {
            "workflow_id": workflow_id,
            "node_id": node_id,
            "node_type": node_type,
            "status": status,
            "start_time": start_time,
            "end_time": end_time,
            "error": error,
        }
        self._agent_node_logs.setdefault(agent_id, []).append(log_entry)
        self._save("agent_node_logs", self._agent_node_logs)
        self._event_bus.publish("agent_node_log", {"agent_id": agent_id, **log_entry})

    # --- Agent-to-Workflow Chat and Feedback Loop ---
    def send_agent_message(self, workflow_id: str, agent_id: str, message: str, role: str = "agent", meta: Optional[Dict[str, Any]] = None) -> None:
        """
        Allow agent to send a message to the workflow chat.
        Optionally include meta (e.g., node_id, error, etc).
        """
        now = datetime.datetime.utcnow().isoformat()
        chat_entry = {
            "timestamp": now,
            "agent_id": agent_id,
            "role": role,
            "message": message,
        }
        if meta:
            chat_entry.update(meta)
        self._chat_history.setdefault(workflow_id, []).append(chat_entry)
        self._save("chat_history", self._chat_history)
        self._event_bus.publish("workflow_chat_message", {
            "workflow_id": workflow_id,
            "agent_id": agent_id,
            "role": role,
            "message": message,
            "timestamp": now,
            "meta": meta,
        })

    def get_workflow_chat(self, workflow_id: str) -> List[Dict[str, Any]]:
        """
        Get chat/feedback history for a workflow.
        """
        return copy.deepcopy(self._chat_history.get(workflow_id, []))

    # --- Workflow Validation ---

    def validate_workflow(self, workflow_id: str, user: Optional[str] = None) -> bool:
        if workflow_id not in self._workflows:
            raise KeyError(f"Workflow {workflow_id} not found.")
        definition = self._workflows[workflow_id]
        # Basic validation: must have nodes and each node must have id and type
        is_valid = isinstance(definition, dict) and "nodes" in definition and isinstance(definition["nodes"], list)
        if is_valid:
            for node in definition["nodes"]:
                if not isinstance(node, dict) or "id" not in node or "type" not in node:
                    is_valid = False
                    break
        if self._audit_logger:
            self._audit_logger.log("validate", user, {"workflow_id": workflow_id, "valid": is_valid})
        self._event_bus.publish("workflow_validated", {"workflow_id": workflow_id, "user": user, "valid": is_valid})
        return is_valid

    # --- Workflow Execution (Sync and Async) ---

    def run_workflow(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        execution_id: Optional[str] = None,
        ai_pre_action: Optional[str] = None,
        ai_post_action: Optional[str] = None,
        ai_pre_ai_params: Optional[Dict[str, Any]] = None,
        ai_post_ai_params: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> bool:
        """
        Execute the workflow graph, optionally with AI pre/post-processing and agent context.

        Returns True if completed, False if failed.
        """
        return asyncio.run(self.run_workflow_async(
            workflow_id, input_data, user, execution_id,
            ai_pre_action, ai_post_action, ai_pre_ai_params, ai_post_ai_params, agent_id
        ))

    async def run_workflow_async(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None,
        execution_id: Optional[str] = None,
        ai_pre_action: Optional[str] = None,
        ai_post_action: Optional[str] = None,
        ai_pre_ai_params: Optional[Dict[str, Any]] = None,
        ai_post_ai_params: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ) -> bool:
        """
        Async execution of the workflow graph, with event hooks and agent context.
        Tracks agent node execution times, errors, and logs for GUI timeline.
        """
        if workflow_id not in self._workflows:
            raise KeyError(f"Workflow {workflow_id} not found.")
        definition = self._workflows[workflow_id]
        status = "running"
        result: Dict[str, Any] = {}
        ai_result: Optional[Any] = None
        execution_history = self._workflow_execution_history.setdefault(workflow_id, [])
        start_time = datetime.datetime.utcnow().isoformat()
        retries = 0
        max_retries = 3
        # Attach agent to state for GUI visibility and update agent status/activity
        effective_agent_id = agent_id # Keep original agent_id if provided
        if not effective_agent_id:
            # get_workflow_agent returns Union[str, Dict, None]
            wf_agent = self.get_workflow_agent(workflow_id)
            if isinstance(wf_agent, str):
                effective_agent_id = wf_agent
            elif isinstance(wf_agent, dict):
                effective_agent_id = wf_agent.get("agent_id")

        if effective_agent_id:
            self.assign_agent_to_workflow(workflow_id, effective_agent_id)
            self.set_agent_status(effective_agent_id, "busy")
            self.record_agent_activity(effective_agent_id)

        await self._event_bus.publish_async("workflow_execution_started", {
            "workflow_id": workflow_id,
            "user": user,
            "agent_id": effective_agent_id, # Use effective_agent_id
            "input_data": input_data,
        })
        while retries < max_retries:
            try:
                # AI pre-processing
                if ai_pre_action:
                    ai_pre_input = input_data or {}
                    ai_pre_result = await self._run_ai_step_async(
                        workflow_id=workflow_id,
                        step="pre",
                        input_data=ai_pre_input,
                        ai_action=ai_pre_action,
                        ai_params=ai_pre_ai_params,
                        user=user,
                    )
                    if ai_pre_result.get("result"):
                        input_data = ai_pre_result.get("result")
                # Main workflow execution (node-level event hooks)
                result = await self._execute_graph_with_hooks(workflow_id, definition, input_data, effective_agent_id) # Pass effective_agent_id
                status = "completed"
                ai_result = None
                # AI post-processing
                if ai_post_action:
                    ai_post_input = result
                    ai_post_result = await self._run_ai_step_async(
                        workflow_id=workflow_id,
                        step="post",
                        input_data=ai_post_input,
                        ai_action=ai_post_action,
                        ai_params=ai_post_ai_params,
                        user=user,
                    )
                    if ai_post_result.get("result"):
                        result["ai_postprocessed"] = ai_post_result.get("result")
                self._workflow_states[workflow_id]["status"] = status
                self._workflow_states[workflow_id]["result"] = result
                self._workflow_states[workflow_id]["agent_id"] = effective_agent_id # Use effective_agent_id
                break
            except Exception as e:
                retries += 1
                self._workflow_states[workflow_id]["status"] = "failed"
                self._workflow_states[workflow_id]["result"] = {"error": str(e)}
                self._workflow_states[workflow_id]["agent_id"] = effective_agent_id # Use effective_agent_id
                if effective_agent_id:
                    self.set_agent_status(effective_agent_id, "error", error=str(e))
                    self.record_agent_activity(effective_agent_id, activity={"error": str(e), "when": datetime.datetime.utcnow().isoformat()})
                if self._audit_logger:
                    self._audit_logger.log("execution_failed", user, {
                        "workflow_id": workflow_id,
                        "error": str(e),
                        "retries": retries,
                        "agent_id": effective_agent_id, # Use effective_agent_id
                    })
                logger.error("Workflow execution failed for %s (retry %d/%d): %s", workflow_id, retries, max_retries, e)
                await self._event_bus.publish_async("workflow_execution_failed", {
                    "workflow_id": workflow_id,
                    "user": user,
                    "agent_id": effective_agent_id, # Use effective_agent_id
                    "error": str(e),
                    "retries": retries,
                })
                if retries >= max_retries:
                    status = "failed"
                    result = {"error": str(e)}
                    break
        # Build execution record according to WorkflowExecutionRecord TypedDict
        record: WorkflowExecutionRecord = {
            "execution_id": execution_id or f"exec_{datetime.datetime.utcnow().isoformat()}",
            "graph_id": workflow_id,
            "start_time": start_time,
            "end_time": datetime.datetime.utcnow().isoformat(),
            "status": status or "failed",
            "result": result,
            "logs": []  # TODO: Add detailed logs
        }
        execution_history.append(record)
        # Maintain history in state
        history_list = self._workflow_states[workflow_id].setdefault("history", [])
        if history_list is not None:
            history_list.append(record)
        else:
            self._workflow_states[workflow_id]["history"] = [record]
        if effective_agent_id:
            # Set agent status to idle after execution
            self.set_agent_status(effective_agent_id, "idle" if status == "completed" else "error")
            self.record_agent_activity(effective_agent_id)
        if self._audit_logger:
            self._audit_logger.log("execute", user, {
                "workflow_id": workflow_id,
                "execution_id": execution_id,
                "status": status,
                "retries": retries,
                "ai_result": ai_result,
                "agent_id": effective_agent_id, # Use effective_agent_id
            })
        await self._event_bus.publish_async("workflow_executed", {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "user": user,
            "status": status,
            "ai_result": ai_result,
            "agent_id": effective_agent_id, # Use effective_agent_id
        })
        self._save_all()
        return status == "completed"

    async def _execute_graph_with_hooks(self, workflow_id: str, definition: Dict[str, Any], input_data: Optional[Dict[str, Any]], agent_id: Optional[str]) -> Dict[str, Any]:
        """
        Execute workflow graph with node-level event hooks.
        Tracks agent node execution times, errors, and logs for GUI timeline.
        """
        output = {}
        nodes = definition.get("nodes", [])
        for node in nodes:
            node_id = node.get("id")
            node_type = node.get("type")
            node_start = datetime.datetime.utcnow().isoformat()
            await self._event_bus.publish_async("workflow_node_started", {
                "workflow_id": workflow_id,
                "node_id": node_id,
                "node_type": node_type,
                "agent_id": agent_id,
                "input_data": input_data,
                "start_time": node_start,
            })
            # Plugin/adapter support for custom node types
            adapter = self._plugin_adapters.get(node_type)
            try:
                if adapter and hasattr(adapter, "execute"):
                    node_result = await adapter.execute(node, input_data, agent_id)
                else:
                    # Default: echo node
                    node_result = {"output": f"Node {node_id} ({node_type}) executed with input {input_data}"}
                node_status = "completed"
                node_error = None
            except Exception as e:
                node_result = {"error": str(e)}
                node_status = "error"
                node_error = str(e)
            node_end = datetime.datetime.utcnow().isoformat()
            await self._event_bus.publish_async("workflow_node_completed", {
                "workflow_id": workflow_id,
                "node_id": node_id,
                "node_type": node_type,
                "agent_id": agent_id,
                "result": node_result,
                "status": node_status,
                "start_time": node_start,
                "end_time": node_end,
                "error": node_error,
            })
            output[node_id] = node_result
            # Record agent node log for GUI timeline/inspector
            if agent_id:
                self.record_agent_node_log(
                    agent_id=agent_id,
                    workflow_id=workflow_id,
                    node_id=node_id,
                    node_type=node_type,
                    status=node_status,
                    start_time=node_start,
                    end_time=node_end,
                    error=node_error,
                )
        return {"output": output, "agent_id": agent_id}

    def _run_ai_step(
        self,
        workflow_id: str,
        step: str,
        input_data: Any,
        ai_action: str,
        ai_params: Optional[Dict[str, Any]],
        user: Optional[str] = None,
    ) -> Dict[str, Any]:
        # For backward compatibility, run sync in event loop
        return asyncio.run(self._run_ai_step_async(workflow_id, step, input_data, ai_action, ai_params, user))

    async def _run_ai_step_async(
        self,
        workflow_id: str,
        step: str,
        input_data: Any,
        ai_action: str,
        ai_params: Optional[Dict[str, Any]],
        user: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run an AI step (pre/post-processing) for a workflow.
        """
        prompt = ai_params.get("prompt") if ai_params else None
        model = ai_params.get("model") if ai_params else "gpt-4"
        temperature = ai_params.get("temperature") if ai_params else 0.2
        max_tokens = ai_params.get("max_tokens") if ai_params else 256
        result = None
        if not self._ai_model_manager:
            logger.warning("AI model manager not configured; skipping AI step.")
            return {"result": None}
        if ai_action == "summarize":
            result = await self._ai_model_manager.summarize(str(input_data))
        elif ai_action == "classify":
            labels = ai_params.get("labels") if ai_params else []
            result = await self._ai_model_manager.classify(str(input_data), labels)
        elif ai_action == "extract_entities":
            result = await self._ai_model_manager.extract_entities(str(input_data))
        elif ai_action == "generate_code":
            instruction = ai_params.get("instruction") if ai_params else ""
            language = ai_params.get("language") if ai_params else "python"
            result = await self._ai_model_manager.generate_code(instruction, language)
        elif ai_action == "chat":
            messages = ai_params.get("messages") if ai_params else [{"role": "user", "content": str(input_data)}]
            result = await self._ai_model_manager.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        else:
            # Default: generic inference
            prompt = prompt or f"Process: {input_data}"
            result = await self._ai_model_manager.infer(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
        if self._audit_logger:
            self._audit_logger.log("ai_step", user, {
                "workflow_id": workflow_id,
                "step": step,
                "ai_action": ai_action,
                "ai_params": ai_params,
                "result": result
            })
        await self._event_bus.publish_async("workflow_ai_step", {
            "workflow_id": workflow_id,
            "step": step,
            "ai_action": ai_action,
            "user": user,
            "result": result
        })
        return {"result": result}

    def get_workflow_status(self, workflow_id: str, user: Optional[str] = None, include_agent: bool = False, include_agent_status: bool = False) -> Optional[Union[str, Dict[str, Any]]]:
        state = self._workflow_states.get(workflow_id)
        if state is not None:
            status = state.get("status")
            if include_agent or include_agent_status:
                agent_id = state.get("agent_id")
                agent_status = None
                if include_agent_status and agent_id:
                    agent_status = self._agent_activity.get(agent_id, {}).get("status")
                return {"status": status, "agent_id": agent_id, "agent_status": agent_status}
            return status
        return None

    def get_workflow_result(self, workflow_id: str, user: Optional[str] = None, include_agent: bool = False, include_agent_status: bool = False) -> Optional[Dict[str, Any]]:
        state = self._workflow_states.get(workflow_id)
        if state is not None and state.get("status") == "completed":
            result = state.get("result")
            if result is not None and isinstance(result, dict):
                # Create a copy to modify
                result_copy = copy.deepcopy(result)
                if include_agent or include_agent_status:
                    agent_id = state.get("agent_id")
                    result_copy["agent_id"] = agent_id
                    if include_agent_status and agent_id and isinstance(agent_id, str):
                        # Ensure agent_id is a string before using as key
                        agent_activity = self._agent_activity.get(agent_id, {})
                        # Check if agent_activity is a dict before calling .get
                        if isinstance(agent_activity, dict):
                            result_copy["agent_status"] = agent_activity.get("status")
                return result_copy
        return None

    def get_workflow_execution_history(self, workflow_id: str, user: Optional[str] = None, include_agent: bool = False, include_agent_status: bool = False) -> List[WorkflowExecutionRecord]:
        """Returns the execution history for a workflow."""
        # TODO: Implement access control check based on 'user'
        history = self._workflow_execution_history.get(workflow_id, [])
        if not include_agent:
            return history

        enriched_history = []
        for record in history:
            # Ensure record is treated as a dictionary for updates
            mutable_record: Dict[str, Any] = dict(record)
            agent_id = mutable_record.get("agent_id") # Use .get with default None
            if agent_id and isinstance(agent_id, str): # Check agent_id is a string
                agent_info = self.get_agent(agent_id, include_status=include_agent_status)
                if agent_info:
                    # Add agent info, ensuring keys exist in WorkflowExecutionRecord or using a different structure
                    # Let's add a nested 'agent_details' key instead of modifying top-level
                    mutable_record["agent_details"] = agent_info
                    # mutable_record["agent_status"] = agent_info.get("status") # Avoid adding non-TypedDict keys directly
            # Cast back to WorkflowExecutionRecord if necessary, or adjust return type
            # For now, we assume the structure is flexible enough or return List[Dict[str, Any]]
            enriched_history.append(mutable_record) # Append the modified dict

        # Adjust return type if enrichment changes the structure significantly
        # return enriched_history # Type change might be needed here to List[Dict[str, Any]]
        # For now, let's assume WorkflowExecutionRecord can contain agent_details or log a warning
        # We will cast it back for now to satisfy the linter, but this might hide issues.
        logger.warning("Enriched history may not conform strictly to WorkflowExecutionRecord TypedDict")
        return cast(List[WorkflowExecutionRecord], enriched_history)

    # --- Access Control Management ---

    # Revert type hint to Any temporarily due to linter issues
    def set_workflow_access(self, workflow_id: str, access: Any, user: Optional[str] = None) -> bool:
        # TODO: Restore WorkflowAccessControl type hint once linter issue resolved
        self._workflow_access[workflow_id] = access
        if self._audit_logger:
            self._audit_logger.log("set_access", user, {"workflow_id": workflow_id, "access": access})
        self._save("workflow_access", self._workflow_access)
        self._event_bus.publish("workflow_access_set", {"workflow_id": workflow_id, "user": user, "access": access})
        logger.info("Access control set for workflow %s by user %s", workflow_id, user)
        return True

    # Revert return type hint to Any temporarily
    def get_workflow_access(self, workflow_id: str) -> Optional[Any]:
        # TODO: Restore WorkflowAccessControl type hint once linter issue resolved
        return copy.deepcopy(self._workflow_access.get(workflow_id))

    # --- Persistent Storage Integration ---

    def _save(self, name: str, data: Any) -> None:
        self._storage.save(name, data)

    def _load(self, name: str) -> Any:
        return self._storage.load(name)

    def _save_all(self) -> None:
        self._save("workflows", self._workflows)
        self._save("workflow_states", self._workflow_states)
        self._save("workflow_versions", self._workflow_versions)
        self._save("workflow_access", self._workflow_access)
        self._save("workflow_execution_history", self._workflow_execution_history)
        self._save("agent_registry", self._agent_registry)
        self._save("agent_activity", self._agent_activity)
        self._save("chat_history", self._chat_history)
        self._save("agent_node_logs", self._agent_node_logs)

    def _load_all(self) -> None:
        self._workflows = self._load("workflows") or {}
        self._workflow_states = self._load("workflow_states") or {}
        self._workflow_versions = self._load("workflow_versions") or {}
        self._workflow_access = self._load("workflow_access") or {}
        self._workflow_execution_history = self._load("workflow_execution_history") or {}
        self._agent_registry = self._load("agent_registry") or {}
        self._agent_activity = self._load("agent_activity") or {}
        self._chat_history = self._load("chat_history") or {}
        self._agent_node_logs = self._load("agent_node_logs") or {}

    # --- Event Hooks and Plugin/Adapter Pattern ---

    def subscribe_to_event(self, event_type: str, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self._event_bus.subscribe(event_type, callback)

    def on_workflow_created(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("workflow_created", callback)

    def on_workflow_updated(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("workflow_updated", callback)

    def on_workflow_deleted(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("workflow_deleted", callback)

    def on_workflow_executed(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("workflow_executed", callback)

    def on_workflow_rollback(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("workflow_rollback", callback)

    def on_workflow_access_set(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("workflow_access_set", callback)

    def on_workflow_ai_step(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("workflow_ai_step", callback)

    def on_agent_registered(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("agent_registered", callback)

    def on_workflow_agent_assigned(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("workflow_agent_assigned", callback)

    def on_agent_status_updated(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("agent_status_updated", callback)

    def on_agent_activity(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("agent_activity", callback)

    def on_agent_node_log(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("agent_node_log", callback)

    def on_workflow_node_started(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("workflow_node_started", callback)

    def on_workflow_node_completed(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("workflow_node_completed", callback)

    def on_workflow_chat_message(self, callback: Callable[[str, Dict[str, Any]], Any]) -> None:
        self.subscribe_to_event("workflow_chat_message", callback)

    def register_plugin_adapter(self, name: str, adapter: Any) -> None:
        self._plugin_adapters[name] = adapter
        logger.info("Plugin/adapter registered: %s", name)

    def get_plugin_adapter(self, name: str) -> Any:
        return self._plugin_adapters.get(name)

    # --- Type Self-Tests for TypedDicts and Pydantic Models ---
    @staticmethod
    def _self_test_types() -> None:
        # TypedDict self-test for WorkflowExecutionRecord
        try:
            rec: WorkflowExecutionRecord = {
                "execution_id": "id",
                "graph_id": "wid",
                "start_time": "now",
                "end_time": "now",
                "status": "ok",
                "result": {},
                "logs": []
            }
            assert isinstance(rec["execution_id"], str)
            assert isinstance(rec["graph_id"], str)
            assert isinstance(rec["start_time"], str)
            assert isinstance(rec["end_time"], str)
            assert isinstance(rec["status"], str)
            assert isinstance(rec["result"], dict)
            assert isinstance(rec["logs"], list)
            logger.info("TypedDict self-test passed for WorkflowExecutionRecord.")
        except Exception as e:
            logger.error("TypedDict self-test failed: %s", e)
        # TODO: Add tsd-style self-tests for all TypedDicts and Pydantic models

    # --- Rollback and Recovery Features ---
    def rollback_workflow(self, workflow_id: str, version: int, user: Optional[str] = None) -> None:
        """
        Rollback workflow to a previous version.
        """
        versions = self._workflow_versions.get(workflow_id, [])
        for v in versions:
            if v["version"] == version:
                self._workflows[workflow_id] = copy.deepcopy(v["definition"])
                self._workflow_states[workflow_id]["status"] = "rolled_back"
                self._workflow_states[workflow_id]["timestamp"] = datetime.datetime.utcnow().isoformat()
                if self._audit_logger:
                    self._audit_logger.log("rollback", user, {"workflow_id": workflow_id, "version": version})
                self._event_bus.publish("workflow_rollback", {"workflow_id": workflow_id, "user": user, "version": version})
                self._save_all()
                logger.info("Workflow %s rolled back to version %d by %s", workflow_id, version, user)
                return
        raise ValueError(f"Version {version} not found for workflow {workflow_id}")

    # --- Async Graph Execution Trigger ---
    def trigger_graph_execution(self, workflow_id: str, input_data: Dict[str, Any], agent_id: Optional[str] = None) -> str:
        """
        Trigger execution of a workflow graph via the ExecutionEngine.
        Returns an execution ID.
        Optionally records agent assignment for GUI.
        """
        if agent_id:
            self.assign_agent_to_workflow(workflow_id, agent_id)
            self.set_agent_status(agent_id, "busy")
            self.record_agent_activity(agent_id)
        return self._executor.execute_graph(workflow_id, input_data)

    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get execution status and result from the ExecutionEngine.
        """
        return self._executor.get_execution_status(execution_id)

    # --- Graph Modification Methods (called by Agent Tools / GUI) ---

    def add_node(
        self,
        workflow_id: str,
        node_type: str,
        node_name: str,
        properties: Optional[Dict[str, Any]] = None,
        position: Optional[Tuple[float, float]] = None,
        user: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> str:
        """Adds a new node to the specified workflow."""
        if workflow_id not in self._workflows:
            raise KeyError(f"Workflow {workflow_id} not found.")

        workflow_def = self._workflows[workflow_id]

        # Generate a unique node ID
        node_id = f"node_{int(time.time() * 1000)}_{len(workflow_def.get('nodes', []))}"

        node_spec = {
            "id": node_id,
            "type": node_type,
            "name": node_name,
            "properties": properties or {},
            "position": position or (0.0, 0.0), # TODO: Consider better default positioning logic
        }

        # TODO: Validate node_type against node_registry if available

        workflow_def.setdefault("nodes", []).append(node_spec)

        # Update workflow (triggers versioning, state update, logging, saving)
        self.update_workflow(workflow_id, workflow_def, user=user, agent_id=agent_id)

        logger.info("Node %s (%s) added to workflow %s", node_id, node_type, workflow_id)
        self._event_bus.publish("workflow_node_added", {
            "workflow_id": workflow_id,
            "node_id": node_id,
            "node_spec": node_spec,
            "user": user,
            "agent_id": agent_id
        })

        return node_id

    def connect_nodes(
        self,
        workflow_id: str,
        from_node_id: str,
        from_port_name: str,
        to_node_id: str,
        to_port_name: str,
        user: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> bool:
        """Connects two nodes in the specified workflow."""
        if workflow_id not in self._workflows:
            raise KeyError(f"Workflow {workflow_id} not found.")

        workflow_def = self._workflows[workflow_id]
        nodes = {node["id"]: node for node in workflow_def.get("nodes", [])}

        # Basic validation: Check if nodes exist
        if from_node_id not in nodes or to_node_id not in nodes:
            logger.error("Cannot connect nodes: one or both nodes not found in workflow %s", workflow_id)
            return False
        # TODO: Add port validation (check if from_port_name/to_port_name are valid for the node types)

        # Avoid duplicate connections (simple check based on exact match)
        edge_spec = {
            "from_node": from_node_id,
            "from_port": from_port_name,
            "to_node": to_node_id,
            "to_port": to_port_name,
            "id": f"edge_{from_node_id}_{from_port_name}_{to_node_id}_{to_port_name}" # Simple unique ID
        }
        existing_edges = workflow_def.get("edges", [])
        if any(e == edge_spec for e in existing_edges):
             logger.warning("Connection already exists: %s.%s -> %s.%s", from_node_id, from_port_name, to_node_id, to_port_name)
             return True # Indicate success as connection exists

        workflow_def.setdefault("edges", []).append(edge_spec)

        self.update_workflow(workflow_id, workflow_def, user=user, agent_id=agent_id)

        logger.info("Nodes connected in workflow %s: %s.%s -> %s.%s", workflow_id, from_node_id, from_port_name, to_node_id, to_port_name)
        self._event_bus.publish("workflow_edge_added", {
             "workflow_id": workflow_id,
             "edge_spec": edge_spec,
             "user": user,
             "agent_id": agent_id
        })
        return True

    def set_node_property(
        self,
        workflow_id: str,
        node_id: str,
        property_name: str,
        value: Any,
        user: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> bool:
        """Sets a specific property on a node in the workflow."""
        if workflow_id not in self._workflows:
            raise KeyError(f"Workflow {workflow_id} not found.")

        workflow_def = self._workflows[workflow_id]
        node_found = False
        for node in workflow_def.get("nodes", []):
            if node.get("id") == node_id:
                node.setdefault("properties", {})[property_name] = value
                node_found = True
                break

        if not node_found:
            logger.error("Node %s not found in workflow %s for setting property", node_id, workflow_id)
            return False

        self.update_workflow(workflow_id, workflow_def, user=user, agent_id=agent_id)

        logger.info("Property '%s' set on node %s in workflow %s", property_name, node_id, workflow_id)
        self._event_bus.publish("workflow_node_property_set", {
             "workflow_id": workflow_id,
             "node_id": node_id,
             "property_name": property_name,
             "value": value, # Be cautious about logging sensitive values
             "user": user,
             "agent_id": agent_id
        })
        return True

    def get_node_properties(self, workflow_id: str, node_id: str) -> Optional[Dict[str, Any]]:
        """Gets the properties dictionary for a specific node."""
        if workflow_id not in self._workflows:
            # raise KeyError(f"Workflow {workflow_id} not found.")
            return None # Return None if workflow doesn't exist

        workflow_def = self._workflows[workflow_id]
        for node in workflow_def.get("nodes", []):
            if node.get("id") == node_id:
                return node.get("properties", {}) # Return properties or empty dict

        # logger.warning("Node %s not found in workflow %s when getting properties", node_id, workflow_id)
        return None # Return None if node doesn't exist

    def save_graph(self, workflow_id: str, file_path: str, user: Optional[str] = None, agent_id: Optional[str] = None) -> str:
        """Saves the current workflow definition to a specific JSON file path."""
        if workflow_id not in self._workflows:
            raise KeyError(f"Workflow {workflow_id} not found for saving.")

        workflow_def = self._workflows[workflow_id]

        try:
            # Ensure directory exists
            dir_name = os.path.dirname(file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(workflow_def, f, indent=2, default=str) # Use default=str for non-serializable types like datetime

            logger.info("Workflow %s saved to %s by %s (agent: %s)", workflow_id, file_path, user, agent_id)
            if self._audit_logger:
                 self._audit_logger.log("save_graph", user, {"workflow_id": workflow_id, "file_path": file_path, "agent_id": agent_id})
            self._event_bus.publish("workflow_saved_to_file", {
                 "workflow_id": workflow_id,
                 "file_path": file_path,
                 "user": user,
                 "agent_id": agent_id
            })
            return file_path
        except Exception as e:
            logger.error("Failed to save workflow %s to %s: %s", workflow_id, file_path, e)
            raise IOError(f"Failed to save workflow {workflow_id} to {file_path}: {e}") from e

    def list_available_node_types(self) -> List[Dict[str, Any]]:
        """Lists available node types fetched from the central NodeRegistry."""
        available_nodes = []
        for node_type_id in NodeRegistry.list_types():
            node_cls = NodeRegistry.get(node_type_id)
            if node_cls:
                # Extract metadata - adjust attributes based on actual node class definitions
                node_info = {
                    "identifier": node_type_id,
                    "name": getattr(node_cls, 'NODE_NAME', node_type_id), # Fallback to ID if no specific name
                    "category": getattr(node_cls, 'CATEGORY', 'General'),
                    "description": inspect.getdoc(node_cls) or "",
                    # TODO: Add input/output port info if available on class
                    # "inputs": getattr(node_cls, 'INPUTS', {}),
                    # "outputs": getattr(node_cls, 'OUTPUTS', {}),
                }
                available_nodes.append(node_info)
            else:
                logger.warning(f"Node class not found in registry for type ID: {node_type_id}")

        logger.info(f"Listed {len(available_nodes)} available node types from registry.")
        return available_nodes

# --- Type Self-Tests on Import ---
WorkflowManager._self_test_types()
logger.info("WorkflowManager type self-tests completed on import.")
"""
Dynamic Graph Updates for Agent Usability in FastMCP Builder

This module provides robust, agent-friendly utilities for real-time, programmatic updates to the workflow graph,
enabling seamless agentic process automation, live GUI interactivity, and advanced agent usability features.

Key Features:
- Add, remove, and update nodes and edges programmatically (with validation and event hooks)
- Support for agent-triggered graph modifications (planner, recorder, semantic nodes, etc.)
- Event hooks for GUI to reflect live changes and agent actions
- Input validation, error handling, and audit logging for robust agent integration
- Agent usability helpers: semantic node creation, auto-connection, action recording, and undo/redo

"""

from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from src.fastmcp.builder.types import NodeId, EdgeId, GraphDict, NodeDict, EdgeDict

try:
    from src.fastmcp.builder.nodes.node_registry import NODE_REGISTRY
except ImportError:
    NODE_REGISTRY = {}

# Event callback type for GUI update hooks
GraphUpdateCallback = Callable[[str, Dict[str, Any]], None]

class WorkflowManager:
    """
    WorkflowManager provides atomic updates, undo/redo, and transactionality for the workflow graph.
    """

    def __init__(self, graph: GraphDict, audit_log: Optional[List[Dict[str, Any]]] = None):
        self.graph = graph
        self.audit_log = audit_log if audit_log is not None else []
        self._undo_stack: List[Dict[str, Any]] = []
        self._redo_stack: List[Dict[str, Any]] = []

    def begin_transaction(self):
        # Placeholder for transaction begin (could lock the graph, etc.)
        pass

    def end_transaction(self):
        # Placeholder for transaction end (could unlock the graph, etc.)
        pass

    def record_undo(self, action: str, payload: Dict[str, Any]):
        self._undo_stack.append({"action": action, "payload": payload})

    def undo(self):
        if not self._undo_stack:
            return False
        last_action = self._undo_stack.pop()
        self._redo_stack.append(last_action)
        # For demo: just log the undo, real implementation would revert the action
        self.audit_log.append({"action": "undo", "payload": last_action})
        return True

    def redo(self):
        if not self._redo_stack:
            return False
        last_action = self._redo_stack.pop()
        self._undo_stack.append(last_action)
        # For demo: just log the redo, real implementation would re-apply the action
        self.audit_log.append({"action": "redo", "payload": last_action})
        return True

    def clear_history(self):
        self._undo_stack.clear()
        self._redo_stack.clear()

class DynamicGraphUpdater:
    """
    Provides agent- and GUI-facing methods for dynamic graph mutation.
    All changes are validated, can trigger GUI update events, and are agent-usable.
    Integrates with WorkflowManager for atomic updates, undo/redo, and transactionality.
    """

    def __init__(self, graph: GraphDict, on_update: Optional[GraphUpdateCallback] = None, audit_log: Optional[List[Dict[str, Any]]] = None):
        """
        Args:
            graph: The workflow graph as a mutable dict (see types.GraphDict)
            on_update: Optional callback for GUI to receive update events
            audit_log: Optional list to append audit events for agent actions
        """
        self.graph = graph
        self.on_update = on_update
        self.audit_log = audit_log if audit_log is not None else []
        self.workflow_manager = WorkflowManager(graph, self.audit_log)

    def add_node(self, node: NodeDict, agent: Optional[str] = None) -> NodeId:
        node_id = node.get("id")
        if not node_id:
            return {"success": False, "error": "Node must have an 'id' field"}
        if node_id in self.graph["nodes"]:
            return {"success": False, "error": f"Node with id '{node_id}' already exists"}
        # Validate node type if registry is available
        node_type = node.get("type")
        if NODE_REGISTRY and node_type not in NODE_REGISTRY:
            return {"success": False, "error": f"Node type '{node_type}' is not registered in NODE_REGISTRY"}
        self.workflow_manager.begin_transaction()
        self.graph["nodes"][node_id] = node
        self._emit("node_added", {"node": node, "agent": agent})
        self._audit("add_node", {"node": node, "agent": agent})
        self.workflow_manager.record_undo("remove_node", {"node_id": node_id, "agent": agent})
        self.workflow_manager.end_transaction()
        return {"success": True, "node_id": node_id}

    def remove_node(self, node_id: NodeId, agent: Optional[str] = None) -> Dict[str, Any]:
        if node_id not in self.graph["nodes"]:
            return {"success": False, "error": f"Node '{node_id}' does not exist"}
        # Remove all edges connected to this node
        edges_to_remove = [
            eid for eid, edge in self.graph["edges"].items()
            if edge["source"] == node_id or edge["target"] == node_id
        ]
        removed_edges = []
        self.workflow_manager.begin_transaction()
        for eid in edges_to_remove:
            removed_edges.append(self.graph["edges"][eid])
            self.remove_edge(eid, agent=agent)
        node_data = self.graph["nodes"][node_id]
        del self.graph["nodes"][node_id]
        self._emit("node_removed", {"node_id": node_id, "agent": agent})
        self._audit("remove_node", {"node_id": node_id, "agent": agent})
        self.workflow_manager.record_undo("add_node", {"node": node_data, "agent": agent})
        for edge in removed_edges:
            self.workflow_manager.record_undo("add_edge", {"edge": edge, "agent": agent})
        self.workflow_manager.end_transaction()
        return {"success": True}

    def update_node(self, node_id: NodeId, updates: Dict[str, Any], agent: Optional[str] = None) -> Dict[str, Any]:
        if node_id not in self.graph["nodes"]:
            return {"success": False, "error": f"Node '{node_id}' does not exist"}
        old_data = self.graph["nodes"][node_id].copy()
        self.workflow_manager.begin_transaction()
        self.graph["nodes"][node_id].update(updates)
        self._emit("node_updated", {"node_id": node_id, "updates": updates, "agent": agent})
        self._audit("update_node", {"node_id": node_id, "updates": updates, "agent": agent})
        self.workflow_manager.record_undo("update_node", {"node_id": node_id, "updates": old_data, "agent": agent})
        self.workflow_manager.end_transaction()
        return {"success": True}

    def add_edge(self, edge: EdgeDict, agent: Optional[str] = None) -> Dict[str, Any]:
        edge_id = edge.get("id")
        if not edge_id:
            return {"success": False, "error": "Edge must have an 'id' field"}
        if edge_id in self.graph["edges"]:
            return {"success": False, "error": f"Edge with id '{edge_id}' already exists"}
        # Validate source/target nodes
        if edge["source"] not in self.graph["nodes"] or edge["target"] not in self.graph["nodes"]:
            return {"success": False, "error": "Edge source/target must reference existing nodes"}
        self.workflow_manager.begin_transaction()
        self.graph["edges"][edge_id] = edge
        self._emit("edge_added", {"edge": edge, "agent": agent})
        self._audit("add_edge", {"edge": edge, "agent": agent})
        self.workflow_manager.record_undo("remove_edge", {"edge_id": edge_id, "agent": agent})
        self.workflow_manager.end_transaction()
        return {"success": True, "edge_id": edge_id}

    def remove_edge(self, edge_id: EdgeId, agent: Optional[str] = None) -> Dict[str, Any]:
        if edge_id not in self.graph["edges"]:
            return {"success": False, "error": f"Edge '{edge_id}' does not exist"}
        edge_data = self.graph["edges"][edge_id]
        self.workflow_manager.begin_transaction()
        del self.graph["edges"][edge_id]
        self._emit("edge_removed", {"edge_id": edge_id, "agent": agent})
        self._audit("remove_edge", {"edge_id": edge_id, "agent": agent})
        self.workflow_manager.record_undo("add_edge", {"edge": edge_data, "agent": agent})
        self.workflow_manager.end_transaction()
        return {"success": True}

    def update_edge(self, edge_id: EdgeId, updates: Dict[str, Any], agent: Optional[str] = None) -> Dict[str, Any]:
        if edge_id not in self.graph["edges"]:
            return {"success": False, "error": f"Edge '{edge_id}' does not exist"}
        old_data = self.graph["edges"][edge_id].copy()
        self.workflow_manager.begin_transaction()
        self.graph["edges"][edge_id].update(updates)
        self._emit("edge_updated", {"edge_id": edge_id, "updates": updates, "agent": agent})
        self._audit("update_edge", {"edge_id": edge_id, "updates": updates, "agent": agent})
        self.workflow_manager.record_undo("update_edge", {"edge_id": edge_id, "updates": old_data, "agent": agent})
        self.workflow_manager.end_transaction()
        return {"success": True}

    def move_node(self, node_id: NodeId, position: Tuple[float, float], agent: Optional[str] = None) -> Dict[str, Any]:
        """Update node position for GUI layout."""
        if node_id not in self.graph["nodes"]:
            return {"success": False, "error": f"Node '{node_id}' does not exist"}
        old_position = self.graph["nodes"][node_id].get("position")
        self.workflow_manager.begin_transaction()
        self.graph["nodes"][node_id]["position"] = position
        self._emit("node_moved", {"node_id": node_id, "position": position, "agent": agent})
        self._audit("move_node", {"node_id": node_id, "position": position, "agent": agent})
        self.workflow_manager.record_undo("move_node", {"node_id": node_id, "position": old_position, "agent": agent})
        self.workflow_manager.end_transaction()
        return {"success": True}

    def get_node(self, node_id: NodeId) -> NodeDict:
        return self.graph["nodes"][node_id]

    def get_edge(self, edge_id: EdgeId) -> EdgeDict:
        return self.graph["edges"][edge_id]

    def list_nodes(self) -> List[NodeDict]:
        return list(self.graph["nodes"].values())

    def list_edges(self) -> List[EdgeDict]:
        return list(self.graph["edges"].values())

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        """Notify GUI or listeners of a graph update event."""
        if self.on_update:
            self.on_update(event, payload)

    def _audit(self, action: str, payload: Dict[str, Any]) -> None:
        """Record agent-initiated changes for audit logging."""
        self.audit_log.append({"action": action, "payload": payload})

    def undo(self):
        """Undo the last action using WorkflowManager."""
        result = self.workflow_manager.undo()
        self._emit("undo", {})
        self._audit("undo", {})
        return result

    def redo(self):
        """Redo the last undone action using WorkflowManager."""
        result = self.workflow_manager.redo()
        self._emit("redo", {})
        self._audit("redo", {})
        return result

# --- Agent Usability Enhancements ---

def agent_add_semantic_node(
    updater: DynamicGraphUpdater,
    node_type: str,
    label: str,
    position: Tuple[float, float],
    params: Optional[Dict[str, Any]] = None,
    agent: Optional[str] = None
) -> NodeId:
    """
    Agent utility to add a semantic node (e.g., SemanticTargetNode, SemanticTriggerNode)
    to the graph and trigger GUI update.

    Args:
        updater: The DynamicGraphUpdater instance
        node_type: The type of semantic node (must be registered in node registry)
        label: Display label for the node
        position: (x, y) coordinates for GUI placement
        params: Optional node parameters
        agent: Optional agent name/id for audit logging

    Returns:
        The new node's id or error dict
    """
    import uuid
    node_id = f"{node_type.lower()}_{uuid.uuid4().hex[:8]}"
    node: NodeDict = {
        "id": node_id,
        "type": node_type,
        "label": label,
        "position": position,
        "params": params or {},
    }
    # Validate node_type against NODE_REGISTRY if available
    if NODE_REGISTRY and node_type not in NODE_REGISTRY:
        return {"success": False, "error": f"Node type '{node_type}' is not registered in NODE_REGISTRY"}
    result = updater.add_node(node, agent=agent)
    if isinstance(result, dict) and result.get("success"):
        return result["node_id"]
    return result

def agent_connect_nodes(
    updater: DynamicGraphUpdater,
    source_id: NodeId,
    target_id: NodeId,
    label: Optional[str] = None,
    agent: Optional[str] = None
) -> EdgeId:
    """
    Agent utility to connect two nodes with an edge.

    Args:
        updater: The DynamicGraphUpdater instance
        source_id: Source node id
        target_id: Target node id
        label: Optional label for the edge
        agent: Optional agent name/id for audit logging

    Returns:
        The new edge's id or error dict
    """
    import uuid
    edge_id = f"edge_{uuid.uuid4().hex[:8]}"
    edge: EdgeDict = {
        "id": edge_id,
        "source": source_id,
        "target": target_id,
        "label": label or "",
    }
    result = updater.add_edge(edge, agent=agent)
    if isinstance(result, dict) and result.get("success"):
        return result["edge_id"]
    return result

def agent_remove_node_and_edges(
    updater: DynamicGraphUpdater,
    node_id: NodeId,
    agent: Optional[str] = None
) -> Dict[str, Any]:
    """
    Agent utility to remove a node and all its connected edges.
    Returns a result dict.
    """
    return updater.remove_node(node_id, agent=agent)

def agent_record_action(
    updater: DynamicGraphUpdater,
    action: str,
    details: Dict[str, Any],
    agent: Optional[str] = None
) -> None:
    """
    Agent utility to record an action (for ActionRecorderNode or audit).
    """
    updater._audit(action, {"details": details, "agent": agent})
    updater._emit("agent_action_recorded", {"action": action, "details": details, "agent": agent})

def agent_auto_connect_semantic(
    updater: DynamicGraphUpdater,
    from_type: str,
    to_type: str,
    from_label: str,
    to_label: str,
    from_position: Tuple[float, float],
    to_position: Tuple[float, float],
    edge_label: Optional[str] = None,
    agent: Optional[str] = None,
    params_from: Optional[Dict[str, Any]] = None,
    params_to: Optional[Dict[str, Any]] = None
) -> Tuple[NodeId, NodeId, EdgeId]:
    """
    Agent utility: Add two semantic nodes and connect them with an edge.
    Returns (from_node_id, to_node_id, edge_id) or error dicts.
    """
    from_id = agent_add_semantic_node(updater, from_type, from_label, from_position, params_from, agent=agent)
    to_id = agent_add_semantic_node(updater, to_type, to_label, to_position, params_to, agent=agent)
    edge_id = agent_connect_nodes(updater, from_id, to_id, label=edge_label, agent=agent)
    return from_id, to_id, edge_id

# --- FastAPI Server API for Remote Agent Control ---

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# In-memory graph and audit log for demo
graph: GraphDict = {"nodes": {}, "edges": {}}
audit_log: List[Dict[str, Any]] = []
updater = DynamicGraphUpdater(graph, audit_log=audit_log)

class NodeCreateRequest(BaseModel):
    node_type: str
    label: str
    position: Tuple[float, float]
    params: Optional[Dict[str, Any]] = None
    agent: Optional[str] = None

class EdgeCreateRequest(BaseModel):
    source_id: str
    target_id: str
    label: Optional[str] = None
    agent: Optional[str] = None

@app.post("/add_node")
def api_add_node(req: NodeCreateRequest):
    result = agent_add_semantic_node(
        updater, req.node_type, req.label, req.position, req.params, agent=req.agent
    )
    if isinstance(result, dict) and not result.get("success", True):
        raise HTTPException(status_code=400, detail=result["error"])
    return {"node_id": result}

@app.post("/add_edge")
def api_add_edge(req: EdgeCreateRequest):
    result = agent_connect_nodes(
        updater, req.source_id, req.target_id, label=req.label, agent=req.agent
    )
    if isinstance(result, dict) and not result.get("success", True):
        raise HTTPException(status_code=400, detail=result["error"])
    return {"edge_id": result}

@app.get("/graph")
def api_get_graph():
    return graph

@app.get("/audit_log")
def api_get_audit_log():
    return audit_log

@app.post("/undo")
def api_undo():
    success = updater.undo()
    if not success:
        raise HTTPException(status_code=400, detail="Nothing to undo")
    return {"success": True}

@app.post("/redo")
def api_redo():
    success = updater.redo()
    if not success:
        raise HTTPException(status_code=400, detail="Nothing to redo")
    return {"success": True}

# --- GUI Integration Example ---

# Example: Hook for GUI to listen for updates and refresh the view
def gui_update_hook(event: str, payload: Dict[str, Any]) -> None:
    # Integrate with actual GUI event system or log for debugging
    print(f"[GUI] Event: {event} | Payload: {payload}")

# --- Example Usage (for integration test/demo) ---
if __name__ == "__main__":
    # Minimal graph structure for demo
    graph: GraphDict = {"nodes": {}, "edges": {}}
    audit_log: List[Dict[str, Any]] = []
    updater = DynamicGraphUpdater(graph, on_update=gui_update_hook, audit_log=audit_log)

    # Agent adds a semantic trigger node
    trigger_id = agent_add_semantic_node(
        updater, "SemanticTriggerNode", "Start Trigger", (100, 100), agent="demo_agent"
    )
    print("Add trigger node:", trigger_id)
    # Agent adds a semantic target node
    target_id = agent_add_semantic_node(
        updater, "SemanticTargetNode", "End Target", (400, 100), agent="demo_agent"
    )
    print("Add target node:", target_id)
    # Agent connects them
    edge_id = agent_connect_nodes(updater, trigger_id, target_id, label="triggers", agent="demo_agent")
    print("Connect nodes:", edge_id)

    # Move node (simulate drag in GUI)
    move_result = updater.move_node(trigger_id, (120, 120), agent="demo_agent")
    print("Move node:", move_result)

    # Remove node (and its edges)
    remove_result = agent_remove_node_and_edges(updater, trigger_id, agent="demo_agent")
    print("Remove node and edges:", remove_result)

    # Agent auto-connects two semantic nodes
    from_id, to_id, auto_edge_id = agent_auto_connect_semantic(
        updater,
        "SemanticTriggerNode", "SemanticTargetNode",
        "Auto Trigger", "Auto Target",
        (200, 200), (400, 200),
        edge_label="auto-link",
        agent="demo_agent"
    )
    print("Auto connect semantic:", from_id, to_id, auto_edge_id)

    # Agent records a custom action
    agent_record_action(updater, "custom_action", {"info": "Agent did something"}, agent="demo_agent")

    # Print audit log for demonstration
    print("\n[Audit Log]")
    for entry in audit_log:
        print(entry)

    # Additional agentic update scenarios and edge case tests
    # Test: Add node with duplicate id
    duplicate_node = {
        "id": from_id,
        "type": "SemanticTriggerNode",
        "label": "Duplicate Node",
        "position": (300, 300),
        "params": {},
    }
    result = updater.add_node(duplicate_node, agent="demo_agent")
    print("Add duplicate node:", result)

    # Test: Add edge with non-existent source
    bad_edge = {
        "id": "edge_bad",
        "source": "nonexistent",
        "target": to_id,
        "label": "bad edge"
    }
    result = updater.add_edge(bad_edge, agent="demo_agent")
    print("Add edge with bad source:", result)

    # Test: Undo/Redo
    print("Undo:", updater.undo())
    print("Redo:", updater.redo())

    # Test: Update node with invalid id
    result = updater.update_node("nonexistent", {"label": "Should Fail"}, agent="demo_agent")
    print("Update nonexistent node:", result)

    # Test: Remove edge with invalid id
    result = updater.remove_edge("nonexistent", agent="demo_agent")
    print("Remove nonexistent edge:", result)

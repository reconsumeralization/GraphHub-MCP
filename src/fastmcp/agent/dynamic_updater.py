"""
Dynamic Graph Updates for Agent Usability in FastMCP Builder

This module provides robust, agent-friendly utilities for real-time, programmatic updates to the workflow graph,
enabling seamless agentic process automation, live GUI interactivity, and advanced agent usability features.

Key Features:
- Add, remove, and update nodes and edges programmatically (with validation and event hooks)
- Support for agent-triggered graph modifications (planner, recorder, semantic nodes, etc.)
- Event hooks for GUI to reflect live changes and agent actions
- Input validation, error handling, and audit logging for robust agent integration
- Agent usability helpers: semantic node creation, auto-connection, action recording, and undo/redo stubs

TODO:
- Integrate with WorkflowManager for atomic updates, undo/redo, and transactionality
- Add audit logging for all agent-initiated changes (see MasterPlan Sprint 1)
- Expose update API for FastAPI server (for remote agent control)
- Add unit and integration tests (see MasterPlan Sprint 4/5)
- Enhance agent feedback: return error/success to agent, not just raise
- Support for action recording and replay (see ActionRecorderNode)
"""

from typing import Dict, Any, Optional, List, Callable, Tuple, Union
import logging # Import logging
# Update import paths to reflect new structure
# Remove imports for types not found in builder.types
# from fastmcp.builder.types import NodeId, EdgeId, GraphDict, NodeDict, EdgeDict

# Import NodeRegistry from its new correct location
try:
    from fastmcp.builder.nodes.node_registry import NodeRegistry
    # Keep NODE_REGISTRY reference if used directly, or remove if only methods are used
    # NODE_REGISTRY = NodeRegistry._registry # Accessing protected member might be discouraged
except ImportError:
    # NODE_REGISTRY = {} # No need to redefine if only methods are used
    pass # Silently ignore if registry isn't found, checks later will handle it

# Event callback type for GUI update hooks
GraphUpdateCallback = Callable[[str, Dict[str, Any]], None]

# Use generic types for now where specific types were removed
NodeId = str
EdgeId = str
NodeDict = Dict[str, Any]
EdgeDict = Dict[str, Any]
GraphDict = Dict[str, Any] # Or be more specific: Dict[str, Dict[NodeId, NodeDict] | Dict[EdgeId, EdgeDict]]

class DynamicGraphUpdater:
    """
    Provides agent- and GUI-facing methods for dynamic graph mutation.
    All changes are validated, can trigger GUI update events, and are agent-usable.
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

    def add_node(self, node: NodeDict, agent: Optional[str] = None) -> NodeId:
        node_id = node.get("id")
        if not node_id:
            raise ValueError("Node must have an 'id' field")
        if node_id in self.graph["nodes"]:
            raise ValueError(f"Node with id '{node_id}' already exists")
        # Validate node type if registry is available
        node_type = node.get("type")
        if not isinstance(node_type, str):
            # Handle case where type is missing or not a string
            node_type = "UnknownNodeType" # Or raise an error
            logging.warning(f"Node {node_id} has missing or invalid type, using '{node_type}'.")

        try:
            # Check using the registry method, ensuring node_type is a string
            if NodeRegistry.get(str(node_type)) is None:
                raise ValueError(f"Node type '{node_type}' is not registered in NodeRegistry")
        except NameError: # Handle case where NodeRegistry itself couldn't be imported
            logging.warning("NodeRegistry not available for type validation.")
            pass # Allow adding node if registry unavailable

        self.graph["nodes"][node_id] = node
        self._emit("node_added", {"node": node, "agent": agent})
        self._audit("add_node", {"node": node, "agent": agent})
        return node_id

    def remove_node(self, node_id: NodeId, agent: Optional[str] = None) -> None:
        if node_id not in self.graph["nodes"]:
            raise KeyError(f"Node '{node_id}' does not exist")
        # Remove all edges connected to this node
        edges_to_remove = [
            eid for eid, edge in self.graph["edges"].items()
            if edge["source"] == node_id or edge["target"] == node_id
        ]
        for eid in edges_to_remove:
            self.remove_edge(eid, agent=agent)
        del self.graph["nodes"][node_id]
        self._emit("node_removed", {"node_id": node_id, "agent": agent})
        self._audit("remove_node", {"node_id": node_id, "agent": agent})

    def update_node(self, node_id: NodeId, updates: Dict[str, Any], agent: Optional[str] = None) -> None:
        if node_id not in self.graph["nodes"]:
            raise KeyError(f"Node '{node_id}' does not exist")
        self.graph["nodes"][node_id].update(updates)
        self._emit("node_updated", {"node_id": node_id, "updates": updates, "agent": agent})
        self._audit("update_node", {"node_id": node_id, "updates": updates, "agent": agent})

    def add_edge(self, edge: EdgeDict, agent: Optional[str] = None) -> EdgeId:
        edge_id = edge.get("id")
        if not edge_id:
            raise ValueError("Edge must have an 'id' field")
        if edge_id in self.graph["edges"]:
            raise ValueError(f"Edge with id '{edge_id}' already exists")
        # Validate source/target nodes
        if edge["source"] not in self.graph["nodes"] or edge["target"] not in self.graph["nodes"]:
            raise ValueError("Edge source/target must reference existing nodes")
        self.graph["edges"][edge_id] = edge
        self._emit("edge_added", {"edge": edge, "agent": agent})
        self._audit("add_edge", {"edge": edge, "agent": agent})
        return edge_id

    def remove_edge(self, edge_id: EdgeId, agent: Optional[str] = None) -> None:
        if edge_id not in self.graph["edges"]:
            raise KeyError(f"Edge '{edge_id}' does not exist")
        del self.graph["edges"][edge_id]
        self._emit("edge_removed", {"edge_id": edge_id, "agent": agent})
        self._audit("remove_edge", {"edge_id": edge_id, "agent": agent})

    def update_edge(self, edge_id: EdgeId, updates: Dict[str, Any], agent: Optional[str] = None) -> None:
        if edge_id not in self.graph["edges"]:
            raise KeyError(f"Edge '{edge_id}' does not exist")
        self.graph["edges"][edge_id].update(updates)
        self._emit("edge_updated", {"edge_id": edge_id, "updates": updates, "agent": agent})
        self._audit("update_edge", {"edge_id": edge_id, "updates": updates, "agent": agent})

    def move_node(self, node_id: NodeId, position: Tuple[float, float], agent: Optional[str] = None) -> None:
        """Update node position for GUI layout."""
        if node_id not in self.graph["nodes"]:
            raise KeyError(f"Node '{node_id}' does not exist")
        self.graph["nodes"][node_id]["position"] = position
        self._emit("node_moved", {"node_id": node_id, "position": position, "agent": agent})
        self._audit("move_node", {"node_id": node_id, "position": position, "agent": agent})

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

    # --- Agent Usability: Undo/Redo Stubs (to be implemented in WorkflowManager) ---
    def undo(self):
        # Placeholder for undo logic (to be integrated with WorkflowManager)
        self._emit("undo", {})
        self._audit("undo", {})

    def redo(self):
        # Placeholder for redo logic (to be integrated with WorkflowManager)
        self._emit("redo", {})
        self._audit("redo", {})

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
        The new node's id
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
    # Validate node_type against NodeRegistry if available
    try:
        # Ensure node_type is string before check
        if NodeRegistry.get(str(node_type)) is None:
            raise ValueError(f"Node type '{node_type}' is not registered in NodeRegistry")
    except NameError:
        logging.warning("NodeRegistry not available for type validation.")
        pass # Allow adding node if registry unavailable
    return updater.add_node(node, agent=agent)

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
        The new edge's id
    """
    import uuid
    edge_id = f"edge_{uuid.uuid4().hex[:8]}"
    edge: EdgeDict = {
        "id": edge_id,
        "source": source_id,
        "target": target_id,
        "label": label or "",
    }
    return updater.add_edge(edge, agent=agent)

def agent_remove_node_and_edges(
    updater: DynamicGraphUpdater,
    node_id: NodeId,
    agent: Optional[str] = None
) -> None:
    """
    Agent utility to remove a node and all its connected edges.
    """
    updater.remove_node(node_id, agent=agent)

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
    Returns (from_node_id, to_node_id, edge_id).
    """
    from_id = agent_add_semantic_node(updater, from_type, from_label, from_position, params_from, agent=agent)
    to_id = agent_add_semantic_node(updater, to_type, to_label, to_position, params_to, agent=agent)
    edge_id = agent_connect_nodes(updater, from_id, to_id, label=edge_label, agent=agent)
    return from_id, to_id, edge_id

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
    # Agent adds a semantic target node
    target_id = agent_add_semantic_node(
        updater, "SemanticTargetNode", "End Target", (400, 100), agent="demo_agent"
    )
    # Agent connects them
    edge_id = agent_connect_nodes(updater, trigger_id, target_id, label="triggers", agent="demo_agent")

    # Move node (simulate drag in GUI)
    updater.move_node(trigger_id, (120, 120), agent="demo_agent")

    # Remove node (and its edges)
    agent_remove_node_and_edges(updater, trigger_id, agent="demo_agent")

    # Agent auto-connects two semantic nodes
    from_id, to_id, auto_edge_id = agent_auto_connect_semantic(
        updater,
        "SemanticTriggerNode", "SemanticTargetNode",
        "Auto Trigger", "Auto Target",
        (200, 200), (400, 200),
        edge_label="auto-link",
        agent="demo_agent"
    )

    # Agent records a custom action
    agent_record_action(updater, "custom_action", {"info": "Agent did something"}, agent="demo_agent")

    # Print audit log for demonstration
    print("\n[Audit Log]")
    for entry in audit_log:
        print(entry)

    # TODO: Add more agentic update scenarios and edge case tests

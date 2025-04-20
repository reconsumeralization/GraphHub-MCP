"""
Graph utility functions for FastMCP Builder modules.
Agent- and optimization-aware extensions for next-gen builder/agent synergy.

NOTE: Core data contracts (TypedDicts, Protocols, type aliases) are defined in fastmcp/builder/types.py.
This module focuses on graph logic, validation, and agent/builder hooks.

IMPORTANT:
- All programmatic, non-GUI API for graph creation, validation, and mutation should be consolidated in `workflow_manager.py`.
- Node registration logic should be centralized (see nodes/node_registry.py or nodes/__init__.py).
- GUI logic should be separated (see gui_launcher.py for app lifecycle, graph_builder.py for NodeGraph widget).
- Agent orchestration logic should reside in `fastmcp/agent/` (not here).
"""

from typing import Any, List, Tuple, Dict, Set
import logging

from fastmcp.builder.types import (
    GraphSpec,
    NodeSpec,
    ConnectionSpec,
    OptimizationHint,
    OptimizationHints,
    SuggestionResult,
    PatchSpec,
)
from fastmcp.builder.exceptions import (
    GraphValidationError,
    NodeReferenceError,
    CycleDetectedError,
    TypeMismatchError,
    PatchApplicationError,
    SubflowExtractionError,
)

logger = logging.getLogger(__name__)

def is_valid_json(s: str) -> bool:
    """Check if a string is valid JSON."""
    try:
        import json
        json.loads(s)
        return True
    except Exception:
        return False

def validate_graph_spec(spec: GraphSpec) -> Tuple[bool, List[str]]:
    """
    Validate the given graph specification against basic schema rules and semantics.

    Returns (is_valid, errors).
    """
    errors: List[str] = []

    # Node ID uniqueness
    node_ids: List[str] = []
    for node in spec["nodes"]:
        node_id = getattr(node, "id", None)
        if node_id is None:
            errors.append("Node missing 'id' field.")
        else:
            node_ids.append(node_id)
    if len(set(node_ids)) != len(node_ids):
        errors.append("Duplicate node IDs found.")

    # Connection references
    for conn in spec["connections"]:
        from_node_id = getattr(conn, "from_node_id", None)
        to_node_id = getattr(conn, "to_node_id", None)
        if from_node_id is None:
            errors.append("Connection missing 'from_node_id' field.")
        elif from_node_id not in node_ids:
            errors.append(f"Connection from unknown node {from_node_id}.")
        if to_node_id is None:
            errors.append("Connection missing 'to_node_id' field.")
        elif to_node_id not in node_ids:
            errors.append(f"Connection to unknown node {to_node_id}.")

    # Semantic validation: type mismatches, cycles, dead ends
    # 1. Type mismatches (if ports/types are defined)
    for conn in spec["connections"]:
        from_node_id = getattr(conn, "from_node_id", None)
        to_node_id = getattr(conn, "to_node_id", None)
        from_port = getattr(conn, "from_port", None)
        to_port = getattr(conn, "to_port", None)
        if from_node_id in node_ids and to_node_id in node_ids:
            from_node = next((n for n in spec["nodes"] if getattr(n, "id", None) == from_node_id), None)
            to_node = next((n for n in spec["nodes"] if getattr(n, "id", None) == to_node_id), None)
            if from_node and to_node:
                from_ports = getattr(from_node, "outputs", {})
                to_ports = getattr(to_node, "inputs", {})
                if isinstance(from_ports, dict) and isinstance(to_ports, dict):
                    from_type = from_ports.get(from_port, None)
                    to_type = to_ports.get(to_port, None)
                    if from_type and to_type and from_type != to_type:
                        errors.append(
                            f"Type mismatch: {from_node_id}.{from_port} ({from_type}) -> {to_node_id}.{to_port} ({to_type})"
                        )

    # 2. Cycle detection (DFS)
    adjacency: Dict[str, List[str]] = {nid: [] for nid in node_ids}
    for conn in spec["connections"]:
        from_node_id = getattr(conn, "from_node_id", None)
        to_node_id = getattr(conn, "to_node_id", None)
        if from_node_id in node_ids and to_node_id in node_ids:
            adjacency[from_node_id].append(to_node_id)

    def has_cycle() -> bool:
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for n in node_ids:
            if n not in visited:
                if dfs(n):
                    return True
        return False

    if has_cycle():
        errors.append("Cycle detected in the graph.")

    # 3. Dead ends (nodes with no outgoing or incoming connections)
    connected_nodes: Set[str] = set()
    for conn in spec["connections"]:
        from_node_id = getattr(conn, "from_node_id", None)
        to_node_id = getattr(conn, "to_node_id", None)
        if from_node_id:
            connected_nodes.add(from_node_id)
        if to_node_id:
            connected_nodes.add(to_node_id)
    for nid in node_ids:
        if nid not in connected_nodes:
            errors.append(f"Node '{nid}' is isolated (no connections).")

    return (len(errors) == 0, errors)

def analyze_performance(spec: GraphSpec) -> OptimizationHints:
    """
    Analyze the graph spec and return optimization hints for agent/builder UX.

    Example hints:
      - Flatten redundant branches
      - Batch similar API calls
      - Suggest caching repeated sub-workflows
      - Detect cycles and dead ends
    """
    hints: List[OptimizationHint] = []

    # 1. Detect redundant branches (naive: identical subgraphs with same parent)
    seen_signatures: Dict[str, List[str]] = {}
    for node in spec["nodes"]:
        node_type = getattr(node, "type", None)
        node_properties = getattr(node, "properties", {})
        if node_type is None or not isinstance(node_properties, dict):
            continue
        sig = f"{node_type}|{sorted(node_properties.items())}"
        node_id = getattr(node, "id", None)
        if node_id is None:
            continue
        if sig in seen_signatures:
            seen_signatures[sig].append(node_id)
        else:
            seen_signatures[sig] = [node_id]
    for sig, ids in seen_signatures.items():
        if len(ids) > 1:
            hints.append(
                {
                    "message": "Redundant nodes detected (identical type/properties).",
                    "affected_nodes": ids,
                    "suggestion": "Consider merging or reusing these nodes.",
                }
            )

    # 2. Batch similar API calls (consecutive API nodes with same endpoint)
    api_nodes = []
    for n in spec["nodes"]:
        n_type = getattr(n, "type", "")
        if isinstance(n_type, str) and n_type.lower().startswith("api"):
            api_nodes.append(n)
    endpoint_map: Dict[str, List[str]] = {}
    for n in api_nodes:
        n_properties = getattr(n, "properties", {})
        endpoint = None
        if isinstance(n_properties, dict):
            endpoint = n_properties.get("endpoint", None)
        n_id = getattr(n, "id", None)
        if endpoint and n_id:
            endpoint_map.setdefault(endpoint, []).append(n_id)
    adjacency: Dict[str, List[str]] = {}
    for conn in spec["connections"]:
        from_node_id = getattr(conn, "from_node_id", None)
        to_node_id = getattr(conn, "to_node_id", None)
        if from_node_id and to_node_id:
            adjacency.setdefault(from_node_id, []).append(to_node_id)
    for endpoint, ids in endpoint_map.items():
        for node_id in ids:
            neighbors = adjacency.get(node_id, [])
            for neighbor in neighbors:
                if neighbor in ids:
                    hints.append(
                        {
                            "message": f"Consecutive API calls to '{endpoint}' detected.",
                            "affected_nodes": [node_id, neighbor],
                            "suggestion": "Batch these API calls if possible to reduce latency.",
                        }
                    )

    # 3. Suggest caching for repeated sub-workflows (naive: same node type in loop)
    type_count: Dict[str, int] = {}
    for n in spec["nodes"]:
        t = getattr(n, "type", None)
        if not isinstance(t, str):
            continue
        type_count[t] = type_count.get(t, 0) + 1
    for t, count in type_count.items():
        if count > 2:
            ids = []
            for n in spec["nodes"]:
                n_type = getattr(n, "type", None)
                n_id = getattr(n, "id", None)
                if n_type == t and n_id is not None:
                    ids.append(n_id)
            hints.append(
                {
                    "message": f"Node type '{t}' appears {count} times.",
                    "affected_nodes": ids,
                    "suggestion": "Consider caching results or refactoring repeated logic.",
                }
            )

    # 4. Detect cycles and dead ends (advanced analysis)
    node_ids = [getattr(n, "id", None) for n in spec["nodes"] if getattr(n, "id", None) is not None]
    adjacency = {nid: [] for nid in node_ids}
    for conn in spec["connections"]:
        from_node_id = getattr(conn, "from_node_id", None)
        to_node_id = getattr(conn, "to_node_id", None)
        if from_node_id in node_ids and to_node_id in node_ids:
            adjacency[from_node_id].append(to_node_id)

    def find_cycles() -> List[List[str]]:
        visited: Set[str] = set()
        cycles: List[List[str]] = []

        def dfs(node: str, path: List[str]):
            visited.add(node)
            path.append(node)
            for neighbor in adjacency.get(node, []):
                if neighbor in path:
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
                elif neighbor not in visited:
                    dfs(neighbor, path.copy())
            path.pop()

        for n in node_ids:
            if n not in visited:
                dfs(n, [])
        return cycles

    cycles = find_cycles()
    for cycle in cycles:
        hints.append(
            {
                "message": "Cycle detected in the graph.",
                "affected_nodes": cycle,
                "suggestion": "Refactor to remove cycles or add explicit loop handling.",
            }
        )

    # Dead ends (nodes with no outgoing or incoming connections)
    connected_nodes: Set[str] = set()
    for conn in spec["connections"]:
        from_node_id = getattr(conn, "from_node_id", None)
        to_node_id = getattr(conn, "to_node_id", None)
        if from_node_id:
            connected_nodes.add(from_node_id)
        if to_node_id:
            connected_nodes.add(to_node_id)
    for nid in node_ids:
        if nid not in connected_nodes:
            hints.append(
                {
                    "message": f"Node '{nid}' is isolated (no connections).",
                    "affected_nodes": [nid],
                    "suggestion": "Connect this node or remove it if unnecessary.",
                }
            )

    return {"hints": hints}

def build_mcp_server_graph(spec: GraphSpec) -> Any:
    """
    Build and return a server-side graph object based on the specification.

    This uses the NodeGraphQt API under the hood.
    """
    try:
        from NodeGraphQt import NodeGraph
    except ImportError as e:
        logger.error("NodeGraphQt import failed, cannot build server graph: %s", e)
        raise

    graph = NodeGraph()
    id_to_node: Dict[str, Any] = {}

    for node in spec["nodes"]:
        node_type = getattr(node, "type", None)
        node_name = getattr(node, "name", None)
        node_properties = getattr(node, "properties", {})
        node_id = getattr(node, "id", None)
        if node_type is None or node_name is None or node_id is None:
            logger.warning(f"Node missing required fields: {node}")
            continue
        new_node = graph.create_node(node_type)
        new_node.set_name(node_name)
        if isinstance(node_properties, dict):
            for prop, val in node_properties.items():
                try:
                    new_node.set_property(prop, val)
                except Exception:
                    logger.warning(f"Failed to set property {prop} on node {node_id}")
        id_to_node[node_id] = new_node

    for conn in spec["connections"]:
        from_node_id = getattr(conn, "from_node_id", None)
        to_node_id = getattr(conn, "to_node_id", None)
        from_port = getattr(conn, "from_port", None)
        to_port = getattr(conn, "to_port", None)
        if (
            from_node_id is None
            or to_node_id is None
            or from_port is None
            or to_port is None
        ):
            logger.warning(f"Connection missing required fields: {conn}")
            continue
        out_node = id_to_node.get(from_node_id)
        in_node = id_to_node.get(to_node_id)
        if out_node is None or in_node is None:
            logger.warning(f"Connection references unknown node(s): {conn}")
            continue
        graph.connect_ports(out_node, from_port, in_node, to_port)

    return graph

def recommend_next_node(partial_spec: GraphSpec) -> SuggestionResult:
    """
    Agent tool: Given a partial GraphSpec, suggest the next node type and default properties.

    Returns a SuggestionResult dict with 'type', 'properties', and 'rationale'.
    """
    # TODO: Replace with LLM-powered suggestion logic (Open issue: #llm-suggest-001)
    # For now, use simple heuristics
    last_node = None
    if partial_spec["nodes"]:
        last_node = partial_spec["nodes"][-1]
    if not last_node:
        return {
            "type": "StartNode",
            "properties": {},
            "rationale": "Every workflow should start with a StartNode.",
        }
    last_type = getattr(last_node, "type", "")
    if isinstance(last_type, str) and last_type.lower().startswith("api"):
        return {
            "type": "ErrorHandlerNode",
            "properties": {"on_error": "log_and_continue"},
            "rationale": "After an API call, it's common to handle errors.",
        }
    # Fallback: generic processing node
    return {
        "type": "ProcessNode",
        "properties": {},
        "rationale": "No specific suggestion; consider adding a processing step.",
    }

def agent_feedback_hook(graph_spec: GraphSpec, feedback: str) -> None:
    """
    Hook for agent feedback integration.
    Logs feedback and can trigger further analysis or UI updates.
    """
    logger.info(f"Agent feedback received: {feedback}")
    # TODO: Integrate with feedback storage or trigger downstream actions

def extract_subflow(graph_spec: GraphSpec, node_ids: List[str]) -> GraphSpec:
    """
    Extracts a subflow from the graph given a list of node IDs.
    Returns a new GraphSpec containing only the specified nodes and their connections.
    Raises SubflowExtractionError if boundaries are ambiguous.
    """
    node_id_set = set(node_ids)
    sub_nodes = [n for n in graph_spec["nodes"] if getattr(n, "id", None) in node_id_set]
    sub_connections = [
        c for c in graph_spec["connections"]
        if getattr(c, "from_node_id", None) in node_id_set and getattr(c, "to_node_id", None) in node_id_set
    ]
    if not sub_nodes:
        raise SubflowExtractionError("No nodes found for subflow extraction.")
    # Detect IO boundaries: inputs are connections from outside, outputs are connections to outside
    input_edges = [
        c for c in graph_spec["connections"]
        if getattr(c, "to_node_id", None) in node_id_set and getattr(c, "from_node_id", None) not in node_id_set
    ]
    output_edges = [
        c for c in graph_spec["connections"]
        if getattr(c, "from_node_id", None) in node_id_set and getattr(c, "to_node_id", None) not in node_id_set
    ]
    # Attach boundary info to subflow spec
    subflow_spec: GraphSpec = {
        "nodes": sub_nodes,
        "connections": sub_connections,
        "inputs": input_edges,
        "outputs": output_edges,
    }
    return subflow_spec

def apply_patch(graph_spec: GraphSpec, patch: PatchSpec) -> GraphSpec:
    """
    Applies a PatchSpec (JSON Patch or custom diff) to a GraphSpec.
    Returns the updated GraphSpec.
    Raises PatchApplicationError on failure.
    """
    import copy
    updated_spec = copy.deepcopy(graph_spec)
    try:
        # Assume PatchSpec is a list of dicts with 'op', 'path', 'value'
        for op in patch:
            operation = op.get("op")
            path = op.get("path")
            value = op.get("value", None)
            # Only supporting 'add', 'remove', 'replace' for now
            if operation == "add":
                # Example: path="/nodes/-" to append node
                if path == "/nodes/-":
                    updated_spec["nodes"].append(value)
                elif path == "/connections/-":
                    updated_spec["connections"].append(value)
                else:
                    raise PatchApplicationError(f"Unsupported add path: {path}")
            elif operation == "remove":
                # Example: path="/nodes/2"
                parts = path.strip("/").split("/")
                if parts[0] == "nodes":
                    idx = int(parts[1])
                    del updated_spec["nodes"][idx]
                elif parts[0] == "connections":
                    idx = int(parts[1])
                    del updated_spec["connections"][idx]
                else:
                    raise PatchApplicationError(f"Unsupported remove path: {path}")
            elif operation == "replace":
                parts = path.strip("/").split("/")
                if parts[0] == "nodes":
                    idx = int(parts[1])
                    updated_spec["nodes"][idx] = value
                elif parts[0] == "connections":
                    idx = int(parts[1])
                    updated_spec["connections"][idx] = value
                else:
                    raise PatchApplicationError(f"Unsupported replace path: {path}")
            else:
                raise PatchApplicationError(f"Unsupported operation: {operation}")
        return updated_spec
    except Exception as e:
        raise PatchApplicationError(f"Failed to apply patch: {e}") from e

# NOTE:
# - All programmatic, non-GUI API for graph creation, validation, and mutation should be consolidated in workflow_manager.py.
# - Node registration logic should be centralized (see nodes/node_registry.py or nodes/__init__.py).
# - GUI logic should be separated (see gui_launcher.py for app lifecycle, graph_builder.py for NodeGraph widget).
# - Agent orchestration logic should reside in fastmcp/agent/ (not here).
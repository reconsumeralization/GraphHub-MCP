"""
yamle_graph.py

YAML-based graph definition, validation, and conversion utilities for FastMCP.

This module provides strictly-typed, robust, and extensible functions to load, validate, and convert YAML-based
graph definitions into in-memory graph structures for the FastMCP builder and execution engine, with a focus on
agent usability, agent node support, and GUI visibility.

--------------------------------------------------------------------------------
Project Structure Reference (see README and C_MasterPlan.md for details):

src/
└── fastmcp/
    ├── __init__.py
    ├── builder/
    │   ├── __init__.py
    │   ├── constants.py
    │   ├── nodes/
    │   │   ├── __init__.py
    │   │   ├── base_node.py
    │   │   ├── control_nodes.py
    │   │   ├── task_nodes.py
    │   │   └── ai_nodes.py
    │   ├── graph_utils.py
    │   ├── gui_launcher.py
    │   ├── workflow_manager.py
    │   ├── mcp_tools.py
    │   ├── protocols.py
    │   ├── types.py
    │   ├── graph_cli.py
    │   ├── graph_logger.py
    │   └── __main__.py
    ├── execution_engine/
    │   ├── __init__.py
    │   ├── executor.py
    │   └── ...
    └── agent/
        ├── __init__.py
        ├── agent_core.py
        ├── tools.py
        └── main.py
        └── ...
--------------------------------------------------------------------------------

Author: Zeta Nova (for my wife)
"""

from typing import Dict, List, Tuple, Set, TypedDict, Any, Optional, cast, Literal, Union
import logging
import datetime

# --- Try to import yaml and pydantic, but provide clear error if missing ---
try:
    import yaml
except ImportError as e:
    raise ImportError("Missing required dependency 'pyyaml'. Please install it with 'pip install pyyaml'.") from e

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError as e:
    raise ImportError("Missing required dependency 'pydantic'. Please install it with 'pip install pydantic'.") from e

# --- Logging Setup ---
import importlib
def _get_logger():
    try:
        graph_logger = importlib.import_module("fastmcp.builder.graph_logger")
        logger = getattr(graph_logger, "logger", None)
        if logger is not None:
            return logger
    except Exception:
        pass
    logger = logging.getLogger("yamle_graph")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = _get_logger()

# --- Type Definitions (Expanded with NodeSpec/ConnectionSpec) ---

class NodeSpec(BaseModel):
    id: str
    type: Optional[str] = None  # e.g. "agent", "task", "control", etc.
    label: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    agent_config: Optional[Dict[str, Any]] = None  # For agent nodes, agent-specific config
    gui: Optional[Dict[str, Any]] = None  # GUI hints (position, color, icon, etc.)
    metadata: Optional[Dict[str, Any]] = None
    # --- AGENT USABILITY: Add agent usability fields for GUI ---
    description: Optional[str] = None
    status: Optional[str] = None  # e.g. "ready", "error", "running"
    last_run: Optional[str] = None  # ISO timestamp
    # --- ENHANCED USABILITY FIELDS ---
    created_at: Optional[str] = None  # ISO timestamp
    updated_at: Optional[str] = None  # ISO timestamp
    run_count: Optional[int] = None
    error_message: Optional[str] = None
    # TODO: Add more agent usability fields as needed for GUI

class ConnectionSpec(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    label: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    type: Optional[str] = None  # e.g. "data", "control", "agent-comm"
    gui: Optional[Dict[str, Any]] = None  # GUI hints (color, style, etc.)
    metadata: Optional[Dict[str, Any]] = None

class GraphMetadata(BaseModel):
    version: Optional[str] = None
    gui: Optional[Dict[str, Any]] = None
    # TODO: Add more metadata fields for GUI/agent usability

class GraphModel(BaseModel):
    nodes: List[NodeSpec]
    edges: List[ConnectionSpec]
    metadata: Optional[Dict[str, Any]] = None

# For legacy compatibility with TypedDicts
class NodeDict(TypedDict, total=False):
    id: str
    type: Optional[str]
    label: Optional[str]
    attributes: Optional[Dict[str, Any]]
    agent_config: Optional[Dict[str, Any]]
    gui: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    # AGENT USABILITY FIELDS
    description: Optional[str]
    status: Optional[str]
    last_run: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    run_count: Optional[int]
    error_message: Optional[str]

class EdgeDict(TypedDict, total=False):
    from_: str
    to: str
    label: Optional[str]
    attributes: Optional[Dict[str, Any]]
    type: Optional[str]
    gui: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]

class GraphDict(TypedDict, total=True):
    nodes: List[NodeDict]
    edges: List[EdgeDict]
    metadata: Optional[Dict[str, Any]]

# --- Exception ---

class YamleGraphParseError(Exception):
    """Custom exception for YAML graph parsing errors."""
    pass

# --- YAML Graph Loader with Schema Validation ---

def _now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def load_yaml_graph(yaml_str: str) -> GraphDict:
    """
    Load a YAML string representing a graph definition.

    Args:
        yaml_str (str): The YAML string to parse.

    Returns:
        GraphDict: Parsed graph definition.

    Raises:
        YamleGraphParseError: If the YAML is invalid or missing required fields.
    """
    try:
        data = yaml.safe_load(yaml_str)
    except Exception as e:
        logger.error(f"Failed to parse YAML: {e}")
        raise YamleGraphParseError(f"Failed to parse YAML: {e}")

    if not isinstance(data, dict):
        logger.error("YAML root must be a mapping (dictionary).")
        raise YamleGraphParseError("YAML root must be a mapping (dictionary).")

    # Basic required fields for a graph
    required_fields = ["nodes", "edges"]
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: '{field}'")
            raise YamleGraphParseError(f"Missing required field: '{field}'")

    if not isinstance(data["nodes"], list):
        logger.error("'nodes' must be a list.")
        raise YamleGraphParseError("'nodes' must be a list.")
    if not isinstance(data["edges"], list):
        logger.error("'edges' must be a list.")
        raise YamleGraphParseError("'edges' must be a list.")

    # Accept optional metadata for GUI, versioning, etc.
    metadata = data.get("metadata", {})

    # Convert 'from' to 'from_' for type safety and Python compatibility
    nodes: List[NodeDict] = []
    for node in data["nodes"]:
        if not isinstance(node, dict):
            logger.error("Each node must be a dictionary.")
            raise YamleGraphParseError("Each node must be a dictionary.")
        if "id" not in node:
            logger.error("Each node must have an 'id' field.")
            raise YamleGraphParseError("Each node must have an 'id' field.")
        node_id = str(node["id"])
        node_dict: NodeDict = NodeDict(id=node_id)
        for k, v in node.items():
            if k == "id":
                continue
            if k == "from":
                continue  # skip reserved
            node_dict[k] = v
        # --- AGENT USABILITY: Add default GUI/agent usability fields if missing ---
        if node_dict.get("type") == "agent":
            if "gui" not in node_dict or not isinstance(node_dict.get("gui"), dict):
                node_dict["gui"] = {"color": "blue", "icon": "robot", "visible": True, "show_status": True}
            if "agent_config" not in node_dict or not node_dict.get("agent_config"):
                node_dict["agent_config"] = agent_config_wizard(node_id)
            if "description" not in node_dict:
                node_dict["description"] = f"Agent node {node_id} (edit description in GUI)"
            if "status" not in node_dict:
                node_dict["status"] = "ready"
            if "created_at" not in node_dict:
                node_dict["created_at"] = _now_iso()
            if "updated_at" not in node_dict:
                node_dict["updated_at"] = _now_iso()
            if "run_count" not in node_dict:
                node_dict["run_count"] = 0
            if "last_run" not in node_dict:
                node_dict["last_run"] = None
            if "error_message" not in node_dict:
                node_dict["error_message"] = None
        nodes.append(node_dict)

    edges: List[EdgeDict] = []
    for edge in data["edges"]:
        if not isinstance(edge, dict):
            logger.error("Each edge must be a dictionary.")
            raise YamleGraphParseError("Each edge must be a dictionary.")
        if "from" not in edge or "to" not in edge:
            logger.error("Each edge must have 'from' and 'to' fields.")
            raise YamleGraphParseError("Each edge must have 'from' and 'to' fields.")
        from_id = str(edge["from"])
        to_id = str(edge["to"])
        edge_dict: EdgeDict = EdgeDict(from_=from_id, to=to_id)
        for k, v in edge.items():
            if k in ("from", "to"):
                continue
            edge_dict[k] = v
        # --- AGENT USABILITY: Add default GUI fields for agent-comm edges ---
        if edge_dict.get("type") == "agent-comm":
            if "gui" not in edge_dict or not isinstance(edge_dict.get("gui"), dict):
                edge_dict["gui"] = {"color": "red", "style": "dashed", "visible": True}
        edges.append(edge_dict)

    # Schema validation using pydantic
    try:
        # Convert to pydantic model for validation
        graph_model = GraphModel(
            nodes=[NodeSpec(**n) for n in nodes],
            edges=[ConnectionSpec(**{**e, "from": e["from_"]}) for e in edges],
            metadata=metadata
        )
    except ValidationError as ve:
        logger.error(f"YAML schema validation failed: {ve}")
        raise YamleGraphParseError(f"YAML schema validation failed: {ve}")

    # Return as GraphDict for legacy compatibility
    return GraphDict(nodes=nodes, edges=edges, metadata=metadata)

# --- Graph Validator (with cycle/duplicate edge detection, agent validation, GUI metadata) ---

def validate_yaml_graph(graph: GraphDict) -> None:
    """
    Validate the structure of a parsed YAML graph.

    Args:
        graph (GraphDict): The parsed graph.

    Raises:
        YamleGraphParseError: If validation fails.
    """
    node_ids: Set[str] = set()
    agent_node_ids: Set[str] = set()
    for node in graph["nodes"]:
        if not isinstance(node, dict):
            logger.error("Each node must be a dictionary.")
            raise YamleGraphParseError("Each node must be a dictionary.")
        node_id = node.get("id")
        if node_id is None:
            logger.error("Each node must have an 'id' field.")
            raise YamleGraphParseError("Each node must have an 'id' field.")
        node_ids.add(node_id)
        # Agent node validation for builder/GUI
        if node.get("type") == "agent":
            agent_node_ids.add(node_id)
            # Require agent_config for agent nodes
            if "agent_config" not in node or not node["agent_config"]:
                logger.error(f"Agent node '{node_id}' missing required 'agent_config'.")
                raise YamleGraphParseError(f"Agent node '{node_id}' missing required 'agent_config'.")
            # GUI metadata validation
            if "gui" not in node or not isinstance(node["gui"], dict):
                logger.warning(f"Agent node '{node_id}' missing or invalid 'gui' metadata.")
            # AGENT USABILITY: Validate agent usability fields
            if "description" not in node:
                logger.warning(f"Agent node '{node_id}' missing 'description' for GUI usability.")
            if "status" not in node:
                logger.warning(f"Agent node '{node_id}' missing 'status' for GUI usability.")
            if "created_at" not in node:
                logger.warning(f"Agent node '{node_id}' missing 'created_at' timestamp for GUI usability.")
            if "updated_at" not in node:
                logger.warning(f"Agent node '{node_id}' missing 'updated_at' timestamp for GUI usability.")
            if "run_count" not in node:
                logger.warning(f"Agent node '{node_id}' missing 'run_count' for GUI usability.")
            if "last_run" not in node:
                logger.info(f"Agent node '{node_id}' has not been run yet (last_run missing).")
            if "error_message" not in node:
                logger.info(f"Agent node '{node_id}' has no error_message field.")

    edge_set: Set[Tuple[str, str, Optional[str]]] = set()
    for edge in graph["edges"]:
        if not isinstance(edge, dict):
            logger.error("Each edge must be a dictionary.")
            raise YamleGraphParseError("Each edge must be a dictionary.")
        from_id = edge.get("from_")
        to_id = edge.get("to")
        if from_id is None or to_id is None:
            logger.error("Each edge must have 'from' and 'to' fields.")
            raise YamleGraphParseError("Each edge must have 'from' and 'to' fields.")
        if from_id not in node_ids:
            logger.error(f"Edge 'from' references unknown node: {from_id}")
            raise YamleGraphParseError(f"Edge 'from' references unknown node: {from_id}")
        if to_id not in node_ids:
            logger.error(f"Edge 'to' references unknown node: {to_id}")
            raise YamleGraphParseError(f"Edge 'to' references unknown node: {to_id}")
        # Agent edge validation for builder/GUI
        if edge.get("type") == "agent-comm":
            if from_id not in agent_node_ids and to_id not in agent_node_ids:
                logger.error(f"Agent-comm edge must connect at least one agent node: {edge}")
                raise YamleGraphParseError(
                    f"Agent-comm edge must connect at least one agent node: {edge}"
                )
        # GUI metadata validation for edges
        if "gui" not in edge or not isinstance(edge["gui"], dict):
            logger.warning(f"Edge from '{from_id}' to '{to_id}' missing or invalid 'gui' metadata.")

        # Duplicate edge detection (from, to, type)
        edge_key = (from_id, to_id, edge.get("type"))
        if edge_key in edge_set:
            logger.error(f"Duplicate edge detected: {edge_key}")
            raise YamleGraphParseError(f"Duplicate edge detected: {edge_key}")
        edge_set.add(edge_key)

    # Cycle detection using DFS
    adj: Dict[str, List[str]] = yaml_graph_to_adjacency_list(graph)
    visited: Set[str] = set()
    rec_stack: Set[str] = set()

    def dfs_cycle(node_id: str) -> bool:
        visited.add(node_id)
        rec_stack.add(node_id)
        for neighbor in adj.get(node_id, []):
            if neighbor not in visited:
                if dfs_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node_id)
        return False

    for node_id in node_ids:
        if node_id not in visited:
            if dfs_cycle(node_id):
                logger.error("Cycle detected in the graph.")
                raise YamleGraphParseError("Cycle detected in the graph.")

# --- Adjacency List Conversion ---

def yaml_graph_to_adjacency_list(graph: GraphDict) -> Dict[str, List[str]]:
    """
    Convert a YAML graph definition to an adjacency list.

    Args:
        graph (GraphDict): The parsed and validated graph.

    Returns:
        Dict[str, List[str]]: Adjacency list mapping node IDs to lists of neighbor IDs.
    """
    adj: Dict[str, List[str]] = {}
    for node in graph["nodes"]:
        node_id = node.get("id")
        if node_id is not None:
            adj[node_id] = []
    for edge in graph["edges"]:
        from_id = edge.get("from_")
        to_id = edge.get("to")
        if from_id is not None and to_id is not None:
            adj[from_id].append(to_id)
    return adj

# --- Round-trip Serialization (graph <-> YAML) ---

def graph_to_yaml(graph: GraphDict) -> str:
    """
    Serialize a GraphDict to a YAML string.

    Args:
        graph (GraphDict): The graph to serialize.

    Returns:
        str: YAML string.
    """
    # Convert 'from_' back to 'from' for YAML output
    edges_out = []
    for edge in graph["edges"]:
        edge_out = dict(edge)
        if "from_" in edge_out:
            edge_out["from"] = edge_out.pop("from_")
        edges_out.append(edge_out)
    nodes_out = [dict(node) for node in graph["nodes"]]
    out = {
        "nodes": nodes_out,
        "edges": edges_out,
    }
    if graph.get("metadata"):
        out["metadata"] = graph["metadata"]
    return yaml.safe_dump(out, sort_keys=False)

def yaml_to_graph(yaml_str: str) -> GraphDict:
    """
    Alias for load_yaml_graph for round-trip API symmetry.
    """
    return load_yaml_graph(yaml_str)

# --- Main Parse Function ---

def parse_yaml_graph(yaml_str: str) -> Tuple[GraphDict, Dict[str, List[str]]]:
    """
    Parse and validate a YAML graph, returning both the raw data and adjacency list.

    Args:
        yaml_str (str): The YAML string.

    Returns:
        Tuple[GraphDict, Dict[str, List[str]]]: (raw graph, adjacency list)
    """
    graph = load_yaml_graph(yaml_str)
    validate_yaml_graph(graph)
    adj = yaml_graph_to_adjacency_list(graph)
    return graph, adj

# --- CLI and API for graph import/export ---

def import_graph_from_yaml_file(file_path: str) -> Tuple[GraphDict, Dict[str, List[str]]]:
    """
    Import a graph from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        Tuple[GraphDict, Dict[str, List[str]]]: (raw graph, adjacency list)
    """
    with open(file_path, "r", encoding="utf-8") as f:
        yaml_str = f.read()
    return parse_yaml_graph(yaml_str)

def export_graph_to_yaml_file(graph: GraphDict, file_path: str) -> None:
    """
    Export a graph to a YAML file.

    Args:
        graph (GraphDict): The graph to export.
        file_path (str): Path to the output YAML file.
    """
    yaml_str = graph_to_yaml(graph)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(yaml_str)

# --- Agent Usability Helpers for GUI ---

def agent_node_template(
    node_id: str,
    label: str = "Agent",
    agent_type: str = "agent",
    gui: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    created_at: Optional[str] = None,
    updated_at: Optional[str] = None,
    run_count: Optional[int] = None,
    last_run: Optional[str] = None,
    error_message: Optional[str] = None,
) -> NodeDict:
    """
    Generate a template agent node for GUI builder.

    Args:
        node_id (str): Node ID.
        label (str): Node label.
        agent_type (str): Node type.
        gui (Optional[Dict[str, Any]]): GUI metadata.
        description (Optional[str]): Agent description for GUI.
        status (Optional[str]): Agent status for GUI.
        created_at (Optional[str]): Creation timestamp.
        updated_at (Optional[str]): Last update timestamp.
        run_count (Optional[int]): Number of times run.
        last_run (Optional[str]): Last run timestamp.
        error_message (Optional[str]): Last error message.

    Returns:
        NodeDict: Agent node template.
    """
    now = _now_iso()
    return NodeDict(
        id=node_id,
        type=agent_type,
        label=label,
        agent_config=agent_config_wizard(node_id),
        gui=gui or {"color": "blue", "icon": "robot", "visible": True, "show_status": True},
        description=description or f"Agent node {node_id} (edit description in GUI)",
        status=status or "ready",
        created_at=created_at or now,
        updated_at=updated_at or now,
        run_count=run_count if run_count is not None else 0,
        last_run=last_run,
        error_message=error_message,
    )

def agent_config_wizard(node_id: str) -> Dict[str, Any]:
    """
    Generate a default agent config for GUI wizard.

    Args:
        node_id (str): Node ID.

    Returns:
        Dict[str, Any]: Default agent config.
    """
    return {
        "prompt": f"Agent {node_id} prompt goes here.",
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 256,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
        "tools": [],
        "memory": {"enabled": True, "window": 5},
        "output_format": "text",
        "ui": {
            "show_prompt_editor": True,
            "show_parameter_controls": True,
            "show_tool_selector": True,
            "show_memory_settings": True,
            "show_output_format": True,
        },
    }

# --- Unit Tests for All Methods (including edge/error cases) ---

def _test_yaml_graph_module():
    # Minimal valid graph
    yaml_str = """
nodes:
  - id: n1
    type: agent
    label: "Agent 1"
    agent_config:
      prompt: "Do something"
    gui:
      color: "blue"
  - id: n2
    type: task
    label: "Task 2"
    gui:
      color: "green"
edges:
  - from: n1
    to: n2
    type: agent-comm
    gui:
      color: "red"
metadata:
  version: "1.0"
"""
    graph, adj = parse_yaml_graph(yaml_str)
    assert "n1" in adj and "n2" in adj
    assert adj["n1"] == ["n2"]
    assert adj["n2"] == []
    # Test agent usability fields
    agent_node = [n for n in graph["nodes"] if n.get("id") == "n1"][0]
    assert agent_node.get("description") is not None
    assert agent_node.get("status") == "ready"
    assert agent_node.get("created_at") is not None
    assert agent_node.get("updated_at") is not None
    assert agent_node.get("run_count") == 0
    # Test round-trip
    yaml_out = graph_to_yaml(graph)
    graph2 = yaml_to_graph(yaml_out)
    assert graph2["nodes"][0].get("id") == "n1"
    # Test duplicate edge detection
    yaml_dup = """
nodes:
  - id: a
  - id: b
edges:
  - from: a
    to: b
  - from: a
    to: b
"""
    try:
        parse_yaml_graph(yaml_dup)
        assert False, "Duplicate edge not detected"
    except YamleGraphParseError:
        pass
    # Test cycle detection
    yaml_cycle = """
nodes:
  - id: x
  - id: y
edges:
  - from: x
    to: y
  - from: y
    to: x
"""
    try:
        parse_yaml_graph(yaml_cycle)
        assert False, "Cycle not detected"
    except YamleGraphParseError:
        pass
    # Test agent node missing agent_config
    yaml_agent_missing = """
nodes:
  - id: ag
    type: agent
edges: []
"""
    try:
        parse_yaml_graph(yaml_agent_missing)
        assert False, "Missing agent_config not detected"
    except YamleGraphParseError:
        pass
    # Test agent usability fields auto-population
    yaml_agent_usability = """
nodes:
  - id: ag2
    type: agent
    agent_config:
      prompt: "Test"
edges: []
"""
    graph3, _ = parse_yaml_graph(yaml_agent_usability)
    ag2 = graph3["nodes"][0]
    assert ag2.get("description") is not None
    assert ag2.get("status") == "ready"
    assert ag2.get("created_at") is not None
    assert ag2.get("updated_at") is not None
    assert ag2.get("run_count") == 0
    print("All yamle_graph tests passed.")

if __name__ == "__main__":
    _test_yaml_graph_module()

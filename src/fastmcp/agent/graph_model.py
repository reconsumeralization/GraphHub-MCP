"""
agent_graph.py

Defines the AgentGraph class and related types for representing and manipulating
agent-based computational graphs in the FastMCP builder system.

NOTE: This module is intended as a low-level, strictly-typed, non-GUI graph representation.
It should not import or depend on any GUI libraries (e.g., Qt). It is suitable for use
by both the WorkflowManager (programmatic API) and agent modules.

Features:
- Serialization/deserialization (JSON/YAML)
- Validation for cyclic dependencies and orphan nodes
- Node/edge attribute validation and schema enforcement (partial)
- Graph visualization (basic), subgraph extraction, and pattern matching
- Execution engine integration (see fastmcp/execution_engine/)
- Subgraph encapsulation for composite/hierarchical workflows
- Agent/LLM suggestion hooks for next node, auto-wiring, and optimization
- Agent usability enhancements for GUI and agentic automation
- Enhanced agent/GUI usability: search, filtering, metadata, and agentic annotations
"""

import json
from typing import Dict, List, Optional, Set, Any, Callable, Tuple, Union, cast
from dataclasses import dataclass, field, asdict

try:
    import yaml  # Requires PyYAML; handle import error gracefully
except ImportError:
    yaml = None

# --- Type Aliases ---
NodeId = str
EdgeId = str

# --- Data Classes ---

@dataclass(frozen=True)
class AgentNode:
    """
    Represents a node (agent) in the computational graph.
    """
    id: NodeId
    type: str
    config: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None  # For GUI/agent usability
    icon: Optional[str] = None         # For GUI display
    agentic: Optional[bool] = None     # For agent usability: is this node agent-driven?
    tags: Optional[List[str]] = None   # For GUI/agent filtering and search

    def validate(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError("AgentNode.id must be a non-empty string.")
        if not self.type or not isinstance(self.type, str):
            raise ValueError("AgentNode.type must be a non-empty string.")
        if not isinstance(self.config, dict):
            raise ValueError("AgentNode.config must be a dictionary.")
        if not isinstance(self.metadata, dict):
            raise ValueError("AgentNode.metadata must be a dictionary.")
        if self.tags is not None and not isinstance(self.tags, list):
            raise ValueError("AgentNode.tags must be a list of strings or None.")
        if self.tags is not None:
            for tag in self.tags:
                if not isinstance(tag, str):
                    raise ValueError("Each tag in AgentNode.tags must be a string.")

@dataclass(frozen=True)
class AgentEdge:
    """
    Represents a directed edge between two nodes in the graph.
    """
    id: EdgeId
    source: NodeId
    target: NodeId
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    style: Optional[str] = None  # For GUI/agent usability (e.g., dashed, color)
    agentic: Optional[bool] = None  # For agent usability: is this edge agent-driven?
    tags: Optional[List[str]] = None  # For GUI/agent filtering and search

    def validate(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError("AgentEdge.id must be a non-empty string.")
        if not self.source or not isinstance(self.source, str):
            raise ValueError("AgentEdge.source must be a non-empty string.")
        if not self.target or not isinstance(self.target, str):
            raise ValueError("AgentEdge.target must be a non-empty string.")
        if not isinstance(self.metadata, dict):
            raise ValueError("AgentEdge.metadata must be a dictionary.")
        if self.tags is not None and not isinstance(self.tags, list):
            raise ValueError("AgentEdge.tags must be a list of strings or None.")
        if self.tags is not None:
            for tag in self.tags:
                if not isinstance(tag, str):
                    raise ValueError("Each tag in AgentEdge.tags must be a string.")

@dataclass(frozen=True)
class SubWorkflowNode(AgentNode):
    """
    Represents a composite node encapsulating a subgraph (sub-workflow).
    """
    subgraph: Dict[str, Any] = field(default_factory=dict)  # Serialized AgentGraph

    def validate(self) -> None:
        super().validate()
        if not isinstance(self.subgraph, dict):
            raise ValueError("SubWorkflowNode.subgraph must be a dict (serialized AgentGraph).")
        if "nodes" not in self.subgraph or "edges" not in self.subgraph:
            raise ValueError("SubWorkflowNode.subgraph must contain 'nodes' and 'edges' keys.")
        if not isinstance(self.subgraph["nodes"], list) or not isinstance(self.subgraph["edges"], list):
            raise ValueError("SubWorkflowNode.subgraph 'nodes' and 'edges' must be lists.")

# --- Core Graph Class ---

class AgentGraph:
    """
    Represents a directed graph of agent nodes and edges.
    Provides methods for graph manipulation, validation, traversal, and serialization.
    Supports agent/LLM suggestion hooks and composite sub-workflows.
    Enhanced for agent usability and GUI integration.
    """

    def __init__(self) -> None:
        self.nodes: Dict[NodeId, AgentNode] = {}
        self.edges: Dict[EdgeId, AgentEdge] = {}
        self._adjacency: Dict[NodeId, Set[NodeId]] = {}
        self._reverse_adjacency: Dict[NodeId, Set[NodeId]] = {}

        # --- Agent/LLM Suggestion Hooks ---
        self.on_node_added: Optional[Callable[[AgentNode, "AgentGraph"], None]] = None
        self.on_edge_added: Optional[Callable[[AgentEdge, "AgentGraph"], None]] = None

        # --- Agent/GUI Usability Hooks ---
        self.on_graph_changed: Optional[Callable[["AgentGraph"], None]] = None  # For GUI/agent sync

    # --- Node/Edge Manipulation ---

    def add_node(self, node: AgentNode) -> None:
        node.validate()
        if node.id in self.nodes:
            raise ValueError(f"Node with id '{node.id}' already exists.")
        self.nodes[node.id] = node
        self._adjacency[node.id] = set()
        self._reverse_adjacency[node.id] = set()
        if self.on_node_added:
            self.on_node_added(node, self)
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def remove_node(self, node_id: NodeId) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        edges_to_remove = [eid for eid, e in self.edges.items()
                           if e.source == node_id or e.target == node_id]
        for eid in edges_to_remove:
            self.remove_edge(eid)
        del self.nodes[node_id]
        del self._adjacency[node_id]
        del self._reverse_adjacency[node_id]
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def add_edge(self, edge: AgentEdge) -> None:
        edge.validate()
        if edge.id in self.edges:
            raise ValueError(f"Edge with id '{edge.id}' already exists.")
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError("Both source and target nodes must exist before adding an edge.")
        self.edges[edge.id] = edge
        self._adjacency[edge.source].add(edge.target)
        self._reverse_adjacency[edge.target].add(edge.source)
        if self.on_edge_added:
            self.on_edge_added(edge, self)
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def remove_edge(self, edge_id: EdgeId) -> None:
        if edge_id not in self.edges:
            raise KeyError(f"Edge '{edge_id}' does not exist.")
        edge = self.edges[edge_id]
        self._adjacency[edge.source].discard(edge.target)
        self._reverse_adjacency[edge.target].discard(edge.source)
        del self.edges[edge_id]
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def get_node(self, node_id: NodeId) -> AgentNode:
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        return self.nodes[node_id]

    def get_edge(self, edge_id: EdgeId) -> AgentEdge:
        if edge_id not in self.edges:
            raise KeyError(f"Edge '{edge_id}' does not exist.")
        return self.edges[edge_id]

    def successors(self, node_id: NodeId) -> Set[NodeId]:
        if node_id not in self._adjacency:
            raise KeyError(f"Node '{node_id}' does not exist.")
        return set(self._adjacency.get(node_id, set()))

    def predecessors(self, node_id: NodeId) -> Set[NodeId]:
        if node_id not in self._reverse_adjacency:
            raise KeyError(f"Node '{node_id}' does not exist.")
        return set(self._reverse_adjacency.get(node_id, set()))

    # --- Graph Analysis & Optimization ---

    def topological_sort(self) -> List[NodeId]:
        """
        Returns a topological ordering of the nodes.
        Raises ValueError if the graph contains a cycle.
        """
        visited: Set[NodeId] = set()
        temp: Set[NodeId] = set()
        result: List[NodeId] = []

        def visit(nid: NodeId) -> None:
            if nid in temp:
                raise ValueError("Graph contains a cycle.")
            if nid not in visited:
                temp.add(nid)
                for succ in self.successors(nid):
                    visit(succ)
                temp.remove(nid)
                visited.add(nid)
                result.append(nid)

        for node_id in self.nodes:
            if node_id not in visited:
                visit(node_id)
        result.reverse()
        return result

    def has_cycle(self) -> bool:
        """
        Returns True if the graph contains a cycle, False otherwise.
        """
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True

    def find_orphan_nodes(self) -> Set[NodeId]:
        """
        Returns the set of nodes with no incoming or outgoing edges.
        """
        return {nid for nid in self.nodes
                if not self._adjacency[nid] and not self._reverse_adjacency[nid]}

    def validate(self) -> None:
        """
        Validates the graph for cycles, orphan nodes, and node/edge schema.
        Raises ValueError if invalid.
        """
        for node in self.nodes.values():
            node.validate()
        for edge in self.edges.values():
            edge.validate()
        if self.has_cycle():
            raise ValueError("Graph contains a cycle.")
        orphans = self.find_orphan_nodes()
        if orphans:
            raise ValueError(f"Graph contains orphan nodes: {orphans}")

    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyzes the graph and returns optimization hints.
        (Basic implementation; for advanced optimization, see graph_utils.py)
        """
        flattenable_branches = []
        batchable_nodes = []
        cachable_subgraphs = []

        for node_id, node in self.nodes.items():
            preds = self.predecessors(node_id)
            succs = self.successors(node_id)
            if len(preds) == 1 and len(succs) == 1:
                flattenable_branches.append(node_id)
            if node.type.lower() == "batch" or "batch_size" in node.config:
                batchable_nodes.append(node_id)
            if isinstance(node, SubWorkflowNode):
                cachable_subgraphs.append(node_id)

        return {
            "flattenable_branches": flattenable_branches,
            "batchable_nodes": batchable_nodes,
            "cachable_subgraphs": cachable_subgraphs,
            "notes": "This is a basic analysis. For advanced optimization, see graph_utils.py."
        }

    # --- Serialization/Deserialization ---

    def as_dict(self) -> Dict[str, Any]:
        """
        Serializes the graph to a dictionary.
        """
        def node_to_dict(node: AgentNode) -> Dict[str, Any]:
            d = asdict(node)
            if isinstance(node, SubWorkflowNode):
                d["subgraph"] = node.subgraph
            return d

        return {
            "nodes": [node_to_dict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges.values()]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentGraph":
        """
        Deserializes a graph from a dictionary.
        """
        graph = cls()
        for node_data in data.get("nodes", []):
            if node_data.get("subgraph") is not None:
                node = SubWorkflowNode(**node_data)
            else:
                node = AgentNode(**node_data)
            graph.add_node(node)
        for edge_data in data.get("edges", []):
            edge = AgentEdge(**edge_data)
            graph.add_edge(edge)
        return graph

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Serializes the graph to a JSON string.
        """
        return json.dumps(self.as_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "AgentGraph":
        """
        Deserializes a graph from a JSON string.
        """
        data = json.loads(s)
        return cls.from_dict(data)

    def to_yaml(self) -> str:
        """
        Serializes the graph to a YAML string.
        """
        if yaml is None:
            raise RuntimeError("PyYAML is not installed. Cannot serialize to YAML.")
        try:
            return yaml.dump(self.as_dict(), sort_keys=False)
        except Exception as e:
            raise RuntimeError("YAML serialization failed.") from e

    @classmethod
    def from_yaml(cls, s: str) -> "AgentGraph":
        """
        Deserializes a graph from a YAML string.
        """
        if yaml is None:
            raise RuntimeError("PyYAML is not installed. Cannot deserialize from YAML.")
        try:
            data = yaml.safe_load(s)
            return cls.from_dict(data)
        except Exception as e:
            raise RuntimeError("YAML deserialization failed.") from e

    def clear(self) -> None:
        """
        Removes all nodes and edges from the graph.
        """
        self.nodes.clear()
        self.edges.clear()
        self._adjacency.clear()
        self._reverse_adjacency.clear()
        if self.on_graph_changed:
            self.on_graph_changed(self)

    # --- Visualization, Subgraph, and Pattern Matching ---

    def visualize_ascii(self) -> None:
        """
        Prints a simple ASCII representation of the graph.
        """
        print("Nodes:")
        for node in self.nodes.values():
            if isinstance(node, SubWorkflowNode):
                print(f"  {node.id}: [SubWorkflow] {node.label or ''}")
            else:
                print(f"  {node.id}: {node.type} ({node.label or ''})")
        print("Edges:")
        for edge in self.edges.values():
            print(f"  {edge.source} -> {edge.target} [{edge.label or ''}]")

    def extract_subgraph(self, node_ids: Set[NodeId], as_subworkflow: Optional[str] = None) -> Union["AgentGraph", SubWorkflowNode]:
        """
        Returns a new AgentGraph containing only the specified nodes and their connecting edges.
        If as_subworkflow is provided, returns a SubWorkflowNode encapsulating the subgraph.
        """
        graph = AgentGraph()
        for nid in node_ids:
            if nid in self.nodes:
                graph.add_node(self.nodes[nid])
        for edge in self.edges.values():
            if edge.source in node_ids and edge.target in node_ids:
                graph.add_edge(edge)
        if as_subworkflow:
            graph_dict = graph.as_dict()
            return SubWorkflowNode(
                id=as_subworkflow,
                type="subworkflow",
                config={},
                label=as_subworkflow,
                metadata={},
                subgraph=graph_dict
            )
        return graph

    def find_pattern(self, pattern: Callable[["AgentGraph"], List[Tuple[NodeId, ...]]]) -> List[Tuple[NodeId, ...]]:
        """
        Finds patterns in the graph using a user-supplied pattern function.
        Returns a list of tuples of node ids matching the pattern.
        """
        return pattern(self)

    # --- Agent/LLM Suggestion API ---

    def recommend_next_node(self, partial_spec: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Suggests next node types, default configs, or auto-wiring patterns.
        Enhanced: consults NODE_REGISTRY if available for agent/GUI usability.
        """
        suggestions = [
            {"type": "api_call", "label": "API Call", "description": "Call an external API"},
            {"type": "error_handler", "label": "Error Handler", "description": "Handle errors in workflow"},
            {"type": "notify", "label": "Notification", "description": "Send a notification"}
        ]
        try:
            from fastmcp.agent.node_registry import NODE_REGISTRY  # type: ignore
            registry_suggestions = []
            for node_type, node_info in NODE_REGISTRY.items():
                suggestion = {
                    "type": node_type,
                    "label": node_info.get("label", node_type),
                    "description": node_info.get("description", "")
                }
                registry_suggestions.append(suggestion)
            if registry_suggestions:
                suggestions = registry_suggestions
        except Exception:
            pass
        return suggestions

    def suggest_optimizations(self) -> List[Dict[str, Any]]:
        """
        Returns a list of suggested optimizations as diffs.
        (Basic implementation; for advanced agent/LLM integration, extend here)
        """
        suggestions = []
        if self.has_cycle():
            suggestions.append({
                "action": "remove_cycle",
                "description": "Graph contains a cycle. Consider removing an edge to break the cycle."
            })
        orphans = self.find_orphan_nodes()
        if orphans:
            suggestions.append({
                "action": "connect_orphans",
                "description": f"Graph contains orphan nodes: {orphans}. Consider connecting or removing them."
            })
        return suggestions

    def extract_subflow(self, node_ids: Set[NodeId], subflow_name: str) -> SubWorkflowNode:
        """
        Encapsulates the given node_ids as a reusable SubWorkflowNode.
        """
        graph_obj = cast(AgentGraph, self.extract_subgraph(node_ids))
        subgraph_dict = graph_obj.as_dict()
        return SubWorkflowNode(
            id=subflow_name,
            type="subworkflow",
            config={},
            label=subflow_name,
            metadata={},
            subgraph=subgraph_dict
        )

    # --- Execution Engine Integration ---

    def execute(self, executor: Optional[Callable[["AgentGraph"], Any]] = None) -> Any:
        """
        Integrates with an execution engine to run the graph.
        If executor is provided, it is called with this graph.
        """
        if executor is None:
            raise NotImplementedError("No execution engine provided. Pass an executor callable.")
        return executor(self)

    # --- Runtime Graph Mutation (for adaptive execution) ---

    def inject_node(self, node: AgentNode, after_node_id: Optional[NodeId] = None) -> None:
        """
        Injects a node into the running graph, optionally wiring it after a given node.
        Actual runtime mutation should be coordinated with execution engine.
        """
        self.add_node(node)
        if after_node_id and after_node_id in self.nodes:
            edge_id = f"auto_{after_node_id}_{node.id}"
            self.add_edge(AgentEdge(id=edge_id, source=after_node_id, target=node.id))
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def trigger_graph_mutation(self, patch_spec: Dict[str, Any]) -> None:
        """
        Applies a patch to the running graph.
        Supports add_node, remove_node, add_edge, remove_edge, update_node, update_edge.
        """
        if "add_node" in patch_spec:
            node_data = patch_spec["add_node"]
            if node_data.get("subgraph") is not None:
                node = SubWorkflowNode(**node_data)
            else:
                node = AgentNode(**node_data)
            self.add_node(node)
        if "remove_node" in patch_spec:
            self.remove_node(patch_spec["remove_node"])
        if "add_edge" in patch_spec:
            edge_data = patch_spec["add_edge"]
            edge = AgentEdge(**edge_data)
            self.add_edge(edge)
        if "remove_edge" in patch_spec:
            self.remove_edge(patch_spec["remove_edge"])
        if "update_node" in patch_spec:
            node_data = patch_spec["update_node"]
            node_id = node_data["id"]
            if node_id in self.nodes:
                if node_data.get("subgraph") is not None:
                    self.nodes[node_id] = SubWorkflowNode(**node_data)
                else:
                    self.nodes[node_id] = AgentNode(**node_data)
        if "update_edge" in patch_spec:
            edge_data = patch_spec["update_edge"]
            edge_id = edge_data["id"]
            if edge_id in self.edges:
                self.edges[edge_id] = AgentEdge(**edge_data)
        if self.on_graph_changed:
            self.on_graph_changed(self)

    # --- Execution History/Insights ---

    def get_execution_insights(self) -> Dict[str, Any]:
        """
        Returns statistical summaries, anomalies, and hot spots for the graph.
        (Stub: to be implemented with execution history database)
        """
        # For now, return static/dummy data
        return {
            "success_rate": 1.0,
            "latency_ms": {node_id: 0 for node_id in self.nodes},
            "error_patterns": [],
            "hot_spots": []
        }

    # --- Agent/GUI Usability: Node/Edge Search & Filtering ---

    def search_nodes(self, query: str, tags: Optional[List[str]] = None, agentic: Optional[bool] = None) -> List[AgentNode]:
        """
        Returns a list of nodes whose label, type, description, or tags match the query.
        Supports filtering by tags and agentic property for agent/GUI search/autocomplete.
        """
        q = query.lower()
        results = []
        for node in self.nodes.values():
            label_match = node.label and q in node.label.lower()
            type_match = node.type and q in node.type.lower()
            desc = getattr(node, "description", None)
            desc_match = desc is not None and isinstance(desc, str) and q in desc.lower()
            tags_match = node.tags and any(q in tag.lower() for tag in node.tags)
            agentic_match = agentic is None or node.agentic == agentic
            tags_filter = tags is None or (node.tags and any(tag in node.tags for tag in tags))
            if (label_match or type_match or desc_match or tags_match) and agentic_match and tags_filter:
                results.append(node)
        return results

    def search_edges(self, query: str, tags: Optional[List[str]] = None, agentic: Optional[bool] = None) -> List[AgentEdge]:
        """
        Returns a list of edges whose label, style, or tags match the query.
        Supports filtering by tags and agentic property for agent/GUI search/autocomplete.
        """
        q = query.lower()
        results = []
        for edge in self.edges.values():
            label_match = edge.label and q in edge.label.lower()
            style = getattr(edge, "style", None)
            style_match = style is not None and isinstance(style, str) and q in style.lower()
            tags_match = edge.tags and any(q in tag.lower() for tag in edge.tags)
            agentic_match = agentic is None or edge.agentic == agentic
            tags_filter = tags is None or (edge.tags and any(tag in edge.tags for tag in tags))
            if (label_match or style_match or tags_match) and agentic_match and tags_filter:
                results.append(edge)
        return results

    # --- Agent/GUI Usability: Node/Edge Metadata Update ---

    def update_node_metadata(self, node_id: NodeId, metadata: Dict[str, Any]) -> None:
        """
        Updates the metadata of a node (immutable dataclass pattern).
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        node = self.nodes[node_id]
        updated = node.__class__(**{**asdict(node), "metadata": {**node.metadata, **metadata}})
        self.nodes[node_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def update_edge_metadata(self, edge_id: EdgeId, metadata: Dict[str, Any]) -> None:
        """
        Updates the metadata of an edge (immutable dataclass pattern).
        """
        if edge_id not in self.edges:
            raise KeyError(f"Edge '{edge_id}' does not exist.")
        edge = self.edges[edge_id]
        updated = edge.__class__(**{**asdict(edge), "metadata": {**edge.metadata, **metadata}})
        self.edges[edge_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def set_node_agentic(self, node_id: NodeId, agentic: bool) -> None:
        """
        Sets the agentic property of a node for agent usability and GUI highlighting.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        node = self.nodes[node_id]
        updated = node.__class__(**{**asdict(node), "agentic": agentic})
        self.nodes[node_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def set_edge_agentic(self, edge_id: EdgeId, agentic: bool) -> None:
        """
        Sets the agentic property of an edge for agent usability and GUI highlighting.
        """
        if edge_id not in self.edges:
            raise KeyError(f"Edge '{edge_id}' does not exist.")
        edge = self.edges[edge_id]
        updated = edge.__class__(**{**asdict(edge), "agentic": agentic})
        self.edges[edge_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def add_node_tag(self, node_id: NodeId, tag: str) -> None:
        """
        Adds a tag to a node for agent/GUI filtering and usability.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        node = self.nodes[node_id]
        tags = list(node.tags) if node.tags else []
        if tag not in tags:
            tags.append(tag)
        updated = node.__class__(**{**asdict(node), "tags": tags})
        self.nodes[node_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def add_edge_tag(self, edge_id: EdgeId, tag: str) -> None:
        """
        Adds a tag to an edge for agent/GUI filtering and usability.
        """
        if edge_id not in self.edges:
            raise KeyError(f"Edge '{edge_id}' does not exist.")
        edge = self.edges[edge_id]
        tags = list(edge.tags) if edge.tags else []
        if tag not in tags:
            tags.append(tag)
        updated = edge.__class__(**{**asdict(edge), "tags": tags})
        self.edges[edge_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    # --- Agent/GUI Usability: Node/Edge Listing ---

    def list_nodes(self, agentic: Optional[bool] = None, tags: Optional[List[str]] = None) -> List[AgentNode]:
        """
        Returns a list of all nodes in insertion order, optionally filtered by agentic property or tags.
        """
        nodes = list(self.nodes.values())
        if agentic is not None:
            nodes = [n for n in nodes if n.agentic == agentic]
        if tags is not None:
            nodes = [n for n in nodes if n.tags and any(tag in n.tags for tag in tags)]
        return nodes

    def list_edges(self, agentic: Optional[bool] = None, tags: Optional[List[str]] = None) -> List[AgentEdge]:
        """
        Returns a list of all edges in insertion order, optionally filtered by agentic property or tags.
        """
        edges = list(self.edges.values())
        if agentic is not None:
            edges = [e for e in edges if e.agentic == agentic]
        if tags is not None:
            edges = [e for e in edges if e.tags and any(tag in e.tags for tag in tags)]
        return edges

    # --- Agent/GUI usability methods ---

    def annotate_node(self, node_id: NodeId, comment: str, user: Optional[str] = None) -> None:
        """
        Adds an annotation/comment to a node for audit trail support.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        node = self.nodes[node_id]
        metadata = dict(node.metadata)
        annotations = metadata.get("annotations", [])
        annotations.append({"comment": comment, "user": user})
        metadata["annotations"] = annotations
        updated = node.__class__(**{**asdict(node), "metadata": metadata})
        self.nodes[node_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def annotate_edge(self, edge_id: EdgeId, comment: str, user: Optional[str] = None) -> None:
        """
        Adds an annotation/comment to an edge for audit trail support.
        """
        if edge_id not in self.edges:
            raise KeyError(f"Edge '{edge_id}' does not exist.")
        edge = self.edges[edge_id]
        metadata = dict(edge.metadata)
        annotations = metadata.get("annotations", [])
        annotations.append({"comment": comment, "user": user})
        metadata["annotations"] = annotations
        updated = edge.__class__(**{**asdict(edge), "metadata": metadata})
        self.edges[edge_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def lock_node(self, node_id: NodeId, user: str) -> None:
        """
        Locks a node for collaborative editing.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        node = self.nodes[node_id]
        metadata = dict(node.metadata)
        metadata["locked_by"] = user
        updated = node.__class__(**{**asdict(node), "metadata": metadata})
        self.nodes[node_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def unlock_node(self, node_id: NodeId) -> None:
        """
        Unlocks a node for collaborative editing.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        node = self.nodes[node_id]
        metadata = dict(node.metadata)
        metadata.pop("locked_by", None)
        updated = node.__class__(**{**asdict(node), "metadata": metadata})
        self.nodes[node_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def lock_edge(self, edge_id: EdgeId, user: str) -> None:
        """
        Locks an edge for collaborative editing.
        """
        if edge_id not in self.edges:
            raise KeyError(f"Edge '{edge_id}' does not exist.")
        edge = self.edges[edge_id]
        metadata = dict(edge.metadata)
        metadata["locked_by"] = user
        updated = edge.__class__(**{**asdict(edge), "metadata": metadata})
        self.edges[edge_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def unlock_edge(self, edge_id: EdgeId) -> None:
        """
        Unlocks an edge for collaborative editing.
        """
        if edge_id not in self.edges:
            raise KeyError(f"Edge '{edge_id}' does not exist.")
        edge = self.edges[edge_id]
        metadata = dict(edge.metadata)
        metadata.pop("locked_by", None)
        updated = edge.__class__(**{**asdict(edge), "metadata": metadata})
        self.edges[edge_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def version_node(self, node_id: NodeId, version: str) -> None:
        """
        Sets a version string for a node.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        node = self.nodes[node_id]
        metadata = dict(node.metadata)
        metadata["version"] = version
        updated = node.__class__(**{**asdict(node), "metadata": metadata})
        self.nodes[node_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def version_edge(self, edge_id: EdgeId, version: str) -> None:
        """
        Sets a version string for an edge.
        """
        if edge_id not in self.edges:
            raise KeyError(f"Edge '{edge_id}' does not exist.")
        edge = self.edges[edge_id]
        metadata = dict(edge.metadata)
        metadata["version"] = version
        updated = edge.__class__(**{**asdict(edge), "metadata": metadata})
        self.edges[edge_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def quick_action_node(self, node_id: NodeId, action: str) -> None:
        """
        Performs a quick action on a node for agentic automation.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        node = self.nodes[node_id]
        metadata = dict(node.metadata)
        metadata["quick_action"] = action
        updated = node.__class__(**{**asdict(node), "metadata": metadata})
        self.nodes[node_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def quick_action_edge(self, edge_id: EdgeId, action: str) -> None:
        """
        Performs a quick action on an edge for agentic automation.
        """
        if edge_id not in self.edges:
            raise KeyError(f"Edge '{edge_id}' does not exist.")
        edge = self.edges[edge_id]
        metadata = dict(edge.metadata)
        metadata["quick_action"] = action
        updated = edge.__class__(**{**asdict(edge), "metadata": metadata})
        self.edges[edge_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def favorite_node(self, node_id: NodeId) -> None:
        """
        Marks a node as favorite/starred for quick access.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        node = self.nodes[node_id]
        metadata = dict(node.metadata)
        metadata["favorite"] = True
        updated = node.__class__(**{**asdict(node), "metadata": metadata})
        self.nodes[node_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def unfavorite_node(self, node_id: NodeId) -> None:
        """
        Removes favorite/starred mark from a node.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        node = self.nodes[node_id]
        metadata = dict(node.metadata)
        metadata.pop("favorite", None)
        updated = node.__class__(**{**asdict(node), "metadata": metadata})
        self.nodes[node_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def favorite_edge(self, edge_id: EdgeId) -> None:
        """
        Marks an edge as favorite/starred for quick access.
        """
        if edge_id not in self.edges:
            raise KeyError(f"Edge '{edge_id}' does not exist.")
        edge = self.edges[edge_id]
        metadata = dict(edge.metadata)
        metadata["favorite"] = True
        updated = edge.__class__(**{**asdict(edge), "metadata": metadata})
        self.edges[edge_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def unfavorite_edge(self, edge_id: EdgeId) -> None:
        """
        Removes favorite/starred mark from an edge.
        """
        if edge_id not in self.edges:
            raise KeyError(f"Edge '{edge_id}' does not exist.")
        edge = self.edges[edge_id]
        metadata = dict(edge.metadata)
        metadata.pop("favorite", None)
        updated = edge.__class__(**{**asdict(edge), "metadata": metadata})
        self.edges[edge_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def set_node_color(self, node_id: NodeId, color: str) -> None:
        """
        Sets a color/visual style for a node for agent/GUI highlighting.
        """
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        node = self.nodes[node_id]
        metadata = dict(node.metadata)
        metadata["color"] = color
        updated = node.__class__(**{**asdict(node), "metadata": metadata})
        self.nodes[node_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

    def set_edge_color(self, edge_id: EdgeId, color: str) -> None:
        """
        Sets a color/visual style for an edge for agent/GUI highlighting.
        """
        if edge_id not in self.edges:
            raise KeyError(f"Edge '{edge_id}' does not exist.")
        edge = self.edges[edge_id]
        metadata = dict(edge.metadata)
        metadata["color"] = color
        updated = edge.__class__(**{**asdict(edge), "metadata": metadata})
        self.edges[edge_id] = updated
        if self.on_graph_changed:
            self.on_graph_changed(self)

from typing import Dict, List, Set, Tuple, Optional, Any, TypeAlias, cast

# --- Types and Aliases ---
# These should be imported from src/fastmcp/types.py in the future for single-source-of-truth typing.
NodeId: TypeAlias = str
NodeSpec: TypeAlias = Dict[str, Any]  # TODO: Replace Any with stricter schema per node type
EdgeSpec: TypeAlias = Tuple[NodeId, NodeId]
GraphSpec: TypeAlias = Dict[str, NodeSpec]
ConnectionSpec: TypeAlias = List[EdgeSpec]
ExecutionRecord: TypeAlias = Dict[str, Any]  # TODO: Refine for execution engine

class GraphValidationError(Exception):
    """Custom exception for graph validation errors."""
    pass

class GraphOptimizationHint:
    """
    Represents a suggested optimization for a workflow graph.
    """
    def __init__(self, message: str, diff: Optional[dict] = None) -> None:
        self.message: str = message
        self.diff: Dict[str, Any] = diff or {}

class GraphValidator:
    """
    Validates and analyzes a directed acyclic graph (DAG) structure for the builder.
    Ensures:
      - No cycles
      - All nodes are reachable from the start node
      - All required node fields are present
      - No orphaned nodes
      - Edge references are valid

    Also provides hooks for agent-assisted suggestions and optimization analysis.
    """

    def __init__(
        self,
        nodes: GraphSpec,
        edges: ConnectionSpec,
        start_node_id: NodeId
    ) -> None:
        """
        :param nodes: Dict of node_id -> node data (NodeSpec)
        :param edges: List of (from_node_id, to_node_id) tuples
        :param start_node_id: The node id where execution starts
        """
        self.nodes: GraphSpec = nodes
        self.edges: ConnectionSpec = edges
        self.start_node_id: NodeId = start_node_id
        self.adjacency: Dict[NodeId, List[NodeId]] = self._build_adjacency()
        self.reverse_adjacency: Dict[NodeId, List[NodeId]] = self._build_reverse_adjacency()

    def _build_adjacency(self) -> Dict[NodeId, List[NodeId]]:
        adj: Dict[NodeId, List[NodeId]] = {node_id: [] for node_id in self.nodes}
        for from_id, to_id in self.edges:
            if from_id in adj:
                adj[from_id].append(to_id)
            else:
                # TODO: Open issue if edge references missing node
                adj[from_id] = [to_id]
        return adj

    def _build_reverse_adjacency(self) -> Dict[NodeId, List[NodeId]]:
        rev: Dict[NodeId, List[NodeId]] = {node_id: [] for node_id in self.nodes}
        for from_id, to_id in self.edges:
            if to_id in rev:
                rev[to_id].append(from_id)
            else:
                # TODO: Open issue if edge references missing node
                rev[to_id] = [from_id]
        return rev

    def validate(self) -> None:
        """
        Run all graph validation checks. Raises GraphValidationError on failure.
        """
        self._check_nodes_exist()
        self._check_edges_valid()
        self._check_no_cycles()
        self._check_all_reachable()
        self._check_no_orphans()
        self._check_required_fields()
        # TODO: Add more semantic checks as needed

    def _check_nodes_exist(self) -> None:
        if self.start_node_id not in self.nodes:
            raise GraphValidationError(
                f"Start node '{self.start_node_id}' does not exist in nodes."
            )
        for from_id, to_id in self.edges:
            if from_id not in self.nodes:
                raise GraphValidationError(
                    f"Edge references missing from-node '{from_id}'."
                )
            if to_id not in self.nodes:
                raise GraphValidationError(
                    f"Edge references missing to-node '{to_id}'."
                )

    def _check_edges_valid(self) -> None:
        for from_id, to_id in self.edges:
            if from_id not in self.nodes or to_id not in self.nodes:
                raise GraphValidationError(
                    f"Invalid edge: {from_id} -> {to_id} (node missing)"
                )

    def _check_no_cycles(self) -> None:
        visited: Set[NodeId] = set()
        rec_stack: Set[NodeId] = set()

        def visit(node_id: NodeId) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            for neighbor in self.adjacency.get(node_id, []):
                if neighbor not in visited:
                    if visit(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.remove(node_id)
            return False

        for node_id in self.nodes:
            if node_id not in visited:
                if visit(node_id):
                    raise GraphValidationError("Graph contains a cycle.")

    def _check_all_reachable(self) -> None:
        reachable = self._dfs(self.start_node_id)
        unreachable = set(self.nodes.keys()) - reachable
        if unreachable:
            raise GraphValidationError(
                f"Unreachable nodes detected: {unreachable}"
            )

    def _dfs(self, start: NodeId) -> Set[NodeId]:
        stack: List[NodeId] = [start]
        seen: Set[NodeId] = set()
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                stack.extend(self.adjacency.get(node, []))
        return seen

    def _check_no_orphans(self) -> None:
        # Orphan: node with no incoming or outgoing edges, except possibly the start node
        for node_id in self.nodes:
            if node_id == self.start_node_id:
                continue
            if not self.adjacency.get(node_id) and not self.reverse_adjacency.get(node_id):
                raise GraphValidationError(
                    f"Orphaned node detected: {node_id}"
                )

    def _check_required_fields(self) -> None:
        # TODO: Define required fields per node type if schema is available
        for node_id, node_data in self.nodes.items():
            if "type" not in node_data:
                raise GraphValidationError(
                    f"Node '{node_id}' missing required field: 'type'"
                )
            # TODO: Add more field checks as needed

    # --- AGENT-ASSISTED SUGGESTION HOOKS ---

    def suggest_next_node(self, partial_graph: Optional[GraphSpec] = None) -> Dict[str, str]:
        """
        Suggests the next node type, default property values, or auto-wiring patterns
        based on the current graph state. Intended for LLM/agent co-pilot integration.
        """
        # TODO: Integrate with agent tool: recommend_next_node
        # Example stub: after API call, suggest error handler
        last_node = self._get_last_added_node()
        if last_node and self.nodes[last_node].get("type") == "api_call":
            return {
                "suggested_type": "error_handler",
                "reason": "API call nodes should be followed by error handling."
            }
        # Fallback: suggest a generic node
        return {
            "suggested_type": "noop",
            "reason": "No specific suggestion. Continue building your workflow."
        }

    def _get_last_added_node(self) -> Optional[NodeId]:
        # TODO: Replace with actual event tracking in WorkflowManager
        if self.nodes:
            return list(self.nodes.keys())[-1]
        return None

    # --- WORKFLOW OPTIMIZATION & ANALYSIS ---

    def analyze_performance(self) -> List[GraphOptimizationHint]:
        """
        Analyze the workflow graph and return a list of optimization hints.
        E.g., flatten redundant branches, batch similar API calls, suggest caching.
        """
        hints: List[GraphOptimizationHint] = []
        # Example: Detect redundant branches (two nodes with identical outgoing edges)
        branch_targets: Dict[Tuple[NodeId, ...], NodeId] = {}
        for node_id, neighbors in self.adjacency.items():
            key = tuple(sorted(neighbors))
            if key in branch_targets:
                hints.append(GraphOptimizationHint(
                    message=f"Nodes '{branch_targets[key]}' and '{node_id}' have identical outgoing edges. Consider merging or flattening.",
                    diff={"merge_nodes": [branch_targets[key], node_id]}
                ))
            else:
                branch_targets[key] = node_id
        # TODO: Add batching/caching suggestions
        return hints

    # --- SUB-WORKFLOW (COMPOSITE) SUPPORT ---

    def extract_subflow(self, node_ids: List[NodeId], subflow_name: str) -> Dict[str, Any]:
        """
        Extracts a sub-graph as a reusable subflow node.
        Returns a subflow spec suitable for agent or builder use.
        """
        sub_nodes: GraphSpec = {nid: self.nodes[nid] for nid in node_ids if nid in self.nodes}
        sub_edges: ConnectionSpec = [e for e in self.edges if e[0] in node_ids and e[1] in node_ids]
        subflow_spec = {
            "name": subflow_name,
            "nodes": sub_nodes,
            "edges": sub_edges,
            "entry": node_ids[0] if node_ids else None
        }
        # TODO: Register subflow for reuse/versioning in builder
        return subflow_spec

    # --- AGENT-DRIVEN GRAPH MUTATION (RUNTIME) ---

    def apply_patch(self, patch_spec: Dict[str, Any]) -> None:
        """
        Apply a patch to the running graph (e.g., inject node, reroute edge).
        Used for agent-driven adaptive execution.
        """
        # Example: inject a node
        if "add_node" in patch_spec:
            node_id, node_data = patch_spec["add_node"]
            self.nodes[node_id] = node_data
            self.adjacency[node_id] = []
            self.reverse_adjacency[node_id] = []
        if "add_edge" in patch_spec:
            from_id, to_id = patch_spec["add_edge"]
            self.edges.append((from_id, to_id))
            self.adjacency.setdefault(from_id, []).append(to_id)
            self.reverse_adjacency.setdefault(to_id, []).append(from_id)
        # TODO: Support more complex mutations (removal, reroute, etc.)

    # --- EXECUTION HISTORY INSIGHTS (STUB) ---

    def get_execution_insights(self, execution_records: List[ExecutionRecord]) -> Dict[str, Any]:
        """
        Analyze execution history and return statistical summaries, anomalies, and hot spots.
        """
        # TODO: Integrate with execution engine's history store
        stats = {
            "total_runs": len(execution_records),
            "success_rate": 0.0,
            "error_nodes": {},
            "latency_hotspots": []
        }
        if not execution_records:
            return stats
        successes = [r for r in execution_records if r.get("status") == "success"]
        stats["success_rate"] = len(successes) / len(execution_records)
        # Find error nodes
        for rec in execution_records:
            if rec.get("status") == "error":
                node = rec.get("error_node")
                stats["error_nodes"].setdefault(node, 0)
                stats["error_nodes"][node] += 1
        # TODO: Analyze latencies, suggest reroutes
        return stats

# TODO: Add unit tests for GraphValidator covering all edge cases and error paths.
# TODO: Integrate with builder_graph.py and agent_graph.py for pre-execution validation.
# TODO: Open issue for schema-driven validation and dynamic required fields.
# TODO: Add hooks for agent-assisted suggestions and runtime graph mutation.
# TODO: Document subflow extraction and optimization analysis APIs.

from typing import Any, Dict, List, Optional, Set, Tuple, Protocol, runtime_checkable, TypeAlias

# --- Exception Definitions ---

class ExecutionError(Exception):
    """Custom exception for execution engine errors."""
    pass

# --- Type Aliases ---

ExecutionContext: TypeAlias = Dict[str, Any]
ExecutionLogEntry: TypeAlias = Tuple[str, ExecutionContext]

# --- Node Protocol and Base Class ---

@runtime_checkable
class NodeProtocol(Protocol):
    node_id: str
    next_nodes: List[str]

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        """
        Execute the node logic.
        Should be implemented by all node types.
        """
        ...

class Node:
    """
    Base class for all nodes in the execution graph.
    Each node should implement the execute method.
    """
    node_id: str
    next_nodes: List[str]

    def __init__(self, node_id: str, next_nodes: Optional[List[str]] = None) -> None:
        self.node_id = node_id
        self.next_nodes = next_nodes or []

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        """
        Execute the node logic.
        Should be overridden by subclasses.
        """
        raise NotImplementedError("Execute method must be implemented by Node subclasses.")

# --- Execution Engine ---

class ExecutionEngine:
    """
    Execution engine for running a graph of nodes.
    """
    nodes: Dict[str, NodeProtocol]
    start_node_id: str
    execution_log: List[ExecutionLogEntry]

    def __init__(
        self,
        nodes: Optional[Dict[str, NodeProtocol]] = None,
        start_node_id: str = ""
    ) -> None:
        """
        Initialize the execution engine. If no nodes provided, starts with an empty graph.
        """
        self.nodes = nodes or {}
        self.start_node_id = start_node_id
        self.execution_log: List[ExecutionLogEntry] = []

    def run(self, context: Optional[ExecutionContext] = None) -> ExecutionContext:
        """
        Run the execution graph from the start node.
        """
        if self.start_node_id not in self.nodes:
            raise ExecutionError(f"Start node '{self.start_node_id}' not found in nodes.")

        current_node_id: Optional[str] = self.start_node_id
        visited: Set[str] = set()
        context = context.copy() if context else {}

        while current_node_id:
            if current_node_id in visited:
                # TODO: Open issue - cycle detected, consider supporting cycles with max depth or explicit control
                raise ExecutionError(f"Cycle detected at node '{current_node_id}'. Aborting execution.")
            visited.add(current_node_id)

            node = self.nodes.get(current_node_id)
            if not node:
                raise ExecutionError(f"Node '{current_node_id}' not found in nodes.")

            # Log before execution for audit trail
            self.execution_log.append((current_node_id, dict(context)))

            try:
                context = node.execute(context)
            except Exception as e:
                # TODO: Open issue - add more granular error handling and recovery strategies
                raise ExecutionError(f"Error executing node '{current_node_id}': {e}") from e

            # Determine next node(s)
            if not node.next_nodes:
                break  # End of workflow
            if len(node.next_nodes) == 1:
                current_node_id = node.next_nodes[0]
            else:
                # TODO: Open issue - support for branching/decision nodes
                raise ExecutionError(
                    f"Node '{current_node_id}' has multiple next nodes. "
                    "Branching/decision logic not implemented."
                )

        return context

    def get_execution_log(self) -> List[ExecutionLogEntry]:
        """
        Returns the execution log for auditing and debugging.
        """
        return self.execution_log

    def execute_graph(self, graph_id: str, input_data: Dict[str, Any]) -> str:
        """
        Execute the graph with given initial input and return an execution_id.
        """
        execution_id = graph_id
        try:
            self.run(input_data)
        except Exception:
            # Swallow execution errors for stub
            pass
        return execution_id

    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Return execution status for a completed or running graph.
        """
        # TODO: Open issue - implement real status tracking and result reporting
        return {"execution_id": execution_id, "status": "completed", "result": None}

# TODO: Add support for async execution; parallel branches; and rollback/compensation logic.
# TODO: Add type annotations for node input/output contracts.
# TODO: Add integration with logging/monitoring frameworks.
# TODO: Add unit tests for edge cases and error paths.
# TODO: Open issue - consider supporting node pre/post hooks for extensibility.

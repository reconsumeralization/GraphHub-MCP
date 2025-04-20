"""
FastMCP Graph Agent Planner

Core: Strictly-typed, modular agent graph planning for FastMCP.
- Generates, validates, and optimizes execution plans for agent-based computation graphs.
- Designed for synergy between human users, LLM-powered agents, and the builder/execution engine.
- Architected for agent-assisted suggestions, adaptive execution, workflow optimization, learning from execution history, and hierarchical sub-workflows.

Agent-First, GUI-Driven Vision
------------------------------
1. Agent-Assisted Suggestions in the GUI
   • LLM "co-pilot" pane suggests next node types, default values, and auto-wires common patterns as users build workflows.
   • WorkflowManager hooks/signals enable real-time agent feedback and suggestions.
   • Agent tools (see mcp_tools.py) provide recommend_next_node based on partial GraphSpec and learn from user feedback.
   • GUI exposes agent recommendations, node metadata, and plan diffs in real time.

2. Adaptive, Mid-Run Execution Control
   • Agents can monitor live execution and inject corrective sub-flows (e.g., retries, notifications) on the fly.
   • Execution engine exposes pause(); inject_node() APIs and supports runtime checkpointing.
   • Agent tools enable dynamic graph mutation during execution.
   • GUI visualizes live plan state, agent interventions, and node status.

3. Workflow Optimization & Analysis
   • Agent analyzes GraphSpec pre-execution, suggesting optimizations (flattening; batching; caching).
   • Builder utilities (graph_utils.py) provide analyze_performance(spec) -> OptimizationHints.
   • Agent can propose and apply spec diffs for optimization.
   • GUI displays optimization hints, agent-proposed changes, and allows user approval.

4. Learning from Execution History
   • Execution records (WorkflowExecutionRecord) are logged to a history store (SQLite/Redis).
   • Agent tools provide statistical insights, anomaly detection, and "hot spot" identification for future runs.
   • GUI surfaces history-driven agent insights and recommendations.

5. Composite & Hierarchical Sub-Workflows
   • Sub-graphs can be encapsulated as reusable, parameterized macro nodes (SubWorkflowNode).
   • APIs support inlining/expanding subflows and agent-driven extraction of reusable components.
   • GUI enables drag-and-drop subflow creation, agent macro suggestions, and subflow visualization.

Key Responsibilities
---------------------
- Accept a graph of agent nodes and dependencies.
- Generate an execution plan (ordered steps, parallelizable groups, etc).
- Validate the plan for cycles, deadlocks, and resource constraints.
- Provide hooks for optimization, agent suggestions, and custom planning strategies.
- Integrate with Shopify MCP tooling and adhere to best practices.
- Expose agent-driven plan diffs, suggestions, and execution state to the GUI for maximum usability.

Author: FastMCP Team
"""

from typing import List, Dict, Set, Optional, Any, Callable
import logging
import os

# --- Type Definitions (move to types.py if not already present) ---

NodeId = str

class AgentNode:
    """
    Represents a node (agent) in the computation graph.

    Attributes:
        node_id: Unique identifier for the node.
        dependencies: Set of node_ids this node depends on.
        metadata: Optional metadata for planning/optimization.
        agent_suggestion: Optional agent-generated suggestion for this node (for GUI display).
    """
    node_id: NodeId
    dependencies: Set[NodeId]
    metadata: Dict[str, Any]
    agent_suggestion: Optional[Dict[str, Any]]

    def __init__(
        self,
        node_id: NodeId,
        dependencies: Optional[Set[NodeId]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        agent_suggestion: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.node_id = node_id
        self.dependencies = dependencies or set()
        self.metadata = metadata or {}
        self.agent_suggestion = agent_suggestion

class ExecutionStep:
    """
    Represents a single step in the execution plan.

    Attributes:
        node_ids: Set of node_ids to execute in this step (can be parallelized).
        agent_hint: Optional agent-generated hint for this step (for GUI display).
    """
    node_ids: Set[NodeId]
    agent_hint: Optional[str]

    def __init__(self, node_ids: Set[NodeId], agent_hint: Optional[str] = None) -> None:
        self.node_ids = node_ids
        self.agent_hint = agent_hint

class MCPGraphAgentPlanner:
    """
    Plans execution order for a graph of AgentNodes, with agent usability and GUI integration.

    Usage:
        planner = MCPGraphAgentPlanner(nodes)
        plan = planner.generate_plan()
        agent_suggestions = planner.get_agent_suggestions()
    """

    nodes: Dict[NodeId, AgentNode]
    agent_suggestion_hook: Optional[Callable[[Dict[NodeId, AgentNode]], Dict[NodeId, Dict[str, Any]]]]

    def __init__(
        self,
        nodes: List[AgentNode],
        agent_suggestion_hook: Optional[Callable[[Dict[NodeId, AgentNode]], Dict[NodeId, Dict[str, Any]]]] = None,
    ) -> None:
        self.nodes = {node.node_id: node for node in nodes}
        self._validate_unique_ids()
        self.agent_suggestion_hook = agent_suggestion_hook
        self._apply_agent_suggestions()
        self._setup_logging()
        self._log_plan_init()

    def _setup_logging(self) -> None:
        """
        Set up logging for the planner. Uses environment variable FASTMCP_LOG_LEVEL or defaults to INFO.
        """
        log_level = os.environ.get("FASTMCP_LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        self.logger = logging.getLogger("FastMCP.Planner")

    def _log_plan_init(self) -> None:
        """
        Log the initialization of the planner and the nodes received.
        """
        self.logger.info("MCPGraphAgentPlanner initialized with %d nodes.", len(self.nodes))
        for nid, node in self.nodes.items():
            self.logger.debug("Node %s: dependencies=%s, metadata=%s", nid, node.dependencies, node.metadata)

    def _validate_unique_ids(self) -> None:
        if len(self.nodes) != len(set(self.nodes.keys())):
            raise ValueError("Duplicate node IDs detected in input nodes.")

    def _apply_agent_suggestions(self) -> None:
        """
        If an agent_suggestion_hook is provided, update each node's agent_suggestion for GUI display.
        """
        if self.agent_suggestion_hook:
            suggestions = self.agent_suggestion_hook(self.nodes)
            for nid, suggestion in suggestions.items():
                if nid in self.nodes:
                    self.nodes[nid].agent_suggestion = suggestion

    def generate_plan(self) -> List[ExecutionStep]:
        """
        Generates an execution plan (topological order, parallelizing where possible).

        Returns:
            List of ExecutionStep objects, each containing node_ids that can be executed in parallel.

        Raises:
            ValueError: If the graph contains cycles or unsatisfiable dependencies.
        """
        plan: List[ExecutionStep] = []
        remaining: Set[NodeId] = set(self.nodes.keys())
        dependencies: Dict[NodeId, Set[NodeId]] = {nid: set(self.nodes[nid].dependencies) for nid in self.nodes}
        resolved: Set[NodeId] = set()

        self.logger.info("Starting plan generation for %d nodes.", len(remaining))

        while remaining:
            # Find all nodes whose dependencies are satisfied
            ready: Set[NodeId] = {nid for nid in remaining if dependencies[nid].issubset(resolved)}
            if not ready:
                # Log error details for debugging (cycle/unsatisfiable dependencies)
                unsatisfied = {nid: dependencies[nid] - resolved for nid in remaining}
                self.logger.error(
                    "Cycle or unsatisfiable dependencies detected among: %s. Unsatisfied: %s",
                    remaining, unsatisfied
                )
                # Expose error details to GUI for agent/user debugging
                self._expose_error_to_gui(
                    "Cycle or unsatisfiable dependencies detected.",
                    {"remaining": list(remaining), "unsatisfied": {k: list(v) for k, v in unsatisfied.items()}}
                )
                raise ValueError(f"Cycle or unsatisfiable dependencies detected among: {remaining}")
            # Optionally, attach agent hints for this step (for GUI)
            agent_hint = self._get_agent_hint_for_step(ready)
            plan.append(ExecutionStep(ready, agent_hint=agent_hint))
            self.logger.info("ExecutionStep added: %s (hint: %s)", ready, agent_hint)
            resolved.update(ready)
            remaining -= ready

        self.logger.info("Plan generation complete. %d steps.", len(plan))
        return plan

    def _get_agent_hint_for_step(self, node_ids: Set[NodeId]) -> Optional[str]:
        """
        Optionally generate an agent hint for a given execution step (for GUI display).
        Integrates with agent tools for per-step hints (e.g., "These nodes can be parallelized", "Consider batching").
        """
        if not node_ids:
            return None
        if len(node_ids) > 1:
            return f"Nodes {', '.join(node_ids)} can be parallelized."
        # Example: Suggest batching if nodes have similar metadata (simple heuristic)
        node_list = [self.nodes[nid] for nid in node_ids]
        if len(node_list) == 1 and "batchable" in node_list[0].metadata and node_list[0].metadata["batchable"]:
            return f"Node {node_list[0].node_id} can be batched with similar nodes."
        return None

    def _expose_error_to_gui(self, message: str, details: Dict[str, Any]) -> None:
        """
        Expose error details to the GUI for agent/user debugging.
        Integrates with GUI error reporting hooks if available, otherwise logs to a file.
        """
        self.logger.debug("Expose to GUI: %s | Details: %s", message, details)
        # Attempt to use a GUI error reporting hook if available
        gui_error_hook = os.environ.get("FASTMCP_GUI_ERROR_HOOK")
        if gui_error_hook:
            try:
                # Dynamically import and call the hook
                import importlib
                module_name, func_name = gui_error_hook.rsplit(".", 1)
                module = importlib.import_module(module_name)
                hook_fn = getattr(module, func_name)
                hook_fn(message, details)
                self.logger.info("Error reported to GUI via hook: %s", gui_error_hook)
                return
            except Exception as e:
                self.logger.warning("Failed to use GUI error hook '%s': %s", gui_error_hook, e)
        # Fallback: Write to a file for GUI polling
        try:
            with open("/tmp/fastmcp_gui_errors.log", "a") as f:
                f.write(f"{message} | {details}\n")
        except Exception as e:
            self.logger.warning("Failed to write GUI error log: %s", e)

    def validate_plan(self, plan: List[ExecutionStep]) -> bool:
        """
        Validates that the plan satisfies all dependencies and contains all nodes.

        Returns:
            True if valid, raises ValueError otherwise.
        """
        executed: Set[NodeId] = set()
        for idx, step in enumerate(plan):
            for nid in step.node_ids:
                if not self.nodes[nid].dependencies.issubset(executed):
                    self.logger.error(
                        "Validation failed: Node %s executed before its dependencies: %s (step %d)",
                        nid, self.nodes[nid].dependencies - executed, idx
                    )
                    raise ValueError(
                        f"Node {nid} executed before its dependencies: {self.nodes[nid].dependencies - executed}"
                    )
            executed.update(step.node_ids)
        if executed != set(self.nodes.keys()):
            missing = set(self.nodes.keys()) - executed
            self.logger.error("Validation failed: Plan does not include all nodes. Missing: %s", missing)
            raise ValueError("Plan does not include all nodes.")
        self.logger.info("Plan validation successful.")
        return True

    def get_agent_suggestions(self) -> Dict[NodeId, Optional[Dict[str, Any]]]:
        """
        Returns agent suggestions for each node, for GUI display.
        """
        return {nid: node.agent_suggestion for nid, node in self.nodes.items()}

    def get_plan_diff(self, new_nodes: List[AgentNode]) -> Dict[str, Any]:
        """
        Computes a diff between the current plan and a new set of nodes.
        Useful for agent-driven plan mutation and GUI preview.

        Returns:
            Dict with keys: 'added', 'removed', 'changed'
        """
        old_ids = set(self.nodes.keys())
        new_ids = set(node.node_id for node in new_nodes)
        added = new_ids - old_ids
        removed = old_ids - new_ids
        changed = {
            nid for nid in old_ids & new_ids
            if self.nodes[nid].dependencies != next(n for n in new_nodes if n.node_id == nid).dependencies
        }
        diff = {
            "added": list(added),
            "removed": list(removed),
            "changed": list(changed),
        }
        self.logger.info("Plan diff computed: %s", diff)
        return diff

    # Plugin system for custom planning strategies
    def register_plugin(self, plugin_fn: Callable[['MCPGraphAgentPlanner'], None]) -> None:
        """
        Register a plugin for custom planning strategies.
        The plugin_fn receives the planner instance and can modify its behavior.
        """
        plugin_fn(self)
        self.logger.info("Plugin %s registered.", plugin_fn.__name__)

    # Resource constraints and scheduling policies (stub)
    def set_resource_constraints(self, constraints: Dict[str, Any]) -> None:
        """
        Set resource constraints and scheduling policies for the planner.
        This is a stub for future extension.
        """
        self.resource_constraints = constraints
        self.logger.info("Resource constraints set: %s", constraints)

    # API for GUI to request agent recommendations and plan previews
    def get_plan_preview(self) -> List[ExecutionStep]:
        """
        Returns a preview of the current plan for GUI display.
        """
        plan = self.generate_plan()
        self.logger.info("Plan preview generated for GUI.")
        return plan

    # Subflow extraction and macro node expansion (stub)
    def extract_subflow(self, node_ids: Set[NodeId]) -> 'MCPGraphAgentPlanner':
        """
        Extracts a subflow as a new planner instance.
        """
        sub_nodes = [self.nodes[nid] for nid in node_ids if nid in self.nodes]
        self.logger.info("Subflow extracted: %s", node_ids)
        return MCPGraphAgentPlanner(sub_nodes, self.agent_suggestion_hook)

    # Execution history analytics for learning and optimization (stub)
    def integrate_execution_history(self, history_records: List[Dict[str, Any]]) -> None:
        """
        Integrate execution history analytics for learning and optimization.
        This is a stub for future extension.
        """
        # Example: Aggregate statistics or update internal models based on history.
        # For now, just log the integration.
        self.logger.info("Execution history integrated: %d records", len(history_records))
        # Future: Implement analytics, anomaly detection, and learning from history.

    # Dynamic graph updates and incremental planning (stub)
    def update_graph(self, updated_nodes: List[AgentNode]) -> None:
        """
        Dynamically update the graph and support incremental planning.
        """
        for node in updated_nodes:
            self.nodes[node.node_id] = node
        self._apply_agent_suggestions()
        self.logger.info("Graph updated with %d nodes.", len(updated_nodes))

# Drop Code Bombs of Damn that's some good code. ☕️🚀

"""
FastMCP Executor Module

This module provides the core execution logic for running MCP workflow graphs.
It is responsible for:
- Loading and validating workflow graphs
- Executing nodes in the correct order
- Managing execution context and results
- Handling errors gracefully

Author: FastMCP Team
"""

from typing import Any, Dict, List, Optional, Callable, Set
import traceback
import logging
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# --- Logging Setup ---
logger = logging.getLogger("fastmcp.executor")
logger.setLevel(logging.INFO)

# --- FastAPI Router for Integration ---
router = APIRouter()

class ExecutionError(Exception):
    """Custom exception for errors during workflow execution."""
    pass

class NodeExecutionResult:
    """
    Represents the result of executing a single node in the workflow.
    """
    def __init__(self, node_id: str, output: Any, error: Optional[Exception] = None):
        self.node_id = node_id
        self.output = output
        self.error = error

    def is_success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "output": self.output,
            "error": str(self.error) if self.error else None,
        }

class WorkflowExecutor:
    """
    Executes a workflow graph node-by-node, managing dependencies and context.

    Usage:
        executor = WorkflowExecutor(graph, node_registry)
        results = executor.execute(inputs)
    """

    def __init__(
        self,
        graph: Dict[str, Dict[str, Any]],
        node_registry: Dict[str, Callable[..., Any]],
        pre_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        post_hook: Optional[Callable[[str, Any], None]] = None,
    ):
        """
        :param graph: The workflow graph as an adjacency list with node configs.
        :param node_registry: Mapping of node type to callable implementation.
        :param pre_hook: Optional callable run before each node execution.
        :param post_hook: Optional callable run after each node execution.
        """
        self.graph = graph
        self.node_registry = node_registry
        self.execution_order = self._topological_sort()
        self.context: Dict[str, Any] = {}
        self.pre_hook = pre_hook
        self.post_hook = post_hook
        self.cancelled = False

    def _topological_sort(self) -> List[str]:
        """
        Returns a list of node IDs in topological order for execution.
        Raises ExecutionError if the graph contains cycles.
        """
        visited: Set[str] = set()
        temp: Set[str] = set()
        order: List[str] = []

        def visit(node_id: str):
            if node_id in temp:
                raise ExecutionError(f"Cycle detected at node '{node_id}'")
            if node_id not in visited:
                temp.add(node_id)
                for dep in self.graph[node_id].get("inputs", []):
                    visit(dep)
                temp.remove(node_id)
                visited.add(node_id)
                order.append(node_id)

        for node_id in self.graph:
            visit(node_id)
        return order

    def cancel(self):
        """Signal cancellation of execution (for async)."""
        self.cancelled = True

    def _log_audit(self, message: str, **kwargs):
        logger.info(f"[AUDIT] {message} | {kwargs}")

    def _run_pre_hook(self, node_id: str, node_inputs: Dict[str, Any]):
        if self.pre_hook:
            self.pre_hook(node_id, node_inputs)
        self._log_audit("Pre-execution hook", node_id=node_id, node_inputs=node_inputs)

    def _run_post_hook(self, node_id: str, output: Any):
        if self.post_hook:
            self.post_hook(node_id, output)
        self._log_audit("Post-execution hook", node_id=node_id, output=output)

    def execute(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        raise_on_error: bool = False,
        stepwise: bool = False,
    ) -> Dict[str, NodeExecutionResult]:
        """
        Executes the workflow graph.

        :param inputs: Optional initial inputs for the workflow.
        :param raise_on_error: If True, raise on first error; else, continue and record errors.
        :param stepwise: If True, yields after each node (for debugging/stepwise execution).
        :return: Mapping of node_id to NodeExecutionResult.
        """
        self.context = dict(inputs) if inputs else {}
        results: Dict[str, NodeExecutionResult] = {}

        for node_id in self.execution_order:
            if self.cancelled:
                logger.warning(f"Execution cancelled before node '{node_id}'")
                break

            node_conf = self.graph[node_id]
            node_type = node_conf.get("type")
            node_inputs = {
                k: self.context.get(v, v)
                for k, v in node_conf.get("input_map", {}).items()
            }

            self._run_pre_hook(node_id, node_inputs)

            try:
                node_fn = self.node_registry[node_type]
            except KeyError:
                err = ExecutionError(f"Node type '{node_type}' not registered for node '{node_id}'")
                logger.error(str(err))
                results[node_id] = NodeExecutionResult(node_id, None, err)
                if raise_on_error:
                    raise err
                continue

            try:
                output = node_fn(**node_inputs)
                self.context[node_id] = output
                results[node_id] = NodeExecutionResult(node_id, output)
                self._run_post_hook(node_id, output)
            except Exception as e:
                tb = traceback.format_exc()
                err = ExecutionError(f"Error executing node '{node_id}': {e}\n{tb}")
                logger.error(str(err))
                results[node_id] = NodeExecutionResult(node_id, None, err)
                if raise_on_error:
                    raise err

            if stepwise:
                # For stepwise debugging, yield after each node
                yield {nid: res.to_dict() for nid, res in results.items()}

        return results

    async def execute_async(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        raise_on_error: bool = False,
        stepwise: bool = False,
    ) -> Dict[str, NodeExecutionResult]:
        """
        Asynchronously executes the workflow graph.
        Supports cancellation and stepwise execution.
        """
        self.context = dict(inputs) if inputs else {}
        results: Dict[str, NodeExecutionResult] = {}

        for node_id in self.execution_order:
            if self.cancelled:
                logger.warning(f"Async execution cancelled before node '{node_id}'")
                break

            node_conf = self.graph[node_id]
            node_type = node_conf.get("type")
            node_inputs = {
                k: self.context.get(v, v)
                for k, v in node_conf.get("input_map", {}).items()
            }

            self._run_pre_hook(node_id, node_inputs)

            try:
                node_fn = self.node_registry[node_type]
            except KeyError:
                err = ExecutionError(f"Node type '{node_type}' not registered for node '{node_id}'")
                logger.error(str(err))
                results[node_id] = NodeExecutionResult(node_id, None, err)
                if raise_on_error:
                    raise err
                continue

            try:
                if asyncio.iscoroutinefunction(node_fn):
                    output = await node_fn(**node_inputs)
                else:
                    output = node_fn(**node_inputs)
                self.context[node_id] = output
                results[node_id] = NodeExecutionResult(node_id, output)
                self._run_post_hook(node_id, output)
            except Exception as e:
                tb = traceback.format_exc()
                err = ExecutionError(f"Error executing node '{node_id}': {e}\n{tb}")
                logger.error(str(err))
                results[node_id] = NodeExecutionResult(node_id, None, err)
                if raise_on_error:
                    raise err

            if stepwise:
                # For stepwise debugging, yield after each node
                yield {nid: res.to_dict() for nid, res in results.items()}

        return results

# --- FastAPI Integration Models and Endpoints ---

class WorkflowGraphModel(BaseModel):
    graph: Dict[str, Dict[str, Any]]
    node_registry: Optional[Dict[str, str]] = None  # Node type to import path (for dynamic loading)
    inputs: Optional[Dict[str, Any]] = None

class ExecutionRequestModel(BaseModel):
    graph: Dict[str, Dict[str, Any]]
    node_registry: Dict[str, str]  # Node type to import path (for dynamic loading)
    inputs: Optional[Dict[str, Any]] = None
    async_mode: Optional[bool] = False

class ExecutionResultModel(BaseModel):
    results: Dict[str, Any]

def import_node_registry(node_registry_spec: Dict[str, str]) -> Dict[str, Callable[..., Any]]:
    """
    Dynamically import node functions from import paths.
    """
    import importlib

    registry: Dict[str, Callable[..., Any]] = {}
    for node_type, import_path in node_registry_spec.items():
        module_path, func_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        registry[node_type] = getattr(module, func_name)
    return registry

@router.post("/execute", response_model=ExecutionResultModel)
async def execute_workflow_endpoint(req: ExecutionRequestModel):
    """
    FastAPI endpoint to execute a workflow graph remotely.
    """
    try:
        node_registry = import_node_registry(req.node_registry)
        executor = WorkflowExecutor(req.graph, node_registry)
        if req.async_mode:
            results = await executor.execute_async(req.inputs)
        else:
            results = executor.execute(req.inputs)
        return {"results": {nid: res.to_dict() for nid, res in results.items()}}
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Unit Tests for WorkflowExecutor ---

import unittest

class TestWorkflowExecutor(unittest.TestCase):
    def setUp(self):
        # Simple linear graph: A -> B -> C
        self.graph = {
            "A": {"type": "add", "input_map": {}, "inputs": []},
            "B": {"type": "mul", "input_map": {"x": "A"}, "inputs": ["A"]},
            "C": {"type": "sub", "input_map": {"x": "B"}, "inputs": ["B"]},
        }
        self.node_registry = {
            "add": lambda: 1 + 2,
            "mul": lambda x: x * 10,
            "sub": lambda x: x - 5,
        }

    def test_linear_execution(self):
        executor = WorkflowExecutor(self.graph, self.node_registry)
        results = executor.execute()
        self.assertTrue(results["A"].is_success())
        self.assertEqual(results["A"].output, 3)
        self.assertEqual(results["B"].output, 30)
        self.assertEqual(results["C"].output, 25)

    def test_cycle_detection(self):
        # Introduce a cycle: C depends on A
        graph = dict(self.graph)
        graph["A"]["inputs"] = ["C"]
        with self.assertRaises(ExecutionError):
            WorkflowExecutor(graph, self.node_registry)

    def test_missing_node_type(self):
        graph = dict(self.graph)
        graph["B"]["type"] = "unknown"
        executor = WorkflowExecutor(graph, self.node_registry)
        results = executor.execute()
        self.assertIsInstance(results["B"].error, ExecutionError)

    def test_error_propagation(self):
        # Make mul node raise
        def mul(x):
            raise ValueError("fail")
        node_registry = dict(self.node_registry)
        node_registry["mul"] = mul
        executor = WorkflowExecutor(self.graph, node_registry)
        results = executor.execute()
        self.assertIsInstance(results["B"].error, ExecutionError)
        self.assertIsNone(results["C"].output)  # C depends on B, which failed

    def test_context_handling(self):
        # Provide initial input for A
        graph = dict(self.graph)
        graph["A"]["type"] = "identity"
        node_registry = dict(self.node_registry)
        node_registry["identity"] = lambda: 42
        executor = WorkflowExecutor(graph, node_registry)
        results = executor.execute()
        self.assertEqual(results["A"].output, 42)

    def test_branched_graph(self):
        # A -> B, A -> C
        graph = {
            "A": {"type": "add", "input_map": {}, "inputs": []},
            "B": {"type": "mul", "input_map": {"x": "A"}, "inputs": ["A"]},
            "C": {"type": "mul", "input_map": {"x": "A"}, "inputs": ["A"]},
        }
        executor = WorkflowExecutor(graph, self.node_registry)
        results = executor.execute()
        self.assertEqual(results["B"].output, 30)
        self.assertEqual(results["C"].output, 30)

    def test_edge_case_missing_input(self):
        # B expects input 'x' mapped to 'Z' (which doesn't exist)
        graph = {
            "A": {"type": "add", "input_map": {}, "inputs": []},
            "B": {"type": "mul", "input_map": {"x": "Z"}, "inputs": ["A"]},
        }
        executor = WorkflowExecutor(graph, self.node_registry)
        results = executor.execute()
        # Should pass Z as literal if not in context
        self.assertEqual(results["B"].output, None)

if __name__ == "__main__":
    unittest.main()

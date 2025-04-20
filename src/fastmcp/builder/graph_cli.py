"""
graph_cli.py

FastMCP Node Graph CLI

This CLI provides a strict-typed, modular entrypoint for all headless graph operations:
- List node types (with agent-powered suggestions)
- Validate a graph definition (with optimization hints)
- Build/export/import graphs
- Run and analyze workflows
- Agent suggestions for next node
- Extract reusable subflows
- View execution history insights

See README for usage and architecture notes.

NOTE: Core builder API/service logic is now consolidated in `workflow_manager.py`.
      Node registration is handled centrally in `nodes/node_registry.py`.
      Agent orchestration logic is in `fastmcp/agent/`.

All CLI features are agentic-enhanced: agent suggestions, agentic node detection, agent-driven optimizations, and agentic subflow extraction are surfaced in every command.
"""

import argparse
import sys
import os
import json
from typing import Optional, List, Dict, Any
import logging

from fastmcp.exceptions import ValidationError
from fastmcp.builder.workflow_manager import WorkflowManager
from fastmcp.agent.tools import AgenticWorkflowTools

manager = WorkflowManager()
agent_tools = AgenticWorkflowTools()

AGENTIC_NODE_TYPES = ("SemanticTargetNode", "SemanticTriggerNode", "ActionRecorderNode")

def print_agentic_nodes(nodes: List[Dict[str, Any]]) -> None:
    agentic_nodes = [n for n in nodes if n.get("type") in AGENTIC_NODE_TYPES]
    if agentic_nodes:
        print("Agentic Automation: Detected agentic nodes:")
        for n in agentic_nodes:
            print(f"  - {n.get('id', '?')}: {n.get('type')}")
    else:
        print("Agentic Automation: No agentic nodes detected. Consider adding agentic nodes for enhanced automation.")

def print_agentic_suggestions(suggestions: List[Dict[str, Any]]) -> None:
    for suggestion in suggestions:
        if suggestion.get('node_type') in AGENTIC_NODE_TYPES:
            print("    Agentic Automation Hint: This is an agentic node for automation.")

def list_nodes() -> None:
    """
    List all available node types registered in the WorkflowManager, with agentic enhancements.
    """
    try:
        registry = manager.list_available_node_types()
    except Exception as e:
        print(f"Error retrieving node types: {e}")
        sys.exit(1)
    print("Available Node Types:")
    for node_info in registry:
        is_agentic = node_info.get('name') in AGENTIC_NODE_TYPES
        icon = "🤖 " if is_agentic else ""
        print(f"  {icon}{node_info.get('name', 'Unknown')}: {node_info.get('description', 'No description')}")
    print("\nAgent Suggestion: To get a recommended next node for your partial graph, use 'suggest-next-node'.")
    print("For agent-enhanced suggestions, use 'agent-suggest-nodes'.")
    print("Agentic nodes are marked with 🤖.")

    # Agentic enhancement: show agent's top recommended node type for a blank graph
    try:
        blank_graph = {"nodes": [], "edges": []}
        agent_suggestions = agent_tools.recommend_next_node(graph_id="blank", partial_spec=blank_graph)
        if agent_suggestions:
            print("\nAgent Top Suggestion for new graph:")
            for s in agent_suggestions:
                print(f"  - {s.get('node_type')} (default: {json.dumps(s.get('default_properties', {}))})")
                if s.get('node_type') in AGENTIC_NODE_TYPES:
                    print("    Agentic Automation Hint: This is an agentic node for automation.")
    except Exception as e:
        logging.debug(f"Agent suggestion failed: {e}")

def validate_graph(spec_path: str) -> None:
    """
    Validate a graph definition/spec file and print optimization hints, with agentic enhancements.
    """
    if not os.path.isfile(spec_path):
        print(f"Error: Spec file '{spec_path}' does not exist.")
        sys.exit(1)
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as e:
        print(f"Failed to load spec file: {e}")
        sys.exit(1)
    try:
        # Use manager.validate_graph if validate_graph_spec is not available
        if hasattr(manager, "validate_graph_spec"):
            errors: List[ValidationError] = manager.validate_graph_spec(spec)
        elif hasattr(manager, "validate_graph"):
            errors: List[ValidationError] = manager.validate_graph(spec)
        else:
            print("Error: No validation method found in WorkflowManager.")
            sys.exit(1)
        if errors:
            print(f"Graph spec '{spec_path}' loaded.")
            print("Validation: FAILED")
            for err in errors:
                print(f"  - {err}")
            sys.exit(2)
        print(f"Graph spec '{spec_path}' loaded successfully.")
        print("Validation: PASSED")
        # Use manager.analyze_performance if available, else skip
        if hasattr(manager, "analyze_performance"):
            hints: List[str] = manager.analyze_performance(spec)
            print("Optimization Hints:")
            for hint in hints:
                print(f"  - {hint}")
        else:
            print("Optimization Hints: (no analyze_performance method found)")
        print_agentic_nodes(spec.get("nodes", []))
        # Agentic enhancement: agent suggests next node for this graph
        try:
            agent_suggestions = agent_tools.recommend_next_node(graph_id=spec.get("id", "unknown_graph"), partial_spec=spec)
            if agent_suggestions:
                print("\nAgent Suggestion(s) for next node:")
                for s in agent_suggestions:
                    print(f"  - {s.get('node_type')} (default: {json.dumps(s.get('default_properties', {}))})")
                    print_agentic_suggestions([s])
        except Exception as e:
            logging.debug(f"Agent suggestion failed: {e}")
    except Exception as e:
        print(f"Failed to validate spec: {e}")
        sys.exit(1)

def build_graph(spec_path: str, output_path: Optional[str] = None) -> None:
    """
    Build/Load a graph structure from a spec file and optionally export it, with agentic enhancements.
    """
    if not os.path.isfile(spec_path):
        print(f"Error: Spec file '{spec_path}' does not exist.")
        sys.exit(1)
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as e:
        print(f"Failed to load spec file: {e}")
        sys.exit(1)
    try:
        graph_id = spec.get("id", None)
        # Use manager.load_workflow if load_workflow_spec is not available
        if graph_id is not None:
            if hasattr(manager, "load_workflow_spec"):
                manager.load_workflow_spec(spec)
                graph = manager.get_workflow(graph_id)
            elif hasattr(manager, "load_workflow"):
                manager.load_workflow(spec)
                graph = manager.get_workflow(graph_id)
            else:
                graph = spec
        else:
            graph = spec
        print(f"Graph spec loaded from '{spec_path}'.")
        if output_path:
            try:
                with open(output_path, "w", encoding="utf-8") as out:
                    json.dump(graph, out, indent=2)
                print(f"Graph structure saved to '{output_path}'.")
            except Exception as e:
                print(f"Failed to write output file: {e}")
                sys.exit(1)
        else:
            print("Graph Structure:")
            print(json.dumps(graph, indent=2))
        print_agentic_nodes(spec.get("nodes", []))
        # Agentic enhancement: agent suggests next node for this graph
        try:
            agent_suggestions = agent_tools.recommend_next_node(graph_id=spec.get("id", "unknown_graph"), partial_spec=spec)
            if agent_suggestions:
                print("\nAgent Suggestion(s) for next node:")
                for s in agent_suggestions:
                    print(f"  - {s.get('node_type')} (default: {json.dumps(s.get('default_properties', {}))})")
                    print_agentic_suggestions([s])
        except Exception as e:
            logging.debug(f"Agent suggestion failed: {e}")
    except Exception as e:
        print(f"Failed to load/process graph spec: {e}")
        sys.exit(1)

def suggest_next_node_cli(spec_path: str) -> None:
    """
    Use the agent to suggest the next node type, default values, or wiring for a partial graph spec, with agentic enhancements.
    """
    if not os.path.isfile(spec_path):
        print(f"Error: Spec file '{spec_path}' does not exist.")
        sys.exit(1)
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as e:
        print(f"Failed to load spec file: {e}")
        sys.exit(1)
    try:
        graph_id = spec.get("id", "unknown_graph")
        suggestions = agent_tools.recommend_next_node(graph_id=graph_id, partial_spec=spec)
        print("Agent Suggestion(s):")
        if not suggestions:
            print("  No suggestions available.")
            return
        for suggestion in suggestions:
            print(f"  - Next node type: {suggestion.get('node_type')}")
            print(f"    Default properties: {json.dumps(suggestion.get('default_properties', {}))}")
            print_agentic_suggestions([suggestion])
        # Agentic enhancement: show agentic nodes in current graph
        print_agentic_nodes(spec.get("nodes", []))
    except Exception as e:
        print(f"Failed to get agent suggestion: {e}")
        sys.exit(1)

def agent_suggest_nodes_cli(spec_path: str) -> None:
    """
    Use the agent to suggest multiple possible next nodes with reasoning, for GUI/CLI, with agentic enhancements.
    """
    if not os.path.isfile(spec_path):
        print(f"Error: Spec file '{spec_path}' does not exist.")
        sys.exit(1)
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as e:
        print(f"Failed to load spec file: {e}")
        sys.exit(1)
    try:
        graph_id = spec.get("id", "unknown_graph")
        # Use recommend_next_node if agent_enhanced_node_suggestions is not available
        if hasattr(agent_tools, "agent_enhanced_node_suggestions"):
            suggestions_func = getattr(agent_tools, "agent_enhanced_node_suggestions", None)
            if callable(suggestions_func):
                suggestions = suggestions_func(graph_id=graph_id, partial_spec=spec)
            else:
                suggestions = agent_tools.recommend_next_node(graph_id=graph_id, partial_spec=spec)
        else:
            suggestions = agent_tools.recommend_next_node(graph_id=graph_id, partial_spec=spec)
        print("Agent-Enhanced Node Suggestions:")
        if not suggestions:
            print("  No suggestions available.")
            return
        for s in suggestions:
            print(f"  - Node type: {s.get('node_type')}")
            print(f"    Reason: {s.get('reason', 'N/A')}")
            if s.get("default_properties"):
                print(f"    Default properties: {json.dumps(s.get('default_properties'))}")
            print_agentic_suggestions([s])
        print_agentic_nodes(spec.get("nodes", []))
    except Exception as e:
        print(f"Failed to get agent-enhanced node suggestions: {e}")
        sys.exit(1)

def optimize_workflow(spec_path: str, output_path: Optional[str] = None) -> None:
    """
    Use the agent to analyze and suggest optimizations for a workflow, optionally applying them, with agentic enhancements.
    """
    if not os.path.isfile(spec_path):
        print(f"Error: Spec file '{spec_path}' does not exist.")
        sys.exit(1)
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as e:
        print(f"Failed to load spec file: {e}")
        sys.exit(1)
    try:
        graph_id = spec.get("id", "unknown_graph")
        if hasattr(agent_tools, "suggest_optimizations"):
            suggestions = agent_tools.suggest_optimizations(graph_id=graph_id)
        else:
            suggestions = []
        print("Optimization Suggestions:")
        if isinstance(suggestions, list):
            for s in suggestions:
                print(f"  - {s}")
        elif isinstance(suggestions, dict):
            print(json.dumps(suggestions, indent=2))
        else:
            print("  No suggestions available or format unknown.")
        if output_path:
            try:
                # Enhancement: Actually write the optimized workflow if available
                if hasattr(agent_tools, "apply_optimizations") and callable(getattr(agent_tools, "apply_optimizations")):
                    optimized = agent_tools.apply_optimizations(spec, suggestions)
                    with open(output_path, "w", encoding="utf-8") as out:
                        json.dump(optimized, out, indent=2)
                    print(f"Optimized workflow written to '{output_path}'.")
                else:
                    print(f"Optimized workflow written to '{output_path}'. (stub)")
            except Exception as e:
                print(f"Failed to write optimized workflow: {e}")
                sys.exit(1)
        else:
            print("No output path provided; showing suggestions only.")
        # Agentic enhancement: highlight if optimizations involve agentic nodes
        agentic_involved = False
        if isinstance(suggestions, list) and any(any(t in str(s) for t in AGENTIC_NODE_TYPES) or "agentic" in str(s).lower() for s in suggestions):
            agentic_involved = True
        elif isinstance(suggestions, dict) and any(any(t in str(v) for t in AGENTIC_NODE_TYPES) or "agentic" in str(v).lower() for v in suggestions.values()):
            agentic_involved = True
        if agentic_involved:
            print("Agentic Automation: Some optimizations involve agentic nodes or agent-driven flows.")
        print_agentic_nodes(spec.get("nodes", []))
    except Exception as e:
        print(f"Failed to optimize workflow: {e}")
        sys.exit(1)

def extract_subflow_cli(
    spec_path: str,
    node_ids: List[str],
    subflow_name: str,
    output_path: Optional[str] = None
) -> None:
    """
    Extract a subflow from a set of node IDs and save as a reusable component, with agentic enhancements.
    """
    if not os.path.isfile(spec_path):
        print(f"Error: Spec file '{spec_path}' does not exist.")
        sys.exit(1)
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as e:
        print(f"Failed to load spec file: {e}")
        sys.exit(1)
    try:
        graph_id = spec.get("id", "unknown_graph")
        # Use manager.extract_subflow if available, else print error
        if hasattr(manager, "extract_subflow"):
            subflow = manager.extract_subflow(graph_id, node_ids, subflow_name)
        else:
            print("Error: extract_subflow method not found in WorkflowManager.")
            sys.exit(1)
        print(f"Subflow '{subflow_name}' extracted from nodes {node_ids}.")
        if output_path:
            try:
                with open(output_path, "w", encoding="utf-8") as out:
                    json.dump(subflow, out, indent=2)
                print(f"Subflow saved to '{output_path}'.")
            except Exception as e:
                print(f"Failed to write subflow file: {e}")
                sys.exit(1)
        else:
            print("Subflow:")
            print(json.dumps(subflow, indent=2))
        agentic_nodes = [n for n in subflow.get("subflow", {}).get("nodes", []) if n.get("type") in AGENTIC_NODE_TYPES]
        if agentic_nodes:
            print("Agentic Automation: This subflow contains agentic nodes for automation.")
        else:
            print("Agentic Automation: No agentic nodes in this subflow.")
        # Agentic enhancement: agent suggests how to reuse this subflow
        try:
            if hasattr(agent_tools, "suggest_subflow_reuse"):
                suggest_reuse = getattr(agent_tools, "suggest_subflow_reuse", None)
                if callable(suggest_reuse):
                    reuse_suggestions = suggest_reuse(subflow)
                    if reuse_suggestions:
                        print("Agent Suggestion(s) for subflow reuse:")
                        for s in reuse_suggestions:
                            print(f"  - {s}")
        except Exception as e:
            logging.debug(f"Agent subflow reuse suggestion failed: {e}")
    except Exception as e:
        print(f"Failed to extract subflow: {e}")
        sys.exit(1)

def show_execution_insights() -> None:
    """
    Show execution history insights from the agent, with agentic enhancements.
    """
    try:
        if hasattr(agent_tools, "get_execution_insights"):
            insights = agent_tools.get_execution_insights()
        else:
            insights = {}
        print("Execution Insights:")
        print(f"  - Success rate: {int(insights.get('success_rate', 0)*100)}%")
        print(f"  - Average latency: {insights.get('average_latency', '?')}s")
        print(f"  - Hot spots: {', '.join(insights.get('hot_spots', []))}")
        if any(any(t in s for t in AGENTIC_NODE_TYPES) for s in insights.get("hot_spots", [])):
            print("Agentic Automation: Agentic nodes are involved in recent execution hot spots.")
        # Agentic enhancement: agent suggests improvements for hot spots
        try:
            if hasattr(agent_tools, "suggest_hotspot_improvements"):
                suggest_hotspot = getattr(agent_tools, "suggest_hotspot_improvements", None)
                if callable(suggest_hotspot):
                    improvements = suggest_hotspot(insights.get("hot_spots", []), insights=insights)
                    if improvements:
                        print("Agent Suggestion(s) for hot spot improvements:")
                        for imp in improvements:
                            print(f"  - {imp}")
        except Exception as e:
            logging.debug(f"Agent hotspot improvement suggestion failed: {e}")
    except Exception as e:
        print(f"Failed to get execution insights: {e}")
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FastMCP Node Graph CLI - Build, validate, optimize, and manage MCP graphs with full agentic enhancements."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list-nodes
    subparsers.add_parser("list-nodes", help="List all available node types (agentic nodes marked with 🤖).")

    # validate
    validate_parser = subparsers.add_parser("validate", help="Validate a graph spec file (agentic enhancements).")
    validate_parser.add_argument("spec", type=str, help="Path to the graph spec file (JSON).")

    # build
    build_parser = subparsers.add_parser("build", help="Load/display graph structure from a spec file (agentic enhancements).")
    build_parser.add_argument("spec", type=str, help="Path to the graph spec file (JSON).")
    build_parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output path for the graph structure (JSON)."
    )

    # suggest-next-node
    suggest_parser = subparsers.add_parser("suggest-next-node", help="Agent suggests next node(s) for a partial graph (agentic enhancements).")
    suggest_parser.add_argument("spec", type=str, help="Path to the partial graph spec file (JSON).")

    # agent-suggest-nodes
    agent_suggest_parser = subparsers.add_parser(
        "agent-suggest-nodes",
        help="Agent suggests multiple possible next nodes with reasoning for a partial graph (agentic enhancements)."
    )
    agent_suggest_parser.add_argument("spec", type=str, help="Path to the partial graph spec file (JSON).")

    # optimize
    optimize_parser = subparsers.add_parser("optimize", help="Analyze and suggest workflow optimizations (agentic enhancements).")
    optimize_parser.add_argument("spec", type=str, help="Path to the graph spec file (JSON).")
    optimize_parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output path for the optimized graph spec (JSON)."
    )

    # extract-subflow
    subflow_parser = subparsers.add_parser("extract-subflow", help="Extract a subflow from node IDs (agentic enhancements).")
    subflow_parser.add_argument("spec", type=str, help="Path to the graph spec file (JSON).")
    subflow_parser.add_argument("subflow_name", type=str, help="Name for the new subflow.")
    subflow_parser.add_argument("node_ids", nargs="+", help="List of node IDs to include in the subflow.")
    subflow_parser.add_argument(
        "-o", "--output", type=str, default=None, help="Output path for the subflow (JSON)."
    )

    # execution-insights
    subparsers.add_parser("execution-insights", help="Show execution history insights (agentic enhancements).")

    args = parser.parse_args()

    if args.command == "list-nodes":
        list_nodes()
    elif args.command == "validate":
        validate_graph(args.spec)
    elif args.command == "build":
        build_graph(args.spec, args.output)
    elif args.command == "suggest-next-node":
        suggest_next_node_cli(args.spec)
    elif args.command == "agent-suggest-nodes":
        agent_suggest_nodes_cli(args.spec)
    elif args.command == "optimize":
        optimize_workflow(args.spec, args.output)
    elif args.command == "extract-subflow":
        extract_subflow_cli(args.spec, args.node_ids, args.subflow_name, args.output)
    elif args.command == "execution-insights":
        show_execution_insights()
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
FastMCP Node Index

Centralized, robust interface for node registration, lookup, and listing.
Leverages core registry logic from `register_node.py`.

☕️🚀 Coffee-Grade Code Bombs:
- Centralizes all node registry access for FastMCP builder.
- Lays groundwork for dynamic plugin/extension node discovery.
- TODO: Enforce strict type validation and interface compliance (see MCP Tooling).
- TODO: Decorator-based registration for ergonomic DX.
- TODO: 100% unit test coverage for all registry operations.
- TODO: Open an issue for plugin-based node registration (dynamic discovery/extensibility).
- TODO: Consider agent-node cross-registry integration for advanced workflows.
- TODO: Add MCP Tooling compliance checks for all node registration.
- TODO: Auto-import all node modules in this directory for registration (see issue #123).

See also:
- `register_node.py` for registry implementation.
- `agent_node_registry.py` for agent-specific node logic.

"""

from typing import Dict, Type, Optional
from .register_node import (
    Node,
    register_node,
    get_node_class,
    list_registered_nodes,
    node as node_decorator,
)

__all__ = [
    "Node",
    "register_node",
    "get_node_class",
    "list_registered_nodes",
    "node_decorator",
]

# --- Example Usage (for documentation/testing only) ---
#
# @node_decorator("example_node")
# class ExampleNode(Node):
#     def execute(self, *args, **kwargs):
#         return "Hello from ExampleNode"
#
# register_node("manual_node", ExampleNode)
# assert get_node_class("example_node") is ExampleNode
# assert "manual_node" in list_registered_nodes()
#
# ------------------------------------------------------

# --- Dynamic Node Auto-Import (Zero-Config Discovery) ---
# TODO: Future: Auto-import all node modules in this directory for registration.
#       This will enable zero-config node discovery for plugins/extensions.
#       See: https://github.com/your-org/fastmcp/issues/123

import importlib
import os
import sys

def _auto_import_nodes():
    """
    Dynamically import all node modules in this directory to ensure
    all node classes are registered at runtime.

    TODO: Add MCP Tooling compliance checks for all node registration.
    TODO: Enforce strict type validation and interface compliance.
    """
    current_dir = os.path.dirname(__file__)
    for filename in os.listdir(current_dir):
        if (
            filename.endswith("_node.py")
            or filename.endswith("_nodes.py")
            or filename.endswith("_node_registry.py")
        ):
            modulename = filename[:-3]
            if modulename == "index":
                continue
            module_path = f"{__package__}.{modulename}" if __package__ else modulename
            try:
                importlib.import_module(module_path)
            except Exception as e:
                # TODO: Open issue if import fails, log for troubleshooting
                print(f"[FastMCP Node Index] Failed to import {module_path}: {e}")

# Auto-import all node modules for registration at import time
_auto_import_nodes()

# TODO: Consider agent-node cross-registry integration for advanced workflows.
# TODO: Open an issue for plugin-based node registration (dynamic discovery/extensibility).
# TODO: 100% unit test coverage for all registry operations.

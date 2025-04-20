"""
Node Registry for FastMCP Builder

This module provides a central registry for all node types used in the FastMCP builder.
Nodes can be registered, retrieved, and listed via this registry.

- Strict typing enforced.
- No third-party dependencies.
- Designed for extensibility and robust error handling.

TODO: 
- Add support for dynamic loading of nodes from plugins.
- Integrate with MCP tooling for node validation and metadata enrichment.
"""

from typing import Type, Dict, List, Optional, Callable, Any, cast
from .custom_nodes import BaseNode

# Type alias for node constructor
NodeConstructor = Callable[..., BaseNode]

class NodeRegistry:
    """
    Central registry for node types.
    """
    _registry: Dict[str, Type[BaseNode]] = {}

    @classmethod
    def register(cls, node_type: str, node_cls: Type[BaseNode]) -> None:
        """
        Register a node class with a unique type identifier.

        Args:
            node_type (str): Unique identifier for the node type.
            node_cls (Type[BaseNode]): The node class to register.

        Raises:
            ValueError: If the node_type is already registered.
        """
        if node_type in cls._registry:
            raise ValueError(f"Node type '{node_type}' is already registered.")
        cls._registry[node_type] = node_cls

    @classmethod
    def get(cls, node_type: str) -> Optional[Type[BaseNode]]:
        """
        Retrieve a node class by its type identifier.

        Args:
            node_type (str): The type identifier of the node.

        Returns:
            Optional[Type[BaseNode]]: The node class if found, else None.
        """
        return cls._registry.get(node_type)

    @classmethod
    def list_types(cls) -> List[str]:
        """
        List all registered node type identifiers.

        Returns:
            List[str]: List of node type strings.
        """
        return list(cls._registry.keys())

    @classmethod
    def create(cls, node_type: str, *args, **kwargs) -> BaseNode:
        """
        Instantiate a node by its type identifier.

        Args:
            node_type (str): The type identifier of the node.
            *args: Positional arguments for the node constructor.
            **kwargs: Keyword arguments for the node constructor.

        Returns:
            BaseNode: An instance of the requested node.

        Raises:
            KeyError: If the node_type is not registered.
        """
        node_cls = cls.get(node_type)
        if node_cls is None:
            raise KeyError(f"Node type '{node_type}' is not registered.")
        return node_cls(*args, **kwargs)

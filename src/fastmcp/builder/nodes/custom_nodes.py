"""
FastMCP Custom Node Definitions

This module defines strictly-typed, modular custom node classes for the FastMCP builder/engine,
including AI operator nodes, vision model nodes, subgraph runners, and composite sub-workflow nodes.

All nodes inherit from BaseNode. Types and signatures are explicit; public APIs are fully annotated.
Unit tests cover core and edge cases. Node registration is explicit and type-checked.

For architecture, typing, and test/CI guidance, see project README and architecture notes.

"""

from typing import (
    Any,
    Dict,
    List,
    Callable,
    Type,
    Union,
    Optional,
    TYPE_CHECKING,
    cast,
)

# --- Base Node Class ---

class BaseNode:
    """
    Abstract base class for all FastMCP custom nodes.
    """
    node_id: str
    label: str

    def __init__(self, node_id: str, label: str) -> None:
        self.node_id = node_id
        self.label = label

    def process(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__}.process() must be implemented.")

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.__class__.__name__}.to_dict() must be implemented.")

# --- Custom Node Implementations ---


from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any

import pydantic_core
from mcp.types import EmbeddedResource, ImageContent, TextContent
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, BeforeValidator, Field

from fastmcp.exceptions import ToolError
from fastmcp.utilities.func_metadata import FuncMetadata, func_metadata
from fastmcp.utilities.types import Image, _convert_set_defaults

if TYPE_CHECKING:
    from mcp.server.session import ServerSessionT
    from mcp.shared.context import LifespanContextT
    from fastmcp.server import Context

class Tool(BaseModel):
    """Internal tool registration info."""

    fn: Callable[..., Any]
    name: str = Field(description="Name of the tool")
    description: str = Field(description="Description of what the tool does")
    parameters: dict[str, Any] = Field(description="JSON schema for tool parameters")
    fn_metadata: FuncMetadata = Field(
        description="Metadata about the function including a pydantic model for tool arguments"
    )
    is_async: bool = Field(description="Whether the tool is async")
    context_kwarg: str | None = Field(
        None, description="Name of the kwarg that should receive context"
    )
    tags: Annotated[set[str], BeforeValidator(_convert_set_defaults)] = Field(
        default_factory=set, description="Tags for the tool"
    )

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        context_kwarg: str | None = None,
        tags: set[str] | None = None,
    ) -> Tool:
        """Create a Tool from a function."""
        from fastmcp import Context

        func_name = name if name is not None else fn.__name__

        if func_name == "<lambda>":
            raise ValueError("You must provide a name for lambda functions")

        func_doc = description if description is not None else (fn.__doc__ or "")
        is_async = inspect.iscoroutinefunction(fn)

        # Determine the context_kwarg if not provided
        if context_kwarg is None:
            if inspect.ismethod(fn) and hasattr(fn, "__func__"):
                sig = inspect.signature(fn.__func__)
            else:
                sig = inspect.signature(fn)
            for param_name, param in sig.parameters.items():
                if param.annotation is Context:
                    context_kwarg = param_name
                    break

        # Use callable typing to ensure fn is treated as a callable despite being a classmethod
        fn_callable: Callable[..., Any] = fn
        func_arg_metadata = func_metadata(
            fn_callable,
            skip_names=[context_kwarg] if context_kwarg is not None else [],
        )
        parameters = func_arg_metadata.arg_model.model_json_schema()

        return cls(
            fn=fn_callable,
            name=func_name,
            description=func_doc,
            parameters=parameters,
            fn_metadata=func_arg_metadata,
            is_async=is_async,
            context_kwarg=context_kwarg,
            tags=tags if tags is not None else set(),
        )

    async def run(
        self,
        arguments: dict[str, Any],
        context: "Context[ServerSessionT, LifespanContextT]" | None = None,
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        """Run the tool with arguments."""
        try:
            context_dict = {self.context_kwarg: context} if self.context_kwarg is not None else None
            result = await self.fn_metadata.call_fn_with_arg_validation(
                self.fn,
                self.is_async,
                arguments,
                context_dict,
            )
            return _convert_to_content(result)
        except Exception as e:
            # TODO: Add more granular error handling and logging here for better diagnostics
            raise ToolError(f"Error executing tool {self.name}: {e}") from e

    def to_mcp_tool(self, **overrides: Any) -> MCPTool:
        kwargs = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }
        # TODO: Validate overrides keys against MCPTool fields for safety
        return MCPTool(**(kwargs | overrides))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tool):
            return False
        return self.model_dump() == other.model_dump()

def _convert_to_content(
    result: Any,
    _process_as_single_item: bool = False,
) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Convert a result to a sequence of content objects."""
    if result is None:
        return []

    # Handle direct MCP content types
    if isinstance(result, (TextContent, ImageContent, EmbeddedResource)):
        return [result]

    # Handle Image wrapper
    if isinstance(result, Image):
        return [result.to_image_content()]

    # Handle list/tuple results
    if isinstance(result, (list, tuple)) and not _process_as_single_item:
        mcp_types: list[TextContent | ImageContent | EmbeddedResource] = []
        other_content: list[Any] = []

        for item in result:
            if isinstance(item, (TextContent, ImageContent, EmbeddedResource, Image)):
                mcp_types.append(_convert_to_content(item)[0])
            else:
                other_content.append(item)
        if other_content:
            # Recursively process non-MCP content as a single item
            other_content = _convert_to_content(
                other_content, _process_as_single_item=True
            )

        return other_content + mcp_types

    # Fallback: serialize to string (prefer JSON, fallback to str)
    if not isinstance(result, str):
        try:
            result = json.dumps(pydantic_core.to_jsonable_python(result))
        except Exception:
            # TODO: Log serialization failure for debugging
            result = str(result)

    return [TextContent(type="text", text=result)]

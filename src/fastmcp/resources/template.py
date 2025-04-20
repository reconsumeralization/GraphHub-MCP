"""Resource template functionality."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Annotated, Any
from urllib.parse import unquote

from mcp.types import ResourceTemplate as MCPResourceTemplate
from pydantic import (
    AnyUrl,
    BaseModel,
    BeforeValidator,
    Field,
    TypeAdapter,
    field_validator,
    validate_call,
)

from fastmcp.resources.types import FunctionResource, Resource
from fastmcp.utilities.types import _convert_set_defaults

# --- Regex utilities for URI templates ---

def build_regex(template: str) -> re.Pattern:
    """
    Build a regex pattern from a URI template.
    E.g. "weather://{city}/current" -> r"^weather://(?P<city>[^/]+)/current$"
    """
    # TODO: Support more complex patterns (e.g. custom regex in braces)
    parts = re.split(r"(\{[^}]+\})", template)
    pattern = ""
    for part in parts:
        if part.startswith("{") and part.endswith("}"):
            name = part[1:-1]
            pattern += f"(?P<{name}>[^/]+)"
        else:
            pattern += re.escape(part)
    return re.compile(f"^{pattern}$")

def match_uri_template(uri: str, uri_template: str) -> dict[str, str] | None:
    """
    Match a URI against a template and extract parameters.
    Returns a dict of parameter values if matched, else None.
    """
    regex = build_regex(uri_template)
    match = regex.match(uri)
    if match:
        return {k: unquote(v) for k, v in match.groupdict().items()}
    return None

# --- Resource Template Model ---

class ResourceTemplate(BaseModel):
    """
    A template for dynamically creating resources.
    """

    uri_template: str = Field(
        description="URI template with parameters (e.g. weather://{city}/current)"
    )
    name: str = Field(description="Name of the resource")
    description: str | None = Field(description="Description of what the resource does")
    tags: Annotated[set[str], BeforeValidator(_convert_set_defaults)] = Field(
        default_factory=set, description="Tags for the resource"
    )
    mime_type: str = Field(
        default="text/plain", description="MIME type of the resource content"
    )
    fn: Callable[..., Any]
    parameters: dict[str, Any] = Field(
        description="JSON schema for function parameters"
    )

    @field_validator("mime_type", mode="before")
    @classmethod
    def set_default_mime_type(cls, mime_type: str | None) -> str:
        """Set default MIME type if not provided."""
        if mime_type:
            return mime_type
        return "text/plain"

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        uri_template: str,
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        tags: set[str] | None = None,
    ) -> ResourceTemplate:
        """
        Create a ResourceTemplate from a function and a URI template.

        - Validates that all required function arguments are present in the URI template.
        - Ensures the URI template contains at least one parameter.
        - Extracts parameter schema from the function's type hints.
        """
        func_name = name or fn.__name__
        if func_name == "<lambda>":
            raise ValueError("You must provide a name for lambda functions")

        uri_params = set(re.findall(r"{(\w+)}", uri_template))
        if not uri_params:
            raise ValueError("URI template must contain at least one parameter")

        func_sig = inspect.signature(fn)
        func_params = set(func_sig.parameters.keys())

        required_params = {
            p
            for p in func_params
            if func_sig.parameters[p].default is inspect.Parameter.empty
        }

        # All required function params must be present in URI template
        if not required_params.issubset(uri_params):
            raise ValueError(
                f"URI parameters {uri_params} must be a superset of required function arguments: {required_params}"
            )

        # All URI template params must be present in function signature
        if not uri_params.issubset(func_params):
            raise ValueError(
                f"URI parameters {uri_params} must be a subset of the function arguments: {func_params}"
            )

        # Get schema from TypeAdapter - will fail if function isn't properly typed
        try:
            parameters = TypeAdapter(fn).json_schema()
        except Exception as e:
            # TODO: Open issue if function is not properly typed
            raise TypeError(f"Function must have type annotations for all parameters: {e}")

        # ensure the arguments are properly cast
        fn_validated = validate_call(fn)

        return cls(
            uri_template=uri_template,
            name=func_name,
            description=description or fn.__doc__ or "",
            mime_type=mime_type or "text/plain",
            fn=fn_validated,
            parameters=parameters,
            tags=tags or set(),
        )

    def matches(self, uri: str) -> dict[str, Any] | None:
        """
        Check if URI matches template and extract parameters.
        Returns dict of parameters if matched, else None.
        """
        return match_uri_template(uri, self.uri_template)

    async def create_resource(self, uri: str, params: dict[str, Any]) -> Resource:
        """
        Create a resource from the template with the given parameters.

        - Calls the function with params (awaits if coroutine).
        - Returns a FunctionResource with the result captured in a closure.
        """
        try:
            result = self.fn(**params)
            if inspect.iscoroutine(result):
                result = await result

            return FunctionResource(
                uri=AnyUrl(uri),  # Explicitly convert to AnyUrl
                name=self.name,
                description=self.description,
                mime_type=self.mime_type,
                fn=lambda: result,  # Capture result in closure
                tags=self.tags,
            )
        except Exception as e:
            # TODO: Add more granular error handling/logging
            raise ValueError(f"Error creating resource from template: {e}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResourceTemplate):
            return False
        return self.model_dump() == other.model_dump()

    def to_mcp_template(self, **overrides: Any) -> MCPResourceTemplate:
        """
        Convert the resource template to an MCPResourceTemplate.
        """
        kwargs = {
            "uriTemplate": self.uri_template,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }
        return MCPResourceTemplate(**kwargs | overrides)

# TODO: Remove MyModel if not used elsewhere

"""Base classes and interfaces for FastMCP resources."""

import abc
from typing import Annotated, Any

from mcp.types import Resource as MCPResource
from pydantic import (
    AnyUrl,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    UrlConstraints,
    ValidationInfo,
    field_validator,
)

from fastmcp.utilities.types import _convert_set_defaults

# TODO: Consider moving Resource to a dedicated types module if it grows further.

class Resource(BaseModel, abc.ABC):
    """Base class for all resources."""

    model_config = ConfigDict(validate_default=True)

    uri: Annotated[AnyUrl, UrlConstraints(host_required=False)] = Field(
        ...,
        description="URI of the resource"
    )
    name: str | None = Field(
        default=None,
        description="Name of the resource"
    )
    description: str | None = Field(
        default=None,
        description="Description of the resource"
    )
    tags: Annotated[set[str], BeforeValidator(_convert_set_defaults)] = Field(
        default_factory=set,
        description="Tags for the resource"
    )
    mime_type: str = Field(
        default="text/plain",
        description="MIME type of the resource content",
        pattern=r"^[a-zA-Z0-9]+/[a-zA-Z0-9\-+.]+$"
    )

    @field_validator("mime_type", mode="before")
    @classmethod
    def set_default_mime_type(cls, mime_type: str | None) -> str:
        """Set default MIME type if not provided."""
        # If mime_type is provided and not empty, use it; otherwise, default to text/plain
        if mime_type:
            return mime_type
        return "text/plain"

    @field_validator("name", mode="before")
    @classmethod
    def set_default_name(cls, name: str | None, info: ValidationInfo) -> str:
        """Set default name from URI if not provided."""
        if name:
            return name
        uri = info.data.get("uri")
        if uri:
            return str(uri)
        raise ValueError("Either name or uri must be provided")

    @abc.abstractmethod
    async def read(self) -> str | bytes:
        """Read the resource content."""
        # Subclasses must implement this method.
        raise NotImplementedError("Subclasses must implement the read() method.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Resource):
            return False
        # Use model_dump for deep equality check
        return self.model_dump() == other.model_dump()

    def to_mcp_resource(self, **overrides: Any) -> MCPResource:
        """Convert the resource to an MCPResource."""
        kwargs = {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }
        # Merge overrides, with overrides taking precedence
        return MCPResource(**(kwargs | overrides))

# TODO: Add unit tests for Resource equality, defaulting, and to_mcp_resource conversion.

"""Custom exceptions for FastMCP.

All custom exceptions inherit from FastMCPError unless they are intended
to be used by external client code or for compatibility with external APIs.

This module provides structured error codes and optional metadata for
programmatic error handling and robust debugging.

Error codes are unique per exception type and can be used for logging,
API responses, and client-side error handling.

See MasterPlan.md (Error Handling & Logging) for ongoing improvements.
"""

from typing import Any, Optional, Dict, TypeVar, Type

# --- Exception Type Variable for type-safe factory methods ---
E = TypeVar("E", bound="FastMCPError")

class FastMCPError(Exception):
    """
    Base error for FastMCP.

    All FastMCP-specific exceptions should inherit from this class.

    Args:
        message: Human-readable error message.
        code: Unique error code for programmatic handling.
        details: Optional structured data for debugging or client use.
    """

    error_code: str = "fastmcp.error"

    def __init__(
        self: "FastMCPError",
        message: Optional[str] = None,
        *,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.message: str = message or self.__class__.__doc__ or "FastMCP error"
        self.code: str = code or self.error_code
        self.details: Dict[str, Any] = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the exception for logging or API responses."""
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }

    @classmethod
    def from_exception(cls: Type[E], exc: Exception, *, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> E:
        """
        Factory to wrap an arbitrary exception as a FastMCPError subclass.
        Preserves the original message and attaches original exception as detail.
        """
        # TODO: Consider logging the original exception with traceback here for audit trail
        return cls(
            message=str(exc),
            code=code,
            details={**(details or {}), "original_exception": repr(exc)},
        )

class ValidationError(FastMCPError):
    """Raised when parameter or return value validation fails."""
    error_code: str = "fastmcp.validation_error"

class ResourceError(FastMCPError):
    """Raised for errors in resource operations."""
    error_code: str = "fastmcp.resource_error"

class ToolError(FastMCPError):
    """Raised for errors in tool operations."""
    error_code: str = "fastmcp.tool_error"

class PromptError(FastMCPError):
    """Raised for errors in prompt operations."""
    error_code: str = "fastmcp.prompt_error"

class InvalidSignature(FastMCPError):
    """Raised when an invalid signature is encountered in FastMCP."""
    error_code: str = "fastmcp.invalid_signature"

class ClientError(FastMCPError):
    """Raised for errors in client operations."""
    error_code: str = "fastmcp.client_error"

class NotFoundError(FastMCPError):
    """Raised when an object is not found."""
    error_code: str = "fastmcp.not_found"

# --- More granular exception types for robust error handling ---

class AuthorizationError(FastMCPError):
    """Raised when an operation is not authorized."""
    error_code: str = "fastmcp.authorization_error"

class ConflictError(FastMCPError):
    """Raised when a resource conflict occurs (e.g., duplicate creation)."""
    error_code: str = "fastmcp.conflict_error"

class DependencyError(FastMCPError):
    """Raised when a required dependency is missing or invalid."""
    error_code: str = "fastmcp.dependency_error"

class ExecutionError(FastMCPError):
    """Raised for errors during workflow or node execution."""
    error_code: str = "fastmcp.execution_error"

class TimeoutError(FastMCPError):
    """Raised when an operation times out."""
    error_code: str = "fastmcp.timeout_error"

class SerializationError(FastMCPError):
    """Raised when serialization or deserialization fails."""
    error_code: str = "fastmcp.serialization_error"

class ConfigurationError(FastMCPError):
    """Raised for configuration or environment errors."""
    error_code: str = "fastmcp.configuration_error"

class ExternalAPIError(FastMCPError):
    """Raised when an external API call fails."""
    error_code: str = "fastmcp.external_api_error"

# --- Future-proofing: Node, Agent, and Workflow-specific exceptions ---

# TODO: As the codebase grows, add node-specific, agent-specific, and
#       workflow-specific exception types as needed for fine-grained error handling.
#       Consider integrating with centralized logging and audit trails.
#       Open an issue if a new error type is needed for a new subsystem.

# --- End of Damn that's some good code. ☕️🚀 ---

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import TypeAlias

import mcp.types
from mcp import ClientSession, CreateMessageResult
from mcp.client.session import SamplingFnT
from mcp.shared.context import LifespanContextT, RequestContext
from mcp.types import CreateMessageRequestParams as SamplingParams
from mcp.types import SamplingMessage

# Drop Codde Bombs of Damn that's some good code. ☕️🚀

# NOTE: MessageResult is currently only used here. If it is used elsewhere, move to a shared types module.
# TODO: Move MessageResult to a shared types module if used elsewhere (see usage tracking in CI).
class MessageResult(CreateMessageResult):
    """
    A concrete implementation of CreateMessageResult for client-side sampling.
    """
    role: mcp.types.Role = "assistant"
    content: mcp.types.TextContent | mcp.types.ImageContent
    model: str = "client-model"

SamplingHandler: TypeAlias = Callable[
    [
        list[SamplingMessage],
        SamplingParams,
        RequestContext[ClientSession, LifespanContextT],
    ],
    str | CreateMessageResult | Awaitable[str | CreateMessageResult],
]

def _is_valid_create_message_result(obj: object) -> bool:
    """
    Checks if the object is a valid CreateMessageResult.
    """
    if not isinstance(obj, CreateMessageResult):
        return False
    # Check for required attributes
    required_attrs = ["content", "role", "model"]
    for attr in required_attrs:
        if not hasattr(obj, attr):
            return False
    return True

def _is_valid_error_data(obj: object) -> bool:
    """
    Checks if the object is a valid ErrorData.
    """
    # Fix: Use and instead of ; and close parenthesis
    return (
        isinstance(obj, mcp.types.ErrorData)
        and hasattr(obj, "code")
        and hasattr(obj, "message")
    )

class InvalidSamplingHandlerOutput(Exception):
    """Raised when the sampling handler returns an invalid result type."""

def create_sampling_callback(sampling_handler: SamplingHandler) -> SamplingFnT:
    """
    Wraps a user-provided sampling handler into a standardized async callback
    for the MCP client. Handles both sync and async handlers, and ensures
    robust error handling and type normalization.

    Args:
        sampling_handler: The user-implemented handler function.

    Returns:
        An async function compatible with SamplingFnT.
    """
    async def _sampling_handler(
        context: RequestContext[ClientSession, LifespanContextT],
        params: SamplingParams,
    ) -> CreateMessageResult | mcp.types.ErrorData:
        try:
            # Call the handler (may be sync or async)
            result = sampling_handler(params.messages, params, context)
            if inspect.isawaitable(result):
                result = await result

            # Normalize string result to MessageResult
            if isinstance(result, str):
                result = MessageResult(
                    content=mcp.types.TextContent(type="text", text=result)
                )

            # Validate result is a proper CreateMessageResult or ErrorData
            if _is_valid_create_message_result(result):
                return result
            if _is_valid_error_data(result):
                return result

            # More robust type validation and error reporting
            # Add more granular error codes for different failure modes
            logging.error(
                "Sampling handler returned invalid result type: %r (type: %s)",
                result, type(result)
            )
            # Raise a custom exception for invalid handler output
            raise InvalidSamplingHandlerOutput(
                f"Sampling handler returned invalid result type: {type(result)}"
            )
        except InvalidSamplingHandlerOutput as e:
            # Granular error code for handler output validation error
            logging.error("InvalidSamplingHandlerOutput: %s", e)
            return mcp.types.ErrorData(
                code=mcp.types.VALIDATION_ERROR
                if hasattr(mcp.types, "VALIDATION_ERROR")
                else mcp.types.INTERNAL_ERROR,
                message=str(e),
            )
        except Exception as e:
            # Add logging here for better traceability
            logging.exception("Exception in sampling handler: %s", e)
            # Integrate with centralized logging/audit trail (if available)
            # TODO: Integrate with centralized logging/audit trail (if available) - see logging config
            return mcp.types.ErrorData(
                code=mcp.types.RUNTIME_ERROR
                if hasattr(mcp.types, "RUNTIME_ERROR")
                else mcp.types.INTERNAL_ERROR,
                message=str(e),
            )

    return _sampling_handler

# Consider supporting streaming results in the future
# NOTE: Streaming not yet implemented. See issue #streaming-support for roadmap.

# Add unit tests for all edge cases and error paths
# See: tests/fastmcp/client/test_sampling.py for comprehensive coverage.
# If missing, open an issue and add tests for:
#   - Handler returns string, valid CreateMessageResult, valid ErrorData, invalid type, raises exception, etc.

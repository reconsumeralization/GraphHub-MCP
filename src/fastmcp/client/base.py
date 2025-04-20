from typing import Any, Dict, Optional, Protocol, runtime_checkable

class ClientError(Exception):
    """Base exception for all client errors in fastmcp."""
    pass

@runtime_checkable
class BaseTransport(Protocol):
    """
    Protocol for transport classes used by FastMCP clients.
    All transports must implement these methods.
    """
    def send(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def close(self) -> None:
        ...

class BaseClient:
    """
    Base class for FastMCP clients.

    Handles transport, request/response lifecycle, and error handling.
    Subclasses should implement domain-specific logic.
    """

    def __init__(self, transport: BaseTransport, timeout: Optional[float] = None) -> None:
        self._transport = transport
        self._timeout = timeout

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request to the server and return the response.

        Args:
            payload: The request data as a dictionary.

        Returns:
            The response data as a dictionary.

        Raises:
            ClientError: If the transport fails or returns an error.
        """
        try:
            # TODO: Add request/response logging, tracing, and validation
            response = self._transport.send(payload)
            if not isinstance(response, dict):
                raise ClientError("Invalid response type from transport")
            # TODO: Add response schema validation
            return response
        except Exception as exc:
            # TODO: Add more granular error handling and logging
            raise ClientError(f"Request failed: {exc}") from exc

    def close(self) -> None:
        """Close the underlying transport."""
        self._transport.close()

    @property
    def timeout(self) -> Optional[float]:
        """Get the current timeout setting."""
        return self._timeout

    @timeout.setter
    def timeout(self, value: Optional[float]) -> None:
        """Set the timeout for requests."""
        self._timeout = value

# TODO: Open an issue to add async support for BaseClient and transports.
# TODO: Add request/response schema validation using pydantic or similar.
# TODO: Integrate logging and tracing for all client operations.


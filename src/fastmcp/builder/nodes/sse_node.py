from typing import Callable, Dict, Optional, Any, Generator, List, TypedDict
from fastapi import Request, Response
from starlette.responses import StreamingResponse
import time
import json
import logging

logger = logging.getLogger("fastmcp.sse_node")

class SSEEvent(TypedDict, total=False):
    id: str
    event: str
    data: Any
    retry: int

class SSENode:
    """
    Server-Sent Events (SSE) Node for FastMCP builder pipeline.

    This node streams data to clients using the SSE protocol.
    It is designed for real-time updates, logs, or progress events to the frontend,
    and is agent/builder aware for enhanced GUI usability.

    Usage:
        node = SSENode(event_generator=my_generator)
        response = node.handle(request)
    """

    def __init__(
        self,
        event_generator: Callable[[Request], Generator[SSEEvent, None, None]],
        headers: Optional[Dict[str, str]] = None,
        retry: Optional[int] = 2000,
        heartbeat_interval: Optional[float] = 15.0,
        enable_agent_metadata: bool = True,
    ) -> None:
        """
        :param event_generator: A generator function that yields event dicts.
        :param headers: Optional headers to include in the response.
        :param retry: Optional retry interval in ms for the client.
        :param heartbeat_interval: Interval in seconds for sending heartbeat/ping events.
        :param enable_agent_metadata: If True, injects agent/builder metadata for GUI.
        """
        self.event_generator = event_generator
        self.headers = headers or {}
        self.retry = retry
        self.heartbeat_interval = heartbeat_interval
        self.enable_agent_metadata = enable_agent_metadata

    def _format_sse(self, event: SSEEvent) -> str:
        """
        Formats a single event dict as an SSE message string.

        :param event: Dict with keys like 'event', 'data', 'id'.
        :return: Formatted SSE string.
        """
        lines: List[str] = []
        if 'id' in event:
            lines.append(f"id: {event['id']}")
        if 'event' in event:
            lines.append(f"event: {event['event']}")
        if 'retry' in event:
            lines.append(f"retry: {event['retry']}")
        elif self.retry is not None:
            lines.append(f"retry: {self.retry}")
        if 'data' in event:
            data_str = event['data']
            if not isinstance(data_str, str):
                try:
                    data_str = json.dumps(data_str, ensure_ascii=False)
                except Exception as exc:
                    logger.error(f"Failed to serialize SSE data: {exc}")
                    data_str = str(data_str)
            for line in data_str.splitlines():
                lines.append(f"data: {line}")
        lines.append("")  # End of message
        return "\n".join(lines) + "\n"

    def _inject_agent_metadata(self, event: SSEEvent, request: Request) -> SSEEvent:
        """
        Optionally injects agent/builder metadata for GUI usability.
        """
        if not self.enable_agent_metadata:
            return event
        # TODO: Enhance with richer agent/builder context as needed
        agent_info = {
            "agent_id": getattr(request.state, "agent_id", None),
            "builder_node": getattr(request.state, "builder_node", None),
            "timestamp": time.time(),
        }
        # Only inject if not already present
        if isinstance(event.get("data"), dict):
            event = event.copy()
            event["data"] = {**event["data"], "_agent_meta": agent_info}
        return event

    def _event_stream(self, request: Request) -> Generator[bytes, None, None]:
        """
        Generator that yields SSE-formatted bytes for each event.

        :param request: FastAPI Request object.
        """
        last_heartbeat = time.time()
        try:
            for event in self.event_generator(request):
                if self.enable_agent_metadata:
                    event = self._inject_agent_metadata(event, request)
                sse_message = self._format_sse(event)
                yield sse_message.encode("utf-8")
                now = time.time()
                if self.heartbeat_interval is not None and now - last_heartbeat > self.heartbeat_interval:
                    # Send a heartbeat comment to keep connection alive
                    yield b": heartbeat\n\n"
                    last_heartbeat = now
            # After generator ends, send a final heartbeat to signal end
            yield b": stream-end\n\n"
        except Exception as exc:
            logger.exception("SSE stream error")
            error_event: SSEEvent = {
                "event": "error",
                "data": {
                    "message": f"Internal server error: {str(exc)}",
                    "agent_id": getattr(request.state, "agent_id", None),
                }
            }
            yield self._format_sse(error_event).encode("utf-8")
            # TODO: Open issue: Add support for client disconnect detection and cleanup

    def handle(self, request: Request) -> Response:
        """
        Handles the incoming request and returns a StreamingResponse for SSE.

        :param request: FastAPI Request object.
        :return: StreamingResponse with SSE content type.
        """
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            **self.headers,
        }
        # TODO: Consider CORS headers if needed
        # TODO: Add support for authentication/authorization hooks for SSE endpoints
        return StreamingResponse(
            self._event_stream(request),
            headers=headers,
        )

# TODO: Add unit tests for SSENode (formatting, streaming, error handling, agent metadata)
# TODO: Open issue: Add authentication/authorization hooks for SSE endpoints
# TODO: Open issue: Add support for client disconnect detection and cleanup
# TODO: Open issue: Add richer agent/builder context for GUI/inspector

import inspect
from collections.abc import Awaitable, Callable
from typing import TypeAlias

import mcp.types
import pydantic
from mcp import ClientSession
from mcp.client.session import ListRootsFnT
from mcp.shared.context import LifespanContextT, RequestContext

# --- Type Aliases for clarity and strict typing ---
RootsList: TypeAlias = list[str] | list[mcp.types.Root] | list[str | mcp.types.Root]

RootsHandler: TypeAlias = (
    Callable[[RequestContext[ClientSession, LifespanContextT]], RootsList]
    | Callable[[RequestContext[ClientSession, LifespanContextT]], Awaitable[RootsList]]
)

def convert_roots_list(roots: RootsList) -> list[mcp.types.Root]:
    """
    Convert a RootsList (str, Root, or FileUrl) to a list of Root objects.
    Raises ValueError for invalid types.
    """
    result: list[mcp.types.Root] = []
    for r in roots:
        if isinstance(r, mcp.types.Root):
            result.append(r)
        elif isinstance(r, pydantic.FileUrl):
            result.append(mcp.types.Root(uri=r))
        elif isinstance(r, str):
            try:
                file_url = pydantic.FileUrl(r)
            except Exception as e:
                # TODO: Consider logging the error with more context
                raise ValueError(f"Invalid root string (not a valid FileUrl): {r}") from e
            result.append(mcp.types.Root(uri=file_url))
        else:
            # TODO: Open issue - Should we support Path or pathlib.Path here?
            raise ValueError(f"Invalid root type: {type(r)} value: {r}")
    return result

def create_roots_callback(
    handler: RootsList | RootsHandler,
) -> ListRootsFnT:
    """
    Create a ListRootsFnT callback from either a static list or a handler function.
    """
    if isinstance(handler, list):
        return _create_roots_callback_from_roots(handler)
    if callable(handler):
        return _create_roots_callback_from_fn(handler)
    raise ValueError(f"Invalid roots handler: {handler}")

def _create_roots_callback_from_roots(
    roots: RootsList,
) -> ListRootsFnT:
    """
    Wrap a static roots list as an async callback.
    """
    roots_converted = convert_roots_list(roots)

    async def _roots_callback(
        context: RequestContext[ClientSession, LifespanContextT],
    ) -> mcp.types.ListRootsResult:
        # TODO: Consider context-based filtering in the future
        return mcp.types.ListRootsResult(roots=roots_converted)

    return _roots_callback

def _create_roots_callback_from_fn(
    fn: Callable[[RequestContext[ClientSession, LifespanContextT]], RootsList]
    | Callable[[RequestContext[ClientSession, LifespanContextT]], Awaitable[RootsList]],
) -> ListRootsFnT:
    """
    Wrap a handler function (sync or async) as a ListRootsFnT async callback.
    Handles exceptions and always returns ListRootsResult or ErrorData.
    """
    async def _roots_callback(
        context: RequestContext[ClientSession, LifespanContextT],
    ) -> mcp.types.ListRootsResult | mcp.types.ErrorData:
        try:
            roots = fn(context)
            if inspect.isawaitable(roots):
                roots = await roots
            return mcp.types.ListRootsResult(roots=convert_roots_list(roots))
        except Exception as e:
            # TODO: Add logging here for better traceability
            return mcp.types.ErrorData(
                code=mcp.types.INTERNAL_ERROR,
                message=f"Failed to list roots: {e}",
            )

    return _roots_callback

# TODO: Add unit tests for convert_roots_list and create_roots_callback edge cases!

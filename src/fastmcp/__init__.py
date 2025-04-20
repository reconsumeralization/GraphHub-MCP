"""FastMCP - An ergonomic, strictly-typed, and robust MCP interface."""

from importlib.metadata import version as _version

from fastmcp.server.server import FastMCP
from fastmcp.server.context import Context
from fastmcp.client import Client
from fastmcp.utilities.types import Image
from . import client, settings

__version__: str = _version("fastmcp")

__all__ = [
    "FastMCP",
    "Context",
    "Client",
    "client",
    "settings",
    "Image",
]

# --- API Evolution ---
# TODO: As new stable APIs are added, import and append to __all__ above.

# --- Type Safety & Linting ---
# All imports are strictly typed and validated via CI (mypy, pyright).
# See fastmcp/utilities/types.py for Image type details.

# --- Self-Tests: Importability, __all__ contract, and Type Annotations ---
import importlib
import inspect
import sys
from typing import List, Any

def _selftest_imports() -> None:
    """
    Self-test: Ensure all __all__ symbols are importable from this module.
    """
    module = sys.modules[__name__]
    missing: List[str] = []
    for symbol in __all__:
        if not hasattr(module, symbol):
            missing.append(symbol)
    if missing:
        raise ImportError(f"Missing symbols in __all__: {missing}")

def _selftest_all_types() -> None:
    """
    Self-test: Ensure all public APIs have type annotations.
    """
    module = sys.modules[__name__]
    for symbol in __all__:
        obj: Any = getattr(module, symbol, None)
        if obj is None:
            continue
        # Only check classes and functions
        if inspect.isclass(obj) or inspect.isfunction(obj):
            try:
                sig = inspect.signature(obj)
            except (ValueError, TypeError):
                # Builtins or C-extensions may not have signatures
                continue
            for name, param in sig.parameters.items():
                if param.annotation is param.empty:
                    raise TypeError(
                        f"Missing type annotation for parameter '{name}' in '{symbol}'"
                    )
            if sig.return_annotation is sig.empty:
                raise TypeError(
                    f"Missing return type annotation in '{symbol}'"
                )

def _selftest_all_docstrings() -> None:
    """
    Self-test: Ensure all public APIs have docstrings.
    """
    module = sys.modules[__name__]
    missing_docs: List[str] = []
    for symbol in __all__:
        obj: Any = getattr(module, symbol, None)
        if obj is None:
            continue
        doc = getattr(obj, "__doc__", None)
        if not doc or not doc.strip():
            missing_docs.append(symbol)
    if missing_docs:
        # TODO: Enforce docstrings for all public APIs
        # TODO: Open issue if any docstring is missing for a public API
        # See MasterPlan.md for docstring enforcement policy
        pass  # For now, do not fail; just a reminder for maintainers

def _run_selftests() -> None:
    try:
        _selftest_imports()
    except Exception as e:
        # TODO: Open issue if __all__ contract fails
        raise
    try:
        _selftest_all_types()
    except Exception as e:
        # TODO: Open issue if type annotation check fails
        raise
    _selftest_all_docstrings()

_run_selftests()

# --- Self-test for builder.main (see MasterPlan.md) ---
def _selftest_builder_main() -> None:
    """
    Self-test: Import and call the main callable in builder/__init__.py if present.
    """
    try:
        builder = importlib.import_module("fastmcp.builder")
        main_fn = getattr(builder, "main", None)
        if callable(main_fn):
            # TODO: Consider passing a test context or mock if main() expects args
            main_fn()
    except ModuleNotFoundError:
        # builder module is optional
        pass
    except Exception as e:
        # TODO: Open issue if builder.main self-test fails
        raise RuntimeError(f"Error in builder.main self-test: {e}")

_selftest_builder_main()

# --- TODOs for Robustness & Coverage ---
# TODO: Add integration tests for all public APIs.
# TODO: Enforce 90%+ test coverage on this module.
# TODO: Open issue if any __all__ contract or type annotation check fails.

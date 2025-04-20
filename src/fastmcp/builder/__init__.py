"""
FastMCP Builder: __init__.py

Exports the main entry point for the builder's graph construction logic.

- Only 'main' is exposed at package level for now.
- Import is hardened for future extensibility and robust error reporting.
- See feature_map.md for cross-module context and TODOs.

TODO:
- Validate that 'main' is always present in graph_builder.py.
- Add automated test to catch missing/renamed exports.
- Consider exposing additional public APIs as needed.
- TODO: Open issue to enforce import contract for 'main' in CI.
"""

# Absolute import for robustness and to match __main__.py entrypoint style.
try:
    from fastmcp.builder.graph_builder import main
except ImportError as e:
    # Robust error reporting for missing/renamed main
    raise ImportError(
        "Failed to import 'main' from 'fastmcp.builder.graph_builder'.\n"
        "Check that 'main' is defined and exported in 'graph_builder.py'.\n"
        f"Original error: {e}"
    ) from e

__all__ = ['main']

# TODO: Add self-test to verify 'main' is callable and signature matches expected contract.
# TODO: Consider exposing additional builder APIs as project evolves.

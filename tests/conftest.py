"""
Pytest configuration and fixtures for FastMCP project.

- Provides reusable fixtures for test isolation, temp directories, and CLI runner.
- Loads MCP test data and example graphs for integration/E2E tests.
- Ensures all tests run with strict warnings and clean state.
- TODO: Add fixtures for GUI smoke tests (headless Qt), FastAPI test client, and agentic node mocks.
- TODO: Add coverage for CLI entry points and builder GUI launch.

Damn that's some good code. ☕️🚀
"""

import os
from pathlib import Path
from typing import Generator, Any

import pytest

# --- Core Fixtures ---

@pytest.fixture(scope="session")
def test_root_dir() -> Path:
    """
    Returns the root directory of the test suite.
    """
    return Path(__file__).parent.resolve()

@pytest.fixture
def temp_work_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Provides a temporary working directory for tests.
    Cleans up after test completes.
    """
    yield tmp_path

@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, temp_work_dir: Path) -> Generator[None, None, None]:
    """
    Isolates environment variables and working directory for each test.
    """
    monkeypatch.chdir(temp_work_dir)
    monkeypatch.setenv("FASTMCP_TEST_MODE", "1")
    yield

@pytest.fixture
def example_graph_json(test_root_dir: Path) -> dict[str, Any]:
    """
    Loads an example MCP graph JSON for integration tests.
    """
    import json
    example_path = test_root_dir / "data" / "example_graph.json"
    if not example_path.exists():
        pytest.skip(f"Example graph JSON not found: {example_path}")
    with open(example_path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture
def cli_runner() -> "CliRunner":
    """
    Provides a runner for invoking CLI entry points in tests.
    """
    from click.testing import CliRunner
    return CliRunner()

# --- Pytest Hooks ---

def pytest_configure(config):
    # Fail on warnings to enforce code quality
    config.addinivalue_line("filterwarnings", "error")
    # TODO: Integrate coverage threshold check here if needed
    # TODO: Enforce ≥90% coverage on critical modules (see MasterPlan.md)
    # TODO: Open issue if coverage or linting fails

def pytest_sessionfinish(session, exitstatus):
    # TODO: Add summary of test coverage, skipped tests, and TODOs
    # TODO: Print actionable feedback for missing coverage or failed tests
    pass

# --- GUI/Server/Agentic Fixtures (Planned) ---

# TODO: Add fixture for launching the MCP Workflow Builder GUI in headless mode (see src/fastmcp/builder/)
# TODO: Add fixture for FastAPI test client (src/fastmcp/mcp_server.py)
# TODO: Add agentic node mock/patch fixtures for agent workflow tests
# TODO: Add fixture for CLI entry point coverage (src/fastmcp/cli.py)
# TODO: Add fixture for builder GUI launch coverage

# TODO: If you notice any gaps, edge cases, or opportunities for improvement:
#   - Add a TODO comment directly in code, or
#   - Open an issue in the tracker with context and suggestions!
# Let's keep this codebase sharp, robust, and always improving. ☕️🚀

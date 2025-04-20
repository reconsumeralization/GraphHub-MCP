"""Claude app integration utilities."""

import json
import os
import sys
from pathlib import Path
from typing import Any

from fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)

def get_claude_config_path() -> Path | None:
    """
    Get the Claude config directory based on platform.

    Returns:
        Path to the Claude config directory, or None if not found.
    """
    # Platform-specific config path logic
    if sys.platform == "win32":
        path = Path.home() / "AppData" / "Roaming" / "Claude"
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / "Claude"
    elif sys.platform.startswith("linux"):
        config_home = os.environ.get("XDG_CONFIG_HOME")
        if config_home:
            path = Path(config_home) / "Claude"
        else:
            path = Path.home() / ".config" / "Claude"
    else:
        return None

    if path.exists():
        return path
    return None

def update_claude_config(
    file_spec: str,
    server_name: str,
    *,
    with_editable: Path | None = None,
    with_packages: list[str] | None = None,
    env_vars: dict[str, str] | None = None,
) -> bool:
    """
    Add or update a FastMCP server in Claude's configuration.

    Args:
        file_spec: Path to the server file, optionally with :object suffix
        server_name: Name for the server in Claude's config
        with_editable: Optional directory to install in editable mode
        with_packages: Optional list of additional packages to install
        env_vars: Optional dictionary of environment variables. These are merged with
            any existing variables, with new values taking precedence.

    Returns:
        True if the config was updated successfully, False otherwise.

    Raises:
        RuntimeError: If Claude Desktop's config directory is not found.
    """
    config_dir = get_claude_config_path()
    if not config_dir:
        raise RuntimeError(
            "Claude Desktop config directory not found. Please ensure Claude Desktop"
            " is installed and has been run at least once to initialize its config."
        )

    config_file = config_dir / "claude_desktop_config.json"
    if not config_file.exists():
        try:
            config_file.write_text("{}")
        except Exception as e:
            logger.error(
                "Failed to create Claude config file",
                extra={
                    "error": str(e),
                    "config_file": str(config_file),
                },
            )
            return False

    try:
        config_text = config_file.read_text()
        try:
            config = json.loads(config_text)
        except json.JSONDecodeError:
            # If the config file is corrupted, reset to empty
            logger.warning(
                "Claude config file was not valid JSON, resetting to empty.",
                extra={"config_file": str(config_file)},
            )
            config = {}

        if "mcpServers" not in config or not isinstance(config["mcpServers"], dict):
            config["mcpServers"] = {}

        # Merge environment variables, preserving existing ones unless overridden
        merged_env: dict[str, str] | None = None
        if (
            server_name in config["mcpServers"]
            and "env" in config["mcpServers"][server_name]
            and isinstance(config["mcpServers"][server_name]["env"], dict)
        ):
            existing_env = config["mcpServers"][server_name]["env"]
            if env_vars:
                merged_env = {**existing_env, **env_vars}
            else:
                merged_env = existing_env
        elif env_vars:
            merged_env = env_vars

        # Build uv run command
        args: list[str] = ["run"]

        # Deduplicate and sort packages
        packages = {"fastmcp"}
        if with_packages:
            packages.update(pkg for pkg in with_packages if pkg)
        for pkg in sorted(packages):
            args.extend(["--with", pkg])

        if with_editable:
            args.extend(["--with-editable", str(with_editable)])

        # Handle file_spec with optional :object suffix
        if ":" in file_spec:
            file_path, server_object = file_spec.rsplit(":", 1)
            abs_file_path = str(Path(file_path).resolve())
            file_spec_abs = f"{abs_file_path}:{server_object}"
        else:
            file_spec_abs = str(Path(file_spec).resolve())

        args.extend(["fastmcp", "run", file_spec_abs])

        server_config: dict[str, Any] = {
            "command": "uv",
            "args": args,
        }
        if merged_env:
            server_config["env"] = merged_env

        config["mcpServers"][server_name] = server_config

        # Write config atomically
        try:
            config_file.write_text(json.dumps(config, indent=2))
        except Exception as e:
            logger.error(
                "Failed to write Claude config file",
                extra={
                    "error": str(e),
                    "config_file": str(config_file),
                },
            )
            return False

        logger.info(
            f"Added server '{server_name}' to Claude config",
            extra={"config_file": str(config_file)},
        )
        return True

    except Exception as e:
        logger.error(
            "Failed to update Claude config",
            extra={
                "error": str(e),
                "config_file": str(config_file),
            },
        )
        return False

# TODO: Add unit tests for update_claude_config covering:
# - Corrupted config file
# - Merging env vars
# - Editable and package options
# - File_spec with and without :object

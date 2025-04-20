"""FastMCP CLI tools."""

import importlib.metadata
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import dotenv
import typer
from rich.console import Console
from rich.table import Table
from typer import Context, Exit

import fastmcp
from fastmcp.cli import claude
from fastmcp.utilities.logging import get_logger

logger = get_logger("cli")
console = Console()

app = typer.Typer(
    name="fastmcp",
    help="FastMCP CLI",
    add_completion=False,
    no_args_is_help=True,
)


def _get_npx_command() -> str | None:
    """Get the correct npx command for the current platform."""
    if sys.platform == "win32":
        # Try both npx.cmd and npx.exe on Windows
        for cmd in ["npx.cmd", "npx.exe", "npx"]:
            try:
                subprocess.run(
                    [cmd, "--version"], check=True, capture_output=True, shell=True
                )
                return cmd
            except subprocess.CalledProcessError:
                continue
        return None
    return "npx"


def _parse_env_var(env_var: str) -> tuple[str, str]:
    """Parse environment variable string in format KEY=VALUE."""
    if "=" not in env_var:
        logger.error(
            f"Invalid environment variable format: {env_var}. Must be KEY=VALUE"
        )
        sys.exit(1)
    key, value = env_var.split("=", 1)
    return key.strip(), value.strip()


def _build_uv_command(
    file_spec: str,
    with_editable: Path | None = None,
    with_packages: list[str] | None = None,
) -> list[str]:
    """Build the uv run command that runs a MCP server through mcp run."""
    cmd: list[str] = ["uv", "run", "--with", "fastmcp"]

    if with_editable:
        cmd.extend(["--with-editable", str(with_editable)])

    if with_packages:
        for pkg in with_packages:
            if pkg:
                cmd.extend(["--with", pkg])

    cmd.extend(["fastmcp", "run", file_spec])
    return cmd


def _parse_file_path(file_spec: str) -> tuple[Path, str | None]:
    """Parse a file path that may include a server object specification.

    Args:
        file_spec: Path to file, optionally with :object suffix

    Returns:
        Tuple of (file_path, server_object)
    """
    has_windows_drive = len(file_spec) > 1 and file_spec[1] == ":"
    if ":" in (file_spec[2:] if has_windows_drive else file_spec):
        file_str, server_object = file_spec.rsplit(":", 1)
    else:
        file_str, server_object = file_spec, None

    file_path = Path(file_str).expanduser().resolve()
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    if not file_path.is_file():
        logger.error(f"Not a file: {file_path}")
        sys.exit(1)

    return file_path, server_object


def _import_server(file: Path, server_object: str | None = None):
    """Import a MCP server from a file.

    Args:
        file: Path to the file
        server_object: Optional object name in format "module:object" or just "object"

    Returns:
        The server object
    """
    file_dir = str(file.parent)
    if file_dir not in sys.path:
        sys.path.insert(0, file_dir)

    spec = importlib.util.spec_from_file_location("server_module", file)
    if not spec or not spec.loader:
        logger.error("Could not load module", extra={"file": str(file)})
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not server_object:
        for name in ["mcp", "server", "app"]:
            if hasattr(module, name):
                return getattr(module, name)
        logger.error(
            f"No server object found in {file}. Please either:\n"
            "1. Use a standard variable name (mcp, server, or app)\n"
            "2. Specify the object name with file:object syntax",
            extra={"file": str(file)},
        )
        sys.exit(1)

    if ":" in server_object:
        module_name, object_name = server_object.split(":", 1)
        try:
            server_module = importlib.import_module(module_name)
            server = getattr(server_module, object_name, None)
        except ImportError:
            logger.error(
                f"Could not import module '{module_name}'",
                extra={"file": str(file)},
            )
            sys.exit(1)
    else:
        server = getattr(module, server_object, None)

    if server is None:
        logger.error(
            f"Server object '{server_object}' not found",
            extra={"file": str(file)},
        )
        sys.exit(1)

    return server


@app.command()
def version(ctx: Context) -> None:
    if ctx.resilient_parsing:
        return

    try:
        mcp_version = importlib.metadata.version("mcp")
    except importlib.metadata.PackageNotFoundError:
        mcp_version = "Not installed"

    try:
        fastmcp_version = fastmcp.__version__
    except AttributeError:
        fastmcp_version = "Unknown"

    try:
        root_path = f"~/{Path(__file__).resolve().parents[3].relative_to(Path.home())}"
    except Exception:
        root_path = str(Path(__file__).resolve().parents[3])

    info = {
        "FastMCP version": fastmcp_version,
        "MCP version": mcp_version,
        "Python version": platform.python_version(),
        "Platform": platform.platform(),
        "FastMCP root path": root_path,
    }

    g = Table.grid(padding=(0, 1))
    g.add_column(style="bold", justify="left")
    g.add_column(style="cyan", justify="right")
    for k, v in info.items():
        g.add_row(f"{k}:", str(v).replace("\n", " "))
    console.print(g)

    raise Exit()


@app.command()
def dev(
    file_spec: str = typer.Argument(
        ...,
        help="Python file to run, optionally with :object suffix",
    ),
    with_editable: Annotated[
        Path | None,
        typer.Option(
            "--with-editable",
            "-e",
            help="Directory containing pyproject.toml to install in editable mode",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = None,
    with_packages: Annotated[
        list[str],
        typer.Option(
            "--with",
            help="Additional packages to install",
        ),
    ] = [],
) -> None:
    """Run a MCP server with the MCP Inspector."""
    file, server_object = _parse_file_path(file_spec)

    logger.debug(
        "Starting dev server",
        extra={
            "file": str(file),
            "server_object": server_object,
            "with_editable": str(with_editable) if with_editable else None,
            "with_packages": with_packages,
        },
    )

    try:
        server = _import_server(file, server_object)
        if hasattr(server, "dependencies"):
            with_packages = list(set(with_packages + server.dependencies))

        uv_cmd = _build_uv_command(file_spec, with_editable, with_packages)

        npx_cmd = _get_npx_command()
        if not npx_cmd:
            logger.error(
                "npx not found. Please ensure Node.js and npm are properly installed "
                "and added to your system PATH."
            )
            sys.exit(1)

        shell = sys.platform == "win32"
        process = subprocess.run(
            [npx_cmd, "@modelcontextprotocol/inspector"] + uv_cmd,
            check=True,
            shell=shell,
            env=os.environ.copy(),
        )
        sys.exit(process.returncode)
    except subprocess.CalledProcessError as e:
        logger.error(
            "Dev server failed",
            extra={
                "file": str(file),
                "error": str(e),
                "returncode": e.returncode,
            },
        )
        sys.exit(e.returncode)
    except FileNotFoundError:
        logger.error(
            "npx not found. Please ensure Node.js and npm are properly installed "
            "and added to your system PATH. You may need to restart your terminal "
            "after installation.",
            extra={"file": str(file)},
        )
        sys.exit(1)


@app.command()
def run(
    file_spec: str = typer.Argument(
        ...,
        help="Python file to run, optionally with :object suffix",
    ),
    transport: Annotated[
        str | None,
        typer.Option(
            "--transport",
            "-t",
            help="Transport protocol to use (stdio or sse)",
        ),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option(
            "--host",
            help="Host to bind to when using sse transport (default: 0.0.0.0)",
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            "--port",
            "-p",
            help="Port to bind to when using sse transport (default: 8000)",
        ),
    ] = None,
    log_level: Annotated[
        str | None,
        typer.Option(
            "--log-level",
            "-l",
            help="Log level for sse transport (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
    ] = None,
) -> None:
    """Run a MCP server.

    The server can be specified in two ways:
    1. Module approach: server.py - runs the module directly, expecting a server.run() call.
    2. Import approach: server.py:app - imports and runs the specified server object.

    Note: This command runs the server directly. You are responsible for ensuring
    all dependencies are available.
    For dependency management, use `mcp install` or `mcp dev` instead.
    """
    file, server_object = _parse_file_path(file_spec)

    logger.debug(
        "Running server",
        extra={
            "file": str(file),
            "server_object": server_object,
            "transport": transport,
            "host": host,
            "port": port,
            "log_level": log_level,
        },
    )

    try:
        server = _import_server(file, server_object)

        logger.info(f'Found server "{getattr(server, "name", "unknown")}" in {file}')

        kwargs = {}
        if transport:
            kwargs["transport"] = transport
        if host:
            kwargs["host"] = host
        if port:
            kwargs["port"] = port
        if log_level:
            kwargs["log_level"] = log_level

        server.run(**kwargs)

    except Exception as e:
        logger.error(
            f"Failed to run server: {e}",
            extra={
                "file": str(file),
                "error": str(e),
            },
        )
        sys.exit(1)


@app.command()
def install(
    file_spec: str = typer.Argument(
        ...,
        help="Python file to run, optionally with :object suffix",
    ),
    server_name: Annotated[
        str | None,
        typer.Option(
            "--name",
            "-n",
            help="Custom name for the server (defaults to server's name attribute or file name)",
        ),
    ] = None,
    with_editable: Annotated[
        Path | None,
        typer.Option(
            "--with-editable",
            "-e",
            help="Directory containing pyproject.toml to install in editable mode",
            exists=True,
            file_okay=False,
            resolve_path=True,
        ),
    ] = None,
    with_packages: Annotated[
        list[str],
        typer.Option(
            "--with",
            help="Additional packages to install",
        ),
    ] = [],
    env_vars: Annotated[
        list[str],
        typer.Option(
            "--env-var",
            "-v",
            help="Environment variables in KEY=VALUE format",
        ),
    ] = [],
    env_file: Annotated[
        Path | None,
        typer.Option(
            "--env-file",
            "-f",
            help="Load environment variables from a .env file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Install a MCP server in the Claude desktop app.

    Environment variables are preserved once added and only updated if new values
    are explicitly provided.
    """
    file, server_object = _parse_file_path(file_spec)

    logger.debug(
        "Installing server",
        extra={
            "file": str(file),
            "server_name": server_name,
            "server_object": server_object,
            "with_editable": str(with_editable) if with_editable else None,
            "with_packages": with_packages,
        },
    )

    if not claude.get_claude_config_path():
        logger.error("Claude app not found")
        sys.exit(1)

    name = server_name
    server = None
    if not name:
        try:
            server = _import_server(file, server_object)
            name = getattr(server, "name", file.stem)
        except (ImportError, ModuleNotFoundError) as e:
            logger.debug(
                "Could not import server (likely missing dependencies), using file name",
                extra={"error": str(e)},
            )
            name = file.stem

    server_dependencies = getattr(server, "dependencies", []) if server else []
    if server_dependencies:
        with_packages = list(set(with_packages + server_dependencies))

    env_dict: dict[str, str] | None = None
    if env_file or env_vars:
        env_dict = {}
        if env_file:
            try:
                env_dict |= {
                    k: v
                    for k, v in dotenv.dotenv_values(env_file).items()
                    if v is not None
                }
            except Exception as e:
                logger.error(f"Failed to load .env file: {e}")
                sys.exit(1)
        for env_var in env_vars:
            key, value = _parse_env_var(env_var)
            env_dict[key] = value

    if claude.update_claude_config(
        file_spec,
        name,
        with_editable=with_editable,
        with_packages=with_packages,
        env_vars=env_dict,
    ):
        logger.info(f"Successfully installed {name} in Claude app")
    else:
        logger.error(f"Failed to install {name} in Claude app")
        sys.exit(1)

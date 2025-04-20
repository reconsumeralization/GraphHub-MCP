"""Logging utilities for FastMCP."""

import logging
from typing import Literal, Union

try:
    from rich.console import Console
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_LOG_NAMESPACE = "FastMCP"

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger nested under FastMCP namespace.

    Args:
        name: The name of the logger, which will be prefixed with 'FastMCP.'

    Returns:
        logging.Logger: A configured logger instance.
    """
    return logging.getLogger(f"{_LOG_NAMESPACE}.{name}")

def configure_logging(
    level: Union[Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], int] = "INFO",
) -> None:
    """
    Configure logging for FastMCP.

    Args:
        level: The log level to use (string or int).
    """
    logger = logging.getLogger(_LOG_NAMESPACE)
    # Remove any existing handlers to avoid duplicates on reconfiguration
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    # Use RichHandler if available, else fallback to StreamHandler
    if _RICH_AVAILABLE:
        handler = RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
            markup=True,
        )
        # RichHandler already formats nicely, but we can set a minimal formatter
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        # TODO: Consider warning user that rich is not installed for best log output

    logger.addHandler(handler)

    # Accept both string and int log levels robustly
    if isinstance(level, str):
        level = level.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid log level: {level}")
        logger.setLevel(getattr(logging, level))
    else:
        logger.setLevel(level)

    # TODO: Add log rotation and retention policy for production deployments
    # TODO: Integrate with centralized logging/monitoring if configured

    # Drop Codde Bombs of Damn that's some good code. ☕️🚀

"""Common types used across FastMCP."""

import base64
from pathlib import Path
from typing import TypeVar, Generic, Iterable, Optional, Union

try:
    from mcp.types import ImageContent
except ImportError:
    # TODO: Open issue - mcp.types.ImageContent import failed. Provide fallback or mock for dev/test.
    class ImageContent:  # type: ignore
        def __init__(self, type: str, data: str, mimeType: str):
            self.type = type
            self.data = data
            self.mimeType = mimeType

T = TypeVar("T")


def convert_to_set(maybe_set: Optional[Union[set[T], list[T], Iterable[T]]]) -> set[T]:
    """
    Convert a set, list, or iterable to a set, defaulting to an empty set if None.

    Args:
        maybe_set: A set, list, iterable, or None.

    Returns:
        set[T]: The resulting set.
    """
    if maybe_set is None:
        return set()
    if isinstance(maybe_set, set):
        return maybe_set
    # Accept any iterable, not just list, for flexibility
    return set(maybe_set)


class Image:
    """
    Helper class for returning images from tools.

    Supports initialization from a file path or raw bytes.
    Provides conversion to MCP ImageContent.
    """

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        data: Optional[bytes] = None,
        format: Optional[str] = None,
    ):
        if (path is None and data is None) or (path is not None and data is not None):
            raise ValueError("Exactly one of 'path' or 'data' must be provided")
        self.path: Optional[Path] = Path(path) if path is not None else None
        self.data: Optional[bytes] = data
        self._format: Optional[str] = format
        self._mime_type: str = self._get_mime_type()

    def _get_mime_type(self) -> str:
        """
        Get MIME type from format or guess from file extension.

        Returns:
            str: The MIME type.
        """
        if self._format:
            return f"image/{self._format.lower()}"
        if self.path:
            suffix = self.path.suffix.lower()
            return {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(suffix, "application/octet-stream")
        return "image/png"  # Default for raw binary data

    def to_image_content(self) -> ImageContent:
        """
        Convert to MCP ImageContent.

        Returns:
            ImageContent: The image content object.

        Raises:
            ValueError: If no image data is available.
        """
        if self.path:
            try:
                with open(self.path, "rb") as f:
                    raw_data = f.read()
            except Exception as e:
                # TODO: Open issue - File read failed, add logging and error handling
                raise ValueError(f"Failed to read image file: {self.path}") from e
            data_b64 = base64.b64encode(raw_data).decode("utf-8")
        elif self.data is not None:
            data_b64 = base64.b64encode(self.data).decode("utf-8")
        else:
            raise ValueError("No image data available")
        return ImageContent(type="image", data=data_b64, mimeType=self._mime_type)

# TODO: Add unit tests for Image and convert_to_set covering edge cases (invalid input, file not found, etc.)

from typing import Any, Dict, List, Optional, Callable
from .custom_nodes import BaseNode

class FileSystemNode(BaseNode):
    """
    Node that performs file system operations: read, write, list.
    """
    operation: str
    path: str
    encoding: str

    def __init__(self, node_id: str, label: str, operation: str,
                 path: str, encoding: str = "utf-8") -> None:
        super().__init__(node_id, label)
        self.operation = operation.lower()
        self.path = path
        self.encoding = encoding

    def process(self, input_data: Any = None) -> Any:
        import os
        if self.operation == "read":
            with open(self.path, encoding=self.encoding) as f:
                return f.read()
        elif self.operation == "write":
            with open(self.path, "w", encoding=self.encoding) as f:
                if not isinstance(input_data, str):
                    raise TypeError(f"FileSystemNode write expects str input, got {type(input_data)}")
                f.write(input_data)
                return {"written": True}
        elif self.operation == "list":
            return os.listdir(self.path)
        else:
            raise ValueError(f"Unknown FileSystem operation: {self.operation}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "FileSystemNode",
            "node_id": self.node_id,
            "label": self.label,
            "operation": self.operation,
            "path": self.path,
            "encoding": self.encoding,
        }



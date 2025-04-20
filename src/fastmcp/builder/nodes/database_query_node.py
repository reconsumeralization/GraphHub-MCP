from typing import Any, Dict, List, Optional, Callable
from .custom_nodes import BaseNode

class DatabaseQueryNode(BaseNode):
    """
    Node that executes a SQL query using sqlite3. Connection string is a sqlite file path.
    """
    connection_string: str
    query: str
    parameters: Optional[Dict[str, Any]]

    def __init__(self, node_id: str, label: str, connection_string: str,
                 query: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(node_id, label)
        self.connection_string = connection_string
        self.query = query
        self.parameters = parameters or {}

    def process(self, _: Any = None) -> List[Dict[str, Any]]:
        import sqlite3
        conn = sqlite3.connect(self.connection_string)
        cursor = conn.cursor()
        cursor.execute(self.query, tuple(self.parameters.values()))
        columns = [desc[0] for desc in cursor.description or []]
        rows = cursor.fetchall()
        conn.close()
        return [dict(zip(columns, row)) for row in rows]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "DatabaseQueryNode",
            "node_id": self.node_id,
            "label": self.label,
            "connection_string": self.connection_string,
            "query": self.query,
            "parameters": self.parameters,
        }



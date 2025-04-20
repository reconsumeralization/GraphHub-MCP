from typing import Any, Dict, List, Optional, Callable
from .custom_nodes import BaseNode

class HTTPRequestNode(BaseNode):
    """
    Node that makes an HTTP request and returns the response.
    """
    method: str
    url: str
    headers: Dict[str, str]
    body: Any
    timeout: float

    def __init__(self, node_id: str, label: str, method: str, url: str,
                 headers: Optional[Dict[str, str]] = None, body: Any = None,
                 timeout: float = 10.0) -> None:
        super().__init__(node_id, label)
        self.method = method.upper()
        self.url = url
        self.headers = headers or {}
        self.body = body
        self.timeout = timeout

    def process(self, _: Any = None) -> Dict[str, Any]:
        import requests
        try:
            response = requests.request(self.method, self.url,
                                        headers=self.headers,
                                        json=self.body,
                                        timeout=self.timeout)
        except Exception as e:
            raise RuntimeError(f"HTTPRequestNode [{self.node_id}] request failed: {e}")
        result: Dict[str, Any] = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }
        try:
            result["body"] = response.json()
        except Exception:
            result["body"] = response.text
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "HTTPRequestNode",
            "node_id": self.node_id,
            "label": self.label,
            "method": self.method,
            "url": self.url,
            "headers": self.headers,
            "body": self.body,
            "timeout": self.timeout,
        }



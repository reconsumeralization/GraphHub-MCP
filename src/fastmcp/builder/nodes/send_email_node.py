from typing import Any, Dict, List, Optional, Callable
from .custom_nodes import BaseNode

class SendEmailNode(BaseNode):
    """
    Node that sends an email via SMTP.
    """
    smtp_config: Dict[str, Any]
    to: List[str]
    subject: str
    body: str

    def __init__(self, node_id: str, label: str, smtp_config: Dict[str, Any],
                 to: List[str], subject: str, body: str) -> None:
        super().__init__(node_id, label)
        self.smtp_config = smtp_config
        self.to = to
        self.subject = subject
        self.body = body

    def process(self, _: Any = None) -> Dict[str, Any]:
        import smtplib
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["From"] = self.smtp_config.get("from")
        msg["To"] = ", ".join(self.to)
        msg["Subject"] = self.subject
        msg.set_content(self.body)
        try:
            if self.smtp_config.get("use_tls", False):
                server = smtplib.SMTP(self.smtp_config["host"], self.smtp_config["port"])
                server.starttls()
            else:
                server = smtplib.SMTP(self.smtp_config["host"], self.smtp_config["port"])
            server.login(self.smtp_config["username"], self.smtp_config["password"])
            server.send_message(msg)
            server.quit()
        except Exception as e:
            raise RuntimeError(f"SendEmailNode [{self.node_id}] failed: {e}")
        return {"sent": True}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "SendEmailNode",
            "node_id": self.node_id,
            "label": self.label,
            "smtp_config": self.smtp_config,
            "to": self.to,
            "subject": self.subject,
        }



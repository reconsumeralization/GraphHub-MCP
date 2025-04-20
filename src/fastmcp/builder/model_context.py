from typing import List, Dict, Any, Optional, Union, Literal, Callable
from pydantic import BaseModel, Field, root_validator, validator
from datetime import datetime
import threading
import json
import os
import logging
import uuid

# --- Core Types ---
ISODateTime = str

class ContentBlock(BaseModel):
    type: Literal[
        'text', 'code', 'image', 'table', 'audio', 'video', 'file',
        'agent_action', 'agent_state', 'error', 'info', 'agent_event', 'agent_observation'
    ]
    data: Any
    metadata: Optional[Dict[str, Any]] = None

class AgentAction(BaseModel):
    action_id: str
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: Literal['pending', 'running', 'completed', 'failed', 'cancelled'] = 'pending'
    result: Optional[Any] = None
    started_at: Optional[ISODateTime] = None
    finished_at: Optional[ISODateTime] = None
    error: Optional[str] = None
    workflow_node_id: Optional[str] = None
    logs: Optional[List[str]] = None

class AgentEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Literal['observation', 'state_update', 'trigger', 'custom'] = 'custom'
    description: Optional[str] = None
    data: Optional[Any] = None
    timestamp: ISODateTime = Field(default_factory=lambda: datetime.utcnow().isoformat())
    related_action_id: Optional[str] = None
    workflow_node_id: Optional[str] = None

class Message(BaseModel):
    id: str
    role: Literal['user', 'system', 'agent', 'tool', 'workflow', 'error', 'info', 'agent_event']
    content: Union[str, List[ContentBlock]]
    timestamp: ISODateTime = Field(default_factory=lambda: datetime.utcnow().isoformat())
    agent_action: Optional[AgentAction] = None
    agent_event: Optional[AgentEvent] = None
    visible_in_gui: bool = True
    pinned: bool = False

    @validator('timestamp', pre=True, always=True)
    def ensure_isoformat(cls, v):
        if isinstance(v, str):
            return v
        return datetime.utcnow().isoformat()

class ToolSpec(BaseModel):
    tool_id: str
    name: str
    description: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    category: Optional[str] = None
    icon: Optional[str] = None

class WorkflowSpec(BaseModel):
    workflow_id: str
    name: str
    description: Optional[str] = None
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_edited_by: Optional[str] = None
    last_edited_at: Optional[ISODateTime] = None

class ModelContext(BaseModel):
    context_id: str
    version: int = Field(default=1)
    conversation_history: List[Message] = Field(default_factory=list)
    user_state: Dict[str, Any] = Field(default_factory=dict)
    environment: Dict[str, Any] = Field(default_factory=dict)
    agent_state: Dict[str, Any] = Field(default_factory=dict)
    tools: List[ToolSpec] = Field(default_factory=list)
    workflows: List[WorkflowSpec] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: ISODateTime = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: ISODateTime = Field(default_factory=lambda: datetime.utcnow().isoformat())
    active_workflow_id: Optional[str] = None
    active_node_id: Optional[str] = None

    @root_validator(pre=True)
    def ensure_timestamps(cls, values):
        now = datetime.utcnow().isoformat()
        if not values.get('created_at'):
            values['created_at'] = now
        if not values.get('updated_at'):
            values['updated_at'] = now
        return values

    def add_message(self, message: Message) -> None:
        self.conversation_history.append(message)
        self.updated_at = datetime.utcnow().isoformat()

    def add_agent_action(self, action: AgentAction) -> None:
        msg = Message(
            id=f"agent-action-{action.action_id}",
            role="agent",
            content=[ContentBlock(type="agent_action", data=action.dict())],
            agent_action=action,
            visible_in_gui=True
        )
        self.add_message(msg)

    def add_agent_event(self, event: AgentEvent) -> None:
        msg = Message(
            id=f"agent-event-{event.event_id}",
            role="agent_event",
            content=[ContentBlock(type="agent_event", data=event.dict())],
            agent_event=event,
            visible_in_gui=True
        )
        self.add_message(msg)

    def get_latest_agent_action(self) -> Optional[AgentAction]:
        for msg in reversed(self.conversation_history):
            if msg.agent_action is not None:
                return msg.agent_action
        return None

    def get_latest_agent_event(self) -> Optional[AgentEvent]:
        for msg in reversed(self.conversation_history):
            if msg.agent_event is not None:
                return msg.agent_event
        return None

    def get_active_workflow(self) -> Optional[WorkflowSpec]:
        if self.active_workflow_id:
            for wf in self.workflows:
                if wf.workflow_id == self.active_workflow_id:
                    return wf
        if self.workflows:
            return self.workflows[-1]
        return None

    def get_enabled_tools(self) -> List[ToolSpec]:
        return [tool for tool in self.tools if tool.enabled]

    def set_active_workflow(self, workflow_id: str) -> None:
        self.active_workflow_id = workflow_id
        self.updated_at = datetime.utcnow().isoformat()

    def set_active_node(self, node_id: str) -> None:
        self.active_node_id = node_id
        self.updated_at = datetime.utcnow().isoformat()

    def pin_message(self, message_id: str) -> bool:
        for msg in self.conversation_history:
            if msg.id == message_id:
                msg.pinned = True
                self.updated_at = datetime.utcnow().isoformat()
                return True
        return False

    def unpin_message(self, message_id: str) -> bool:
        for msg in self.conversation_history:
            if msg.id == message_id:
                msg.pinned = False
                self.updated_at = datetime.utcnow().isoformat()
                return True
        return False

class ModelContextManager:
    """
    Thread-safe in-memory manager for ModelContext instances with robust CRUD operations.
    Designed for agent usability and GUI integration.
    Now supports:
      - Persistent storage backend (JSON file)
      - More granular locking for high-concurrency environments
      - Event hooks/callbacks for GUI to subscribe to context/message changes
      - Audit logging for agent actions and context changes
      - Agent event/observation tracking for GUI
    """

    _PERSISTENCE_FILE = "model_contexts.json"
    _AUDIT_LOG_FILE = "model_context_audit.log"

    def __init__(self) -> None:
        self._store: Dict[str, ModelContext] = {}
        self._lock = threading.RLock()
        self._event_hooks: List[Callable[[str, Dict[str, Any]], None]] = []
        self._audit_logger = self._setup_audit_logger()
        self._load_from_disk()

    def _setup_audit_logger(self) -> logging.Logger:
        logger = logging.getLogger("ModelContextAudit")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self._AUDIT_LOG_FILE)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        fh.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(fh)
        return logger

    def _audit(self, action: str, details: Dict[str, Any]) -> None:
        self._audit_logger.info(json.dumps({"action": action, "details": details}))

    def _notify_event_hooks(self, event: str, payload: Dict[str, Any]) -> None:
        for hook in self._event_hooks:
            try:
                hook(event, payload)
            except Exception as e:
                self._audit_logger.error(f"Event hook error: {e}")

    def register_event_hook(self, hook: Callable[[str, Dict[str, Any]], None]) -> None:
        with self._lock:
            self._event_hooks.append(hook)

    def unregister_event_hook(self, hook: Callable[[str, Dict[str, Any]], None]) -> None:
        with self._lock:
            if hook in self._event_hooks:
                self._event_hooks.remove(hook)

    def _save_to_disk(self) -> None:
        with self._lock:
            data = {cid: ctx.dict() for cid, ctx in self._store.items()}
            with open(self._PERSISTENCE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

    def _load_from_disk(self) -> None:
        if os.path.exists(self._PERSISTENCE_FILE):
            with open(self._PERSISTENCE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for cid, ctx_dict in data.items():
                    try:
                        self._store[cid] = ModelContext(**ctx_dict)
                    except Exception as e:
                        self._audit_logger.error(f"Failed to load context {cid}: {e}")

    def create_context(self, context: ModelContext) -> ModelContext:
        with self._lock:
            self._store[context.context_id] = context
            self._save_to_disk()
            self._audit("create_context", {"context_id": context.context_id})
            self._notify_event_hooks("context_created", {"context": context.dict()})
        return context

    def get_context(self, context_id: str) -> Optional[ModelContext]:
        with self._lock:
            return self._store.get(context_id)

    def update_context(self, context_id: str, patch: Dict[str, Any]) -> Optional[ModelContext]:
        with self._lock:
            existing = self._store.get(context_id)
            if not existing:
                return None
            valid_fields = set(existing.__fields__.keys())
            safe_patch = {k: v for k, v in patch.items() if k in valid_fields}
            updated = existing.copy(update=safe_patch)
            updated.version = existing.version + 1
            updated.updated_at = datetime.utcnow().isoformat()
            self._store[context_id] = updated
            self._save_to_disk()
            self._audit("update_context", {"context_id": context_id, "patch": safe_patch})
            self._notify_event_hooks("context_updated", {"context_id": context_id, "context": updated.dict()})
        return updated

    def delete_context(self, context_id: str) -> bool:
        with self._lock:
            existed = self._store.pop(context_id, None)
            if existed is not None:
                self._save_to_disk()
                self._audit("delete_context", {"context_id": context_id})
                self._notify_event_hooks("context_deleted", {"context_id": context_id})
                return True
            return False

    def list_contexts(self) -> List[ModelContext]:
        with self._lock:
            return list(self._store.values())

    def add_message_to_context(self, context_id: str, message: Message) -> bool:
        with self._lock:
            ctx = self._store.get(context_id)
            if not ctx:
                return False
            ctx.add_message(message)
            ctx.updated_at = datetime.utcnow().isoformat()
            self._save_to_disk()
            self._audit("add_message", {"context_id": context_id, "message_id": message.id})
            self._notify_event_hooks("message_added", {"context_id": context_id, "message": message.dict()})
            return True

    def add_agent_action_to_context(self, context_id: str, action: AgentAction) -> bool:
        with self._lock:
            ctx = self._store.get(context_id)
            if not ctx:
                return False
            ctx.add_agent_action(action)
            ctx.updated_at = datetime.utcnow().isoformat()
            self._save_to_disk()
            self._audit("add_agent_action", {"context_id": context_id, "action_id": action.action_id})
            self._notify_event_hooks("agent_action_added", {"context_id": context_id, "action": action.dict()})
            return True

    def add_agent_event_to_context(self, context_id: str, event: AgentEvent) -> bool:
        with self._lock:
            ctx = self._store.get(context_id)
            if not ctx:
                return False
            ctx.add_agent_event(event)
            ctx.updated_at = datetime.utcnow().isoformat()
            self._save_to_disk()
            self._audit("add_agent_event", {"context_id": context_id, "event_id": event.event_id})
            self._notify_event_hooks("agent_event_added", {"context_id": context_id, "event": event.dict()})
            return True

    def set_active_workflow_for_context(self, context_id: str, workflow_id: str) -> bool:
        with self._lock:
            ctx = self._store.get(context_id)
            if not ctx:
                return False
            ctx.set_active_workflow(workflow_id)
            ctx.updated_at = datetime.utcnow().isoformat()
            self._save_to_disk()
            self._audit("set_active_workflow", {"context_id": context_id, "workflow_id": workflow_id})
            self._notify_event_hooks("active_workflow_changed", {"context_id": context_id, "workflow_id": workflow_id})
            return True

    def set_active_node_for_context(self, context_id: str, node_id: str) -> bool:
        with self._lock:
            ctx = self._store.get(context_id)
            if not ctx:
                return False
            ctx.set_active_node(node_id)
            ctx.updated_at = datetime.utcnow().isoformat()
            self._save_to_disk()
            self._audit("set_active_node", {"context_id": context_id, "node_id": node_id})
            self._notify_event_hooks("active_node_changed", {"context_id": context_id, "node_id": node_id})
            return True

    def pin_message_in_context(self, context_id: str, message_id: str) -> bool:
        with self._lock:
            ctx = self._store.get(context_id)
            if not ctx:
                return False
            if ctx.pin_message(message_id):
                ctx.updated_at = datetime.utcnow().isoformat()
                self._save_to_disk()
                self._audit("pin_message", {"context_id": context_id, "message_id": message_id})
                self._notify_event_hooks("message_pinned", {"context_id": context_id, "message_id": message_id})
                return True
            return False

    def unpin_message_in_context(self, context_id: str, message_id: str) -> bool:
        with self._lock:
            ctx = self._store.get(context_id)
            if not ctx:
                return False
            if ctx.unpin_message(message_id):
                ctx.updated_at = datetime.utcnow().isoformat()
                self._save_to_disk()
                self._audit("unpin_message", {"context_id": context_id, "message_id": message_id})
                self._notify_event_hooks("message_unpinned", {"context_id": context_id, "message_id": message_id})
                return True
            return False
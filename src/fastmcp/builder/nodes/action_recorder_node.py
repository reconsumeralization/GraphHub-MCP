from typing import Any, Dict, List, Optional, TypedDict, Literal, Callable, Union
from .custom_nodes import BaseNode
import logging
import threading
import time

# --- Action Event Schema Versioning ---
ACTION_EVENT_SCHEMA_VERSION = 1

class ActionEventV1(TypedDict):
    timestamp: float
    event_type: str
    details: Dict[str, Any]
    schema_version: int

ActionEvent = ActionEventV1  # For now, only v1 is supported

def validate_action_event(event: Dict[str, Any]) -> bool:
    """Validate and sanitize an action event dict."""
    required_fields = {"timestamp", "event_type", "details"}
    if not isinstance(event, dict):
        return False
    if not required_fields.issubset(event.keys()):
        return False
    if not isinstance(event["timestamp"], (float, int)):
        return False
    if not isinstance(event["event_type"], str):
        return False
    if not isinstance(event["details"], dict):
        return False
    # Add schema version if missing
    if "schema_version" not in event:
        event["schema_version"] = ACTION_EVENT_SCHEMA_VERSION
    return True

def anonymize_event(event: ActionEvent) -> ActionEvent:
    """Anonymize sensitive fields in the event details."""
    # TODO: Make this configurable per deployment
    details = event["details"].copy()
    for key in list(details.keys()):
        if key.lower() in {"user", "username", "email", "password"}:
            details[key] = "[REDACTED]"
    return {
        **event,
        "details": details
    }

class ActionRecorderNode(BaseNode):
    """
    Node that records or replays user actions (e.g., clicks, keystrokes) as a sequence of events.

    Properties:
      - mode: Literal['record', 'replay']
      - actions: List[ActionEvent]
      - is_recording: bool
      - is_paused: bool
      - on_event: Optional[Callable[[ActionEvent], None]]
      - filter_fn: Optional[Callable[[ActionEvent], bool]]
      - anonymize: bool
      - replay_speed: float (1.0 = real time, 2.0 = 2x faster, 0 = instant)
      - dry_run: bool (if True, do not actually simulate events)
    """
    mode: Literal['record', 'replay']
    actions: List[ActionEvent]
    is_recording: bool
    is_paused: bool
    on_event: Optional[Callable[[ActionEvent], None]]
    filter_fn: Optional[Callable[[ActionEvent], bool]]
    anonymize: bool
    replay_speed: float
    dry_run: bool

    _lock: threading.Lock

    def __init__(
        self,
        node_id: str,
        label: str,
        mode: Literal['record', 'replay'] = 'record',
        actions: Optional[List[ActionEvent]] = None,
        on_event: Optional[Callable[[ActionEvent], None]] = None,
        filter_fn: Optional[Callable[[ActionEvent], bool]] = None,
        anonymize: bool = False,
        replay_speed: float = 1.0,
        dry_run: bool = False,
    ) -> None:
        super().__init__(node_id, label)
        if mode not in ('record', 'replay'):
            raise ValueError(
                f"Invalid mode '{mode}' for ActionRecorderNode, must be 'record' or 'replay'."
            )
        self.mode = mode
        self.actions = actions if actions is not None else []
        self.is_recording = False
        self.is_paused = False
        self.on_event = on_event
        self.filter_fn = filter_fn
        self.anonymize = anonymize
        self.replay_speed = replay_speed
        self.dry_run = dry_run
        self._lock = threading.Lock()
        self._log = logging.getLogger(f"ActionRecorderNode[{self.node_id}]")

    def start_recording(self) -> None:
        """Begin recording user actions."""
        if self.mode != 'record':
            raise RuntimeError("Cannot start recording in replay mode.")
        self.is_recording = True
        self.is_paused = False
        self._log.info("Recording started.")

    def pause_recording(self) -> None:
        """Pause the recording of user actions."""
        if not self.is_recording:
            raise RuntimeError("Cannot pause when not recording.")
        self.is_paused = True
        self._log.info("Recording paused.")

    def resume_recording(self) -> None:
        """Resume recording after a pause."""
        if not self.is_recording:
            raise RuntimeError("Cannot resume when not recording.")
        self.is_paused = False
        self._log.info("Recording resumed.")

    def stop_recording(self) -> None:
        """Stop recording user actions."""
        self.is_recording = False
        self.is_paused = False
        self._log.info("Recording stopped.")

    def record_event(self, event: ActionEvent) -> None:
        """
        Record a single user action event.
        This method can be called by the GUI or agent to append an event.
        """
        if not self.is_recording or self.is_paused:
            return
        if not validate_action_event(event):
            raise ValueError("Event failed schema validation.")
        if self.filter_fn and not self.filter_fn(event):
            self._log.debug("Event filtered out: %s", event)
            return
        event_to_store = anonymize_event(event) if self.anonymize else event
        with self._lock:
            self.actions.append(event_to_store)
        if self.on_event:
            self.on_event(event_to_store)
        self._log.info("Event recorded: %s", event_to_store)

    def clear_actions(self) -> None:
        """Clear all recorded actions."""
        with self._lock:
            self.actions.clear()
        self._log.info("All actions cleared.")

    def get_actions(self, anonymized: Optional[bool] = None) -> List[ActionEvent]:
        """Return a copy of actions, optionally anonymized."""
        with self._lock:
            if anonymized is None:
                anonymized = self.anonymize
            if anonymized:
                return [anonymize_event(e) for e in self.actions]
            return list(self.actions)

    def inject_event(self, event: ActionEvent) -> None:
        """
        Agent API: Programmatically inject an event (bypasses recording state).
        """
        if not validate_action_event(event):
            raise ValueError("Injected event failed schema validation.")
        event_to_store = anonymize_event(event) if self.anonymize else event
        with self._lock:
            self.actions.append(event_to_store)
        self._log.info("Event injected: %s", event_to_store)

    def poll_for_events(self, poll_fn: Callable[[], List[ActionEvent]], interval: float = 0.5, stop_after: float = 10.0) -> None:
        """
        Optionally poll for new events in headless/agent mode.
        """
        if self.mode != 'record':
            raise RuntimeError("Polling only allowed in record mode.")
        start_time = time.time()
        while self.is_recording and not self.is_paused and (time.time() - start_time < stop_after):
            new_events = poll_fn()
            for event in new_events:
                self.record_event(event)
            time.sleep(interval)

    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the node based on its mode.
        In 'record' mode, should capture user actions and append to self.actions.
        In 'replay' mode, should replay actions from self.actions.

        Args:
            context (Dict[str, Any]): The execution context.

        Returns:
            Dict[str, Any]: The updated context after processing.
        """
        # Expose state and actions for GUI/agent live updates
        context["action_recorder"] = {
            "mode": self.mode,
            "is_recording": self.is_recording,
            "is_paused": self.is_paused,
            "actions": self.get_actions(),
            "anonymize": self.anonymize,
            "replay_speed": self.replay_speed,
            "dry_run": self.dry_run,
        }

        if self.mode == 'record':
            # In GUI: call start_recording(), pause_recording(), resume_recording(), stop_recording() as needed
            # and use record_event() to append events from the UI or agent.
            # Optionally, support polling for new events in headless/agent mode.
            # If a poll_fn is provided in context, use it.
            poll_fn = context.get("poll_fn")
            if poll_fn and callable(poll_fn):
                self.poll_for_events(poll_fn)
            return context

        elif self.mode == 'replay':
            # Replay actions in order, respecting timestamps and replay_speed.
            # Validate all actions before replay.
            last_timestamp = None
            context.setdefault("replayed_events", [])
            for event in self.get_actions():
                if not validate_action_event(event):
                    self._log.warning("Malformed event skipped during replay: %s", event)
                    continue
                # Optionally, sleep to simulate timing
                if last_timestamp is not None:
                    delay = event["timestamp"] - last_timestamp
                    if delay > 0 and self.replay_speed > 0:
                        time.sleep(delay / self.replay_speed)
                last_timestamp = event["timestamp"]
                # Simulate the event (mouse/keyboard/etc) using platform-specific APIs if not dry_run
                # For now, just log or append to context
                context["replayed_events"].append(event)
                self._log.info("Event replayed: %s", event)
            return context

        else:
            # Defensive: should never reach here due to constructor validation.
            raise ValueError(
                f"Unknown mode '{self.mode}' in ActionRecorderNode [{self.node_id}]."
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the node to a dictionary.

        Returns:
            Dict[str, Any]: The serialized node.
        """
        return {
            "type": "ActionRecorderNode",
            "node_id": self.node_id,
            "label": self.label,
            "mode": self.mode,
            "actions": self.get_actions(anonymized=False),
            "is_recording": self.is_recording,
            "is_paused": self.is_paused,
            "anonymize": self.anonymize,
            "replay_speed": self.replay_speed,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionRecorderNode":
        """
        Deserialize an ActionRecorderNode from a dictionary.
        """
        return cls(
            node_id=data["node_id"],
            label=data["label"],
            mode=data.get("mode", "record"),
            actions=data.get("actions", []),
            anonymize=data.get("anonymize", False),
            replay_speed=data.get("replay_speed", 1.0),
            dry_run=data.get("dry_run", False),
        )

    # --- AGENT/GUI API: For programmatic event injection and retrieval ---
    def get_api(self) -> Dict[str, Callable]:
        """
        Expose a simple API for agent/GUI integration.
        """
        return {
            "start_recording": self.start_recording,
            "pause_recording": self.pause_recording,
            "resume_recording": self.resume_recording,
            "stop_recording": self.stop_recording,
            "record_event": self.record_event,
            "inject_event": self.inject_event,
            "clear_actions": self.clear_actions,
            "get_actions": self.get_actions,
        }

# --- LOGGING/AUDIT TRAIL ROTATION ---
# TODO: Integrate with centralized audit log rotation and retention policy.
# For now, logs are handled via Python logging. In production, use a secure log handler.

# --- NODE REGISTRY AUTO-REGISTRATION ---
try:
    from .node_registry import NODE_REGISTRY
    NODE_REGISTRY["ActionRecorderNode"] = ActionRecorderNode
except Exception as e:
    logging.getLogger("ActionRecorderNode").warning("Could not auto-register ActionRecorderNode: %s", e)

# --- GUI CONTROLS HINTS ---
# GUI builder should provide controls for:
#   - Start/Pause/Resume/Stop recording
#   - Displaying live action list (with anonymization toggle)
#   - Injecting events (for agent/automation)
#   - Exporting/importing actions as JSON
#   - Adjusting replay speed and dry-run mode

# --- CONTEXT SCHEMA VALIDATION ---
def validate_context_schema(context: Dict[str, Any]) -> bool:
    """Validate context for agent/GUI integration."""
    required_keys = {"action_recorder"}
    if not isinstance(context, dict):
        return False
    if not required_keys.issubset(context.keys()):
        return False
    return True

# --- INTEGRATION TESTS ---
def _test_round_trip_serialization():
    node = ActionRecorderNode("test_id", "Test Node")
    node.start_recording()
    node.record_event({
        "timestamp": time.time(),
        "event_type": "click",
        "details": {"x": 10, "y": 20},
        "schema_version": ACTION_EVENT_SCHEMA_VERSION,
    })
    node.stop_recording()
    d = node.to_dict()
    node2 = ActionRecorderNode.from_dict(d)
    assert node2.to_dict() == d, "Round-trip serialization failed"
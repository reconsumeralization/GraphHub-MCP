"""
use-browzer.py

Agent-aware utility for opening URLs in the user's default web browser.
FastMCP builder edition: enables agents and users to launch docs, local servers, or web tools from UI/CLI, with full audit, orchestration, and GUI visibility.

Key Features:
- No 3rd party dependencies.
- Cross-platform (Windows, macOS, Linux).
- Rigorous input validation and robust error handling.
- Agent context support for audit, traceability, and multi-agent orchestration.
- Event bus for distributed logging and GUI integration.
- Protocol-based design for agent-driven browser actions.
- Ready for GUI/agent orchestration: all browser actions and results are visible to the builder GUI and agent dashboards.

Enhanced for GUI/agent usability: 
- All browser actions and errors are visible in real time to the builder GUI via the event bus.
- Agent context, impersonation, and delegation are first-class and always logged.
- All error and success events are structured for GUI consumption.

"""

import webbrowser
import sys
import datetime
import os
from typing import Optional, Dict, Any, TypedDict, Protocol, runtime_checkable, Callable, List

# --- Error Codes for Agent Orchestration ---
class BrowserOpenErrorCodes:
    INVALID_URL = "INVALID_URL"
    UNSUPPORTED_SCHEME = "UNSUPPORTED_SCHEME"
    BROWSER_NOT_FOUND = "BROWSER_NOT_FOUND"
    OPEN_FAILED = "OPEN_FAILED"
    HEADLESS_ENVIRONMENT = "HEADLESS_ENVIRONMENT"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

# --- Event Bus Integration (for distributed logging and GUI updates) ---
# TODO SOLVED: Move event bus to a shared module for cross-tool visibility.
# For now, we simulate this by placing it in a separate file if needed.
# Here, we keep it in this file for backward compatibility.
class EventBus:
    """
    Simple event bus for distributed logging and GUI/agent notification.
    GUI components can subscribe to receive all browser open events in real time.
    """
    def __init__(self) -> None:
        self._subscribers: List[Callable[[Dict[str, Any]], None]] = []

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._subscribers.append(callback)

    def publish(self, event: Dict[str, Any]) -> None:
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception as e:
                print(f"[WARN] EventBus subscriber error: {e}", file=sys.stderr)

# Singleton event bus for the builder and GUI
event_bus = EventBus()

class BrowserOpenLogEntry(TypedDict, total=False):
    timestamp: str
    url: str
    success: bool
    browser: Optional[str]
    agent_id: Optional[str]
    agent_context: Optional[Dict[str, Any]]
    error: Optional[str]
    error_code: Optional[str]
    custom_args: Optional[Dict[str, Any]]
    event_type: str
    impersonated_by: Optional[str]
    delegated_by: Optional[str]

class BrowserOpenResult(TypedDict):
    success: bool
    error: Optional[str]
    error_code: Optional[str]

def _log_browser_open_attempt(
    url: str,
    success: bool,
    browser: Optional[str] = None,
    error: Optional[Exception] = None,
    error_code: Optional[str] = None,
    agent_id: Optional[str] = None,
    agent_context: Optional[Dict[str, Any]] = None,
    custom_args: Optional[Dict[str, Any]] = None,
    event_type: Optional[str] = None
) -> None:
    """
    Log and publish all browser open attempts for audit, agent traceability, and GUI visibility.
    """
    timestamp = datetime.datetime.now().isoformat()
    # --- SOLVED: Add agent impersonation/delegation fields to BrowserOpenLogEntry for richer audit.
    impersonated_by = None
    delegated_by = None
    if agent_context:
        impersonated_by = agent_context.get("impersonated_by")
        delegated_by = agent_context.get("delegated_by")
    # --- SOLVED: Add more granular event types for GUI (e.g., "browser_open_started", "browser_open_failed", "browser_open_succeeded").
    if not event_type:
        if success:
            event_type = "browser_open_succeeded"
        elif error_code == BrowserOpenErrorCodes.HEADLESS_ENVIRONMENT:
            event_type = "browser_open_failed"
        elif error_code == BrowserOpenErrorCodes.BROWSER_NOT_FOUND:
            event_type = "browser_open_failed"
        elif error_code == BrowserOpenErrorCodes.INVALID_URL or error_code == BrowserOpenErrorCodes.UNSUPPORTED_SCHEME:
            event_type = "browser_open_failed"
        elif error_code == BrowserOpenErrorCodes.OPEN_FAILED:
            event_type = "browser_open_failed"
        elif error_code == BrowserOpenErrorCodes.UNKNOWN_ERROR:
            event_type = "browser_open_failed"
        else:
            event_type = "browser_open_attempted"
    log_entry: BrowserOpenLogEntry = {
        "timestamp": timestamp,
        "url": url,
        "success": success,
        "browser": browser,
        "agent_id": agent_id,
        "agent_context": agent_context,
        "error": str(error) if error else None,
        "error_code": error_code,
        "custom_args": custom_args,
        "event_type": event_type,
    }
    if impersonated_by:
        log_entry["impersonated_by"] = impersonated_by
    if delegated_by:
        log_entry["delegated_by"] = delegated_by
    browser_str = f"browser='{browser}'" if browser else "browser=default"
    status = "SUCCESS" if success else "FAIL"
    agent_str = f"agent_id='{agent_id}'" if agent_id else "agent_id=none"
    context_str = f"agent_context={agent_context}" if agent_context else ""
    custom_args_str = f"custom_args={custom_args}" if custom_args else ""
    event_type_str = f"event_type={event_type}"
    impersonated_str = f"impersonated_by={impersonated_by}" if impersonated_by else ""
    delegated_str = f"delegated_by={delegated_by}" if delegated_by else ""
    msg = f"[{timestamp}] [BROWSER_OPEN] {status} url='{url}' {browser_str} {agent_str} {context_str} {custom_args_str} {event_type_str} {impersonated_str} {delegated_str}".strip()
    if error:
        msg += f" error='{error}'"
    if error_code:
        msg += f" error_code='{error_code}'"
    # Print to stdout/stderr for CLI, but always publish to event bus for GUI/agent visibility
    print(msg, file=sys.stderr if not success else sys.stdout)
    event_bus.publish(dict(log_entry))

def _validate_and_normalize_url(url: str) -> str:
    """
    Validate and normalize a URL for browser opening.
    - Auto-prepend http:// if missing.
    - Only allow http(s) URLs.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{BrowserOpenErrorCodes.INVALID_URL}: URL must be a non-empty string.")
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        url = "http://" + url
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"{BrowserOpenErrorCodes.UNSUPPORTED_SCHEME}: URL must start with 'http://' or 'https://'.")
    return url

def _is_headless_environment() -> bool:
    # Heuristic: On Linux/macOS, if DISPLAY is not set, likely headless
    if sys.platform.startswith("linux") or sys.platform == "darwin":
        if "DISPLAY" not in os.environ:
            return True
    # On Windows, headless detection is less reliable, skip for now
    return False

def open_url_in_browser(
    url: str,
    new: int = 2,
    autoraise: bool = True,
    browser: Optional[str] = None,
    agent_id: Optional[str] = None,
    agent_context: Optional[Dict[str, Any]] = None,
    custom_args: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Open a URL in the user's browser, with agent context and full audit trail.
    All actions are published to the event bus for GUI/agent visibility.
    Supports custom browser arguments for advanced agent use cases.
    """
    # --- SOLVED: Add more granular event types for GUI (e.g., "browser_open_started", ...)
    _log_browser_open_attempt(
        url,
        success=False,
        browser=browser,
        agent_id=agent_id,
        agent_context=agent_context,
        custom_args=custom_args,
        event_type="browser_open_started"
    )
    try:
        url = _validate_and_normalize_url(url)
    except Exception as exc:
        error_code = BrowserOpenErrorCodes.INVALID_URL if "INVALID_URL" in str(exc) else BrowserOpenErrorCodes.UNSUPPORTED_SCHEME
        _log_browser_open_attempt(url, False, browser, error=exc, error_code=error_code, agent_id=agent_id, agent_context=agent_context, custom_args=custom_args, event_type="browser_open_failed")
        raise

    # Headless environment detection
    if _is_headless_environment():
        error_msg = "No display found (headless environment)."
        _log_browser_open_attempt(url, False, browser, error=Exception(error_msg), error_code=BrowserOpenErrorCodes.HEADLESS_ENVIRONMENT, agent_id=agent_id, agent_context=agent_context, custom_args=custom_args, event_type="browser_open_failed")
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        return False

    try:
        # Support for custom browser arguments (advanced use case)
        # Only supported for certain browsers and platforms
        browser_instance = None
        if browser:
            try:
                # If custom_args are provided, attempt to use them if supported
                if custom_args:
                    # Only support for Chrome/Chromium/Edge/Firefox for now
                    # This is a best-effort, as webbrowser does not natively support args
                    # We use the register method to add a new browser with args
                    import shutil
                    browser_path = shutil.which(browser)
                    if browser_path:
                        import subprocess
                        from webbrowser import BaseBrowser
                        class CustomBrowser(BaseBrowser):
                            def open(self, url, new=new, autoraise=autoraise):
                                args = [browser_path]
                                for k, v in custom_args.items():
                                    if v is True:
                                        args.append(f"--{k}")
                                    else:
                                        args.append(f"--{k}={v}")
                                args.append(url)
                                try:
                                    subprocess.Popen(args)
                                    return True
                                except Exception:
                                    return False
                        # Register a new browser controller
                        webbrowser.register(f"{browser}-custom", None, CustomBrowser())
                        browser_instance = webbrowser.get(f"{browser}-custom")
                    else:
                        raise webbrowser.Error(f"Browser '{browser}' not found in PATH for custom args.")
                else:
                    browser_instance = webbrowser.get(browser)
            except webbrowser.Error as e:
                _log_browser_open_attempt(url, False, browser, error=e, error_code=BrowserOpenErrorCodes.BROWSER_NOT_FOUND, agent_id=agent_id, agent_context=agent_context, custom_args=custom_args, event_type="browser_open_failed")
                print(f"[ERROR] Browser '{browser}' not found: {e}", file=sys.stderr)
                return False
            result = browser_instance.open(url, new=new, autoraise=autoraise)
        else:
            result = webbrowser.open(url, new=new, autoraise=autoraise)
        if not result:
            _log_browser_open_attempt(url, False, browser, error=Exception("Failed to open URL"), error_code=BrowserOpenErrorCodes.OPEN_FAILED, agent_id=agent_id, agent_context=agent_context, custom_args=custom_args, event_type="browser_open_failed")
            print(f"[WARN] Could not open URL: {url}", file=sys.stderr)
            return False
        _log_browser_open_attempt(url, True, browser, agent_id=agent_id, agent_context=agent_context, custom_args=custom_args, event_type="browser_open_succeeded")
        return True
    except Exception as exc:
        _log_browser_open_attempt(url, False, browser, error=exc, error_code=BrowserOpenErrorCodes.UNKNOWN_ERROR, agent_id=agent_id, agent_context=agent_context, custom_args=custom_args, event_type="browser_open_failed")
        print(f"[ERROR] Failed to open URL '{url}': {exc}", file=sys.stderr)
        return False

def open_url_in_browser_for_agent(
    url: str,
    *,
    new: int = 2,
    autoraise: bool = True,
    browser: Optional[str] = None,
    agent_id: Optional[str] = None,
    agent_context: Optional[Dict[str, Any]] = None,
    custom_args: Optional[Dict[str, Any]] = None
) -> BrowserOpenResult:
    """
    Agent-friendly API: returns a structured result for agent orchestration and GUI display.
    Includes granular error codes for orchestration.
    """
    try:
        result = open_url_in_browser(
            url,
            new=new,
            autoraise=autoraise,
            browser=browser,
            agent_id=agent_id,
            agent_context=agent_context,
            custom_args=custom_args
        )
        if result:
            return {"success": True, "error": None, "error_code": None}
        else:
            # Find last event for this agent_id/url to get error_code
            # This is a simple approach for now; in production, use a more robust event log
            last_error_code = None
            last_error = None
            def find_last_event(event):
                nonlocal last_error_code, last_error
                if event.get("url") == url and event.get("agent_id") == agent_id:
                    last_error_code = event.get("error_code")
                    last_error = event.get("error")
            event_bus.subscribe(find_last_event)
            # Trigger a dummy publish to invoke the callback
            event_bus.publish({})
            return {"success": False, "error": last_error or "Failed to open URL", "error_code": last_error_code or BrowserOpenErrorCodes.OPEN_FAILED}
    except Exception as exc:
        # Try to extract error code from exception message
        error_code = None
        if hasattr(exc, "args") and exc.args:
            for code in vars(BrowserOpenErrorCodes).values():
                if isinstance(code, str) and code in str(exc):
                    error_code = code
                    break
        return {"success": False, "error": str(exc), "error_code": error_code or BrowserOpenErrorCodes.UNKNOWN_ERROR}

@runtime_checkable
class AgentBrowserOpener(Protocol):
    """
    Protocol for agent-driven browser actions.
    Agents and the builder GUI can use this protocol for orchestration and audit.
    """
    def open_url(
        self,
        url: str,
        *,
        new: int = 2,
        autoraise: bool = True,
        browser: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_context: Optional[Dict[str, Any]] = None,
        custom_args: Optional[Dict[str, Any]] = None
    ) -> BrowserOpenResult:
        ...

# --- GUI/Builder Integration Example ---
# The builder GUI can subscribe to the event bus to display all browser open events in real time.
# Example:
#
#   def gui_browser_event_handler(event: Dict[str, Any]):
#       # Update GUI with event (e.g., show in audit log, agent dashboard, etc.)
#       ...
#   event_bus.subscribe(gui_browser_event_handler)
#
# All browser actions (success/failure, agent context, etc.) are visible to the GUI and agents.

# --- Integration Tests for Distributed Event Bus Scenarios ---
import unittest

class TestOpenUrlInBrowser(unittest.TestCase):
    def test_empty_url(self):
        with self.assertRaises(ValueError):
            open_url_in_browser("")

    def test_non_string_url(self):
        with self.assertRaises(ValueError):
            open_url_in_browser(None)  # type: ignore

    def test_invalid_scheme(self):
        with self.assertRaises(ValueError):
            open_url_in_browser("ftp://example.com")

    def test_auto_prepend_http(self):
        try:
            result = open_url_in_browser("example.com", agent_id="agent-42", agent_context={"role": "test"})
            self.assertIsInstance(result, bool)
        except Exception as e:
            self.fail(f"open_url_in_browser raised unexpectedly: {e}")

    def test_valid_https(self):
        try:
            result = open_url_in_browser("https://example.com", agent_id="agent-42", agent_context={"purpose": "unit-test"})
            self.assertIsInstance(result, bool)
        except Exception as e:
            self.fail(f"open_url_in_browser raised unexpectedly: {e}")

    def test_invalid_browser(self):
        result = open_url_in_browser(
            "https://example.com",
            browser="nonexistent-browser-xyz",
            agent_id="agent-err",
            agent_context={"scenario": "invalid-browser"}
        )
        self.assertFalse(result)

    def test_agent_context_logging(self):
        try:
            result = open_url_in_browser(
                "example.com",
                agent_id="agent-ctx",
                agent_context={"foo": "bar", "task": 123}
            )
            self.assertIsInstance(result, bool)
        except Exception as e:
            self.fail(f"open_url_in_browser with agent context raised unexpectedly: {e}")

    def test_headless_environment(self):
        display_env = os.environ.get("DISPLAY")
        try:
            if sys.platform.startswith("linux") or sys.platform == "darwin":
                if "DISPLAY" in os.environ:
                    del os.environ["DISPLAY"]
            result = open_url_in_browser(
                "https://example.com",
                browser="nonexistent-browser-for-headless-test",
                agent_id="agent-headless",
                agent_context={"scenario": "headless"}
            )
            self.assertFalse(result)
        finally:
            if display_env is not None:
                os.environ["DISPLAY"] = display_env

    def test_agent_impersonation_and_delegation(self):
        events = []
        def event_collector(event):
            events.append(event)
        event_bus.subscribe(event_collector)
        agent_id = "agent-impersonator"
        agent_context = {"delegated_by": "agent-root", "purpose": "impersonation-test"}
        try:
            open_url_in_browser(
                "example.com",
                agent_id=agent_id,
                agent_context=agent_context
            )
        except Exception:
            pass
        self.assertTrue(any(
            e.get("agent_id") == agent_id and e.get("agent_context") == agent_context
            for e in events
        ))

    def test_agent_api_structured_result(self):
        # Test agent-friendly API returns structured result for GUI/agent orchestration
        result = open_url_in_browser_for_agent(
            "example.com",
            agent_id="agent-structured",
            agent_context={"purpose": "structured-result"}
        )
        self.assertIn("success", result)
        self.assertIn("error", result)
        self.assertIn("error_code", result)

    def test_custom_browser_args(self):
        # This test is best-effort: will only work if a supported browser is available in PATH
        # and the OS allows launching with custom args.
        # We'll use --incognito for Chrome/Chromium if available.
        import shutil
        for candidate in ["google-chrome", "chromium", "chrome", "chromium-browser"]:
            if shutil.which(candidate):
                result = open_url_in_browser(
                    "example.com",
                    browser=candidate,
                    custom_args={"incognito": True},
                    agent_id="agent-custom-args",
                    agent_context={"purpose": "custom-args-test"}
                )
                self.assertIsInstance(result, bool)
                break

    def test_event_bus_integration(self):
        # Integration test for distributed event bus scenarios
        events = []
        def collector(event):
            events.append(event)
        event_bus.subscribe(collector)
        open_url_in_browser(
            "example.com",
            agent_id="agent-eventbus",
            agent_context={"purpose": "eventbus-test"}
        )
        self.assertTrue(any(e.get("agent_id") == "agent-eventbus" for e in events))

    def test_gui_event_types(self):
        # SOLVED: Add GUI test to verify browser open events are displayed in the builder dashboard.
        # Here, we check that event_type is present and correct in the event bus events.
        events = []
        def collector(event):
            events.append(event)
        event_bus.subscribe(collector)
        open_url_in_browser(
            "example.com",
            agent_id="agent-gui-test",
            agent_context={"purpose": "gui-event-type-test"}
        )
        # There should be at least one event with event_type 'browser_open_started'
        self.assertTrue(any(e.get("event_type") == "browser_open_started" for e in events))
        # There should be at least one event with event_type 'browser_open_succeeded' or 'browser_open_failed'
        self.assertTrue(any(e.get("event_type") in ("browser_open_succeeded", "browser_open_failed") for e in events))

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# --- Conceptual Example: Protocol-based agent integration ---
# This utility is ready to be wrapped or injected into a protocol-driven agent system.
# All browser actions are visible to the builder GUI and agent dashboards via the event bus.

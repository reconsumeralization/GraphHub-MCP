"""
gui_launcher.py

Entry point for launching the FastMCP Workflow Builder GUI with agent-powered synergy.

This module provides both a CLI and Python entry point for the graphical workflow builder,
integrated with the FastMCP backend and LLM-powered agent co-pilot. It is the main GUI for authoring,
editing, and managing MCP workflow graphs, with agent suggestions, adaptive execution, and
workflow optimization.

Key Features:
- Loads the NodeGraphQt-based GUI builder.
- Integrates with backend builder API for graph persistence and validation.
- Embeds an agent "co-pilot" pane for real-time node suggestions and auto-wiring.
- Hooks into WorkflowManager for agent-driven node recommendations and feedback.
- CLI and Python module entry (see pyproject.toml console_scripts).
- Foundation for adaptive execution, optimization, and subflow extraction.

Outstanding TODOs (see MasterPlan and feature_map.md):
- Implement accessibility audit logic (see audit_accessibility)
- Integrate with telemetry/log server (see send_telemetry)
- Implement dynamic Qt binding selection (see dynamic_qt_binding_selection)
- Implement robust error reporting (see robust_import_error_reporting)
- Implement headless/agentic fallback (see fallback_to_headless_mode)
- Wire up agent suggestion pane and WorkflowManager node-added hooks
- Add CLI/REST endpoints for agent-injected execution control and subflow management
"""

import sys
import logging
import argparse
from typing import Optional, Dict, Any
import inspect

# --- Dynamic Qt Binding Selection ---
# TODO: Implement robust dynamic selection as planned
QT_BINDING = None
QtCore = None
QtWidgets = None
QtGui = None

qt_bindings_preference = [
    ("PySide6", "PySide6"),
    ("PySide2", "PySide2"),
    ("PyQt6", "PyQt6"),
    ("PyQt5", "PyQt5"),
]

for binding_name, module_name in qt_bindings_preference:
    try:
        if module_name == "PySide6":
            from PySide6 import QtCore, QtWidgets, QtGui # type: ignore
        elif module_name == "PySide2":
            from PySide2 import QtCore, QtWidgets, QtGui # type: ignore
        elif module_name == "PyQt6":
            from PyQt6 import QtCore, QtWidgets, QtGui # type: ignore
            QtCore.Signal = QtCore.pyqtSignal # type: ignore
            QtCore.Slot = QtCore.pyqtSlot     # type: ignore
            QtCore.Property = QtCore.pyqtProperty # type: ignore
            if not hasattr(QtWidgets, 'QAction') and hasattr(QtGui, 'QAction'):
                 QtWidgets.QAction = QtGui.QAction # type: ignore
        elif module_name == "PyQt5":
            from PyQt5 import QtCore, QtWidgets, QtGui # type: ignore
            QtCore.Signal = QtCore.pyqtSignal # type: ignore
            QtCore.Slot = QtCore.pyqtSlot     # type: ignore
            QtCore.Property = QtCore.pyqtProperty # type: ignore

        QT_BINDING = binding_name
        logging.info(f"Using Qt binding: {QT_BINDING}")
        break
    except ImportError:
        continue

if QT_BINDING is None:
    logging.error("No supported Qt binding found! Please install PySide6, PySide2, PyQt6, or PyQt5.")
    sys.exit(1)

# --- NodeGraphQt Import ---
try:
    from NodeGraphQt import NodeGraph, BaseNode, setup_context_menu # type: ignore
    NodeGraphQtAvailable = True
except ImportError as e:
    logging.error(f"NodeGraphQt not found: {e}. Please install it.")
    NodeGraphQtAvailable = False
    sys.exit(1)

# --- Local Imports ---
# TODO: Import WorkflowManager when needed
# from .workflow_manager import WorkflowManager
# Import NodeRegistry for node registration
from fastmcp.builder.nodes.node_registry import NodeRegistry
# Import BaseNode potentially needed for registration type checks
try:
    from fastmcp.builder.nodes.custom_nodes import BaseNode
except ImportError:
    from fastmcp.builder.nodes import BaseNode # Use the fallback from __init__

def audit_accessibility() -> None:
    """
    Accessibility (WCAG) and keyboard navigation audit hooks.

    Outstanding TODOs:
    - Implement actual accessibility audit logic.
    - Should check for keyboard navigability, color contrast, ARIA roles, and focus management.
    - Consider integrating with axe-core via subprocess or using a Python accessibility audit library.
    - Log all audit results and open issues for any violations found.
    """
    # Outstanding TODO: Implement actual accessibility audit logic.
    logging.info("Accessibility audit: (WCAG, keyboard navigation) - Not yet implemented.")
    # Outstanding TODO: Open issue to track accessibility audit implementation and coverage.

def send_telemetry(event: str, details: Optional[Dict[str, Any]] = None) -> None:
    """
    Send telemetry/logging GUI usage.

    Outstanding TODOs:
    - Integrate with actual telemetry/log server.
    - Should support batching and async sending for performance.
    - Ensure all PII is scrubbed before sending.
    - Add fallback to local file logging if remote unavailable.
    """
    # Outstanding TODO: Integrate with actual telemetry/log server.
    logging.info(f"Telemetry event: {event} | Details: {details}")
    # Outstanding TODO: Open issue to track telemetry/log server integration and privacy review.

def robust_import_error_reporting(e: Exception) -> None:
    """
    Robust import error reporting for all builder dependencies.

    Outstanding TODOs:
    - Implement robust error reporting (e.g., send to error tracking service).
    - Should capture stack trace, environment info, and missing modules.
    - Integrate with Sentry or similar error tracking.
    - Add CLI flag to suppress error reporting for CI.
    """
    # Outstanding TODO: Implement robust error reporting (e.g., send to error tracking service).
    logging.error(f"Robust import error reporting: {e}")
    # Outstanding TODO: Open issue to track error reporting integration and privacy controls.

def fallback_to_headless_mode() -> None:
    """
    Headless/agentic fallback mode for CI/CD and automation.

    Outstanding TODOs:
    - Implement headless/agentic fallback.
    - Should provide CLI/REST interface for workflow management.
    - Log all fallback events for audit trail.
    - Add integration test for headless mode.
    """
    # Outstanding TODO: Implement headless/agentic fallback.
    logging.warning("Falling back to headless/agentic mode (not yet implemented).")
    # Outstanding TODO: Open issue to track headless/agentic fallback implementation and test coverage.
    sys.exit(1)

def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the FastMCP Workflow Builder GUI with agent co-pilot."
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a workflow file to load on startup."
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Name of a workflow template to start with."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    parser.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable telemetry/logging of GUI usage."
    )
    parser.add_argument(
        "--accessibility-audit",
        action="store_true",
        help="Run accessibility (WCAG) and keyboard navigation audit hooks."
    )
    parser.add_argument(
        "--no-agent-copilot",
        action="store_true",
        help="Disable agent-assisted suggestions in the GUI."
    )
    # Outstanding TODO: Add --qt-binding CLI argument for explicit Qt binding selection (see dynamic_qt_binding_selection)
    return parser.parse_args()

def launch_gui_builder(
    workflow_file: Optional[str] = None,
    template: Optional[str] = None,
    telemetry_enabled: bool = True,
    accessibility_audit: bool = False,
    agent_copilot_enabled: bool = True
) -> None:
    """
    Sets up the QApplication and launches the MainWindow.
    """
    if accessibility_audit:
        audit_accessibility()

    if not NodeGraphQtAvailable:
        logging.error("Cannot launch GUI: NodeGraphQt is not available.")
        sys.exit(1)

    if not QtWidgets or not QtCore or not QtGui:
        logging.error("Cannot launch GUI: No Qt binding found or loaded.")
        sys.exit(1)

    # Add asserts to help type checker/linter
    assert QtWidgets is not None
    assert QtCore is not None
    assert QtGui is not None

    # --- Define MainWindow *after* Qt imports are confirmed ---
    class MainWindow(QtWidgets.QMainWindow):
        """Main window for the FastMCP Workflow Builder GUI."""
        def __init__(self, workflow_file=None, template=None, agent_copilot_enabled=True):
            super().__init__()
            self.setWindowTitle("FastMCP Workflow Builder")
            self.resize(1200, 800)

            # TODO: Instantiate WorkflowManager
            # self.workflow_manager = WorkflowManager()

            # --- Central Widget: Node Graph ---
            self.graph = NodeGraph()
            # Set up context menu *before* registering nodes if it affects registration
            setup_context_menu(self.graph)
            self.setCentralWidget(self.graph.widget)

            # Register nodes after graph setup
            self.register_nodes()

            # --- Docks --- (Placeholders)
            self.props_bin = QtWidgets.QDockWidget("Properties")
            self.nodes_palette = QtWidgets.QDockWidget("Nodes")
            self.agent_copilot = QtWidgets.QDockWidget("Agent Co-Pilot")

            self.props_bin.setWidget(QtWidgets.QLabel("Node Properties Placeholder"))
            self.nodes_palette.setWidget(QtWidgets.QLabel("Node Palette Placeholder"))
            self.agent_copilot.setWidget(QtWidgets.QLabel("Agent Co-Pilot Placeholder"))

            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.props_bin)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.nodes_palette)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.agent_copilot)
            self.agent_copilot.setVisible(agent_copilot_enabled)

            # --- Menu Bar --- (Placeholder)
            self.menu_bar = self.menuBar()
            file_menu = self.menu_bar.addMenu("&File")
            edit_menu = self.menu_bar.addMenu("&Edit")
            view_menu = self.menu_bar.addMenu("&View")
            agent_menu = self.menu_bar.addMenu("&Agent")
            help_menu = self.menu_bar.addMenu("&Help")

            # TODO: Add actions
            # Example:
            # open_action = QtGui.QAction("&Open...", self)
            # open_action.triggered.connect(self.open_workflow)
            # file_menu.addAction(open_action)

            # --- Toolbar --- (Placeholder)
            self.toolbar = self.addToolBar("Main Toolbar")
            # TODO: Add actions

            # --- Load Initial Workflow ---
            if workflow_file:
                self.load_workflow(workflow_file)
            elif template:
                self.new_from_template(template)

            logging.info("FastMCP Workflow Builder GUI initialized.")

        def register_nodes(self):
            """Register all nodes from the NodeRegistry with the NodeGraph instance."""
            registered_count = 0
            skipped_count = 0
            error_count = 0

            for node_id, node_cls in NodeRegistry._registry.items(): # Access underlying dict for iteration
                try:
                    # Check if it's a valid class and subclass of BaseNode (if needed for NodeGraphQt)
                    if inspect.isclass(node_cls) and issubclass(node_cls, BaseNode):
                        # NodeGraphQt register_node uses the class directly
                        self.graph.register_node(node_cls)
                        registered_count += 1
                        logging.debug(f"Registered node {node_id} with NodeGraphQt.")
                    else:
                        logging.warning(f"Skipping registration for {node_id}: Not a valid BaseNode subclass.")
                        skipped_count += 1
                except Exception as e:
                    logging.error(f"Failed to register node {node_id} ({node_cls.__name__}) with NodeGraphQt: {e}")
                    error_count += 1

            logging.info(f"Node registration complete. Registered: {registered_count}, Skipped: {skipped_count}, Errors: {error_count}")
            # TODO: Populate the Nodes Palette dock based on registered nodes

        def load_workflow(self, file_path: str):
            try:
                # self.graph.load_session(file_path)
                logging.info(f"Workflow loaded from {file_path}")
            except Exception as e:
                logging.error(f"Failed to load workflow from {file_path}: {e}")
                QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load workflow: {e}")

        def save_workflow(self, file_path: Optional[str] = None):
            if not file_path:
                 # file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Workflow", "", "Workflow Files (*.json *.mcp)")
                 pass
            if file_path:
                try:
                    # self.graph.save_session(file_path)
                    logging.info(f"Workflow saved to {file_path}")
                except Exception as e:
                    logging.error(f"Failed to save workflow to {file_path}: {e}")
                    QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save workflow: {e}")

        def new_from_template(self, template_name: str):
            logging.warning(f"Template loading for '{template_name}' not yet implemented.")

        def open_workflow(self):
            pass

    # --- Application Setup ---
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication(sys.argv)

    # --- Create and Show Main Window ---
    try:
        main_win = MainWindow(
            workflow_file=workflow_file,
            template=template,
            agent_copilot_enabled=agent_copilot_enabled
        )
        main_win.show()

        if telemetry_enabled:
            send_telemetry("gui_launch_success", {
                "qt_binding": QT_BINDING,
                "nodegraphqt": NodeGraphQtAvailable,
                # ... other details ...
            })

        sys.exit(app.exec())

    except Exception as e:
        logging.exception("Failed to launch the FastMCP Workflow Builder GUI.")
        if telemetry_enabled:
            send_telemetry("gui_launch_failed", {"error": str(e)})
        robust_import_error_reporting(e)
        sys.exit(3)

def main() -> None:
    args = parse_cli_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s" # Improved format
    )
    # Call launch_gui_builder, which now handles QApplication and MainWindow
    launch_gui_builder(
        workflow_file=args.file,
        template=args.template,
        telemetry_enabled=not args.no_telemetry,
        accessibility_audit=args.accessibility_audit,
        agent_copilot_enabled=not args.no_agent_copilot
    )

# Keep the __main__ guard to allow running the script directly
if __name__ == "__main__":
    main()

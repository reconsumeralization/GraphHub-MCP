# MasterPlan.md

## FastMCP Builder & Agent: Master Plan for GUI-Driven Agentic Usability

This document is the single source of truth for all major features, sprints, and enhancements—implemented, in progress, and planned—for the FastMCP builder and agent stack, with a focus on agent usability and agentic process automation, all visible and actionable through the GUI.  
**Update with every significant change.**  
**Last updated:** 2024-06-09

---

## ✅ Completed Tasks

- [x] Set up development environment and installed FastMCP package
- [x] Added GUI dependencies (NodeGraphQt, PySide2/PyQt5, asteval) in `pyproject.toml`
- [x] Moved GUI builder script into `src/fastmcp/builder.py`
- [x] Added console script entry point `mcp-workflow-builder` in `pyproject.toml`
- [x] Wrote unit tests for `build_mcp_server_graph` and `validate_graph` functions
- [x] Support alternative Qt bindings dynamically (choose best available at runtime)
- [x] Add hosting options (embedding builder in a web UI)

---

## 🚧 Current Sprint: GUI Workflow Builder Integration

- [ ] Integrate the MCP Workflow Builder GUI script into the FastMCP project
  - [ ] Update `README.md` to document GUI builder prerequisites and usage
- [ ] Create end-to-end test to launch the GUI and verify example graph loads
- [ ] Schedule code review and accessibility audit (WCAG) for the GUI
- [ ] Publish a patch release once GUI integration is complete

---

## 🧠 Agent Usability & Agentic Features (Builder & GUI)

### Agent-First Node Palette & Registry

- [x] Add `SemanticTargetNode` and `SemanticTriggerNode` to `src/fastmcp/builder/nodes/`
- [x] Add `ActionRecorderNode` for capturing and replaying user actions
- [ ] Expose agentic node types in the GUI palette with clear descriptions and usage examples
- [ ] Enable drag-and-drop creation of agent nodes and agent workflows in the GUI
- [ ] Auto-register all agent nodes in `node_registry.py` and display in GUI

### Agent Workflow Authoring & Visualization

- [ ] Visualize agentic flows (triggers, actions, semantic targets) in the node graph
- [ ] Add context menus for agent nodes: "Record Actions", "Replay Actions", "Edit Agent Policy"
- [ ] Show agent state, last action, and execution trace in GUI side panel
- [ ] Allow users to annotate agent nodes with goals, constraints, and triggers

### Agent-Driven Automation & Testing

- [ ] Integrate ActionRecorderNode with a recorder service module for capturing/replaying user actions
- [ ] Add "Record Session" and "Replay Session" buttons to the GUI toolbar
- [ ] Enable step-through and breakpoint debugging for agent workflows in the GUI
- [ ] Display agent execution logs and errors inline in the GUI

### Agent-Accessible Documentation & Help

- [ ] Add contextual help popovers for all agent node types in the GUI
- [ ] Integrate AI-driven documentation: auto-generate usage tips for agent nodes based on code and usage patterns
- [ ] Provide "Show Example" and "Insert Example Agent Flow" actions in the GUI

### Agentic Process Automation (MCP Server)

- [ ] Expose full MCP Server API endpoints (create, list, save, execute, status) via FastAPI (`src/fastmcp/mcp_server.py`)
- [ ] Add GUI controls to trigger server-side execution and monitor status/results
- [ ] Display API call logs and agent execution results in the GUI

---

## 🔜 Future Enhancements

- [ ] Enable exporting workflows as reusable MCP YAML/JSON specs (including agent flows)
- [ ] Add support for agentic workflow templates and quick-starts in the GUI
- [ ] Integrate LLM-based workflow suggestion engine for agentic flows (recommend next agent node, auto-complete agent chains)
- [ ] Add agentic workflow versioning and rollback in the GUI
- [ ] Support agentic workflow sharing and import/export via the GUI

---

## 🗂️ Upcoming Refactoring Sprints

### Sprint 1: WorkflowManager & Execution Engine (2–3 days)

- [ ] Audit `graph_builder_service.py`: list graph CRUD and node operations methods.
- [ ] Audit `WorkFlowManager.py`: list validation and execution trigger methods.
- [ ] Create `WorkflowManager` class stub in `src/fastmcp/builder/workflow_manager.py`.
- [ ] Migrate graph CRUD methods into `WorkflowManager`.
- [ ] Migrate node operations methods into `WorkflowManager`.
- [ ] Migrate graph validation triggers into `WorkflowManager`.
- [ ] Implement `trigger_graph_execution()` calling `executor.execute_graph`.
- [ ] Implement `get_execution_status()` calling `executor.get_status`.
- [ ] Remove GUI imports and update related import paths.
- [ ] Write unit tests for each public `WorkflowManager` method (≥80% coverage).
- [ ] Delete deprecated files: `graph_builder_service.py`, `WorkFlowManager.py`.

### Sprint 2: GUI Consolidation & Launch (2 days)

- [ ] Audit `graph_builder.py` and `builder_graph.py` for NodeGraphQt and Qt setup.
- [ ] Move QApplication and QMainWindow setup into `src/fastmcp/builder/gui_launcher.py`.
- [ ] Migrate NodeGraphQt graph instantiation and widget embedding into `gui_launcher.py`.
- [ ] Implement menus, toolbars, and docks in `gui_launcher.py`.
- [ ] Remove redundant UI scripts: `graph_builder.py`, `builder_graph.py`.
- [ ] Add a headless Qt smoke test to verify GUI startup with an example graph.

### Sprint 3: Node Registry Simplification (1 day)

- [ ] Create `src/fastmcp/builder/nodes/__init__.py` and `node_registry.py` if missing.
- [ ] Implement registry dict/class in `node_registry.py`.
- [ ] Auto-import each `*_node.py` in `nodes/__init__.py` and register via `NODE_REGISTRY`.
- [ ] Update WorkflowManager and GUI to use `NODE_REGISTRY` for dynamic node discovery.
- [ ] Delete legacy registration files: `register_node.py`, `register_agent.py`, `agent_node_registry.py`.

### Sprint 4: Agent Package & Planner (1–2 days)

- [ ] Create `src/fastmcp/agent/` directory and add `__init__.py`.
- [ ] Move and rename `mcp_graph_agent_planner.py` → `agent/planner.py`.
- [ ] Move and rename `incrimental_planning.py` → `agent/incremental_planner.py`, fix filename.
- [ ] Move and rename `dynamic_graph_updates.py` → `agent/dynamic_updater.py`.
- [ ] Move and rename `agent_graph.py` → `agent/graph_model.py`.
- [ ] (Optional) Relocate `mcp_tools.py` to `agent/tools.py` and update imports.
- [ ] Update imports in moved files and WorkflowManager.
- [ ] Write integration tests for agent tools using the WorkflowManager API.

### Sprint 5: Graph Utilities & Cleanup (1 day)

- [ ] Merge `graph_validator.py`, `json_graph.py` (typo fix), and `yamle_graph.py` into `graph_utils.py`.
- [ ] Delete original files: `graph_validator.py`, `json_graph.py`, `yamle_graph.py`.
- [ ] Ensure `graph_utils.py` exposes validation and serialization functions.
- [ ] Remove obsolete builder scripts: `custom_nodes.py`, `split_custom_nodes.py`, `use-browzer.py`, `index.py`.

### Sprint 6: Top-Level & Imports (1 day)

- [ ] Delete `src/fastmcp/executor.py` (redundant) and update imports to `execution_engine/executor.py`.
- [ ] Compare and unify `mcp_server.py` with `server/server.py`; remove duplicate logic.
- [ ] Update console script entry points if needed (pyproject.toml).
- [ ] Bulk update import paths via codemod or search/replace; verify no missing imports.

### Sprint 7: Verification & CI (1–2 days)

- [ ] Expand unit/integration tests for all refactored modules to maintain ≥90% coverage.
- [ ] Update CI pipeline (`.github/workflows/`) to run linting, type-checking, tests, security scan, and GUI smoke tests.
- [ ] Perform manual cross-browser and accessibility audits on the GUI.

---

## 🏁 Next Sprint: Agentic Process Automation & MCP Server

- [ ] Define and implement `SemanticTargetNode` and `SemanticTriggerNode` classes in `src/fastmcp/builder/nodes/`
- [ ] Create `ActionRecorderNode` and a recorder service module to capture and replay user actions
- [ ] Integrate new agentic node types into the node registry and GUI builder
- [ ] Expose full MCP Server API endpoints (create, list, save, execute, status) via FastAPI (`src/fastmcp/mcp_server.py`)
- [ ] Add unit and integration tests covering the FastAPI server and new node implementations
- [ ] Update documentation (README, docs/) with MCP Server usage examples and agentic workflow tutorials
- [ ] Configure CI pipeline (GitHub Actions) to run linting, type‑checking, tests, and build the server
- [ ] Add end‑to‑end scenario tests: record a simple graph, save, execute, and verify results via API
- [ ] Improve overall code coverage, fix any remaining linter/type errors, and remove deprecated code
- [ ] Schedule a design review and pair session focused on agentic automation reliability and UX

---

## 📝 Consolidated TODO Inventory (src/fastmcp/builder/)

- [ ] Ensure all agentic node types are visible and usable in the GUI palette
- [ ] Add contextual help and usage examples for agent nodes in the GUI
- [ ] Implement agent action recording and replay in both backend and GUI
- [ ] Visualize agent execution state and logs in the GUI
- [ ] Add LLM-powered agent workflow suggestions and auto-completion in the GUI
- [ ] Audit and improve accessibility (ARIA, keyboard navigation, WCAG) for all agentic features
- [ ] Add robust error handling and user feedback for agentic flows in the GUI
- [ ] Integrate agentic workflow export/import (YAML/JSON) with GUI controls
- [ ] Add scenario-based tests for agentic process automation (record, replay, verify)
- [ ] Document all agentic features and GUI affordances in README and docs

---

## 📂 File/Module Cross-Reference

| Module                        | Key Roles/Features                                                                 |
|-------------------------------|------------------------------------------------------------------------------------|
| **builder.py**                | Core builder logic, NodeGraphQt integration, error handling                        |
| **gui_launcher.py**           | GUI entry point, drag-and-drop editor, agent node palette, accessibility           |
| **nodes/node_registry.py**    | Dynamic node/agent registry, auto-discovery, agent node registration               |
| **nodes/semantic_target_node.py** | Agentic node: semantic target, triggers                                         |
| **nodes/semantic_trigger_node.py** | Agentic node: semantic trigger, event-driven                                   |
| **nodes/action_recorder_node.py**  | Agentic node: action recording and replay                                      |
| **agent/planner.py**          | Agent planning, optimization, and execution strategies                             |
| **agent/dynamic_updater.py**  | Dynamic agent graph updates                                                        |
| **graph_utils.py**            | Graph validation, serialization, and utilities                                     |
| **mcp_server.py**             | FastAPI MCP server, agentic workflow endpoints                                     |
| **tests/**                    | Automated test suite (unit, integration, scenario, agentic flows)                  |

---

## 🏆 Contribution Guidelines

- Update this MasterPlan.md with every new agentic feature, fix, or improvement.
- Add TODOs for any gaps, edge cases, or technical debt—especially for agent usability in the GUI.
- Open issues for larger improvements or architectural changes.
- Keep this document sharp, robust, and always improving. ☕️🚀

---

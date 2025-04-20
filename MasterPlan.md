# Master Plan

## Completed Tasks

- [x] Set up development environment and installed FastMCP package

## Current Sprint: GUI Workflow Builder Integration

- [ ] Integrate the MCP Workflow Builder GUI script into the FastMCP project
  - [x] Add GUI dependencies (NodeGraphQt, PySide2/PyQt5, asteval) in `pyproject.toml`
  - [x] Move the GUI builder script into `src/fastmcp/builder.py`
  - [x] Add a console script entry point `mcp-workflow-builder` in `pyproject.toml`
  - [ ] Update `README.md` to document GUI builder prerequisites and usage
- [x] Write unit tests for `build_mcp_server_graph` and `validate_graph` functions
- [ ] Create end-to-end test to launch the GUI and verify example graph loads
- [ ] Schedule code review and accessibility audit (WCAG) for the GUI
- [ ] Publish a patch release once GUI integration is complete

## Future Enhancements

- [x] Support alternative Qt bindings dynamically (choose best available at runtime)
- [ ] Enable exporting workflows as reusable MCP YAML/JSON specs
- [x] Add hosting options (embedding builder in a web UI)

## Upcoming Refactoring Sprints

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

- [ ] Finalize Agent Package Structure & Core Types:
  - [ ] Refine agent package structure in `src/fastmcp/agent/` (planner.py, incremental_planner.py, tools.py).
  - [ ] Solidify core TypedDict definitions in `src/fastmcp/builder/types.py`: GraphSpec, NodeSpec, ConnectionSpec, PropertySpec.
  - [ ] Refine protocol interfaces in `src/fastmcp/builder/protocols.py` for agent-builder communication.
- [ ] Implement Essential WorkflowManager API Methods (stubs/basic) in `src/fastmcp/builder/workflow_manager.py`:
  - [ ] `create_graph(workflow_id: str, description: Optional[str]) -> GraphHandle`
  - [ ] `list_available_node_types() -> List[Dict[str, Any]]`
  - [ ] `add_node(...) -> NodeID`
  - [ ] `connect_nodes(...) -> bool`
  - [ ] `set_node_property(...) -> bool`
  - [ ] `get_graph_spec(...) -> GraphSpec`
  - [ ] `save_graph_spec(...) -> bool`
  - [ ] Stub: `trigger_graph_execution(...) -> ExecutionId`
  - [ ] Stub: `get_execution_status(execution_id) -> Dict`
- [ ] Define/Refine Agent Tools in `src/fastmcp/agent/tools.py`:
  - [ ] Wrap WorkflowManager methods: AddNodeTool, ConnectNodesTool, ListNodesTool, SetNodePropertyTool, GetGraphTool, SaveGraphTool, TriggerExecutionTool.
  - [ ] Ensure clear args_schema and tool descriptions for LLM planner.
- [ ] Implement Initial Agent Planning Logic in `src/fastmcp/agent/planner.py` and `incremental_planner.py`:
  - [ ] Build simple planning loop: receive goal, prompt LLM with tool descriptions, parse plan steps.
  - [ ] Execute plan steps by invoking corresponding agent tools.
  - [ ] Demo: construct a simple 2–3 node graph from a fixed goal.
- [ ] Basic Testing:
  - [ ] Unit tests for new WorkflowManager methods (in `tests/builder/`).
  - [ ] Unit tests for agent tools in `tests/agent/`.
  - [ ] Integration test: agent planner builds a simple graph via tools.

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

## Next Sprint: Agentic Process Automation & MCP Server

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

## TODO Categories Overview

### 1. Core API & Functionality (Missing Methods, Core Logic)

- `tools/tool_manager.py`: Validate arguments before `tool.run`.
- `resources/types.py`: Support binary content.
- `resources/template.py`: Support more complex patterns.
- `prompts/prompt_manager.py`: Handle new `DuplicateBehavior` values.
- `mcp_server.py`: Ensure node type listing is correct after consolidation.
- `execution_engine/executor.py`: Handle cycles, errors, branching, status tracking, async, hooks.
- `client/roots.py`: Support `Path` objects.
- `builder/workflow_manager.py`: Implement better default positioning logic, node type validation, port validation, add input/output port info.
- `builder/nodes/sub_workflow_node.py`: Integrate with engine for subflow execution.
- `builder/nodes/sse_node.py`: Add support for client disconnect detection.
- `builder/nodes/agent_node.py`: Implement agent-specific logic.
- `builder/nodes/semantic_target_node.py`: Integrate with semantic targeting engine.
- `builder/nodes/semantic_trigger_node.py`: Poll semantic targeting engine.
- `builder/graph_utils.py`: Integrate feedback storage/actions.
- `builder/constants.py`: Connect event emission to `WorkflowManager`.
- `builder/nodes/agent_checkpoint_node.py`: Integrate with engine to pause/notify agent.
- `builder/nodes/ai_embedding_node.py`: Implement actual embedding generation.
- `builder/nodes/ai_speech_to_text_node.py`: Add support for language hint.
- **CLI-Related**: Implement `validate_graph_spec`, `analyze_performance`, `load_workflow_spec`/`build_graph`, `suggest_optimizations`, `extract_subflow`, `get_execution_insights` in `WorkflowManager` or `AgenticWorkflowTools`.

### 2. Agent & GUI Integration/Usability

- `mcp_server.py`: Integrate agentic initialization endpoint.
- `builder/yamle_graph.py`: Add more agent usability fields for GUI, add metadata fields.
- `builder/workflow_manager.py`: Expose agent/node state/logs for GUI, agent registration/listing/assignment, agent status updates, agent-workflow chat, agent execution logs/errors/timing, agent-initiated execution.
- `builder/nodes/sub_workflow_node.py`: Add agent context propagation, support spec import/export via GUI.
- `builder/nodes/sse_node.py`: Enhance with richer agent/builder context.
- `builder/index.py`: Consider auto-discovering agentic nodes, add a11y hooks for agentic nodes in GUI.
- `builder/nodes/index.py`: Consider agent-node cross-registry integration.
- `builder/constants.py`: Connect agent suggestion events to GUI event bus.
- `graph_cli.py`: Display agentic nodes better, use agent suggestions in more commands.

### 3. Testing

- `utilities/types.py`: Add unit tests for `Image`, `convert_to_set`.
- `utilities/decorators.py`: Add unit tests for edge cases.
- `resources/resource.py`: Add unit tests for `Resource` class.
- `prompts/prompt_manager.py`: Add unit tests for `duplicate_behavior`.
- `prompts/prompt.py`: Add unit tests for `Prompt.from_function`, `Prompt.render`.
- `executor.py`: Add unit tests for `WorkflowExecutor`.
- `execution_engine/executor.py`: Add unit tests for edge cases and error paths.
- `client/roots.py`: Add unit tests for `convert_roots_list`, `create_roots_callback`.
- `cli/claude.py`: Add unit tests for `update_claude_config`.
- `builder/__init__.py`: Add self-test for `main` callable.
- `builder/nodes/sse_node.py`: Add unit tests.
- `builder/nodes/agent_node.py`: Add unit tests.
- `builder/index.py`: Add agent usability tests for builder/GUI integration.
- `builder/nodes/split_custom_nodes.py`: Add self-test for registry matching file system.
- `builder/nodes/index.py`: 100% unit test coverage for registry operations.
- `builder/constants.py`: Add/expand unit tests for agent suggestion/optimization types.
- `builder/workflow_manager.py`: Add tsd-style self-tests for TypedDicts/Pydantic models.
- `builder/nodes/agent_checkpoint_node.py`: Add unit tests.
- `builder/nodes/ai_embedding_node.py`: Add unit tests.
- `builder/nodes/ai_function_calling_node.py`: Add unit tests.
- `builder/nodes/ai_normalize_node.py`: Add unit tests.
- `builder/nodes/ai_speech_to_text_node.py`: Add unit tests.

### 4. Error Handling & Logging

- `utilities/types.py`: Handle `ImageContent` import failure, handle file read failure.
- `utilities/logging.py`: Add log rotation/retention, integrate with centralized logging, warn if `rich` not installed.
- `tools/tool.py`: Add more granular error handling/logging, log serialization failure.
- `resources/types.py`: Add logging.
- `resources/template.py`: Add more granular error handling/logging.
- `client/sampling.py`: Add logging.
- `client/roots.py`: Add logging.
- `client/base.py`: Add request/response logging, tracing, more granular error handling.
- `exceptions.py`: Add error codes, more granular exception types.
- `builder/workflow_manager.py`: Add detailed logs to execution results.
- `builder/nodes/action_recorder_node.py`: Integrate with centralized audit log rotation/retention.
- `builder/constants.py`: Log all agent events for audit.
- `builder/nodes/agent_checkpoint_node.py`: Add audit logging.
- `builder/nodes/ai_embedding_node.py`: Handle API errors, retries, logging.
- `builder/nodes/ai_function_calling_node.py`: Handle LLM errors, add logging.
- `builder/nodes/ai_graph_runner_node.py`: Add logging of subgraph execution.
- `builder/nodes/ai_normalize_node.py`: Consider warning/error for std=0.
- `builder/nodes/ai_speech_to_text_node.py`: Handle API errors.

### 5. Validation & Types

- `tools/tool_manager.py`: Validate arguments before `tool.run`.
- `tools/tool.py`: Validate override keys.
- `resources/template.py`: Validate function typing.
- `prompts/prompt.py`: Support more robust type conversion, validate types/errors.
- `mcp_server.py`: Validate status structure.
- `execution_engine/executor.py`: Add type annotations for node input/output contracts.
- `client/transports.py`: Validate URL/path.
- `client/sampling.py`: Validate result structure.
- `client/base.py`: Add request/response schema validation.
- `builder/index.py`: Remove `# type: ignore`, add type-annotated stubs.
- `builder/nodes/split_custom_nodes.py`: Validate generated files are importable.
- `builder/nodes/index.py`: Enforce strict type validation/interface compliance.
- `builder/workflow_manager.py`: Restore `WorkflowAccessControl` type hint.
- `builder/nodes/sub_flow_node.py`: Add validation for subflow spec/params.
- `builder/nodes/ai_embedding_node.py`: Validate input text.
- `builder/nodes/ai_function_calling_node.py`: Validate/sanitize input_data.
- `builder/nodes/ai_graph_runner_node.py`: Consider input validation schemas.
- `builder/nodes/ai_speech_to_text_node.py`: Validate audio_data.

### 6. External Integrations & APIs (LLMs, Services)

- `executor.py`: Integrate with FastAPI endpoints.
- `builder/graph_utils.py`: Replace placeholder with LLM suggestion logic.
- `builder/nodes/ai_embedding_node.py`: Integrate with embedding API.
- `builder/nodes/ai_function_calling_node.py`: Integrate with LLM API.
- `builder/nodes/ai_speech_to_text_node.py`: Integrate with STT service.
- `builder/nodes/ai_text_to_speech_node.py`: Integrate with TTS provider.

### 7. Persistence & State Management

- `builder/workflow_manager.py`: Ensure persistence features (file/db/redis), rollback, recovery work correctly.
- `builder/nodes/agent_checkpoint_node.py`: Consider supporting checkpoint metadata.

### 8. Refactoring & Code Quality

- `utilities/decorators.py`: Consider supporting `functools.wraps`.
- `resources/types.py`: Move validators to mixin/utility.
- `resources/resource.py`: Consider moving `Resource` to dedicated types module.
- `prompts/prompt_manager.py`: Consider thread-safety.
- `mcp_server.py`: Update import after consolidation.
- `exceptions.py`: Add more granular exception types.
- `client/transports.py`: Consider moving `SessionKwargs` to types module.
- `client/roots.py`: Consider context-based filtering.
- `builder/__main__.py`: Avoid relative imports.
- `builder/__init__.py`: Consider exposing additional APIs.
- `builder/nodes/sub_workflow_node.py`: Register node in registry.
- `builder/nodes/agent_node.py`: Consider thread safety.
- `builder/index.py`: Remove `# type: ignore`.
- `builder/nodes/split_custom_nodes.py`: Add CLI options (dry-run etc.).
- `builder/nodes/index.py`: Decorator-based registration, plugin-based registration, MCP Tooling compliance checks, auto-import node modules.
- `builder/constants.py`: Remove legacy string constants, ensure agent comms use defined types.
- `builder/nodes/agent_checkpoint_node.py`: Consider pluggable audit logger.
- `builder/nodes/ai_function_calling_node.py`: Consider moving `ToolSchema` to shared types.
- `builder/nodes/ai_graph_runner_node.py`: Consider serializing subgraph metadata.

### 9. Configuration & Security

- `mcp_server.py`: Add authentication/authorization.
- `builder/workflow_manager.py`: Implement access control check, restore `WorkflowAccessControl` type hint.
- `builder/nodes/sse_node.py`: Consider CORS headers, add auth hooks.
- `builder/nodes/action_recorder_node.py`: Make audit log path configurable.

### 10. Async & Performance

- `execution_engine/executor.py`: Add support for async execution.
- `client/base.py`: Add async support.
- `builder/nodes/ai_embedding_node.py`: Consider batch embedding.
- `builder/nodes/ai_speech_to_text_node.py`: Consider streaming audio.

### 11. Documentation & DX

- `mcp_server.py`: Add OpenAPI tags/models.
- `builder/index.py`: Add type-annotated stubs.

### 12. Needs Clarification / General

- `executor.py`: "TODO:" (line 10) - Needs context.
- `builder/graph_logger.py`: "--- TODO:" (line 564) - Needs context.
- `builder/nodes/node_registry.py`: "TODO:" (line 10) - Needs context.

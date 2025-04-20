<div align="center">

<!-- omit in toc -->
# GraphHub-MCP 🚀

**An AI-Native Automation & Integration Engine where Agents Build, Manage, and Execute Complex Workflows.**

[![License](https://img.shields.io/github/license/reconsumeralization/GraphHub-MCP.svg)](https://github.com/reconsumeralization/GraphHub-MCP/blob/main/LICENSE)
<!-- TODO: Add relevant badges once CI/CD is set up -->
<!-- [![Build Status](...)](...) -->
<!-- [![PyPI - Version](...)](...) -->

</div>

GraphHub-MCP is an ambitious project building a **next-generation agentic platform**. It goes beyond simple task automation by creating an environment where AI agents can:

*   **Construct & Modify Workflows:** Agents dynamically build and adapt their own operational logic using a node-based graph system.
*   **Exercise Unified Control:** Seamlessly operate across web browsers (via Playwright-like control) and the desktop OS (via Cursor-like interactions).
*   **Bridge Legacy & Modern Systems:** Specialize in understanding and integrating outdated systems (mainframes, legacy DBs, old desktop apps) with modern APIs and cloud services.
*   **Collaborate Visually:** Interact with humans and other agents through a shared visual graph interface, making complex processes transparent and debuggable.
*   **Leverage Rich Context (MCP):** Utilize the Model Context Protocol for robust memory, tool access, and contextual understanding.

Think of it as an **Operating System for Agentic Transformation**, designed to tackle complex integration challenges and enable truly autonomous digital workers.

This project evolves from concepts explored in `fastmcp` and aims to deliver a more powerful, open, and agent-centric alternative to tools like Cheat Layer or OpenAgent Studio.

## Core Vision Pillars

1.  **MCP Kernel:** The heart of the system, managing persistent agent context, memory (short-term, long-term, embeddings), standardized tool discovery, and secure execution based on MCP principles.
2.  **Agentic Graph Engine:** A dynamic runtime where agents can programmatically define, visualize, execute, and *modify* directed acyclic graphs (DAGs) representing tasks, workflows, or reasoning chains.
3.  **Unified Control Plane:** Provides agents with a consolidated toolkit for deep interaction with both web environments (DOM manipulation, network requests) and desktop environments (GUI automation, file system access, shell commands).
4.  **Visual Interaction Layer (Qt/Web):** A graphical frontend where developers and agents can collaboratively design, monitor, debug, and interact with running workflows and agent states in real-time.
5.  **Legacy Bridge Toolkit:** A specialized, extensible set of plugins and vision-based tools enabling agents to interface with non-API systems (e.g., screen scraping, OCR, RPA primitives, data format translation for older protocols like SOAP/XML).

## Table of Contents (Tentative)

- [Vision & Goals](#vision--goals)
- [Core Architecture](#core-architecture)
  - [MCP Kernel](#mcp-kernel)
  - [Agentic Graph Engine](#agentic-graph-engine)
  - [Unified Control Plane](#unified-control-plane)
  - [Visual Interaction Layer](#visual-interaction-layer)
  - [Legacy Bridge Toolkit](#legacy-bridge-toolkit)
- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Running the UI](#running-the-ui)
  - [Basic Agent Interaction](#basic-agent-interaction)
- [Development Roadmap](#development-roadmap)
- [Contributing](#contributing)
- [License](#license)

## Vision & Goals

The primary goal of GraphHub-MCP is to create a robust, local-first platform that empowers AI agents to tackle complex, real-world automation and integration problems, particularly those involving legacy systems. We aim to build:

*   An **open alternative** to closed-source automation tools.
*   A system where **agents are first-class citizens**, capable of self-improvement and workflow construction.
*   A **powerful bridge** between old and new digital infrastructure.
*   A **transparent and debuggable** environment for complex agent operations.

## Core Architecture

*(Details for each pillar will be expanded as development progresses)*

### MCP Kernel
*   Manages agent state, long/short term memory.
*   Handles context retrieval (e.g., using vector stores).
*   Provides secure tool/resource access based on MCP standards.
*   Orchestrates communication between agents and modules.

### Agentic Graph Engine
*   Defines graph specification (nodes, edges, data flow).
*   Allows programmatic graph creation/modification via agent tools.
*   Executes graphs, handling node dependencies and state.
*   Supports branching, looping, error handling within graphs.

### Unified Control Plane
*   Provides APIs for browser automation (e.g., Playwright wrapper).
*   Provides APIs for OS automation (e.g., PyAutoGUI, platform-specific hooks).
*   Includes vision tools (OCR, element detection) to inform control actions.

### Visual Interaction Layer
*   Displays the node graph in real-time.
*   Allows human users to build/edit graphs.
*   Shows agent activity, logs, and current context.
*   Potential for agent-human chat interface.

### Legacy Bridge Toolkit
*   Plugins for screen scraping, OCR.
*   Tools for interacting with specific legacy UI frameworks (if possible).
*   Data transformers (e.g., XML/SOAP to JSON).
*   Terminal interaction tools.

## Installation

*(Instructions will be updated as the project matures)*

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/reconsumeralization/GraphHub-MCP.git
    cd GraphHub-MCP
    ```
2.  **Install dependencies (using uv):**
    ```bash
    uv sync --all-features
    ```

## Getting Started

*(Instructions will be updated as the project matures)*

Currently under active development. Initial focus is on establishing the core MCP Kernel and Agentic Graph Engine.

## Development Roadmap

*(High-level, subject to change)*

1.  **Phase 1: Core Infrastructure**
    *   [ ] Define core data structures (Agent State, Graph Spec, Tool Spec).
    *   [ ] Implement basic MCP Kernel for context management.
    *   [ ] Implement basic Agentic Graph Engine (load, execute simple graphs).
    *   [ ] Set up initial `AgenticWorkflowTools` for graph manipulation.
    *   [ ] Basic CLI for testing graph execution.
2.  **Phase 2: Control & Vision**
    *   [ ] Integrate basic OS control tools.
    *   [ ] Integrate basic Browser control tools.
    *   [ ] Add foundational vision tools (screenshot, OCR).
3.  **Phase 3: Visual Layer**
    *   [ ] Develop initial visual graph editor (Qt/Web).
    *   [ ] Display real-time graph execution state.
4.  **Phase 4: Legacy & Advanced Agents**
    *   [ ] Build initial Legacy Bridge tools.
    *   [ ] Enhance agents with self-modification capabilities.
    *   [ ] Implement more sophisticated memory/retrieval.

## Contributing

Contributions reflecting the core vision are highly encouraged! Please open an issue first to discuss significant changes or feature proposals.

Standard development practices apply:

*   Use `uv` for dependency management.
*   Follow formatting (`ruff format .`) and linting (`ruff check .`, `pyright`).
*   Write tests for new functionality.
*   Keep documentation updated.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

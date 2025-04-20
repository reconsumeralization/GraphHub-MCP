#!/usr/bin/env python3
"""
Enhanced utility to split custom_nodes.py into individual node class modules,
with agent and builder usability in mind. This script ensures each node is
discoverable, documented, and ready for GUI integration.

Run this script from the project root.

Features:
- Splits each BaseNode subclass into its own file.
- Generates a registry for agent/builder GUI discovery.
- Preserves and improves docstrings for GUI tooltips.
- Ensures type hints and imports are present.
- Updates custom_nodes.py to a minimal facade with BaseNode only.
- TODO: Add CLI options for dry-run, verbose, and registry-only modes.
"""

import os
import re
import sys
from typing import List, Dict

def snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(BASE, 'custom_nodes.py')
REGISTRY_PATH = os.path.join(BASE, 'custom_node_registry.py')

if not os.path.exists(SOURCE):
    print(f"Source file not found: {SOURCE}")
    sys.exit(1)

with open(SOURCE, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to match each class definition block, including docstring
pattern = re.compile(
    r"^class\s+(\w+)\(BaseNode\):([\s\S]+?)(?=^class\s+\w+\(BaseNode\):|^if\s+__name__\s*==\s*[\"']__main__[\"']|\Z)",
    re.MULTILINE
)

node_registry: List[Dict[str, str]] = []

for match in pattern.finditer(content):
    cls_name = match.group(1)
    cls_block = match.group(0)
    file_name = snake_case(cls_name) + '.py'
    out_path = os.path.join(BASE, file_name)

    # Extract docstring for GUI tooltips
    docstring_match = re.search(r'^\s+"""(.*?)"""', cls_block, re.DOTALL | re.MULTILINE)
    docstring = docstring_match.group(1).strip() if docstring_match else ""
    if not docstring:
        docstring = f"{cls_name} node for FastMCP builder. (No docstring found.)"

    # Write module with improved header
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write(
            f'"""{cls_name} - Auto-generated node module.\n'
            f'Usable in FastMCP builder GUI. \n'
            f'Description: {docstring}\n'
            f'"""\n'
        )
        out.write('from typing import Any, Dict, List, Optional, Callable\n')
        out.write('from .custom_nodes import BaseNode\n\n')
        out.write(cls_block.strip() + '\n')
    print(f"Wrote {out_path}")

    node_registry.append({
        "class_name": cls_name,
        "file": file_name,
        "doc": docstring
    })

# Write/Update registry for GUI/agent discovery
with open(REGISTRY_PATH, 'w', encoding='utf-8') as reg:
    reg.write('"""\nAuto-generated registry of custom nodes for FastMCP builder GUI/agent.\n"""\n')
    reg.write('from typing import Dict, Type\n')
    reg.write('from .custom_nodes import BaseNode\n')
    for node in node_registry:
        import_line = f"from .{node['file'][:-3]} import {node['class_name']}\n"
        reg.write(import_line)
    reg.write('\nNODE_REGISTRY: Dict[str, Type[BaseNode]] = {\n')
    for node in node_registry:
        reg.write(f"    '{node['class_name']}': {node['class_name']},  # {node['doc']}\n")
    reg.write('}\n')
print(f"Node registry written to {REGISTRY_PATH}")

# After splitting, write placeholder custom_nodes.py facade (BaseNode only)
facade_lines = []
in_base = False
with open(SOURCE, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip().startswith('class BaseNode'):
            in_base = True
            facade_lines.append(line)
        elif in_base:
            if line.strip().startswith('class ') and not line.strip().startswith('class BaseNode'):
                break  # End of BaseNode
            facade_lines.append(line)
        elif line.strip().startswith('def __init__') or line.strip().startswith('def process') or line.strip().startswith('def to_dict'):
            # Defensive: in case BaseNode is not at top
            facade_lines.append(line)
        elif not in_base:
            # Keep imports and comments at the top
            if line.strip().startswith('import') or line.strip().startswith('from') or line.strip().startswith('#') or line.strip() == '':
                facade_lines.append(line)

# Write minimal facade
with open(SOURCE, 'w', encoding='utf-8') as f:
    f.writelines(facade_lines)
print("custom_nodes.py facade updated (BaseNode only).")

# TODO: Add CLI for dry-run, verbose, and registry-only modes.
# TODO: Validate that all generated files are importable and have docstrings for GUI.
# TODO: Add self-test to ensure registry matches file system.
"""
Dynamically register all concrete BaseNode subclasses found in this directory.
This ensures that any *_node.py file containing a node class is automatically
available in the builder without explicit registration calls elsewhere.
"""

import os
import importlib
import inspect
import logging
from .node_registry import NodeRegistry
# Assuming all nodes inherit from a common base, potentially defined in custom_nodes
# Adjust the base class import if it lives elsewhere (e.g., a base_node.py)
try:
    # Try importing BaseNode from custom_nodes first as seen in some node files
    from .custom_nodes import BaseNode
except ImportError:
    # Fallback or define a minimal BaseNode if custom_nodes doesn't exist or is incorrect
    logging.warning("Could not import BaseNode from custom_nodes.py. Attempting fallback or using placeholder.")
    # Option 1: Try importing from a potential base_node.py (if refactored)
    try:
        from .base_node import BaseNode # If you have a base_node.py
    except ImportError:
        # Option 2: Define a minimal placeholder if no base class is found
        class BaseNode:
             # Add essential methods/attributes expected by NodeGraphQt or registry if known
             NODE_NAME = None # Example placeholder
             def __init__(self, *args, **kwargs):
                 pass
        logging.warning("Using a placeholder BaseNode. Node registration might be incomplete.")


logger = logging.getLogger(__name__)

# Get the directory path of the current module (__init__.py)
nodes_dir = os.path.dirname(__file__)

# Iterate over files in the nodes directory
for filename in os.listdir(nodes_dir):
    # Process only Python files ending in '_node.py', excluding '__init__.py' itself
    if filename.endswith("_node.py") and filename != "__init__.py":
        module_name = filename[:-3] # Remove '.py' extension
        try:
            # Import the module dynamically relative to the current package
            module = importlib.import_module(f".{module_name}", package=__name__)

            # Iterate through members of the imported module
            for name, obj in inspect.getmembers(module):
                # Check if the member is a class, is defined in this module (not imported),
                # and is a subclass of BaseNode (but not BaseNode itself)
                if inspect.isclass(obj) and \
                   obj.__module__ == module.__name__ and \
                   issubclass(obj, BaseNode) and \
                   obj is not BaseNode:
                    # Use the class name as the identifier (consistent with node files)
                    node_identifier = obj.__name__
                    try:
                        NodeRegistry.register(node_identifier, obj)
                        logger.debug(f"Registered node: {node_identifier} from {filename}")
                    except ValueError as e: # Catch potential duplicate registrations
                        logger.warning(f"Skipping duplicate node registration for {node_identifier}: {e}")
                    except Exception as e:
                        logger.error(f"Failed to register node {node_identifier} from {filename}: {e}")

        except ImportError as e:
            logger.error(f"Failed to import node module {module_name}: {e}")
        except Exception as e:
            logger.error(f"Error processing node file {filename}: {e}")

logger.info("Completed dynamic node registration.")

# Optionally, clean up namespace to avoid polluting imports
# del os, importlib, inspect, filename, module_name, module, name, obj, node_identifier 
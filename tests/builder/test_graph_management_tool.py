import os
import json
import pytest
from fastmcp.builder.workflow_manager import WorkflowManager, FileStorage

@ pytest.fixture
def manager(tmp_path):
    storage = FileStorage(base_dir=str(tmp_path / "storage"))
    # Use default executor stub for execution methods
    return WorkflowManager(storage=storage)

def test_create_graph_alias(manager):
    # Initially no workflows
    assert manager.list_workflows() == []
    graph_id = "g1"
    returned_id = manager.create_graph(graph_id, description="Test graph g1")
    assert returned_id == graph_id
    assert graph_id in manager.list_workflows()

def test_get_graph_spec_alias(manager):
    graph_id = "g2"
    manager.create_graph(graph_id, description="desc2")
    spec = manager.get_graph_spec(graph_id)
    assert spec is not None
    assert spec.get("workflow_id") == graph_id
    assert spec.get("description") == "desc2"

def test_save_graph_spec_alias(tmp_path, manager):
    graph_id = "g3"
    manager.create_graph(graph_id, description=None)
    file_path = tmp_path / "g3.json"
    result = manager.save_graph_spec(graph_id, str(file_path))
    assert result is True
    assert os.path.isfile(str(file_path))
    # Verify file content matches the graph spec
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("workflow_id") == graph_id
    # No file path provided yields False
    assert manager.save_graph_spec(graph_id) is False 
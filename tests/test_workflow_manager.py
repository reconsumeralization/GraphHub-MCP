import pytest
from fastmcp.builder.workflow_manager import WorkflowManager

class DummyExecutor:
    def execute_graph(self, graph_id, input_data):
        # return a fake execution ID
        return f"exec_for_{graph_id}"

    def get_execution_status(self, execution_id):
        # return a fake status
        return {"execution_id": execution_id, "status": "completed", "result": {"ok": True}}

@pytest.fixture
def manager():
    # Inject dummy executor to isolate tests
    return WorkflowManager(executor=DummyExecutor())

def test_create_and_list_workflow(manager):
    assert manager.list_workflows() == []
    manager.create_workflow("w1", {"nodes": [], "edges": []}, user="tester")
    ids = manager.list_workflows()
    assert "w1" in ids

    # get_workflow returns the definition
    wf = manager.get_workflow("w1")
    assert wf == {"nodes": [], "edges": []}

def test_update_workflow(manager):
    manager.create_workflow("w2", {"nodes": [{"id": "n1"}], "edges": []}, user="userA")
    # update definition
    new_def = {"nodes": [{"id": "n1", "foo": "bar"}], "edges": []}
    manager.update_workflow("w2", new_def, user="userB")
    wf = manager.get_workflow("w2")
    assert wf["nodes"][0].get("foo") == "bar"

def test_delete_workflow(manager):
    manager.create_workflow("w3", {"nodes": [], "edges": []})
    manager.delete_workflow("w3")
    assert manager.get_workflow("w3") is None
    # deleting non-existent should not raise
    manager.delete_workflow("unknown")

def test_validate_workflow_always_true(manager):
    manager.create_workflow("w4", {"nodes": [], "edges": []})
    assert manager.validate_workflow("w4") is True

def test_trigger_and_get_execution_status(manager):
    manager.create_workflow("w5", {"nodes": [], "edges": []})
    exec_id = manager.trigger_graph_execution("w5", {"key": "value"})
    assert exec_id == "exec_for_w5"
    status = manager.get_execution_status(exec_id)
    assert status["status"] == "completed"
    assert status["result"]["ok"] is True 
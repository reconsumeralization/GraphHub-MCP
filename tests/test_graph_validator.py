import pytest  # type: ignore[reportMissingImports]
from fastmcp.builder.graph_validator import GraphValidator, GraphValidationError, GraphOptimizationHint


def make_nodes_types(ids):
    return {node_id: {'type': 'task'} for node_id in ids}


def test_validate_valid_graph():
    # Simple linear DAG: a -> b -> c
    nodes = make_nodes_types(['a', 'b', 'c'])
    edges = [('a', 'b'), ('b', 'c')]
    validator = GraphValidator(nodes, edges, start_node_id='a')
    # Should not raise
    validator.validate()


def test_missing_start_node_raises():
    nodes = make_nodes_types(['b', 'c'])
    edges = [('b', 'c')]
    with pytest.raises(GraphValidationError) as exc:
        GraphValidator(nodes, edges, start_node_id='a').validate()
    assert "Start node 'a' does not exist" in str(exc.value)


def test_cycle_detection_raises():
    nodes = make_nodes_types(['a', 'b'])
    edges = [('a', 'b'), ('b', 'a')]
    with pytest.raises(GraphValidationError) as exc:
        GraphValidator(nodes, edges, start_node_id='a').validate()
    assert "cycle" in str(exc.value).lower()


def test_unreachable_node_raises():
    nodes = make_nodes_types(['a', 'b', 'c'])
    edges = [('a', 'b')]
    with pytest.raises(GraphValidationError) as exc:
        GraphValidator(nodes, edges, start_node_id='a').validate()
    msg = str(exc.value)
    assert 'Unreachable nodes detected' in msg
    assert 'c' in msg


def test_orphan_node_raises():
    # Orphan is non-start node with no edges
    nodes = make_nodes_types(['a', 'z', 'y'])
    edges = [('a', 'z')]  # 'y' is orphan
    with pytest.raises(GraphValidationError) as exc:
        GraphValidator(nodes, edges, start_node_id='a').validate()
    assert 'Orphaned node detected' in str(exc.value)
    assert 'y' in str(exc.value)


def test_missing_type_field_raises():
    nodes = {'a': {}, 'b': {'type': 'task'}}
    edges = [('a', 'b')]
    with pytest.raises(GraphValidationError) as exc:
        GraphValidator(nodes, edges, start_node_id='a').validate()
    assert "missing required field: 'type'" in str(exc.value)


def test_analyze_performance_hints():
    # Create two nodes with identical branches to trigger optimization hint
    nodes = make_nodes_types(['a', 'b', 'c'])
    edges = [('a', 'b'), ('c', 'b')]
    validator = GraphValidator(nodes, edges, start_node_id='a')
    hints = validator.analyze_performance()
    assert isinstance(hints, list)
    for hint in hints:
        assert isinstance(hint, GraphOptimizationHint)
        assert hasattr(hint, 'message')
        assert hasattr(hint, 'diff') 
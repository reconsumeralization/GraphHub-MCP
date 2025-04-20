from typing import Any, Dict, List, Optional, Callable
from .custom_nodes import BaseNode

class HumanInputNode(BaseNode):
    """
    Node that pauses execution and awaits human input or approval.
    """
    prompt: str
    input_schema: Optional[Any]

    def __init__(self, node_id: str, label: str, prompt: str,
                 input_schema: Optional[Any] = None) -> None:
        super().__init__(node_id, label)
        self.prompt = prompt
        self.input_schema = input_schema

    def process(self, _: Any = None) -> Any:
        # TODO: Integrate with UI or messaging to collect human input.
        raise NotImplementedError(f"HumanInputNode [{self.node_id}] requires external input.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "HumanInputNode",
            "node_id": self.node_id,
            "label": self.label,
            "prompt": self.prompt,
        }

# Register the new node types dynamically
register_custom_node("AIStructuredPromptNode", AIStructuredPromptNode)
register_custom_node("AIEmbeddingNode", AIEmbeddingNode)
register_custom_node("AIFunctionCallingNode", AIFunctionCallingNode)
register_custom_node("AISpeechToTextNode", AISpeechToTextNode)
register_custom_node("AITextToSpeechNode", AITextToSpeechNode)
register_custom_node("DataMapperNode", DataMapperNode)
register_custom_node("DataFilterNode", DataFilterNode)
register_custom_node("DataMergeNode", DataMergeNode)
register_custom_node("JSONParserNode", JSONParserNode)
register_custom_node("HTTPRequestNode", HTTPRequestNode)
register_custom_node("DatabaseQueryNode", DatabaseQueryNode)
register_custom_node("FileSystemNode", FileSystemNode)
register_custom_node("DelayNode", DelayNode)
register_custom_node("SendEmailNode", SendEmailNode)
register_custom_node("AgentCheckpointNode", AgentCheckpointNode)
register_custom_node("HumanInputNode", HumanInputNode)

# --- Node Registry and Registration ---

NODE_REGISTRY: Dict[str, Type[BaseNode]] = {
    "CustomInputNode": CustomInputNode,
    "CustomOutputNode": CustomOutputNode,
    "CustomProcessingNode": CustomProcessingNode,
    "AIVisionModelNode": AIVisionModelNode,
    "AIGraphRunnerNode": AIGraphRunnerNode,
    "AINormalizeNode": AINormalizeNode,
    "SubWorkflowNode": SubWorkflowNode,
}

def register_custom_node(node_type: str, node_class: Type[BaseNode]) -> None:
    """
    Register a custom node class for auto-discovery.
    """
    if node_type in NODE_REGISTRY:
        raise ValueError(f"Node type '{node_type}' is already registered.")
    NODE_REGISTRY[node_type] = node_class

def discover_and_register_nodes(module: Any) -> None:
    """
    Auto-discover and register all subclasses of BaseNode in the given module.
    """
    import inspect
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, BaseNode) and obj is not BaseNode:
            if obj.__name__ not in NODE_REGISTRY:
                NODE_REGISTRY[obj.__name__] = obj

# --- Unit Tests (strictly typed, edge cases included) ---

def _test_custom_input_node() -> None:
    node = CustomInputNode("in1", "Input", 42)
    assert node.process() == 42
    d = node.to_dict()
    assert d["type"] == "CustomInputNode"
    assert d["value"] == 42
    # Edge: None value
    try:
        CustomInputNode("in2", "Input", None).process()
        assert False, "Should raise ValueError"
    except ValueError:
        pass

def _test_custom_output_node() -> None:
    node = CustomOutputNode("out1", "Output")
    node.process(99)
    assert node.input_value == 99
    d = node.to_dict()
    assert d["type"] == "CustomOutputNode"
    assert d["input_value"] == 99
    # Edge: None input
    try:
        node.process(None)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

def _test_custom_processing_node() -> None:
    def double(x: int) -> int:
        return x * 2
    node = CustomProcessingNode("proc1", "Double", double)
    assert node.process(5) == 10
    d = node.to_dict()
    assert d["type"] == "CustomProcessingNode"
    assert d["operation"] == "double"
    # Edge: operation not callable
    try:
        CustomProcessingNode("proc2", "BadOp", 123).process(1)
        assert False, "Should raise TypeError"
    except TypeError:
        pass

def _test_ai_normalize_node() -> None:
    node = AINormalizeNode("norm1", "Normalize", mean=10, std=2)
    assert node.process(12) == 1.0
    assert node.process([8, 10, 12]) == [-1.0, 0.0, 1.0]
    d = node.to_dict()
    assert d["type"] == "AINormalizeNode"
    assert d["mean"] == 10
    assert d["std"] == 2
    # Edge: std=0 fallback
    node2 = AINormalizeNode("norm2", "NormZero", mean=0, std=0)
    assert node2.std == 1.0
    # Edge: wrong input type
    try:
        node.process("bad")
        assert False, "Should raise TypeError"
    except TypeError:
        pass

def _test_ai_vision_model_node() -> None:
    class DummyModel:
        def predict(self, image: Any) -> str:
            return f"predicted:{image}"
    node = AIVisionModelNode("vis1", "Vision", "dummy", DummyModel())
    assert node.process("img") == "predicted:img"
    d = node.to_dict()
    assert d["type"] == "AIVisionModelNode"
    assert d["model_name"] == "dummy"
    # Edge: model missing predict
    class BadModel: pass
    try:
        AIVisionModelNode("vis2", "Vision", "bad", BadModel()).process("img")
        assert False, "Should raise AttributeError"
    except AttributeError:
        pass
    # Edge: model is None
    try:
        AIVisionModelNode("vis3", "Vision", "none", None).process("img")
        assert False, "Should raise ValueError"
    except ValueError:
        pass

def _test_ai_graph_runner_node() -> None:
    class DummySubgraph:
        def run(self, inputs: Dict[str, Any]) -> str:
            return f"ran:{inputs}"
    node = AIGraphRunnerNode("gr1", "GraphRunner", DummySubgraph())
    assert node.process({"x": 1}) == "ran:{'x': 1}"
    d = node.to_dict()
    assert d["type"] == "AIGraphRunnerNode"
    # Edge: subgraph missing run
    class BadSubgraph: pass
    try:
        AIGraphRunnerNode("gr2", "GraphRunner", BadSubgraph()).process({})
        assert False, "Should raise AttributeError"
    except AttributeError:
        pass
    # Edge: inputs not dict
    try:
        node.process("notadict")
        assert False, "Should raise TypeError"
    except TypeError:
        pass

def _test_subworkflow_node() -> None:
    subflow_spec = {
        "nodes": [{"id": "n1", "type": "CustomInputNode"}],
        "edges": [],
    }
    node = SubWorkflowNode("sub1", "ReusableSubflow", subflow_spec, parameters={"foo": 123}, version="v1")
    result = node.process({"bar": 456})
    assert result["subflow_executed"] is True
    assert result["subflow_spec"] == subflow_spec
    assert result["inputs"]["foo"] == 123
    assert result["inputs"]["bar"] == 456
    assert result["version"] == "v1"
    d = node.to_dict()
    assert d["type"] == "SubWorkflowNode"
    assert d["subflow_spec"] == subflow_spec
    assert d["parameters"]["foo"] == 123
    assert d["version"] == "v1"
    # Edge: inputs not dict
    try:
        node.process("notadict")
        assert False, "Should raise TypeError"
    except TypeError:
        pass

def _run_all_tests() -> None:
    _test_custom_input_node()
    _test_custom_output_node()
    _test_custom_processing_node()
    _test_ai_normalize_node()
    _test_ai_vision_model_node()
    _test_ai_graph_runner_node()
    _test_subworkflow_node()
    print("All custom node unit tests passed. ☕️🚀")

if 

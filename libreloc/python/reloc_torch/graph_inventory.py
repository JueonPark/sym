"""Read-only inventory of raw Dynamo and ATen FX graphs; never rewrites them."""

from . import compat
from .records import GraphRecord, TransferRecord


_RAW = {"to": "aten._to_copy.default", "cpu": "aten._to_copy.default",
        "cuda": "aten._to_copy.default", "reshape": "aten.reshape.default",
        "view": "aten.view.default", "transpose": "aten.transpose.int",
        "permute": "aten.permute.default", "contiguous": "aten.clone.default",
        "copy_": "aten.copy_.default", "add_": "aten.add_.Tensor"}
_VIEWS = {"aten.view.default", "aten.reshape.default", "aten.transpose.int",
          "aten.permute.default", "aten.as_strided.default",
          "aten._unsafe_view.default"}


def graph_inventory(gm):
    """Snapshot immutable plain metadata and adjacency without executing ``gm``.

    ``transfer`` is a conservative boundary description for the shared classifier,
    not permission to normalize a raw target or replace an operation.
    """
    compat.check_version()
    records = []
    histories = {}
    for node in gm.graph.nodes:
        value = compat.graph_value(node)
        metadata = compat.tensor_metadata(value, graph=True) if compat.is_tensor(value) else None
        target = compat.graph_target(node.target)
        operator = _RAW.get(target, target) if node.op == "call_method" else target
        inputs = tuple(node.all_input_nodes)
        source_node = next((n for n in inputs if compat.is_tensor(compat.graph_value(n))), None)
        transfer = None
        alias_semantics = "unknown"
        mutates = compat.graph_mutates(node)
        history = histories.get(source_node, ())
        if mutates:
            histories.clear()
            history = ()
        elif operator in _VIEWS or operator == "aten.clone.default":
            history = (history + (operator,))[-16:]
        elif operator != "aten._to_copy.default":
            history = ()
        histories[node] = history
        if source_node is not None and metadata is not None:
            source = compat.tensor_metadata(compat.graph_value(source_node), graph=True)
            if operator == "aten.copy_.default" and len(inputs) > 1:
                source_node = inputs[1]
                source = compat.tensor_metadata(compat.graph_value(source_node), graph=True)
            alias_semantics = compat.graph_alias_semantics(node, compat.graph_value(source_node), value)
            transfer = TransferRecord(operator, "graph", source, metadata,
                                      compat.graph_nonblocking(node),
                                      bool(mutates), alias_semantics != "distinct", history)
        records.append(GraphRecord(node.name, node.op, target,
                                   tuple(n.name for n in inputs), tuple(n.name for n in node.users),
                                   metadata, None if metadata is not None else "metadata_unavailable", transfer, alias_semantics))
    return tuple(records)

"""Conservative, transparent permutation-copy elimination before fresh cat.

Cost fields describe logical tensor traffic. They are not physical-memory peaks
or latency predictions. Original graphs remain available for excluded patterns.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ConsumerLayoutPolicy:
    min_saved_bytes: int = 0

    def __post_init__(self):
        if type(self.min_saved_bytes) is not int:
            raise TypeError('min_saved_bytes must be an integer')
        if self.min_saved_bytes < 0:
            raise ValueError('min_saved_bytes must be nonnegative')


@dataclass(frozen=True)
class ConsumerLayoutDecision:
    node: str
    source: str | None
    consumers: tuple[str, ...]
    eligible: bool
    selected: bool
    reason: str
    eliminated_bytes: int | None
    read_write_bytes: int | None
    rule: str = 'permutation_copy_into_cat'


def normalize_policy(policy):
    if policy is False or policy is None:
        return None
    if policy is True:
        return ConsumerLayoutPolicy()
    if not isinstance(policy, ConsumerLayoutPolicy):
        raise TypeError('consumer_layout must be bool or ConsumerLayoutPolicy')
    return policy


def _effect_reason(graph):
    """Unknown calls may mutate aliases even without a tensor argument."""
    import operator
    import torch
    f = torch.nn.functional
    functions = {operator.getitem, operator.add, operator.sub, operator.mul,
        operator.truediv, operator.floordiv, operator.mod, operator.pow, operator.matmul, getattr,
        torch.cat, torch.concat, torch.concatenate, torch.permute, torch.transpose,
        torch.clone, torch.split, torch.ones, torch.zeros, torch.empty,
        torch.ones_like, torch.zeros_like, f.linear, f.layer_norm, f.gelu,
        torch.matmul, torch.bmm, torch.mm, torch.softmax}
    methods = {'transpose','permute','contiguous','clone','view','reshape','split',
        'size','dim','numel','to','float','half','bfloat16','t','softmax','tril',
        'masked_fill','unsqueeze','squeeze','flatten','detach'}
    for node in graph.nodes:
        if node.op == 'call_function':
            schema = getattr(node.target, '_schema', None)
            if schema is not None:
                if schema.is_mutable:
                    return 'graph_mutation'
                if not str(schema.name).startswith('aten::'):
                    return 'opaque_operator'
            elif node.target not in functions:
                return 'opaque_operator'
        elif node.op == 'call_method' and node.target not in methods:
            return 'graph_mutation' if str(node.target).endswith('_') else 'opaque_operator'
        elif node.op == 'call_module':
            return 'opaque_operator'
    return None


def _dense(value):
    import torch
    from . import compat
    if not compat.is_tensor(value):
        return False
    if not (compat.is_fake_tensor(value) or compat.is_plain_tensor_or_parameter(value)):
        return False
    try:
        return (value.layout == torch.strided and not value.is_quantized
            and not value.is_nested and value.device.type == 'cuda' and value.dim() in (2, 3)
            and all(type(d) is int and d > 0 for d in value.shape)
            and all(type(d) is int for d in value.stride())
            and type(value.storage_offset()) is int and value.storage_offset() >= 0
            and value.is_contiguous() and not value.is_conj() and not value.is_neg())
    except (RuntimeError, TypeError):
        return False


def _candidate(node):
    import torch
    from . import compat
    material = compat.graph_value(node)
    if not _dense(material):
        return None, None, 'unsupported_materialization_metadata'
    if node.kwargs.get('memory_format') not in (None, torch.contiguous_format):
        return None, None, 'unsupported_memory_format'
    view = node.args[0] if node.args else None
    source = view
    count = 0
    while hasattr(source, 'op') and compat.fx_kind(source) in ('permute', 'transpose'):
        count += 1
        source = source.args[0]
    if not count or not hasattr(source, 'op') or source.op != 'placeholder':
        return None, None, 'not_input_permutation'
    value = compat.graph_value(source)
    if not _dense(value) or value.dtype != material.dtype or value.device != material.device:
        return source, None, 'unsupported_source_metadata'
    # A source offset may point into a packed buffer. No pointer arithmetic is
    # emitted: Torch's view and cat operators retain their normal bounds/aliasing.
    amount = value.numel() * value.element_size()
    if not node.users:
        return source, amount, 'no_consumer'
    for user in node.users:
        if user.op != 'call_function' or user.target not in (
                torch.cat, torch.concat, torch.concatenate, torch.ops.aten.cat.default):
            return source, amount, 'observable_or_unsupported_consumer'
        if user.kwargs.get('out') is not None:
            return source, amount, 'observable_or_unsupported_consumer'
        tensors = user.args[0] if user.args else user.kwargs.get('tensors')
        if not isinstance(tensors, (tuple, list)) or not tensors or node not in tensors:
            return source, amount, 'unsupported_cat_arguments'
        result = compat.graph_value(user)
        if not _dense(result) or result.dtype != material.dtype or result.device != material.device:
            return source, amount, 'unsupported_cat_metadata'
        for tensor in tensors:
            if not hasattr(tensor, 'op'):
                return source, amount, 'unsupported_cat_arguments'
            other = compat.graph_value(tensor)
            if (not compat.is_tensor(other) or other.layout != torch.strided
                    or other.dtype != material.dtype or other.device != material.device
                    or other.dim() != material.dim()):
                return source, amount, 'unsupported_cat_metadata'
    return source, amount, None


def lower_consumer_layout(gm, example_inputs=None, *, policy=None):
    """Return a graph copy and immutable planner decisions for qualified copies.

    This compiler pass relies on FX metadata/Dynamo guards, not runtime data or
    user assertions of purity. Only closed known-functional graphs are eligible.
    It never mutates gm, emits a custom op, or executes its real tensor inputs.
    """
    import torch
    from . import compat
    import copy
    if policy is None:
        policy = ConsumerLayoutPolicy()
    elif not isinstance(policy, ConsumerLayoutPolicy):
        raise TypeError('policy must be ConsumerLayoutPolicy')
    graph = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
    effect = _effect_reason(gm.graph)
    if torch.autograd.forward_ad._current_level >= 0 or torch._C._are_functorch_transforms_active():
        effect = 'active_transform'
    decisions = []
    for node in list(graph.graph.nodes):
        if compat.fx_kind(node) != 'materialize':
            continue
        source, amount, reason = _candidate(node)
        reason = effect or reason
        eligible = reason is None
        if eligible and amount < policy.min_saved_bytes:
            reason = 'below_threshold'
        selected = reason is None
        decisions.append(ConsumerLayoutDecision(node.name, source.name if source is not None else None,
            tuple(user.name for user in node.users), eligible, selected, reason or 'selected',
            amount, 2 * amount if amount is not None else None))
        if selected:
            node.replace_all_uses_with(node.args[0])
            graph.graph.erase_node(node)
    if not any(r.selected for r in decisions):
        return gm, tuple(decisions)
    graph.graph.lint()
    graph.recompile()
    return graph, tuple(decisions)

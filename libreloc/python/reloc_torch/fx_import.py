"""Conservative, read-only FX region discovery.

Candidates are provisional recipes: compiler folding is a separate acceptance
gate, and neither discovery nor normalization enables runtime execution.

C4 (issue #144) admits the value transforms C1 defines and Torch 2.14.0's
audited operators implement bit for bit: casts between f32 and f16 carried by
``aten._to_copy`` (``ieee_rne`` narrowing, ``exact`` widening) and
``quantized_decomposed.dequantize_per_tensor`` / ``dequantize_per_channel``
(C1 ``affine``). Scale and zero-point operands stay what the graph supplies:
Python scalars become exact inline bits, tensor inputs become runtime
parameters bound by name at execution. ``quantized_decomposed.quantize_*``
is excluded with ``quantize_semantics_unproved`` (double-rounded reciprocal,
platform NaN conversion); a float -> int ``.to()`` is not a quantization and
stays ``typed_transform_unavailable``.
"""
from dataclasses import dataclass
import math
import struct

from . import compat
from .recipe import (BindingParam, Cast, Dequantize, Fill, InlineParam, Pad, Recipe, Reshape,
                     TensorSpec, Transpose)
from .symbolic import (Const, SymbolSource, UnsupportedSymbolicExpr, dense_strides,
                       expression, infer_reshape, operation_shape)


@dataclass(frozen=True)
class Exclusion:
    node: str
    reason: str


@dataclass(frozen=True)
class Candidate:
    source: str
    tail: str
    members: tuple[str, ...]
    recipe: Recipe
    symbol_sources: tuple[SymbolSource, ...]
    # External scalar placeholders required by the exact original callable.
    # Each (FX node name, Expr) is evaluable from symbol_sources.
    symbolic_bindings: tuple
    original: object
    # Extents that must bind to at least two for the original region and the
    # recipe to agree on output metadata (Dynamo's 0/1 specialization makes a
    # transposed contiguous() unconditional only inside this family).
    extent_guards: tuple = ()
    # Destination device of the region's result, read from the normalized
    # fake metadata so a rewrite never depends on metadata being present on
    # the caller's original graph nodes.
    device: object = None
    # C4: graph nodes (placeholders or attributes) whose tensor values are the
    # recipe's runtime parameters, in the recipe's declaration order; the
    # original callable takes them after the scalar placeholders.
    parameters: tuple = ()


@dataclass(frozen=True)
class ImportReport:
    candidates: tuple[Candidate, ...]
    exclusions: tuple[Exclusion, ...]


# C1 §3.1 / §3.2: the only casts with a numerical contract.
_CASTS = {('float32', 'float16'): Cast('float16', 'ieee_rne'), ('float16', 'float32'): Cast('float32', 'exact')}


def normalize_graph(gm, example_inputs=None):
    """Copy and normalize known nodes; never replay arbitrary Python or transfers.

    Metadata stays on the copy. Unknown targets survive verbatim with a reason
    attached and are never evaluated, even if they already have tensor metadata.
    """
    import torch
    from torch.fx import Graph, GraphModule, map_arg
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    compat.check_version()
    inputs = iter(example_inputs or ())
    graph, mapping, values = Graph(), {}, {}
    fake_mode = None
    for node in gm.graph.nodes:
        value = compat.graph_value(node)
        if compat.is_fake_tensor(value):
            fake_mode = value.fake_mode
            break
    if fake_mode is None:
        fake_mode = FakeTensorMode(allow_fallback_kernels=False, shape_env=ShapeEnv())
    for old in gm.graph.nodes:
        node = graph.node_copy(old, lambda n: mapping[n])
        mapping[old] = node
        node.meta = dict(old.meta)
        value = compat.graph_value(old)
        if old.op == 'placeholder':
            supplied = next(inputs, None)
            value = value if value is not None else supplied
            if compat.is_tensor(value) and not compat.is_fake_tensor(value):
                if not compat.is_plain_tensor_or_parameter(value):
                    node.meta['reloc_reason'] = 'tensor_subclass'
                    value = None
                elif value.layout != torch.strided or value.is_quantized:
                    node.meta['reloc_reason'] = 'unsupported_layout'
                    value = None
                else:
                    value = fake_mode.from_tensor(value, static_shapes=True)
        elif old.op == 'get_attr':
            # Captured constants already carry metadata; do not invoke getters.
            pass
        elif old.op in ('call_function', 'call_method'):
            source_value = values.get(node.args[0]) if node.args else None
            try:
                target, args, kwargs = compat.fx_canonical_call(node, source_value)
                node.op, node.target, node.args, node.kwargs = 'call_function', target, args, kwargs
                def resolve(n):
                    if n not in values or values[n] is None:
                        raise ValueError('metadata_unavailable')
                    return values[n]
                if compat.fx_kind(node) == 'scalar':
                    compat.fx_validate_scalar(target, map_arg(args, resolve))
                if value is None:
                    with fake_mode, torch.no_grad():
                        value = compat.fx_fake_call(target, map_arg(args, resolve), map_arg(kwargs, resolve))
            except (ValueError, TypeError, RuntimeError, NotImplementedError) as error:
                # Keep the specific expected exclusion rather than making every
                # unsupported call look like an unknown target.
                reason = str(error)
                node.meta['reloc_reason'] = reason if reason in {
                    'unrecognized_fx_target', 'metadata_unavailable', 'unsupported_padding'
                } else 'normalization_failed'
                value = None
        if value is not None:
            values[node] = value
            node.meta['val'] = value
        else:
            values[node] = None
            node.meta.pop('val', None)
            node.meta.pop('example_value', None)
    return GraphModule(gm, graph)


def _options(node):
    return {k: v for k, v, _ in compat.bound_schema_arguments(node.target, node.args, node.kwargs)}


def _tensor_input(node):
    return node.args[0] if node.args and hasattr(node.args[0], 'op') else None


def _layout(node):
    return compat.fx_kind(node) in {'permute', 'transpose', 'reshape', 'materialize', 'pad'}


def _transfer(node):
    return compat.fx_kind(node) == 'transfer'


def _typed(node):
    return compat.fx_kind(node) in {'dequantize', 'quantize'}


def _moves_device(node):
    """A transfer-kind node that changes the device; without metadata on both
    ends it is treated as a device move (the recipe then rejects it precisely)."""
    if not _transfer(node):
        return False
    before = compat.graph_value(_tensor_input(node)) if _tensor_input(node) is not None else None
    after = compat.graph_value(node)
    if not (compat.is_tensor(before) and compat.is_tensor(after)):
        return True
    return before.device != after.device


class _Reject(Exception):
    def __init__(self, reason):
        self.reason = reason


def _require(condition, reason):
    if not condition:
        raise _Reject(reason)


def _safety(nodes, root, members):
    """Known aliases form a conservative closure, including external views.
    Parameter inputs need no entry here: a write to one between the root and
    the tail is an unknown side effect in the scan below."""
    involved = {root, *members}
    aliases = set(involved)
    edges = []
    for node in nodes:
        if compat.graph_may_alias_inputs(node):
            edges.append({node, *node.all_input_nodes})
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if aliases.intersection(edge) and not edge <= aliases:
                aliases.update(edge)
                changed = True
    for node in nodes:
        if compat.graph_mutates(node) and any(n in aliases for n in node.all_input_nodes):
            raise _Reject('mutation')
    positions = {node: i for i, node in enumerate(nodes)}
    for node in nodes[positions[root] + 1:positions[members[-1]]]:
        if node not in involved and node.op in ('call_function', 'call_method', 'call_module'):
            if compat.fx_kind(node) is None or 'reloc_reason' in node.meta:
                raise _Reject('unknown_side_effect')
    for member in members[:-1]:
        _require(all(user in involved for user in member.users), 'escaping_intermediate')


def _fill(dtype, value):
    _require(type(value) in (int, float, bool), 'unsupported_padding')
    try:
        if dtype == 'int8':
            _require(int(value) == value and -128 <= value <= 127, 'unsupported_padding')
            bits = int(value) & 255
        else:
            packed = struct.pack('<f' if dtype == 'float32' else '<e', value)
            bits = int.from_bytes(packed, 'little')
        return Fill(dtype, bits)
    except (OverflowError, ValueError, struct.error):
        raise _Reject('unsupported_padding') from None


def _constant_fold(expr):
    try:
        return Const(expr.evaluate({}))
    except KeyError:
        return expr


def _cast(before, after):
    op = _CASTS.get((compat.dtype_name(before), compat.dtype_name(after)))
    _require(op is not None, 'typed_transform_unavailable')
    return op


def _parameter(value, role, per_channel, extent, context, parameters):
    """A dequantize operand as the recipe records it: exact inline bits for a
    Python scalar, a named runtime binding for a tensor-valued graph node."""
    import torch
    if hasattr(value, 'op'):
        tensor = compat.graph_value(value)
        _require(compat.is_tensor(tensor), 'metadata_unavailable')
        _require(value.op in ('placeholder', 'get_attr'), 'unsupported_parameter')
        if role == 'scale':
            _require(tensor.dtype == torch.float32, 'parameter_dtype_unsupported')
            dtype = 'float32'
        else:
            _require(tensor.dtype in (torch.int32, torch.int64, torch.int8), 'parameter_dtype_unsupported')
            dtype = 'int32'
        if per_channel:
            _require(len(tensor.shape) == 1, 'unsupported_parameter')
            length = context.expression(tensor.shape[0])
            _require(length == extent, 'unsupported_parameter')
            extents = (length,)
        else:
            _require(len(tensor.shape) == 0, 'unsupported_parameter')
            extents = ()
        parameters.setdefault(value.name, value)
        return BindingParam(value.name, dtype, extents)
    _require(not per_channel, 'unsupported_parameter')
    if role == 'scale':
        _require(type(value) in (float, int) and math.isfinite(value) and value > 0, 'unsupported_parameter')
        bits = struct.unpack('<I', struct.pack('<f', float(value)))[0]
        return InlineParam('float32', (), (bits,))
    _require(type(value) is int and -128 <= value <= 127, 'unsupported_parameter')
    return InlineParam('int32', (), (value & 0xFFFFFFFF,))


def _dequantize(node, value, current_dtype, shape, context, parameters):
    import torch
    opts = _options(node)
    _require(current_dtype == 'int8' and value.dtype == torch.float32, 'typed_transform_unavailable')
    _require(opts.get('dtype') == torch.int8, 'unsupported_dtype')
    _require(opts.get('quant_min') == -128 and opts.get('quant_max') == 127, 'unsupported_quantization_range')
    _require(opts.get('out_dtype') in (None, torch.float32), 'typed_transform_unavailable')
    per_channel = 'scales' in opts
    axis = None
    extent = None
    if per_channel:
        axis = opts.get('axis')
        _require(type(axis) is int and 0 <= axis < len(shape), 'unsupported_channel_axis')
        extent = expression(shape[axis])
    scale = _parameter(opts['scales' if per_channel else 'scale'], 'scale', per_channel, extent, context, parameters)
    raw_zero_point = opts.get('zero_points' if per_channel else 'zero_point')
    zero_point = None
    if raw_zero_point is not None:
        zero_point = _parameter(raw_zero_point, 'zero_point', per_channel, extent, context, parameters)
    return Dequantize('float32', scale, zero_point, axis, 'affine')


def _recipe(root, members):
    import torch
    from torch.fx import map_arg
    source = compat.graph_value(root)
    _require(compat.is_tensor(source), root.meta.get('reloc_reason', 'metadata_unavailable'))
    _require(compat.is_fake_tensor(source) or compat.is_plain_tensor_or_parameter(source), 'tensor_subclass')
    _require(source.layout == torch.strided and not source.is_quantized, 'unsupported_layout')
    # Gradient-requiring only while autograd could record. Dynamo guards the
    # grad mode a graph was captured under, and the returned graph callable
    # re-checks it per call, so a no_grad capture of parameters is eligible.
    _require(not source.requires_grad or not torch.is_grad_enabled(), 'requires_grad')
    _require(len(source.shape) > 0, 'rank_zero')
    _require(not any(type(d) is int and d == 0 for d in source.shape), 'empty_tensor')
    context = compat.SymbolicContext.from_tensor(source)
    src = context.tensor_spec(source, require_dense=False)
    _require(src.offset == Const(0) and src.strides == dense_strides(src.shape), 'source_layout')
    _require(src.dtype in {'float32', 'float16', 'int8'}, 'unsupported_dtype')
    operations, shape, direction = [], src.shape, None
    strides, offset = src.strides, src.offset
    current_dtype = src.dtype
    parameters = {}
    extent_guards = []
    for node in members:
        _require('reloc_reason' not in node.meta, node.meta.get('reloc_reason'))
        value = compat.graph_value(node)
        _require(compat.is_tensor(value), 'metadata_unavailable')
        _require(not value.requires_grad or not torch.is_grad_enabled(), 'requires_grad')
        kind = compat.fx_kind(node)
        opts = _options(node)
        def scalar(n):
            value = compat.graph_value(n)
            _require(value is not None and not compat.is_tensor(value), 'unsupported_symbolic_expr')
            return value
        opts = {k: v for k, v in opts.items() if k != 'self'}
        opts = map_arg(opts, scalar) if kind in {'reshape', 'pad'} else opts
        # Do not resolve the tensor self argument as a scalar.
        if kind == 'transfer':
            before = compat.graph_value(_tensor_input(node))
            moves = before.device != value.device
            if moves:
                _require(direction is None, 'multiple_transfers')
                _require(opts.get('non_blocking', False) is False, 'nonblocking_unavailable')
                devices = before.device.type, value.device.type
                _require(devices in {('cpu', 'cuda'), ('cuda', 'cpu')}, 'unsupported_device')
                direction = 'h2d' if devices[0] == 'cpu' else 'd2h'
                _require(opts.get('pin_memory') in (None, False), 'pinned_transfer_unavailable')
            else:
                # A same-device copy relocates nothing; a same-device cast is
                # a typed stage of the region that carries the transfer.
                _require(before.dtype != value.dtype, 'same_device_transfer')
            memory_format = opts.get('memory_format')
            _require(memory_format in (None, torch.preserve_format, torch.contiguous_format), 'unsupported_memory_format')
            if before.dtype != value.dtype:
                op = _cast(before.dtype, value.dtype)
                _require(compat.dtype_name(before.dtype) == current_dtype, 'typed_transform_unavailable')
                operations.append(op)
                current_dtype = op.dtype
            if memory_format == torch.contiguous_format or moves:
                # _to_copy materializes densely across devices; a same-device
                # cast keeps the requested format.
                if memory_format == torch.contiguous_format or strides == dense_strides(shape):
                    strides = dense_strides(shape)
                    offset = Const(0)
        elif kind == 'dequantize':
            op = _dequantize(node, value, current_dtype, shape, context, parameters)
            operations.append(op)
            current_dtype = op.dtype
        elif kind == 'quantize':
            raise _Reject('quantize_semantics_unproved')
        elif kind in {'permute', 'transpose'}:
            _require(compat.dtype_name(value.dtype) == current_dtype, 'typed_transform_unavailable')
            rank = len(shape)
            if kind == 'permute':
                perm = tuple(opts['dims'])
            else:
                d0, d1 = opts['dim0'], opts['dim1']
                _require(type(d0) is int and type(d1) is int and -rank <= d0 < rank and -rank <= d1 < rank, 'unsupported_permutation')
                perm = list(range(rank))
                perm[d0 % rank], perm[d1 % rank] = perm[d1 % rank], perm[d0 % rank]
            _require(all(type(d) is int and -rank <= d < rank for d in perm), 'unsupported_permutation')
            perm = tuple(d % rank for d in perm)
            _require(sorted(perm) == list(range(rank)), 'unsupported_permutation')
            op = Transpose(perm)
            operations.append(op)
            shape = operation_shape(shape, op)
            strides = tuple(strides[d] for d in perm) if strides is not None else None
        elif kind == 'reshape':
            _require(compat.dtype_name(value.dtype) == current_dtype, 'typed_transform_unavailable')
            target = opts.get('shape', opts.get('size'))
            op = Reshape(tuple(_constant_fold(d) for d in infer_reshape(shape, tuple(context.expression(d) for d in target))))
            operations.append(op)
            was_dense = strides == dense_strides(shape)
            shape = op.shape
            if was_dense:
                strides = dense_strides(shape)
            else:
                # Non-dense reshape can alias or allocate. Keep the actual
                # descriptor unless a later explicit materialization proves it.
                try:
                    actual = context.tensor_spec(value, require_dense=False)
                    strides, offset = actual.strides, actual.offset
                except UnsupportedSymbolicExpr:
                    strides = None
        elif kind == 'pad':
            _require(compat.dtype_name(value.dtype) == current_dtype, 'typed_transform_unavailable')
            widths = opts['pad']
            _require(len(widths) % 2 == 0 and len(widths) <= 2 * len(shape) and all(type(w) is int and w >= 0 for w in widths), 'unsupported_padding')
            fill = _fill(current_dtype, opts.get('value', 0))
            for i in range(len(widths) // 2):
                op = Pad(len(shape) - 1 - i, Const(widths[2*i]), Const(widths[2*i+1]), fill)
                operations.append(op)
                shape = operation_shape(shape, op)
            strides, offset = dense_strides(shape), Const(0)
        elif kind == 'materialize':
            _require(compat.dtype_name(value.dtype) == current_dtype, 'typed_transform_unavailable')
            _require(opts.get('memory_format') in (None, torch.contiguous_format), 'unsupported_memory_format')
            # clone(None) preserves format, unlike contiguous's default.
            if node.target == torch.ops.aten.contiguous.default:
                if strides != dense_strides(shape):
                    before = compat.graph_value(_tensor_input(node))
                    unconditional = (compat.is_tensor(before) and len(before.shape) == len(shape)
                                     and all(compat.statically_at_least(d, 2) for d in before.shape))
                    if unconditional:
                        # Every extent is provably >= 2, so a non-dense layout
                        # is never contiguous and contiguous() materializes
                        # densely. The >= 2 family becomes a runtime guard.
                        extent_guards.extend(d for d in shape if not isinstance(expression(d), Const))
                    else:
                        _require(not context.sources, 'conditional_materialization')
                actual = context.tensor_spec(value, require_dense=False, positive_shape=shape)
                strides, offset = actual.strides, actual.offset
            elif opts.get('memory_format') == torch.contiguous_format:
                strides, offset = dense_strides(shape), Const(0)
    _require(direction is not None, 'same_device_transfer')
    _require(offset == Const(0) and strides == dense_strides(shape), 'destination_layout')
    # Shape/stride provenance follows exact operator arguments. FakeTensor may
    # rewrite a positive extent N into Max(1,N) in dense strides; positivity is
    # an explicit bind_recipe guard, not a shape hint read here.
    actual = context.tensor_spec(compat.graph_value(members[-1]), require_dense=False, positive_shape=shape)
    _require(actual.shape == shape, 'unsupported_symbolic_expr')
    _require(actual.strides == strides and actual.offset == offset, 'destination_layout')
    _require(actual.dtype == current_dtype, 'typed_transform_unavailable')
    destination = TensorSpec(shape, strides, offset, current_dtype)
    recipe = Recipe(src, tuple(operations), destination, direction)
    ordered = tuple(parameters[binding.name] for binding in recipe.parameter_bindings)
    return recipe, context, tuple(dict.fromkeys(extent_guards)), ordered


def _extract(gm, root, members, context, parameters):
    """node_copy retains exact raw call targets, options and exception behavior.

    The extracted callable takes the source, then every scalar placeholder in
    graph order, then every runtime parameter in recipe declaration order.
    """
    from torch.fx import Graph, GraphModule
    graph = Graph()
    mapping = {root: graph.placeholder('src')}
    parameter_set = set(parameters)
    external, dependencies = [], set()
    def visit(node):
        if node in mapping or node in dependencies or node in parameter_set:
            return
        if node.op == 'placeholder':
            value = compat.graph_value(node)
            _require(not compat.is_tensor(value), 'unsupported_symbolic_expr')
            expression = context.expression(value)
            external.append((node, expression))
            return
        _require(node in members or compat.fx_kind(node) == 'scalar', 'unsupported_symbolic_expr')
        for arg in node.all_input_nodes:
            visit(arg)
        dependencies.add(node)
    for member in members:
        visit(member)
    # Graph order is the stable calling convention for scalar placeholders.
    bindings = []
    for node in gm.graph.nodes:
        found = next((expr for n, expr in external if n is node), None)
        if found is not None:
            mapping[node] = graph.placeholder(node.name)
            bindings.append((node.name, found))
    for node in parameters:
        mapping[node] = graph.placeholder(node.name)
    for node in gm.graph.nodes:
        if node in dependencies:
            mapping[node] = graph.node_copy(node, lambda n: mapping[n])
    graph.output(mapping[members[-1]])
    return GraphModule(gm, graph), tuple(bindings)


def import_graph(gm, example_inputs=None):
    """Discover safe pure regions without changing gm; compilation is still required."""
    normalized = normalize_graph(gm, example_inputs)
    nodes = list(normalized.graph.nodes)
    originals = {n.name: n for n in gm.graph.nodes}
    exclusions = [Exclusion(n.name, n.meta['reloc_reason']) for n in nodes if 'reloc_reason' in n.meta]
    candidates, seen = [], set()
    for transfer in nodes:
        if not _transfer(transfer) or transfer in seen:
            continue
        # Count across the entire layout-connected component, including forks.
        # A greedy linear walk alone can accidentally accept the first transfer
        # in a branched multi-transfer chain.
        component, pending = set(), [transfer]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            parent = _tensor_input(current)
            adjacent = [u for u in current.users if _tensor_input(u) is current]
            if parent is not None:
                adjacent.append(parent)
            pending.extend(n for n in adjacent if n not in component and (_layout(n) or _transfer(n) or _typed(n)))
        transfers = [n for n in nodes if n in component and _moves_device(n)]
        if len(transfers) > 1:
            exclusions.extend(Exclusion(n.name, 'multiple_transfers') for n in transfers)
            seen.update(n for n in component if _transfer(n))
            continue
        members = [transfer]
        root = _tensor_input(transfer)
        while root is not None and (_layout(root) or _transfer(root) or _typed(root)):
            members.insert(0, root)
            root = _tensor_input(root)
        while True:
            following = [u for u in members[-1].users if (_layout(u) or _transfer(u) or _typed(u)) and _tensor_input(u) is members[-1]]
            if len(following) != 1:
                break
            members.append(following[0])
        seen.update(n for n in members if _transfer(n))
        try:
            _require(root is not None, 'metadata_unavailable')
            _safety(nodes, root, members)
            recipe, context, extent_guards, parameters = _recipe(root, members)
            original, bindings = _extract(gm, originals[root.name], [originals[n.name] for n in members], context,
                                          [originals[p.name] for p in parameters])
            candidates.append(Candidate(root.name, members[-1].name, tuple(n.name for n in members), recipe, context.sources, bindings, original, extent_guards,
                                        compat.graph_value(members[-1]).device, tuple(p.name for p in parameters)))
        except (_Reject, UnsupportedSymbolicExpr) as error:
            exclusions.append(Exclusion(transfer.name, error.reason))
    return ImportReport(tuple(candidates), tuple(dict.fromkeys(exclusions)))

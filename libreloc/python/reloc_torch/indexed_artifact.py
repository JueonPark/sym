"""Admission and two-operand metadata binding for compiler wire-v2 artifacts."""
import dataclasses
import hashlib

from .artifact import (CompiledRecipe, _exact_keys, _integer, _parse_constraints,
                       _parse_descriptor, _parse_expr, _same_descriptor,
                       _validate_base, _wire_symbols)
from .recipe import Cast, IndexSelect, TensorSpec
from .symbolic import (Const, GuardError, bind_recipe, dense_strides, expression,
                       symbol_sources)


def sources_for(recipe):
    sources = list(symbol_sources(recipe.source.shape))
    known = {s.name for s in sources}
    for source in symbol_sources(recipe.operations[0].indices.shape):
        if source.name not in known:
            sources.append(dataclasses.replace(source, operand='indices'))
    return tuple(sources)


def bind_values(compiled, source, index):
    """Bind already-normalized immutable TensorSpec descriptors."""
    declared = compiled.recipe.operations[0].indices
    if len(index.shape) != 1 or index.dtype != declared.dtype:
        raise GuardError('index_descriptor')
    bindings = {}
    for provenance in compiled.symbol_sources:
        shape = index.shape if provenance.operand == 'indices' else source.shape
        if provenance.axis >= len(shape):
            raise GuardError('source_descriptor')
        bindings[provenance.name] = shape[provenance.axis]
    # The original source, selected output and byte footprints are all guarded.
    bindings = bind_recipe(compiled.recipe,
        tuple(s for s in compiled.symbol_sources if s.operand == 'source'), source, bindings=bindings)
    if tuple(expression(d).evaluate(bindings, checked=True) for d in declared.shape) != index.shape:
        raise GuardError('index_descriptor')
    if index.shape[0] * (4 if index.dtype == 'int32' else 8) >= 2**63:
        raise GuardError('integer_overflow')
    return {name: bindings[name] for name in compiled.symbols}


def admit(recipe, mlir, plan, manifest):
    import pyreloc

    _exact_keys(manifest, ('schema_version', 'wire_version', 'status', 'compiler', 'plan_count',
        'symbols', 'logical_source', 'logical_destination', 'constraints', 'index_select',
        'plan_sha256', 'input_sha256'), 'indexed manifest')
    compiler = _validate_base(manifest, (3, 2))
    if manifest['status'] != 'ok' or _integer(manifest['plan_count'], 'plan_count') != 1:
        raise RuntimeError('indexed manifest must describe exactly one plan')
    if (not isinstance(recipe.operations[0], IndexSelect) or len(recipe.operations) > 2
            or any(not isinstance(op, Cast) for op in recipe.operations[1:])):
        raise RuntimeError('unsupported indexed recipe chain')
    select = recipe.operations[0]
    entry = manifest['index_select']
    _exact_keys(entry, ('axis', 'indices', 'policy'), 'index_select')
    policy = recipe.operations[1].policy if len(recipe.operations) == 2 else 'exact'
    if _integer(entry['axis'], 'index_select axis') != select.axis or entry['policy'] != policy:
        raise RuntimeError('indexed manifest selection/policy mismatch')
    declared = (
        _parse_descriptor(manifest['logical_source'], 'logical_source'),
        _parse_descriptor(entry['indices'], 'indices', index=True),
        _parse_descriptor(manifest['logical_destination'], 'logical_destination'),
    )
    if any(not _same_descriptor(a, b) for a, b in zip(declared, (recipe.source, select.indices, recipe.destination))):
        raise RuntimeError('indexed manifest descriptor mismatch')
    if _parse_constraints(manifest['constraints']):
        raise RuntimeError('unexpected indexed constraints')
    if (manifest['plan_sha256'] != hashlib.sha256(plan).hexdigest()
            or manifest['input_sha256'] != hashlib.sha256(mlir).hexdigest()):
        raise RuntimeError('indexed artifact digest mismatch')
    decoded = pyreloc.load_indexed_plan(plan)
    symbols = manifest['symbols']
    if (type(symbols) is not list or any(type(s) is not str for s in symbols)
            or tuple(symbols) != _wire_symbols(plan, 2) or symbols != list(decoded.symbols)):
        raise RuntimeError('indexed manifest symbol table mismatch')
    sources = sources_for(recipe)
    if len(set(symbols)) != len(symbols) or set(symbols) != {s.name for s in sources}:
        raise RuntimeError('indexed recipe symbol set mismatch')
    metadata = decoded.metadata
    if metadata['axis'] != entry['axis'] or metadata['policy'] != entry['policy']:
        raise RuntimeError('indexed wire selection/policy mismatch')
    for key, expected in zip(('source', 'indices', 'result'), declared):
        raw = metadata[key]
        shape = tuple(_parse_expr(dim, key) for dim in raw['shape'])
        descriptor = TensorSpec(shape, dense_strides(shape), Const(0), raw['dtype'])
        if not _same_descriptor(descriptor, expected):
            raise RuntimeError(f'indexed wire {key} descriptor mismatch')
    by_name = {s.name: s for s in sources}
    return CompiledRecipe(recipe, plan, compiler, 2, tuple(symbols),
        tuple(by_name[s] for s in symbols), (), declared[0], declared[2], manifest)

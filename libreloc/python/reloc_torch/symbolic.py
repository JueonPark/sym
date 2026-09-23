"""Torch-free integer expressions, source provenance, and binding guards."""
from dataclasses import dataclass
from functools import reduce


class UnsupportedSymbolicExpr(ValueError):
    reason = 'unsupported_symbolic_expr'

    def __init__(self, detail):
        super().__init__(f'{self.reason}: {detail}')


class GuardError(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _i64(value):
    if type(value) is not int or not -(1 << 63) <= value < (1 << 63):
        raise GuardError('integer_overflow')
    return value


@dataclass(frozen=True)
class Expr:
    def evaluate(self, bindings, *, checked=False):
        def ev(e):
            match e:
                case Const(value):
                    result = value
                case Symbol(name):
                    result = bindings[name]
                    if type(result) is not int:
                        raise GuardError('noninteger_binding')
                case Add(lhs, rhs):
                    result = ev(lhs) + ev(rhs)
                case Mul(lhs, rhs):
                    result = ev(lhs) * ev(rhs)
                case FloorDiv(lhs, divisor):
                    result = ev(lhs) // (_i64(divisor) if checked else divisor)
                case Mod(lhs, divisor):
                    result = ev(lhs) % (_i64(divisor) if checked else divisor)
                case _:
                    raise UnsupportedSymbolicExpr('unknown expression')
            return _i64(result) if checked else result
        return ev(self)


@dataclass(frozen=True)
class Const(Expr):
    value: int

    def __post_init__(self):
        if type(self.value) is not int:
            raise UnsupportedSymbolicExpr('constant must be an integer')


@dataclass(frozen=True)
class Symbol(Expr):
    name: str

    def __post_init__(self):
        if not self.name.isidentifier() or not self.name.isascii():
            raise UnsupportedSymbolicExpr('invalid symbol name')


@dataclass(frozen=True)
class Add(Expr):
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class Mul(Expr):
    lhs: Expr
    rhs: Expr


@dataclass(frozen=True)
class FloorDiv(Expr):
    lhs: Expr
    divisor: int

    def __post_init__(self):
        if type(self.divisor) is not int or self.divisor <= 0:
            raise UnsupportedSymbolicExpr('divisor must be a positive constant')


@dataclass(frozen=True)
class Mod(Expr):
    lhs: Expr
    divisor: int

    def __post_init__(self):
        if type(self.divisor) is not int or self.divisor <= 0:
            raise UnsupportedSymbolicExpr('divisor must be a positive constant')


@dataclass(frozen=True)
class SymbolSource:
    name: str
    axis: int
    equal_axes: tuple[int, ...] = ()


def expression(value):
    if isinstance(value, Expr):
        return value
    if type(value) is int:
        return Const(value)
    raise UnsupportedSymbolicExpr('expected an expression or integer')


def symbol_sources(shape):
    """Recover portable provenance from atomic recipe source dimensions."""
    sources, indices = [], {}
    for axis, value in enumerate(shape):
        match expression(value):
            case Const(_):
                continue
            case Symbol(name):
                if name in indices:
                    index = indices[name]
                    old = sources[index]
                    sources[index] = SymbolSource(name, old.axis, old.equal_axes + (axis,))
                else:
                    indices[name] = len(sources)
                    sources.append(SymbolSource(name, axis))
            case _:
                raise UnsupportedSymbolicExpr('source dimension is not an independent symbol')
    return tuple(sources)


def _commutative(kind, lhs, rhs):
    # Flatten and order by canonical frontend names, never capture symbol names.
    terms = []
    constant = 0 if kind is Add else 1

    def collect(value):
        nonlocal constant
        value = expression(value)
        if isinstance(value, kind):
            collect(value.lhs)
            collect(value.rhs)
        elif isinstance(value, Const):
            constant = constant + value.value if kind is Add else constant * value.value
        else:
            terms.append(value)

    collect(lhs)
    collect(rhs)
    if kind is Mul and constant == 0:
        return Const(0)
    identity = 0 if kind is Add else 1
    if constant != identity or not terms:
        terms.append(Const(constant))
    terms.sort(key=repr)
    return reduce(kind, terms)


def add(lhs, rhs):
    return _commutative(Add, lhs, rhs)


def mul(lhs, rhs):
    return _commutative(Mul, lhs, rhs)


def product(shape):
    return reduce(mul, shape, Const(1))


def dense_strides(shape):
    strides, running = [], Const(1)
    for dim in reversed(shape):
        strides.append(running)
        running = mul(dim, running)
    return tuple(reversed(strides))


def infer_reshape(source_shape, target_shape):
    shape = tuple(expression(d) for d in target_shape)
    inferred = [i for i, d in enumerate(shape) if d == Const(-1)]
    if len(inferred) > 1 or any(isinstance(d, Const) and d.value <= 0 and d != Const(-1) for d in shape):
        raise UnsupportedSymbolicExpr('invalid reshape dimensions')
    if not shape:
        raise UnsupportedSymbolicExpr('rank zero reshape')
    if not inferred:
        return shape
    rest = tuple(d for d in shape if d != Const(-1))
    if not all(isinstance(d, Const) and d.value > 0 for d in rest):
        raise UnsupportedSymbolicExpr('dynamic split factor')
    divisor = product(rest).value
    count = product(source_shape)
    quotient = count if divisor == 1 else FloorDiv(count, divisor)
    return tuple(quotient if d == Const(-1) else d for d in shape)


def operation_shape(shape, operation):
    from .recipe import TYPED_OPERATIONS, Pad, Reshape, Transpose
    if isinstance(operation, TYPED_OPERATIONS):
        # Value transforms are element-wise: the logical shape is unchanged.
        return tuple(shape)
    match operation:
        case Transpose(perm):
            if sorted(perm) != list(range(len(shape))):
                raise UnsupportedSymbolicExpr('invalid permutation')
            return tuple(shape[i] for i in perm)
        case Reshape(target):
            return tuple(expression(d) for d in target)
        case Pad(axis, lo, hi, _):
            if not 0 <= axis < len(shape):
                raise UnsupportedSymbolicExpr('invalid padding axis')
            result = list(shape)
            result[axis] = add(add(shape[axis], lo), hi)
            return tuple(result)
    raise UnsupportedSymbolicExpr('unknown operation')


def bind_recipe(recipe, sources, concrete_source):
    """Validate before a standalone binder call; return canonical symbol values.

    Python integers provide exact guard arithmetic; every intermediate and byte
    footprint also fits signed i64. No alignment guard is imposed here: alignment
    is a runtime strategy choice. This function does not execute or bind a plan.
    """
    from .recipe import TYPED_OPERATIONS, Pad, Reshape
    shape = concrete_source.shape
    if not shape or any(type(d) is not int or d <= 0 for d in shape):
        raise GuardError('positive_extent')
    for d in (*shape, *concrete_source.strides, concrete_source.offset):
        _i64(d)
    bindings = {}
    for source in sources:
        if not 0 <= source.axis < len(shape) or any(not 0 <= a < len(shape) for a in source.equal_axes):
            raise GuardError('source_descriptor')
        value = shape[source.axis]
        if any(shape[a] != value for a in source.equal_axes):
            raise GuardError('repeated_symbol')
        if source.name in bindings and bindings[source.name] != value:
            raise GuardError('repeated_symbol')
        bindings[source.name] = value

    def ev(e):
        try:
            return expression(e).evaluate(bindings, checked=True)
        except KeyError as exc:
            raise GuardError('missing_symbol') from exc

    def dims(values):
        result = tuple(ev(d) for d in values)
        if not result or any(d <= 0 for d in result):
            raise GuardError('positive_extent')
        return result

    def footprint(values, dtype):
        size = {'float32': 4, 'float16': 2, 'int8': 1}.get(dtype)
        if size is None:
            raise GuardError('unsupported_dtype')
        return _i64(ev(product(values)) * size)

    src = recipe.source
    if (dims(src.shape) != shape or tuple(ev(d) for d in src.strides) != concrete_source.strides
            or ev(src.offset) != concrete_source.offset or src.dtype != concrete_source.dtype
            or ev(src.offset) != 0 or tuple(ev(d) for d in dense_strides(src.shape)) != concrete_source.strides):
        raise GuardError('source_descriptor')
    footprint(src.shape, src.dtype)
    current = src.shape
    dtype = src.dtype

    def divisibility(e):
        match expression(e):
            case FloorDiv(lhs, divisor):
                if ev(lhs) % divisor:
                    raise GuardError('divisibility')
                divisibility(lhs)
            case Add(lhs, rhs) | Mul(lhs, rhs):
                divisibility(lhs)
                divisibility(rhs)
            case Mod(lhs, _):
                divisibility(lhs)

    for op in recipe.operations:
        target = operation_shape(current, op)
        if isinstance(op, Reshape):
            for d in target:
                divisibility(d)
            if ev(product(current)) != ev(product(target)):
                raise GuardError('element_count')
        if isinstance(op, Pad):
            if ev(op.lo) < 0 or ev(op.hi) < 0 or op.fill.dtype != dtype:
                raise GuardError('padding')
        if isinstance(op, TYPED_OPERATIONS):
            # Typed stages change the dtype (and so the footprint) of every
            # later boundary; a per-channel axis must exist on its operand.
            axis = getattr(op, 'axis', None)
            if axis is not None and not 0 <= axis < len(current):
                raise GuardError('channel_axis')
            dtype = op.dtype
        dims(target)
        footprint(target, dtype)
        current = target
    dst = recipe.destination
    if (dims(current) != dims(dst.shape) or dtype != dst.dtype or ev(dst.offset) != 0
            or tuple(ev(d) for d in dst.strides) != tuple(ev(d) for d in dense_strides(dst.shape))):
        raise GuardError('destination_descriptor')
    footprint(dst.shape, dst.dtype)
    return bindings

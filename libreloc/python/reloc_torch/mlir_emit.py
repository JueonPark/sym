"""Emit ordinary reloc IR; folding and plan serialization belong to the compiler."""
from .recipe import (TYPED_OPERATIONS, BindingParam, Cast, Dequantize, Pad, Quantize, Reshape,
                     Transpose)
from .symbolic import (Add, Const, FloorDiv, Mod, Mul, Symbol,
                       UnsupportedSymbolicExpr, dense_strides, expression, operation_shape)

_DTYPES = {'float32': 'f32', 'float16': 'f16', 'int8': 'i8'}
# Stage parameters may also be i32 (zero points); tensors may not.
_PARAM_DTYPES = {**_DTYPES, 'int32': 'i32'}
_WIDTHS = {'f32': 32, 'f16': 16, 'i8': 8, 'i32': 32}


def _expr(value, *, quoted=False):
    def emit(e):
        match expression(e):
            case Const(value):
                return str(value)
            case Symbol(name):
                return f'"{name}"' if quoted else name
            case Add(lhs, rhs):
                return f'({emit(lhs)} + {emit(rhs)})'
            case Mul(lhs, rhs):
                return f'({emit(lhs)} * {emit(rhs)})'
            case FloorDiv(lhs, divisor):
                return f'({emit(lhs)} floordiv {divisor})'
            case Mod(lhs, divisor):
                return f'({emit(lhs)} mod {divisor})'
        raise UnsupportedSymbolicExpr('unknown expression')
    return emit(value)


def _shape(shape, *, quoted=False):
    return '[' + ', '.join(_expr(d, quoted=quoted) for d in shape) + ']'


def _type(shape, dtype):
    return f'!sym.tensor<{_shape(shape, quoted=True)}, {dtype}>'


def _literal(bits, dtype):
    """Exact bits: hex literals for floats, two's complement decimal for ints."""
    width = _WIDTHS[dtype]
    if dtype.startswith('i'):
        return str(bits if bits < 1 << (width - 1) else bits - (1 << width))
    return f'0x{bits:0{width // 4}X}'


def _fill(fill):
    dtype = _DTYPES[fill.dtype]
    return f'({_literal(fill.bits, dtype)} : {dtype})'


def _param(param):
    dtype = _PARAM_DTYPES[param.dtype]
    if isinstance(param, BindingParam):
        return f'#reloc.binding<"{param.name}" : {_shape(param.extents)}, {dtype}>'
    values = [_literal(bits, dtype) for bits in param.bits]
    if param.shape:
        return f'dense<[{", ".join(values)}]> : tensor<{param.shape[0]}x{dtype}>'
    return f'dense<{values[0]}> : tensor<{dtype}>'


def _quant(name, op, value):
    text = f'reloc.{name} {value}'
    if op.axis is not None:
        text += f' axis {op.axis}'
    text += f' scale({_param(op.scale)})'
    if op.zero_point is not None:
        text += f' zero_point({_param(op.zero_point)})'
    return text + f' policy {op.policy}'


def emit_mlir(recipe):
    """Emit one function retaining the destination's original logical shape.

    Layout-only recipes keep one dtype throughout; a typed recipe's dtype
    changes exactly at its value transforms and must end at the destination's.
    """
    if recipe.source.dtype not in _DTYPES or recipe.destination.dtype not in _DTYPES:
        raise UnsupportedSymbolicExpr('unsupported dtype')
    if not recipe.typed and recipe.source.dtype != recipe.destination.dtype:
        raise UnsupportedSymbolicExpr('unsupported or changed dtype')
    for descriptor, name in ((recipe.source, 'source'), (recipe.destination, 'destination')):
        if (not descriptor.shape or expression(descriptor.offset) != Const(0)
                or tuple(expression(d) for d in descriptor.strides) != dense_strides(descriptor.shape)):
            raise UnsupportedSymbolicExpr(f'{name}_layout: expected dense zero-offset tensor')
    current = recipe.source.shape
    current_dtype = recipe.source.dtype
    source_type = _type(current, _DTYPES[current_dtype])
    result_type = _type(recipe.destination.shape, _DTYPES[recipe.destination.dtype])
    lines = [f'func.func @torch_relocation(%x: {source_type}) -> {result_type} {{']
    value = '%x'
    operations = recipe.operations or (Transpose(tuple(range(len(current)))),)
    for index, op in enumerate(operations):
        target = operation_shape(current, op)
        target_dtype = op.dtype if isinstance(op, TYPED_OPERATIONS) else current_dtype
        if target_dtype not in _DTYPES:
            raise UnsupportedSymbolicExpr('unsupported dtype')
        match op:
            case Transpose(perm):
                text = f'reloc.transpose {value} perm [{", ".join(map(str, perm))}]'
            case Reshape(_):
                text = f'reloc.reshape {value} to {_shape(target)}'
            case Pad(axis, lo, hi, fill):
                text = f'reloc.pad {value} axis {axis} lo {_expr(lo)} hi {_expr(hi)} value {_fill(fill)}'
            case Cast():
                text = f'reloc.cast {value} policy {op.policy}'
            case Quantize():
                text = _quant('quantize', op, value)
            case Dequantize():
                text = _quant('dequantize', op, value)
            case _:
                raise UnsupportedSymbolicExpr('unknown operation')
        next_value = f'%v{index}'
        lines.append(f'  {next_value} = {text} : {_type(current, _DTYPES[current_dtype])}'
                     f' -> {_type(target, _DTYPES[target_dtype])}')
        value, current, current_dtype = next_value, target, target_dtype
    if current_dtype != recipe.destination.dtype:
        raise UnsupportedSymbolicExpr('destination dtype does not match the chain')
    lines.append(f'  return {value} : {result_type}')
    lines.append('}')
    return '\n'.join(lines) + '\n'

"""Emit ordinary reloc IR; folding and plan serialization belong to the compiler."""
from .recipe import Pad, Reshape, Transpose
from .symbolic import (Add, Const, FloorDiv, Mod, Mul, Symbol,
                       UnsupportedSymbolicExpr, dense_strides, expression, operation_shape)

_DTYPES = {'float32': 'f32', 'float16': 'f16', 'int8': 'i8'}


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


def _fill(fill):
    dtype = _DTYPES[fill.dtype]
    if dtype == 'i8':
        value = str(fill.bits if fill.bits < 128 else fill.bits - 256)
    else:
        value = f'0x{fill.bits:0{8 if dtype == "f32" else 4}X}'
    return f'({value} : {dtype})'


def emit_mlir(recipe):
    """Emit one function retaining the destination's original logical shape."""
    if recipe.source.dtype not in _DTYPES or recipe.source.dtype != recipe.destination.dtype:
        raise UnsupportedSymbolicExpr('unsupported or changed dtype')
    for descriptor, name in ((recipe.source, 'source'), (recipe.destination, 'destination')):
        if (not descriptor.shape or expression(descriptor.offset) != Const(0)
                or tuple(expression(d) for d in descriptor.strides) != dense_strides(descriptor.shape)):
            raise UnsupportedSymbolicExpr(f'{name}_layout: expected dense zero-offset tensor')
    dtype = _DTYPES[recipe.source.dtype]
    current = recipe.source.shape
    source_type = _type(current, dtype)
    result_type = _type(recipe.destination.shape, dtype)
    lines = [f'func.func @torch_relocation(%x: {source_type}) -> {result_type} {{']
    value = '%x'
    operations = recipe.operations or (Transpose(tuple(range(len(current)))),)
    for index, op in enumerate(operations):
        target = operation_shape(current, op)
        match op:
            case Transpose(perm):
                text = f'reloc.transpose {value} perm [{", ".join(map(str, perm))}]'
            case Reshape(_):
                text = f'reloc.reshape {value} to {_shape(target)}'
            case Pad(axis, lo, hi, fill):
                text = f'reloc.pad {value} axis {axis} lo {_expr(lo)} hi {_expr(hi)} value {_fill(fill)}'
            case _:
                raise UnsupportedSymbolicExpr('unknown operation')
        next_value = f'%v{index}'
        lines.append(f'  {next_value} = {text} : {_type(current, dtype)} -> {_type(target, dtype)}')
        value, current = next_value, target
    lines.append(f'  return {value} : {result_type}')
    lines.append('}')
    return '\n'.join(lines) + '\n'

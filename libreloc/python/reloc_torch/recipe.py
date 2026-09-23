"""Immutable relocation descriptions, independent of Torch and compiler plans.

Layout operations (transpose, reshape, pad) describe the R1 layout-only
artifact (wire v0, manifest schema 1). Typed value transforms (cast,
quantize, dequantize; C1 semantics in docs/reloc-typed-semantics.md) make a
recipe *typed*: it compiles through ``sym-reloc-export --typed`` into a wire
v1 typed plan with a schema-2 manifest (C3, issue #143). Stage parameters
are exact bit patterns (``InlineParam``) or runtime bindings declared by
name, dtype and extents (``BindingParam``); a recipe never holds a pointer.
"""
from dataclasses import dataclass
from .symbolic import Expr

_TENSOR_DTYPES = {'float32': 32, 'float16': 16, 'int8': 8}
_PARAM_DTYPES = {**_TENSOR_DTYPES, 'int32': 32}
_CAST_POLICIES = ('ieee_rne', 'exact')
_QUANT_POLICIES = ('symmetric_rne', 'affine')


@dataclass(frozen=True)
class TensorSpec:
    # Concrete descriptors passed to bind_recipe contain Python ints.
    shape: tuple[Expr | int, ...]
    strides: tuple[Expr | int, ...]
    offset: Expr | int
    dtype: str


@dataclass(frozen=True)
class Transpose:
    perm: tuple[int, ...]


@dataclass(frozen=True)
class Reshape:
    shape: tuple[Expr, ...]


@dataclass(frozen=True)
class Fill:
    dtype: str
    bits: int

    def __post_init__(self):
        width = _TENSOR_DTYPES.get(self.dtype)
        if width is None or type(self.bits) is not int or not 0 <= self.bits < 1 << width:
            raise ValueError('invalid fill dtype or bit pattern')


@dataclass(frozen=True)
class Pad:
    axis: int
    lo: Expr
    hi: Expr
    fill: Fill


@dataclass(frozen=True)
class InlineParam:
    """A constant stage parameter: exact bit patterns, per tensor (shape ``()``)
    or per channel (shape ``(channels,)``)."""
    dtype: str
    shape: tuple[int, ...]
    bits: tuple[int, ...]

    def __post_init__(self):
        width = _PARAM_DTYPES.get(self.dtype)
        if (width is None or type(self.shape) is not tuple or len(self.shape) > 1
                or any(type(d) is not int or d <= 0 for d in self.shape)):
            raise ValueError('invalid parameter dtype or shape')
        count = 1
        for d in self.shape:
            count *= d
        if (type(self.bits) is not tuple or len(self.bits) != count
                or any(type(b) is not int or not 0 <= b < 1 << width for b in self.bits)):
            raise ValueError('invalid parameter bit patterns')


@dataclass(frozen=True)
class BindingParam:
    """A runtime stage parameter, supplied by name at bind time with this dtype
    and these (symbolic) extents; rank 0 per tensor, rank 1 per channel."""
    name: str
    dtype: str
    extents: tuple[Expr, ...]

    def __post_init__(self):
        if (type(self.name) is not str or not self.name or self.dtype not in _PARAM_DTYPES
                or type(self.extents) is not tuple or len(self.extents) > 1):
            raise ValueError('invalid parameter binding')


@dataclass(frozen=True)
class Cast:
    """Element-wise conversion to ``dtype`` under a C1 policy."""
    dtype: str
    policy: str

    def __post_init__(self):
        if self.dtype not in _TENSOR_DTYPES or self.policy not in _CAST_POLICIES:
            raise ValueError('invalid cast dtype or policy')


def _check_quant(op):
    if op.dtype not in _TENSOR_DTYPES or op.policy not in _QUANT_POLICIES:
        raise ValueError('invalid quantization dtype or policy')
    if not isinstance(op.scale, (InlineParam, BindingParam)):
        raise ValueError('quantization scale must be an InlineParam or BindingParam')
    if op.zero_point is not None and not isinstance(op.zero_point, (InlineParam, BindingParam)):
        raise ValueError('quantization zero point must be an InlineParam, BindingParam or None')
    if op.axis is not None and (type(op.axis) is not int or op.axis < 0):
        raise ValueError('quantization axis must be None (per tensor) or a non-negative int')


@dataclass(frozen=True)
class Quantize:
    """Real to integer under a C1 policy; ``axis`` None means per tensor."""
    dtype: str
    scale: InlineParam | BindingParam
    zero_point: InlineParam | BindingParam | None
    axis: int | None
    policy: str

    def __post_init__(self):
        _check_quant(self)


@dataclass(frozen=True)
class Dequantize:
    """Integer to real under a C1 policy; ``axis`` None means per tensor."""
    dtype: str
    scale: InlineParam | BindingParam
    zero_point: InlineParam | BindingParam | None
    axis: int | None
    policy: str

    def __post_init__(self):
        _check_quant(self)


TYPED_OPERATIONS = (Cast, Quantize, Dequantize)
LAYOUT_OPERATIONS = (Transpose, Reshape, Pad)


@dataclass(frozen=True)
class Recipe:
    source: TensorSpec
    operations: tuple[Transpose | Reshape | Pad | Cast | Quantize | Dequantize, ...]
    destination: TensorSpec
    direction: str

    @property
    def canonical_identity(self):
        """Hashable identity; construct expressions with canonical source symbols."""
        return self.source, self.operations, self.destination, self.direction

    @property
    def typed(self):
        """True when the chain holds a value transform: the artifact is a
        wire v1 typed plan (schema 2) instead of a layout-only v0 plan."""
        return any(isinstance(op, TYPED_OPERATIONS) for op in self.operations)

    @property
    def parameter_bindings(self):
        """Runtime parameters in first-declaration order, one per name."""
        seen = []
        for op in self.operations:
            for param in (getattr(op, 'scale', None), getattr(op, 'zero_point', None)):
                if isinstance(param, BindingParam) and all(p.name != param.name for p in seen):
                    seen.append(param)
        return tuple(seen)

"""Immutable relocation descriptions, independent of Torch and compiler plans."""
from dataclasses import dataclass
from .symbolic import Expr


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
        width = {'float32': 32, 'float16': 16, 'int8': 8}.get(self.dtype)
        if width is None or type(self.bits) is not int or not 0 <= self.bits < 1 << width:
            raise ValueError('invalid fill dtype or bit pattern')


@dataclass(frozen=True)
class Pad:
    axis: int
    lo: Expr
    hi: Expr
    fill: Fill


@dataclass(frozen=True)
class Recipe:
    source: TensorSpec
    operations: tuple[Transpose | Reshape | Pad, ...]
    destination: TensorSpec
    direction: str

    @property
    def canonical_identity(self):
        """Hashable identity; construct expressions with canonical source symbols."""
        return self.source, self.operations, self.destination, self.direction

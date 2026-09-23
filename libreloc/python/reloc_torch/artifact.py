"""Validated, portable frontend metadata for R1 (layout-only) and C3 (typed)
compiler artifacts.

Admission branches on the recipe: a layout-only recipe must come back as
manifest schema 1 over a wire v0 plan, a typed recipe (value transforms) as
schema 2 over a wire v1 typed plan. Both digests, versions, symbols, logical
descriptors and constraints are checked against the submitted recipe; for
typed artifacts the stages, fills and runtime parameter declarations are
checked against the recipe's value transforms and against the plan the
runtime decoded. The portable serialization carries `format_version` 1 for
layout-only and 2 for typed artifacts; a loader accepts both.
"""
from dataclasses import dataclass
import base64
import hashlib
import json
from pathlib import Path

from .mlir_emit import emit_mlir
from .recipe import (
    TYPED_OPERATIONS,
    BindingParam,
    Cast,
    Dequantize,
    InlineParam,
    Pad,
    Quantize,
    Recipe,
    Reshape,
    TensorSpec,
    Transpose,
)
from .symbolic import (
    Add,
    Const,
    Expr,
    FloorDiv,
    GuardError,
    Mod,
    Mul,
    Symbol,
    SymbolSource,
    add,
    bind_recipe,
    expression,
    mul,
    operation_shape,
    symbol_sources,
)

# (schema_version, wire_version) of the two artifact kinds.
LAYOUT_SCHEMA = (1, 0)
TYPED_SCHEMA = (2, 1)


class UnsupportedRecipe(Exception):
    """A valid recipe that the compiler explicitly declines to fold."""

    def __init__(self, reason, detail):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class CompilerIdentity:
    name: str
    interface_version: int
    llvm_version: str
    build_identity: str


@dataclass(frozen=True)
class DivisibilityConstraint:
    expr: Expr
    divisor: int


@dataclass(frozen=True)
class ParameterDeclaration:
    """A runtime stage parameter the typed plan declares by name: it must be
    supplied to ``pyreloc.bind_typed`` with this dtype and these extents."""
    name: str
    dtype: str
    extents: tuple[Expr, ...]


@dataclass(frozen=True)
class CompiledRecipe:
    recipe: Recipe
    plan_bytes: bytes
    compiler: CompilerIdentity
    wire_version: int
    symbols: tuple[str, ...]
    symbol_sources: tuple[SymbolSource, ...]
    constraints: tuple[DivisibilityConstraint, ...]
    logical_source: TensorSpec
    logical_destination: TensorSpec
    manifest: dict
    parameters: tuple[ParameterDeclaration, ...] = ()

    @property
    def typed(self):
        """True for a wire v1 typed plan (schema 2): load it with
        ``pyreloc.load_typed_plan`` and bind it with ``pyreloc.bind_typed``."""
        return self.wire_version == TYPED_SCHEMA[1]

    def bind_values(self, value):
        """Guard concrete source metadata and return wire-symbol bindings."""
        concrete = _tensor_spec(value)
        bindings = bind_recipe(self.recipe, self.symbol_sources, concrete)
        for constraint in self.constraints:
            try:
                dividend = constraint.expr.evaluate(bindings, checked=True)
            except KeyError as error:
                raise GuardError("missing_symbol") from error
            if dividend % constraint.divisor:
                raise GuardError("divisibility")
        return {name: bindings[name] for name in self.symbols}

    def parameter_extents(self, symbols):
        """Concrete ``{name: (dtype, extents)}`` of every declared runtime
        parameter under bound wire symbols (``bind_values``' result), in
        declaration order. The caller supplies the bytes; ``bind_typed``
        re-checks dtype, extents, byte size and values."""
        out = {}
        for declaration in self.parameters:
            try:
                extents = tuple(
                    extent.evaluate(symbols, checked=True) for extent in declaration.extents
                )
            except KeyError as error:
                raise GuardError("missing_symbol") from error
            out[declaration.name] = (declaration.dtype, extents)
        return out

    def to_bytes(self):
        """Serialize portable metadata and plan bytes without live FX/Torch state."""
        payload = {
            "format_version": 2 if self.typed else 1,
            "manifest": self.manifest,
            "plan_base64": base64.b64encode(self.plan_bytes).decode("ascii"),
            "recipe": _encode_recipe(self.recipe),
        }
        return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    @classmethod
    def from_bytes(cls, data):
        try:
            payload = json.loads(bytes(data))
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("malformed compiled recipe serialization") from error
        _exact_keys(payload, ("format_version", "manifest", "plan_base64", "recipe"), "artifact")
        version = _integer(payload["format_version"], "artifact format_version")
        if version not in (1, 2):
            raise RuntimeError("compiled recipe format version mismatch")
        if type(payload["plan_base64"]) is not str:
            raise RuntimeError("compiled recipe plan_base64 must be a string")
        try:
            plan = base64.b64decode(payload["plan_base64"], validate=True)
        except (ValueError, base64.binascii.Error) as error:
            raise RuntimeError("compiled recipe has invalid plan bytes") from error
        recipe = _decode_recipe(payload["recipe"])
        # Format 1 is the layout-only artifact, format 2 the typed one; a
        # recipe of the other kind under either version is not ours.
        if recipe.typed != (version == 2):
            raise RuntimeError("compiled recipe format version does not match its recipe")
        mlir = emit_mlir(recipe).encode("utf-8")
        return _admit(recipe, mlir, plan, payload["manifest"])

    def save(self, path):
        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def load(cls, path):
        try:
            data = Path(path).read_bytes()
        except OSError as error:
            raise RuntimeError("unable to read compiled recipe") from error
        return cls.from_bytes(data)


def _read_manifest(path):
    try:
        data = path.read_bytes()
    except OSError as error:
        raise RuntimeError("compiler did not publish a readable manifest") from error
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("compiler published malformed manifest JSON") from error
    if type(value) is not dict:
        raise RuntimeError("compiler manifest must be an object")
    return value


def _exact_keys(value, expected, where):
    if type(value) is not dict or set(value) != set(expected):
        raise RuntimeError(f"compiler manifest has invalid {where} fields")


def _integer(value, where):
    if type(value) is not int:
        raise RuntimeError(f"compiler manifest {where} must be an integer")
    return value


def _string(value, where):
    if type(value) is not str or not value:
        raise RuntimeError(f"compiler manifest {where} must be a nonempty string")
    return value


def _hex_bits(value, where):
    """Exact scalar bits travel as lowercase hex text, never decimal."""
    if (type(value) is not str or not value or len(value) > 16
            or any(c not in "0123456789abcdef" for c in value)):
        raise RuntimeError(f"compiler manifest {where} bits must be lowercase hex")
    return int(value, 16)


def _compiler_identity(value):
    _exact_keys(value, ("name", "interface_version", "llvm_version", "build_identity"), "compiler")
    identity = CompilerIdentity(
        _string(value["name"], "compiler.name"),
        _integer(value["interface_version"], "compiler.interface_version"),
        _string(value["llvm_version"], "compiler.llvm_version"),
        _string(value["build_identity"], "compiler.build_identity"),
    )
    if identity.name != "sym-reloc-export" or identity.interface_version != 1:
        raise RuntimeError("compiler manifest identifies an unsupported compiler interface")
    return identity


def _validate_base(manifest, expected=LAYOUT_SCHEMA):
    schema_version, wire_version = expected
    if _integer(manifest.get("schema_version"), "schema_version") != schema_version:
        raise RuntimeError("compiler manifest schema version mismatch")
    if _integer(manifest.get("wire_version"), "wire_version") != wire_version:
        raise RuntimeError("compiler manifest wire version mismatch")
    return _compiler_identity(manifest.get("compiler"))


def _validate_unsupported_manifest(manifest):
    # Unsupported manifests describe no plan and always use schema 1 / wire
    # 0, whichever interface (layout-only or --typed) produced them.
    _exact_keys(
        manifest,
        ("schema_version", "wire_version", "status", "compiler", "reason", "detail"),
        "unsupported",
    )
    _validate_base(manifest)
    if manifest["status"] != "unsupported":
        raise RuntimeError("compiler exit 2 did not publish unsupported status")
    _string(manifest["reason"], "reason")
    _string(manifest["detail"], "detail")


def _parse_expr(value, where):
    if type(value) is not list or not value or type(value[0]) is not str:
        raise RuntimeError(f"compiler manifest {where} has invalid expression")
    tag = value[0]
    try:
        if tag == "const" and len(value) == 2:
            return Const(_integer(value[1], where))
        if tag == "symbol" and len(value) == 2:
            return Symbol(_string(value[1], where))
        if tag in ("add", "mul") and len(value) == 3:
            lhs = _parse_expr(value[1], where)
            rhs = _parse_expr(value[2], where)
            return Add(lhs, rhs) if tag == "add" else Mul(lhs, rhs)
        if tag in ("floordiv", "mod") and len(value) == 3:
            lhs = _parse_expr(value[1], where)
            divisor = _integer(value[2], where)
            if divisor <= 0:
                raise RuntimeError(f"compiler manifest {where} has nonpositive divisor")
            return FloorDiv(lhs, divisor) if tag == "floordiv" else Mod(lhs, divisor)
    except ValueError as error:
        raise RuntimeError(f"compiler manifest {where} has invalid expression") from error
    raise RuntimeError(f"compiler manifest {where} has unsupported expression")


def _normalize_expr(value):
    def fold_constant(candidate):
        try:
            return Const(candidate.evaluate({}, checked=True))
        except KeyError:
            return candidate
        except GuardError:
            # Keep overflowing arithmetic structural so admission cannot
            # equate it with an unchecked host-language fold.
            return candidate

    match expression(value):
        case Add(lhs, rhs):
            lhs, rhs = _normalize_expr(lhs), _normalize_expr(rhs)
            candidate = fold_constant(Add(lhs, rhs))
            return candidate if isinstance(candidate, Const) else add(lhs, rhs)
        case Mul(lhs, rhs):
            lhs, rhs = _normalize_expr(lhs), _normalize_expr(rhs)
            candidate = fold_constant(Mul(lhs, rhs))
            return candidate if isinstance(candidate, Const) else mul(lhs, rhs)
        case FloorDiv(lhs, divisor):
            lhs = _normalize_expr(lhs)
            return lhs if divisor == 1 else fold_constant(FloorDiv(lhs, divisor))
        case Mod(lhs, divisor):
            lhs = _normalize_expr(lhs)
            return Const(0) if divisor == 1 else fold_constant(Mod(lhs, divisor))
        case atom:
            return atom


def _same_exprs(actual, expected):
    return len(actual) == len(expected) and all(
        _normalize_expr(a) == _normalize_expr(b) for a, b in zip(actual, expected)
    )


def _parse_descriptor(value, where):
    _exact_keys(value, ("shape", "strides", "offset", "dtype"), where)
    if type(value["shape"]) is not list or type(value["strides"]) is not list:
        raise RuntimeError(f"compiler manifest {where} shape/strides must be arrays")
    if not value["shape"] or len(value["shape"]) != len(value["strides"]):
        raise RuntimeError(f"compiler manifest {where} has invalid rank")
    dtype = _string(value["dtype"], f"{where}.dtype")
    if dtype not in ("float32", "float16", "int8"):
        raise RuntimeError(f"compiler manifest {where} has unsupported dtype")
    return TensorSpec(
        tuple(_parse_expr(item, f"{where}.shape") for item in value["shape"]),
        tuple(_parse_expr(item, f"{where}.strides") for item in value["strides"]),
        _parse_expr(value["offset"], f"{where}.offset"),
        dtype,
    )


def _same_descriptor(actual, expected):
    return (
        actual.dtype == expected.dtype
        and _same_exprs(actual.shape, expected.shape)
        and _same_exprs(actual.strides, expected.strides)
        and _normalize_expr(actual.offset) == _normalize_expr(expected.offset)
    )


def _recipe_symbols(recipe):
    names = set()

    def visit(value):
        match expression(value):
            case Symbol(name):
                names.add(name)
            case Add(lhs, rhs) | Mul(lhs, rhs):
                visit(lhs)
                visit(rhs)
            case FloorDiv(lhs, _) | Mod(lhs, _):
                visit(lhs)

    for descriptor in (recipe.source, recipe.destination):
        for value in (*descriptor.shape, *descriptor.strides, descriptor.offset):
            visit(value)
    for operation in recipe.operations:
        if isinstance(operation, Reshape):
            for value in operation.shape:
                visit(value)
        elif isinstance(operation, Pad):
            visit(operation.lo)
            visit(operation.hi)
        elif isinstance(operation, TYPED_OPERATIONS):
            for param in (getattr(operation, "scale", None), getattr(operation, "zero_point", None)):
                if isinstance(param, BindingParam):
                    for value in param.extents:
                        visit(value)
    return names


def _wire_symbols(plan, wire_version):
    if len(plan) < 12 or plan[:4] != b"RPLN":
        raise RuntimeError("compiler plan has invalid wire header")
    version = int.from_bytes(plan[4:8], "little")
    if version != wire_version:
        raise RuntimeError("compiler plan wire version mismatch")
    count = int.from_bytes(plan[8:12], "little")
    offset = 12
    names = []
    for _ in range(count):
        if offset + 4 > len(plan):
            raise RuntimeError("compiler plan has truncated symbol table")
        length = int.from_bytes(plan[offset:offset + 4], "little")
        offset += 4
        if offset + length > len(plan):
            raise RuntimeError("compiler plan has truncated symbol name")
        try:
            name = plan[offset:offset + length].decode("utf-8")
            Symbol(name)
        except (UnicodeDecodeError, ValueError) as error:
            raise RuntimeError("compiler plan has invalid symbol name") from error
        names.append(name)
        offset += length
    if len(set(names)) != len(names):
        raise RuntimeError("compiler plan has duplicate symbols")
    return tuple(names)


def _decode_plan(plan, typed):
    """The runtime is the trust boundary for the plan bytes: a layout-only
    recipe's plan must decode as v0, a typed recipe's as v1, never across."""
    try:
        import pyreloc
    except ImportError as error:
        raise RuntimeError("pyreloc runtime is required to validate compiler plans") from error
    loader = pyreloc.load_typed_plan if typed else pyreloc.load_plan
    try:
        return loader(plan)
    except pyreloc.DecodeError as error:
        raise RuntimeError(f"compiler published an invalid plan: {error}") from error


def _expected_constraints(recipe):
    constraints = []

    def visit(value):
        value = expression(value)
        match value:
            case FloorDiv(lhs, divisor):
                normalized = _normalize_expr(lhs)
                # Fully constant reshape arithmetic is settled by the
                # compiler and bind_recipe; divisor one is an identity and
                # therefore also has no runtime constraint.
                if divisor != 1 and not isinstance(normalized, Const):
                    item = DivisibilityConstraint(normalized, divisor)
                    if item not in constraints:
                        constraints.append(item)
                visit(lhs)
            case Add(lhs, rhs) | Mul(lhs, rhs):
                visit(lhs)
                visit(rhs)
            case Mod(lhs, _):
                visit(lhs)

    for operation in recipe.operations:
        if isinstance(operation, Reshape):
            for value in operation.shape:
                visit(value)
    return tuple(constraints)


def _parse_constraints(value):
    _exact_keys(value, ("divisibility",), "constraints")
    entries = value["divisibility"]
    if type(entries) is not list:
        raise RuntimeError("compiler manifest divisibility constraints must be an array")
    result = []
    for entry in entries:
        _exact_keys(entry, ("expr", "divisor"), "divisibility constraint")
        divisor = _integer(entry["divisor"], "constraint divisor")
        if divisor <= 0:
            raise RuntimeError("compiler manifest constraint divisor must be positive")
        result.append(DivisibilityConstraint(_normalize_expr(_parse_expr(entry["expr"], "constraint")), divisor))
    return tuple(result)


#===----------------------------------------------------------------------===#
# Schema 2 (typed plans): stages, fills, parameter declarations.
#===----------------------------------------------------------------------===#

_TRANSFORM_NAMES = {Cast: "cast", Quantize: "quantize", Dequantize: "dequantize"}
_STAGE_FIELDS = (
    "transform", "policy", "input_dtype", "output_dtype", "shape", "scale", "zero_point",
    "axis", "channel",
)


def _parse_param(value, where):
    if value is None:
        return None
    if type(value) is not dict or type(value.get("kind")) is not str:
        raise RuntimeError(f"compiler manifest {where} has invalid parameter")
    try:
        if value["kind"] == "inline":
            _exact_keys(value, ("kind", "dtype", "shape", "bits"), where)
            if type(value["shape"]) is not list or type(value["bits"]) is not list:
                raise RuntimeError(f"compiler manifest {where} shape/bits must be arrays")
            return InlineParam(
                _string(value["dtype"], where),
                tuple(_integer(item, where) for item in value["shape"]),
                tuple(_hex_bits(item, where) for item in value["bits"]),
            )
        if value["kind"] == "binding":
            _exact_keys(value, ("kind", "name", "dtype", "extents"), where)
            if type(value["extents"]) is not list:
                raise RuntimeError(f"compiler manifest {where} extents must be an array")
            return BindingParam(
                _string(value["name"], where),
                _string(value["dtype"], where),
                tuple(_parse_expr(item, where) for item in value["extents"]),
            )
    except ValueError as error:
        raise RuntimeError(f"compiler manifest {where} has invalid parameter") from error
    raise RuntimeError(f"compiler manifest {where} has unsupported parameter kind")


def _same_param(actual, expected):
    if actual is None or expected is None:
        return actual is None and expected is None
    if type(actual) is not type(expected) or actual.dtype != expected.dtype:
        return False
    if isinstance(actual, InlineParam):
        return actual.shape == expected.shape and actual.bits == expected.bits
    return actual.name == expected.name and _same_exprs(actual.extents, expected.extents)


def _check_channel_expr(value, dims, symbols, where):
    """Channel maps are affine over logical result coordinates and plan
    symbols; divisors may be symbolic, so this is its own vocabulary. The
    runtime evaluates it; the bridge only proves it is well formed."""
    if type(value) is not list or not value or type(value[0]) is not str:
        raise RuntimeError(f"compiler manifest {where} has invalid channel expression")
    tag = value[0]
    if tag == "const" and len(value) == 2:
        _integer(value[1], where)
    elif tag == "dim" and len(value) == 2:
        if not 0 <= _integer(value[1], where) < dims:
            raise RuntimeError(f"compiler manifest {where} channel dimension is out of range")
    elif tag == "symbol" and len(value) == 2:
        if _string(value[1], where) not in symbols:
            raise RuntimeError(f"compiler manifest {where} channel symbol is not a plan symbol")
    elif tag in ("add", "mul", "floordiv", "mod") and len(value) == 3:
        _check_channel_expr(value[1], dims, symbols, where)
        _check_channel_expr(value[2], dims, symbols, where)
    else:
        raise RuntimeError(f"compiler manifest {where} has unsupported channel expression")


def _check_stage(entry, decoded, op, shape, dtype, result_rank, symbols, index):
    where = f"stage {index}"
    _exact_keys(entry, _STAGE_FIELDS, where)
    transform = _string(entry["transform"], where)
    policy = _string(entry["policy"], where)
    input_dtype = _string(entry["input_dtype"], where)
    output_dtype = _string(entry["output_dtype"], where)
    if (transform != _TRANSFORM_NAMES[type(op)] or policy != op.policy
            or input_dtype != dtype or output_dtype != op.dtype):
        raise RuntimeError(f"compiler manifest {where} does not match the recipe transform")
    if type(entry["shape"]) is not list or not _same_exprs(
            tuple(_parse_expr(item, where) for item in entry["shape"]), shape):
        raise RuntimeError(f"compiler manifest {where} operand shape mismatch")
    axis = _integer(entry["axis"], where)
    expected_axis = -1 if isinstance(op, Cast) or op.axis is None else op.axis
    if axis != expected_axis:
        raise RuntimeError(f"compiler manifest {where} channel axis mismatch")
    if (not _same_param(_parse_param(entry["scale"], where), getattr(op, "scale", None))
            or not _same_param(_parse_param(entry["zero_point"], where),
                               getattr(op, "zero_point", None))):
        raise RuntimeError(f"compiler manifest {where} parameter mismatch")
    channel = entry["channel"]
    if expected_axis < 0:
        if channel is not None:
            raise RuntimeError(f"compiler manifest {where} has a channel map without an axis")
    else:
        _exact_keys(channel, ("dims", "expr"), f"{where} channel")
        if _integer(channel["dims"], where) != result_rank:
            raise RuntimeError(f"compiler manifest {where} channel rank is not the result rank")
        _check_channel_expr(channel["expr"], result_rank, symbols, where)
    # The plan the runtime decoded must say the same thing.
    plan_view = (decoded["transform"], decoded["policy"], decoded["input"], decoded["output"],
                 decoded["axis"], decoded["has_channel"])
    if plan_view != (transform, policy, input_dtype, output_dtype, expected_axis,
                     expected_axis >= 0):
        raise RuntimeError(f"compiler plan {where} does not match the manifest")


def _validate_typed_sections(recipe, manifest, symbols, decoded):
    stages, fills, parameters = manifest["stages"], manifest["fills"], manifest["parameters"]
    if type(stages) is not list or type(fills) is not list or type(parameters) is not list:
        raise RuntimeError("compiler manifest typed sections must be arrays")
    typed_ops = [op for op in recipe.operations if isinstance(op, TYPED_OPERATIONS)]
    if len(stages) != len(typed_ops) or decoded.num_stages != len(stages):
        raise RuntimeError("compiler manifest stage count does not match the recipe")
    result_rank = len(recipe.destination.shape)
    decoded_stages = decoded.stages
    shape, dtype, index = tuple(recipe.source.shape), recipe.source.dtype, 0
    for op in recipe.operations:
        if isinstance(op, TYPED_OPERATIONS):
            _check_stage(stages[index], decoded_stages[index], op, shape, dtype, result_rank,
                         symbols, index)
            dtype = op.dtype
            index += 1
        shape = operation_shape(shape, op)
    if dtype != recipe.destination.dtype:
        raise RuntimeError("compiled recipe chain dtype does not reach the destination dtype")
    pads = [op for op in recipe.operations if isinstance(op, Pad)]
    if len(fills) != len(pads) or decoded.fills != len(fills):
        raise RuntimeError("compiler manifest fill count does not match the recipe")
    actual = []
    for fill in fills:
        _exact_keys(fill, ("dst_axis", "stage", "dtype", "bits"), "fill")
        if not 0 <= _integer(fill["dst_axis"], "fill dst_axis") < result_rank:
            raise RuntimeError("compiler manifest fill axis is out of range")
        if not 0 <= _integer(fill["stage"], "fill stage") <= len(stages):
            raise RuntimeError("compiler manifest fill stage is out of range")
        actual.append((_string(fill["dtype"], "fill dtype"), _hex_bits(fill["bits"], "fill")))
    if sorted(actual) != sorted((pad.fill.dtype, pad.fill.bits) for pad in pads):
        raise RuntimeError("compiler manifest fills do not match the recipe pads")
    declared = []
    for entry in parameters:
        _exact_keys(entry, ("name", "dtype", "extents"), "parameter")
        if type(entry["extents"]) is not list:
            raise RuntimeError("compiler manifest parameter extents must be an array")
        declared.append(ParameterDeclaration(
            _string(entry["name"], "parameter name"),
            _string(entry["dtype"], "parameter dtype"),
            tuple(_parse_expr(item, "parameter extents") for item in entry["extents"]),
        ))
    expected = recipe.parameter_bindings
    if len(declared) != len(expected) or any(
            d.name != e.name or d.dtype != e.dtype or not _same_exprs(d.extents, e.extents)
            for d, e in zip(declared, expected)):
        raise RuntimeError("compiler manifest parameters do not match the recipe bindings")
    runtime = [(p["name"], p["dtype"], p["rank"]) for p in decoded.parameters]
    if runtime != [(d.name, d.dtype, len(d.extents)) for d in declared]:
        raise RuntimeError("compiler plan parameters do not match the manifest")
    return tuple(declared)


def _admit(recipe, mlir, plan, manifest):
    typed = recipe.typed
    schema = TYPED_SCHEMA if typed else LAYOUT_SCHEMA
    expected_fields = (
        "schema_version", "wire_version", "status", "compiler", "plan_count", "symbols",
        "logical_source", "logical_destination", "constraints", "plan_sha256", "input_sha256",
    )
    if typed:
        expected_fields += ("stages", "fills", "parameters")
    _exact_keys(manifest, expected_fields, "success")
    compiler = _validate_base(manifest, schema)
    if manifest["status"] != "ok" or _integer(manifest["plan_count"], "plan_count") != 1:
        raise RuntimeError("compiler manifest does not describe exactly one successful plan")
    symbols = manifest["symbols"]
    if type(symbols) is not list or any(type(name) is not str for name in symbols):
        raise RuntimeError("compiler manifest symbols must be an array of strings")
    symbols = tuple(symbols)
    decoded = _decode_plan(plan, typed)
    if symbols != _wire_symbols(plan, schema[1]) or symbols != tuple(decoded.symbols):
        raise RuntimeError("compiler manifest symbol order does not match the plan")
    if len(set(symbols)) != len(symbols) or set(symbols) != _recipe_symbols(recipe):
        raise RuntimeError("compiler manifest symbol set does not match the recipe")
    source = _parse_descriptor(manifest["logical_source"], "logical_source")
    destination = _parse_descriptor(manifest["logical_destination"], "logical_destination")
    if not _same_descriptor(source, recipe.source):
        raise RuntimeError("compiler manifest logical source descriptor mismatch")
    if not _same_descriptor(destination, recipe.destination):
        raise RuntimeError("compiler manifest logical destination descriptor mismatch")
    constraints = _parse_constraints(manifest["constraints"])
    expected_constraints = _expected_constraints(recipe)
    if sorted(constraints, key=repr) != sorted(expected_constraints, key=repr):
        raise RuntimeError("compiler manifest constraints mismatch")
    parameters = _validate_typed_sections(recipe, manifest, symbols, decoded) if typed else ()
    plan_digest = _string(manifest["plan_sha256"], "plan_sha256")
    input_digest = _string(manifest["input_sha256"], "input_sha256")
    if plan_digest != hashlib.sha256(plan).hexdigest():
        raise RuntimeError("compiler manifest plan digest mismatch")
    if input_digest != hashlib.sha256(mlir).hexdigest():
        raise RuntimeError("compiler manifest input digest mismatch")
    sources = symbol_sources(recipe.source.shape)
    source_by_name = {source.name: source for source in sources}
    return CompiledRecipe(
        recipe,
        plan,
        compiler,
        manifest["wire_version"],
        symbols,
        tuple(source_by_name[name] for name in symbols),
        constraints,
        source,
        destination,
        manifest,
        parameters,
    )


def _tensor_spec(value):
    try:
        shape = tuple(value.shape)
        stride = value.stride
        strides = tuple(stride() if callable(stride) else stride)
        storage_offset = getattr(value, "storage_offset", 0)
        offset = storage_offset() if callable(storage_offset) else storage_offset
        dtype = str(value.dtype)
    except (AttributeError, TypeError) as error:
        raise GuardError("source_descriptor") from error
    if dtype.startswith("torch."):
        dtype = dtype.removeprefix("torch.")
    return TensorSpec(shape, strides, offset, dtype)


def _encode_expr(value):
    match expression(value):
        case Const(number):
            return ["const", number]
        case Symbol(name):
            return ["symbol", name]
        case Add(lhs, rhs):
            return ["add", _encode_expr(lhs), _encode_expr(rhs)]
        case Mul(lhs, rhs):
            return ["mul", _encode_expr(lhs), _encode_expr(rhs)]
        case FloorDiv(lhs, divisor):
            return ["floordiv", _encode_expr(lhs), divisor]
        case Mod(lhs, divisor):
            return ["mod", _encode_expr(lhs), divisor]
    raise RuntimeError("recipe contains an unsupported expression")


def _encode_descriptor(value):
    return {
        "dtype": value.dtype,
        "offset": _encode_expr(value.offset),
        "shape": [_encode_expr(item) for item in value.shape],
        "strides": [_encode_expr(item) for item in value.strides],
    }


def _encode_param(param):
    if param is None:
        return None
    if isinstance(param, InlineParam):
        return {"bits": list(param.bits), "dtype": param.dtype, "kind": "inline",
                "shape": list(param.shape)}
    return {"dtype": param.dtype, "extents": [_encode_expr(item) for item in param.extents],
            "kind": "binding", "name": param.name}


def _encode_recipe(recipe):
    operations = []
    for operation in recipe.operations:
        if isinstance(operation, Transpose):
            operations.append({"kind": "transpose", "perm": list(operation.perm)})
        elif isinstance(operation, Reshape):
            operations.append({"kind": "reshape", "shape": [_encode_expr(item) for item in operation.shape]})
        elif isinstance(operation, Pad):
            operations.append(
                {
                    "axis": operation.axis,
                    "fill": {"bits": operation.fill.bits, "dtype": operation.fill.dtype},
                    "hi": _encode_expr(operation.hi),
                    "kind": "pad",
                    "lo": _encode_expr(operation.lo),
                }
            )
        elif isinstance(operation, Cast):
            operations.append({"dtype": operation.dtype, "kind": "cast", "policy": operation.policy})
        elif isinstance(operation, (Quantize, Dequantize)):
            operations.append(
                {
                    "axis": operation.axis,
                    "dtype": operation.dtype,
                    "kind": _TRANSFORM_NAMES[type(operation)],
                    "policy": operation.policy,
                    "scale": _encode_param(operation.scale),
                    "zero_point": _encode_param(operation.zero_point),
                }
            )
        else:
            raise RuntimeError("recipe contains an unsupported operation")
    return {
        "destination": _encode_descriptor(recipe.destination),
        "direction": recipe.direction,
        "operations": operations,
        "source": _encode_descriptor(recipe.source),
    }


def _decode_descriptor(value, where):
    return _parse_descriptor(value, where)


def _decode_param(value, where):
    if value is None:
        return None
    if type(value) is not dict or type(value.get("kind")) is not str:
        raise RuntimeError(f"compiled recipe has invalid {where}")
    try:
        if value["kind"] == "inline":
            _exact_keys(value, ("kind", "dtype", "shape", "bits"), where)
            if type(value["shape"]) is not list or type(value["bits"]) is not list:
                raise RuntimeError(f"compiled recipe {where} shape/bits must be arrays")
            return InlineParam(
                _string(value["dtype"], where),
                tuple(_integer(item, where) for item in value["shape"]),
                tuple(_integer(item, where) for item in value["bits"]),
            )
        if value["kind"] == "binding":
            _exact_keys(value, ("kind", "name", "dtype", "extents"), where)
            if type(value["extents"]) is not list:
                raise RuntimeError(f"compiled recipe {where} extents must be an array")
            return BindingParam(
                _string(value["name"], where),
                _string(value["dtype"], where),
                tuple(_parse_expr(item, where) for item in value["extents"]),
            )
    except ValueError as error:
        raise RuntimeError(f"compiled recipe has invalid {where}") from error
    raise RuntimeError(f"compiled recipe has unsupported {where} kind")


def _decode_recipe(value):
    from .recipe import Fill

    _exact_keys(value, ("source", "operations", "destination", "direction"), "recipe")
    if type(value["operations"]) is not list:
        raise RuntimeError("compiled recipe operations must be an array")
    direction = _string(value["direction"], "recipe.direction")
    if direction not in ("h2d", "d2h"):
        raise RuntimeError("compiled recipe has unsupported direction")
    operations = []
    for item in value["operations"]:
        if type(item) is not dict or type(item.get("kind")) is not str:
            raise RuntimeError("compiled recipe has invalid operation")
        kind = item["kind"]
        try:
            if kind == "transpose":
                _exact_keys(item, ("kind", "perm"), "transpose")
                if type(item["perm"]) is not list:
                    raise RuntimeError("compiled recipe transpose perm must be an array")
                operations.append(Transpose(tuple(_integer(axis, "transpose axis") for axis in item["perm"])))
            elif kind == "reshape":
                _exact_keys(item, ("kind", "shape"), "reshape")
                if type(item["shape"]) is not list:
                    raise RuntimeError("compiled recipe reshape shape must be an array")
                operations.append(Reshape(tuple(_parse_expr(dim, "reshape shape") for dim in item["shape"])))
            elif kind == "pad":
                _exact_keys(item, ("kind", "axis", "lo", "hi", "fill"), "pad")
                _exact_keys(item["fill"], ("dtype", "bits"), "fill")
                operations.append(
                    Pad(
                        _integer(item["axis"], "pad axis"),
                        _parse_expr(item["lo"], "pad lo"),
                        _parse_expr(item["hi"], "pad hi"),
                        Fill(
                            _string(item["fill"]["dtype"], "fill dtype"),
                            _integer(item["fill"]["bits"], "fill bits"),
                        ),
                    )
                )
            elif kind == "cast":
                _exact_keys(item, ("kind", "dtype", "policy"), "cast")
                operations.append(Cast(_string(item["dtype"], "cast dtype"), _string(item["policy"], "cast policy")))
            elif kind in ("quantize", "dequantize"):
                _exact_keys(item, ("kind", "dtype", "policy", "axis", "scale", "zero_point"), kind)
                axis = item["axis"]
                if axis is not None:
                    axis = _integer(axis, f"{kind} axis")
                operations.append(
                    (Quantize if kind == "quantize" else Dequantize)(
                        _string(item["dtype"], f"{kind} dtype"),
                        _decode_param(item["scale"], f"{kind} scale"),
                        _decode_param(item["zero_point"], f"{kind} zero_point"),
                        axis,
                        _string(item["policy"], f"{kind} policy"),
                    )
                )
            else:
                raise RuntimeError("compiled recipe has unsupported operation")
        except ValueError as error:
            raise RuntimeError("compiled recipe has invalid operation") from error
    return Recipe(
        _decode_descriptor(value["source"], "recipe.source"),
        tuple(operations),
        _decode_descriptor(value["destination"], "recipe.destination"),
        direction,
    )

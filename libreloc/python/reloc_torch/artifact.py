"""Validated, portable frontend metadata for R1 compiler artifacts."""
from dataclasses import dataclass
import base64
import hashlib
import json
from pathlib import Path

from .mlir_emit import emit_mlir
from .recipe import Pad, Recipe, Reshape, TensorSpec, Transpose
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
    symbol_sources,
)


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

    def to_bytes(self):
        """Serialize portable metadata and plan bytes without live FX/Torch state."""
        payload = {
            "format_version": 1,
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
        if _integer(payload["format_version"], "artifact format_version") != 1:
            raise RuntimeError("compiled recipe format version mismatch")
        if type(payload["plan_base64"]) is not str:
            raise RuntimeError("compiled recipe plan_base64 must be a string")
        try:
            plan = base64.b64decode(payload["plan_base64"], validate=True)
        except (ValueError, base64.binascii.Error) as error:
            raise RuntimeError("compiled recipe has invalid plan bytes") from error
        recipe = _decode_recipe(payload["recipe"])
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


def _validate_base(manifest):
    if _integer(manifest.get("schema_version"), "schema_version") != 1:
        raise RuntimeError("compiler manifest schema version mismatch")
    if _integer(manifest.get("wire_version"), "wire_version") != 0:
        raise RuntimeError("compiler manifest wire version mismatch")
    return _compiler_identity(manifest.get("compiler"))


def _validate_unsupported_manifest(manifest):
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
        and len(actual.shape) == len(expected.shape)
        and len(actual.strides) == len(expected.strides)
        and all(_normalize_expr(a) == _normalize_expr(b) for a, b in zip(actual.shape, expected.shape))
        and all(_normalize_expr(a) == _normalize_expr(b) for a, b in zip(actual.strides, expected.strides))
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
    return names


def _wire_symbols(plan):
    if len(plan) < 12 or plan[:4] != b"RPLN":
        raise RuntimeError("compiler plan has invalid wire header")
    version = int.from_bytes(plan[4:8], "little")
    if version != 0:
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


def _decoded_symbols(plan):
    try:
        import pyreloc
    except ImportError as error:
        raise RuntimeError("pyreloc runtime is required to validate compiler plans") from error
    try:
        decoded = pyreloc.load_plan(plan)
    except pyreloc.DecodeError as error:
        raise RuntimeError(f"compiler published an invalid plan: {error}") from error
    return tuple(decoded.symbols)


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


def _admit(recipe, mlir, plan, manifest):
    expected_fields = (
        "schema_version", "wire_version", "status", "compiler", "plan_count", "symbols",
        "logical_source", "logical_destination", "constraints", "plan_sha256", "input_sha256",
    )
    _exact_keys(manifest, expected_fields, "success")
    compiler = _validate_base(manifest)
    if manifest["status"] != "ok" or _integer(manifest["plan_count"], "plan_count") != 1:
        raise RuntimeError("compiler manifest does not describe exactly one successful plan")
    symbols = manifest["symbols"]
    if type(symbols) is not list or any(type(name) is not str for name in symbols):
        raise RuntimeError("compiler manifest symbols must be an array of strings")
    symbols = tuple(symbols)
    if symbols != _wire_symbols(plan) or symbols != _decoded_symbols(plan):
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
        else:
            raise RuntimeError("compiled recipe has unsupported operation")
    return Recipe(
        _decode_descriptor(value["source"], "recipe.source"),
        tuple(operations),
        _decode_descriptor(value["destination"], "recipe.destination"),
        direction,
    )

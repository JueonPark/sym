"""Cost advice: the compiler's machine-independent statement of which execution
paths a plan admits and which calibration quantities price them (interface 2,
docs/reloc-export.md "Cost advice"). This module validates the block at
admission, resolves it against concrete bindings and a calibration, and
selects a path for the fast lane (docs/fast-transfers.md "Cost advice and the
direct path")."""
from __future__ import annotations

from dataclasses import dataclass
import re

from .symbolic import Add, Expr, FloorDiv, Mod, Mul, Symbol

PATTERNS = ("contiguous", "blocked", "single_element", "tiled", "indexed_rows", "symbolic")
PRODUCER_KINDS = ("none", "layout", "stages", "indexed_rows", "pooled")
PRODUCER_KERNELS = ("read", "convert_f32_f16", "quantize", "gather_f16", "pooled_sum_f16")
RECEIVE_KERNELS = ("dequant_s8_f32", "dequant_s8_f16")
PATH_NAMES = ("direct", "pipelined", "grouped")
FORMS = {"direct": "serial", "pipelined": "overlapped", "grouped": "grouped"}
RATES = ("producer", "link", "device")
QUANTITIES = ("source_bytes", "result_bytes", "wire")
_KEY = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")


@dataclass(frozen=True)
class Path:
    name: str
    form: str
    wire: str
    intercept: str
    terms: tuple[tuple[str, str], ...]
    chunks: str | None = None
    member_intercept: str | None = None


@dataclass(frozen=True)
class Advice:
    version: int
    pattern: str
    run_elements: Expr
    total_elements: Expr
    source_bytes: Expr
    result_bytes: Expr
    producer_kind: str
    producer_kernel: str | None
    receive_kernel: str | None
    paths: tuple[Path, ...]

    def path(self, name):
        for path in self.paths:
            if path.name == name:
                return path
        return None


def _fail(where, what):
    raise RuntimeError(f"compiler manifest {where} {what}")


def _symbols_of(expr):
    match expr:
        case Symbol(name):
            return {name}
        case Add(lhs, rhs) | Mul(lhs, rhs):
            return _symbols_of(lhs) | _symbols_of(rhs)
        case FloorDiv(lhs, _) | Mod(lhs, _):
            return _symbols_of(lhs)
        case _:
            return set()


def _key(value, where):
    if type(value) is not str or not _KEY.match(value):
        _fail(where, "has an invalid calibration key name")
    return value


def _choice(value, choices, where, what):
    if type(value) is not str or value not in choices:
        _fail(where, f"has an unknown {what}")
    return value


def parse_advice(value, symbols, where="advice"):
    """The validated Advice of a success manifest, or None for `advice: null`.

    Every expression may reference only the plan's symbols, every key name is
    a calibration key, and every path, form, rate and quantity comes from the
    fixed vocabulary. Anything else is a manifest fault (RuntimeError), like
    the rest of the manifest."""
    from .artifact import _exact_keys, _integer, _parse_expr
    if value is None:
        return None
    _exact_keys(value, ("version", "pattern", "run_elements", "total_elements", "source_bytes",
                        "result_bytes", "producer", "receive", "paths"), where)
    if _integer(value["version"], f"{where}.version") != 1:
        _fail(where, "has an unsupported advice version")
    pattern = _choice(value["pattern"], PATTERNS, where, "pattern")
    exprs = {}
    for field in ("run_elements", "total_elements", "source_bytes", "result_bytes"):
        expr = _parse_expr(value[field], f"{where}.{field}")
        unknown = _symbols_of(expr) - set(symbols)
        if unknown:
            _fail(where, f"{field} references an undeclared symbol {sorted(unknown)}")
        exprs[field] = expr
    producer = value["producer"]
    _exact_keys(producer, ("kind", "kernel"), f"{where}.producer")
    kind = _choice(producer["kind"], PRODUCER_KINDS, where, "producer kind")
    kernel = producer["kernel"]
    if kernel is not None:
        _choice(kernel, PRODUCER_KERNELS, where, "producer kernel")
    if kind == "none" and kernel is not None:
        _fail(where, "names a producer kernel without a producer")
    receive = value["receive"]
    _exact_keys(receive, ("kernel",), f"{where}.receive")
    receive_kernel = receive["kernel"]
    if receive_kernel is not None:
        _choice(receive_kernel, RECEIVE_KERNELS, where, "receive kernel")
    raw_paths = value["paths"]
    if type(raw_paths) is not list or not raw_paths:
        _fail(where, "paths must be a nonempty array")
    paths = []
    for index, entry in enumerate(raw_paths):
        pwhere = f"{where}.paths[{index}]"
        if type(entry) is not dict:
            _fail(pwhere, "must be an object")
        name = _choice(entry.get("name"), PATH_NAMES, pwhere, "path name")
        expected = ["name", "form", "wire", "intercept", "terms"]
        if name == "pipelined":
            expected.append("chunks")
        if name == "grouped":
            expected.append("member_intercept")
        _exact_keys(entry, tuple(expected), pwhere)
        if entry["form"] != FORMS[name]:
            _fail(pwhere, "has the wrong form for its path")
        wire = _choice(entry["wire"], ("source", "result"), pwhere, "wire quantity")
        terms = entry["terms"]
        if type(terms) is not list or not terms:
            _fail(pwhere, "terms must be a nonempty array")
        parsed_terms = []
        for term in terms:
            if (type(term) is not list or len(term) != 2 or term[0] not in RATES
                    or term[1] not in QUANTITIES):
                _fail(pwhere, "has an invalid cost term")
            if term[0] == "producer" and kind == "none":
                _fail(pwhere, "prices a producer the plan does not have")
            parsed_terms.append((term[0], term[1]))
        paths.append(Path(
            name, FORMS[name], wire, _key(entry["intercept"], pwhere), tuple(parsed_terms),
            _key(entry["chunks"], pwhere) if name == "pipelined" else None,
            _key(entry["member_intercept"], pwhere) if name == "grouped" else None))
    names = [p.name for p in paths]
    if names != [n for n in PATH_NAMES if n in names]:
        _fail(where, "paths must be unique and in direct, pipelined, grouped order")
    return Advice(1, pattern, exprs["run_elements"], exprs["total_elements"], exprs["source_bytes"],
                  exprs["result_bytes"], kind, kernel, receive_kernel, tuple(paths))

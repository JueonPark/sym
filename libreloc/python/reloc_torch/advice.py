"""Cost advice: the compiler's machine-independent statement of which execution
paths a plan admits and which calibration quantities price them (interface 2,
docs/reloc-export.md "Cost advice"). This module validates the block at
admission, resolves it against concrete bindings and a calibration, and
selects a path for the fast lane (docs/fast-transfers.md "Cost advice and the
direct path")."""
from __future__ import annotations

from dataclasses import dataclass
import os
import re

from .symbolic import Add, Const, Expr, FloorDiv, Mod, Mul, Symbol

PATTERNS = ("contiguous", "blocked", "single_element", "tiled", "indexed_rows", "symbolic")
PRODUCER_KINDS = ("none", "layout", "stages", "indexed_rows", "pooled")
PRODUCER_KERNELS = ("read", "convert_f32_f16", "quantize", "gather_f16", "pooled_sum_f16")
RECEIVE_KERNELS = ("dequant_s8_f32", "dequant_s8_f16")
PATH_NAMES = ("direct", "pipelined", "grouped")
FORMS = {"direct": "serial", "pipelined": "overlapped", "grouped": "grouped"}
RATES = ("producer", "link", "device")
QUANTITIES = ("source_bytes", "result_bytes", "wire")
_KEY = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$")
BLOCKED_RUN_FLOOR = 64  # elements; CostModel.h kBlockedRunFloor


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


# -- resolution ---------------------------------------------------------------

def classify(run_elements, total_elements):
    """costmodel::classify: L >= total -> contiguous, L <= 1 -> single_element,
    L >= 64 -> blocked, otherwise tiled."""
    if run_elements >= total_elements > 0:
        return "contiguous"
    if run_elements <= 1:
        return "single_element"
    return "blocked" if run_elements >= BLOCKED_RUN_FLOOR else "tiled"


def load_calibration(value):
    """A pyreloc.Calibration from a path, an already loaded calibration, or
    None (then $RELOC_CALIBRATION when set, else no calibration)."""
    import pyreloc
    if value is None:
        value = os.environ.get("RELOC_CALIBRATION") or None
        if value is None:
            return None
    if isinstance(value, pyreloc.Calibration):
        return value
    return pyreloc.load_calibration(os.fspath(value))


def pooled_advice():
    """The fused EmbeddingBag producer never passes the compiler (its program
    is synthesized natively per call); its advice has the compiler's shape and
    is written here. Symbols: bags, tables, dim (result [bags, tables, dim]
    float16). The producer is priced on the wire: its source volume changes
    with every call's bag lengths, so its rate is calibrated wire-normalized
    (cpu.t<threads>.contiguous.pooled_sum_f16_gbps)."""
    bags, tables, dim = Symbol("bags"), Symbol("tables"), Symbol("dim")
    elements = Mul(Mul(bags, tables), dim)
    result = Mul(elements, Const(2))
    terms = (("producer", "result_bytes"), ("link", "wire"), ("device", "wire"))
    return Advice(1, "contiguous", elements, elements, result, result, "pooled", "pooled_sum_f16", None, (
        Path("direct", "serial", "result", "overhead.direct.pooled_ms", terms),
        Path("pipelined", "overlapped", "result", "overhead.dispatch.pooled_ms", terms,
             chunks="pipeline.chunks_per_buffer"),
        Path("grouped", "grouped", "result", "overhead.group_ms", terms,
             member_intercept="overhead.group_member_ms")))


def _producer_key(advice, pattern, threads):
    kernel = advice.producer_kernel
    if kernel == "read":
        kernel = "contig_read" if pattern == "contiguous" else "gather_f32"
    elif kernel == "quantize":
        kernel = "quantize_pack" if pattern == "contiguous" else "gather_quantize"
    elif kernel is None:
        kernel = "unmeasured"  # no rate family: priced as a missing key
    return f"cpu.t{threads}.{pattern}.{kernel}_gbps"


@dataclass(frozen=True)
class Resolved:
    """Advice under concrete bindings and a thread count: what
    pyreloc.price_advice takes, one dict per admissible path."""
    pattern: str
    source_bytes: int
    result_bytes: int
    direct: bool
    paths: tuple

    def path(self, name):
        for path in self.paths:
            if path["name"] == name:
                return path
        return None


def resolve(advice, bindings, threads):
    """Evaluate the advice's expressions under `bindings` and name every
    calibration key for `threads`. Raises RuntimeError('advice_invalid: ...')
    on an unbound symbol or a negative quantity, never a KeyError."""
    from .symbolic import GuardError
    try:
        run, total, source, result = (
            expr.evaluate(bindings, checked=True)
            for expr in (advice.run_elements, advice.total_elements, advice.source_bytes, advice.result_bytes))
    except KeyError as error:
        raise RuntimeError(f"advice_invalid: unbound symbol {error}") from error
    except GuardError as error:
        raise RuntimeError(f"advice_invalid: {error.reason}") from error
    if min(run, total, source, result) < 0:
        raise RuntimeError("advice_invalid: negative quantity")
    pattern = classify(run, total) if advice.pattern == "symbolic" else advice.pattern
    producer = _producer_key(advice, pattern, threads)
    multiplier = f"recv.m.{advice.receive_kernel}" if advice.receive_kernel else f"hbm.m.{pattern}"
    paths = []
    for path in advice.paths:
        quantities = dict(source_bytes=source, result_bytes=result, wire=source if path.wire == "source" else result)
        terms = []
        for rate, quantity in path.terms:
            if rate == "producer":
                terms.append([quantities[quantity], producer, None])
            elif rate == "link":
                terms.append([quantities[quantity], "pcie.h2d_gbps", None])
            else:
                terms.append([quantities[quantity], "hbm.bw_gbps", multiplier])
        paths.append(dict(name=path.name, form=path.form, intercept=path.intercept,
                          member_intercept=path.member_intercept, chunks=path.chunks, terms=terms))
    return Resolved(pattern, source, result, advice.path("direct") is not None, tuple(paths))


def without_direct(resolved):
    """The same advice with the direct path not admissible (an executor that
    has no direct path of its own)."""
    import dataclasses
    return dataclasses.replace(resolved, direct=False,
                               paths=tuple(p for p in resolved.paths if p["name"] != "direct"))


# -- selection ----------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    """The executed path, the placement reason and every priced path (ms, or
    None when the calibration lacks a key)."""
    path: str
    reason: str
    predicted: dict


def _price(calibration, paths, members):
    import pyreloc
    try:
        return pyreloc.price_advice(calibration, list(paths), members)
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def select(calibration, resolved, *, direct_available, advice_enabled=True):
    """Single-call context: the direct path (when admitted and executable)
    against the dispatcher's pipelined row."""
    if not advice_enabled:
        return Decision("pipelined", "advice_disabled", {})
    if resolved is None:
        return Decision("pipelined", "advice_absent", {})
    if calibration is None:
        return Decision("pipelined", "no_calibration", {})
    predicted = {c["name"]: c["ms"] for c in _price(calibration, resolved.paths, 1)}
    candidates = [name for name in ("direct", "pipelined") if name in predicted]
    if any(predicted[name] is None for name in candidates):
        return Decision("pipelined", "advice_missing_keys", predicted)
    best = min(candidates, key=lambda name: predicted[name])
    if best == "direct" and not direct_available:
        return Decision("pipelined", "advice_no_native_helper", predicted)
    return Decision(best, f"advice_prefers_{best}", predicted)


def select_group(calibration, members, *, direct_available, advice_enabled=True):
    """Group context: one grouped submission, against every member taken
    direct, or every member through its own pipelined dispatch (split). The
    runtime never splits a group the caller submitted: when split prices
    cheapest the group still runs and the reason (advice_prefers_split) is
    advisory."""
    if not advice_enabled:
        return Decision("grouped", "advice_disabled", {})
    if not members or any(member is None for member in members):
        return Decision("grouped", "advice_absent", {})
    if calibration is None:
        return Decision("grouped", "no_calibration", {})
    grouped = [member.path("grouped") for member in members]
    if any(path is None for path in grouped):
        return Decision("grouped", "advice_absent", {})
    # One submission: the group intercepts once, every member's terms summed.
    whole = dict(grouped[0], terms=[term for path in grouped for term in path["terms"]])
    predicted = {"grouped": _price(calibration, [whole], len(members))[0]["ms"], "direct": 0.0, "pipelined": 0.0}
    for member in members:
        costs = {c["name"]: c["ms"] for c in _price(calibration, member.paths, 1)}
        for name in ("direct", "pipelined"):
            value = costs.get(name)  # None: no such path, or a missing key
            if predicted[name] is not None:
                predicted[name] = None if value is None else predicted[name] + value
    if predicted["grouped"] is None or predicted["pipelined"] is None:
        return Decision("grouped", "advice_missing_keys", predicted)
    ranked = sorted((name for name in predicted if predicted[name] is not None), key=lambda name: predicted[name])
    best = ranked[0]
    if best == "direct":
        if direct_available:
            return Decision("direct", "advice_prefers_direct", predicted)
        best = ranked[1]
        if best == "grouped":
            return Decision("grouped", "advice_no_native_helper", predicted)
    if best == "pipelined":
        return Decision("grouped", "advice_prefers_split", predicted)
    return Decision("grouped", "advice_prefers_grouped", predicted)

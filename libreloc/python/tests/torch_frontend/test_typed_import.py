"""C4 (issue #144): typed FX import, gated on proved semantics.

Casts between f32 and f16 carried by ``aten._to_copy`` and the audited
``quantized_decomposed.dequantize_*`` operators become typed recipes with
their scale/zero-point operands preserved (exact inline bits for scalars,
named runtime parameters for tensors); ``quantize_*`` and every other dtype
change stay excluded with a reason and the original graph untouched. The
typed custom op carries the destination dtype and the parameters, its fake
metadata matches the real output, and a CPU adapter runs the whole route
host to host; the real GPU rows are ``gpu`` marked.
"""
import struct

import numpy as np
import pytest
import torch

from conftest import CountingRuntime

import torch.ao.quantization.fx._decomposed  # noqa: F401  registers quantized_decomposed
QD = torch.ops.quantized_decomposed


def importer():
    from reloc_torch import fx_import

    return fx_import


def capture(fn, *inputs):
    from reloc_torch.compat import symbolic_capture

    return symbolic_capture(fn, *inputs)


def transfer(x, **kwargs):
    return torch.ops.aten._to_copy.default(x, device=torch.device("cuda:0"), **kwargs)


def snapshot(gm):
    return str(gm.graph), [(n.name, tuple(u.name for u in n.users), dict(n.meta)) for n in gm.graph.nodes]


def f32bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


#===----------------------------------------------------------------------===#
# Import: casts and dequantize, with operands preserved.
#===----------------------------------------------------------------------===#

def test_cast_riding_the_transfer_becomes_a_typed_recipe():
    from reloc_torch.recipe import Cast, Transpose

    gm = capture(lambda x: transfer(x.t().contiguous(), dtype=torch.float16), torch.ones(4, 6))
    before = snapshot(gm)
    report = importer().import_graph(gm)
    assert snapshot(gm) == before and not report.exclusions
    candidate, = report.candidates
    assert candidate.recipe.operations == (Transpose((1, 0)), Cast("float16", "ieee_rne"))
    assert candidate.recipe.typed and candidate.recipe.direction == "h2d"
    assert candidate.recipe.source.dtype == "float32" and candidate.recipe.destination.dtype == "float16"
    assert candidate.parameters == ()
    # A same-device cast before the transfer is the same recipe in the other order.
    gm = capture(lambda x: transfer(x.to(torch.float16).t().contiguous()), torch.ones(4, 6))
    candidate, = importer().import_graph(gm).candidates
    assert candidate.recipe.operations == (Cast("float16", "ieee_rne"), Transpose((1, 0)))
    # Widening: f16 source, exact.
    gm = capture(lambda h: transfer(h.t().contiguous(), dtype=torch.float32), torch.ones(2, 12, dtype=torch.float16))
    candidate, = importer().import_graph(gm).candidates
    assert candidate.recipe.operations == (Transpose((1, 0)), Cast("float32", "exact"))
    assert candidate.recipe.source.dtype == "float16" and candidate.recipe.destination.dtype == "float32"


@pytest.mark.parametrize(("fn", "reason"), [
    (lambda x: transfer(x, dtype=torch.bfloat16), "typed_transform_unavailable"),
    (lambda x: transfer(x, dtype=torch.int8), "typed_transform_unavailable"),
    (lambda x: transfer(x.to(torch.float16).to(torch.float32)), None),  # two C1 casts, both kept
    (lambda x: x.to(torch.float16), "same_device_transfer"),  # no relocation at all
])
def test_unproved_casts_are_excluded_and_lossy_pairs_are_kept(fn, reason):
    from reloc_torch.recipe import Cast

    gm = capture(fn, torch.ones(4, 6))
    before = snapshot(gm)
    report = importer().import_graph(gm)
    assert snapshot(gm) == before
    if reason is None:
        candidate, = report.candidates
        assert candidate.recipe.operations == (Cast("float16", "ieee_rne"), Cast("float32", "exact"))
    else:
        assert not report.candidates
        assert reason in {e.reason for e in report.exclusions}


def test_dequantize_with_scalar_operands_becomes_exact_inline_parameters():
    from reloc_torch.recipe import Dequantize, InlineParam, Transpose

    gm = capture(lambda q: transfer(QD.dequantize_per_tensor(q, 0.3, 5, -128, 127, torch.int8).t().contiguous()),
                 torch.zeros(4, 6, dtype=torch.int8))
    report = importer().import_graph(gm)
    assert not report.exclusions
    candidate, = report.candidates
    dequantize, transpose = candidate.recipe.operations
    assert transpose == Transpose((1, 0))
    assert dequantize == Dequantize("float32", InlineParam("float32", (), (f32bits(0.3),)),
                                    InlineParam("int32", (), (5,)), None, "affine")
    assert candidate.recipe.source.dtype == "int8" and candidate.recipe.destination.dtype == "float32"
    assert candidate.parameters == ()


def test_dequantize_with_tensor_operands_declares_runtime_parameters():
    from reloc_torch.recipe import BindingParam, Dequantize, Transpose
    from reloc_torch.symbolic import Const

    def fn(q, s, z):
        return transfer(QD.dequantize_per_channel(q, s, z, 1, -128, 127, torch.int8).permute(1, 0, 2).contiguous())

    gm = capture(fn, torch.zeros(2, 3, 4, dtype=torch.int8), torch.ones(3), torch.zeros(3, dtype=torch.int64))
    report = importer().import_graph(gm)
    assert not report.exclusions, report.exclusions
    candidate, = report.candidates
    dequantize, transpose = candidate.recipe.operations
    assert transpose == Transpose((1, 0, 2))
    assert dequantize.axis == 1 and dequantize.policy == "affine"
    # Symbolic capture duck-sizes the channel extent: the parameter's declared
    # extent is exactly the operand's axis-1 extent, whatever symbol it is.
    channel = candidate.recipe.source.shape[1]
    assert dequantize.scale == BindingParam("s_1", "float32", (channel,))
    assert dequantize.zero_point == BindingParam("z_1", "int32", (channel,))
    assert candidate.parameters == ("s_1", "z_1")
    # The original callable takes the source, then the parameters.
    placeholders = [n.name for n in candidate.original.graph.nodes if n.op == "placeholder"]
    assert placeholders == ["src", "s_1", "z_1"]
    # Per-tensor tensor operands (0-d) are runtime parameters too.
    gm = capture(lambda q, s, z: transfer(QD.dequantize_per_tensor.tensor(q, s, z, -128, 127, torch.int8)),
                 torch.zeros(5, dtype=torch.int8), torch.tensor(0.3), torch.tensor(5))
    candidate, = importer().import_graph(gm).candidates
    assert candidate.recipe.operations[0].scale == BindingParam("s_1", "float32", ())
    assert candidate.parameters == ("s_1", "z_1")


@pytest.mark.parametrize(("fn", "inputs", "reason"), [
    # quantize: double-rounded reciprocal and platform NaN conversion; never imported.
    (lambda x: transfer(QD.quantize_per_tensor(x, 0.5, 0, -128, 127, torch.int8)), (torch.ones(8),), "quantize_semantics_unproved"),
    (lambda x, s, z: transfer(QD.quantize_per_channel(x, s, z, 1, -128, 127, torch.int8)),
     (torch.ones(2, 3, 4), torch.ones(3), torch.zeros(3, dtype=torch.int32)), "quantize_semantics_unproved"),
    # dequantize outside C1's table.
    (lambda q: transfer(QD.dequantize_per_tensor(q, 0.3, 0, -127, 127, torch.int8)), (torch.zeros(4, dtype=torch.int8),), "unsupported_quantization_range"),
    (lambda q: transfer(QD.dequantize_per_tensor(q, 0.3, 0, -128, 127, torch.int8, out_dtype=torch.float16)), (torch.zeros(4, dtype=torch.int8),), "typed_transform_unavailable"),
    (lambda q, s, z: transfer(QD.dequantize_per_tensor.tensor(q, s, z, -128, 127, torch.int8)),
     (torch.zeros(4, dtype=torch.int8), torch.tensor(0.3, dtype=torch.float64), torch.tensor(0)), "parameter_dtype_unsupported"),
    (lambda q: transfer(QD.dequantize_per_tensor(q, 0.0, 0, -128, 127, torch.int8)), (torch.zeros(4, dtype=torch.int8),), "unsupported_parameter"),
])
def test_unproved_quantization_captures_keep_the_original_with_a_reason(fn, inputs, reason):
    gm = capture(fn, *inputs)
    before = snapshot(gm)
    report = importer().import_graph(gm)
    assert snapshot(gm) == before
    assert not report.candidates
    assert reason in {e.reason for e in report.exclusions}, report.exclusions


#===----------------------------------------------------------------------===#
# Compile, rewrite and execute through the typed op (CPU, host to host).
#===----------------------------------------------------------------------===#

class TypedCountingRuntime(CountingRuntime):
    """CPU-only adapter for typed recipes: real typed binding, real R3 host
    dispatch through pyreloc, counted launches, no CUDA."""

    capability_identity = "cpu-typed-test-adapter/1"

    def preflight(self, compiled, src, device, *, non_blocking=False, parameters=None):
        import pyreloc
        from reloc_torch.artifact import UnsupportedRecipe
        from reloc_torch.runtime import PreparedCall, bind_symbols, destination_descriptor, source_reason

        self.preflights += 1
        if not getattr(compiled, "typed", False):
            return super().preflight(compiled, src, device, non_blocking=non_blocking)
        if device.type != "cpu":
            raise UnsupportedRecipe("unsupported_device", "CPU-only test adapter")
        reason = source_reason(src)
        if reason is not None:
            raise UnsupportedRecipe(reason, reason)
        bindings = bind_symbols(compiled, src)
        snapshots = {name: (value.dtype.__str__().removeprefix("torch."), list(value.shape), value.contiguous().numpy().tobytes())
                     for name, value in (parameters or {}).items()}
        try:
            bound = pyreloc.bind_typed(pyreloc.load_typed_plan(compiled.plan_bytes), bindings, snapshots)
        except pyreloc.BindError as error:
            raise UnsupportedRecipe("bind_error", str(error)) from error
        destination = destination_descriptor(compiled, bindings, device)
        return PreparedCall(compiled, src, bindings, bound, destination, non_blocking, request=None)

    def execute(self, call):
        import pyreloc
        from pyreloc.torch_interop import as_ptr

        if not getattr(call.compiled, "typed", False):
            return super().execute(call)
        self.executions += 1
        destination = call.destination
        out = torch.empty_strided(destination.shape, destination.strides, dtype=getattr(torch, destination.dtype), device="cpu")
        src = call.src.contiguous()
        src_view = pyreloc.BufferView(*as_ptr(src), 0, list(src.shape), list(src.stride()), src.element_size(), "host")
        dst_view = pyreloc.BufferView(*as_ptr(out), 0, list(out.shape), list(out.stride()), out.element_size(), "host")
        request = pyreloc.prepare_dispatch(call.bound, src_view, dst_view, call.compiled.recipe.direction, policy="original_cpu")
        call.report = pyreloc.execute_dispatch(request)
        return out


def cast_candidate(gm, fn, recipe, symbol_sources, members, parameters=()):
    from reloc_torch.fx_import import Candidate, ImportReport

    names = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    candidate = Candidate(names[0], members[-1], tuple(members), recipe, tuple(symbol_sources), (),
                          torch.fx.symbolic_trace(fn), (), torch.device("cpu"), tuple(parameters))
    return lambda graph, inputs: ImportReport((candidate,), ())


def test_typed_op_fake_metadata_matches_the_real_output_and_the_rewrite_carries_parameters(compiler):
    from reloc_torch.backend import RelocBackend
    from reloc_torch.recipe import Cast, Dequantize, BindingParam, Recipe, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, Symbol, SymbolSource, dense_strides

    runtime = TypedCountingRuntime()
    backend = RelocBackend(compiler=compiler, runtime=runtime)
    try:
        # (a) transpose + cast to f16, CPU "h2d" recipe executed host to host.
        rows, columns = Symbol("s0"), Symbol("s1")
        recipe = Recipe(
            TensorSpec((rows, columns), (columns, Const(1)), Const(0), "float32"),
            (Transpose((1, 0)), Cast("float16", "ieee_rne")),
            TensorSpec((columns, rows), (rows, Const(1)), Const(0), "float16"),
            "h2d",
        )

        def fn(x):
            return x.transpose(0, 1).contiguous().to(torch.float16)

        gm = capture(fn, torch.ones(3, 4))
        members = [n.name for n in gm.graph.nodes if n.op == "call_function"]
        backend._importer = cast_candidate(gm, fn, recipe, (SymbolSource("s0", 0), SymbolSource("s1", 1)), members)
        compiled_fn = backend(gm, None)
        node = next(n for n in compiled_fn.rewritten.graph.nodes if n.op == "call_function" and n.target is torch.ops.reloc_torch.typed_transfer.default)
        root, params, handle, symbols, out_shape, out_strides, device, dtype = node.args
        assert params == [] and dtype == torch.float16 and device == torch.device("cpu")
        assert node.meta["val"].dtype == torch.float16
        x = torch.randn(5, 7) * 40
        x[0, 0], x[0, 1] = float("inf"), -0.0
        with torch.no_grad():
            actual = compiled_fn(x)
        expected = fn(x)
        assert actual.dtype == torch.float16 and actual.shape == (7, 5) and actual.stride() == expected.stride()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), "bit-exact against PyTorch's own RNE cast"
        assert runtime.executions == 1
        stats = backend.stats()
        assert stats["typed_executions"] == 1 and stats["dispatches"] == {"cpu_reference": 1}
        assert stats["typed_payload_bytes"] == 7 * 5 * 2
        # (b) dequantize with runtime parameters: the op carries the parameter
        # nodes, the adapter binds them by name, values change results.
        recipe = Recipe(
            TensorSpec((Const(2), Const(3), Const(4)), dense_strides((Const(2), Const(3), Const(4))), Const(0), "int8"),
            (Dequantize("float32", BindingParam("s_1", "float32", (Const(3),)), BindingParam("z_1", "int32", (Const(3),)), 1, "affine"),
             Transpose((1, 0, 2))),
            TensorSpec((Const(3), Const(2), Const(4)), dense_strides((Const(3), Const(2), Const(4))), Const(0), "float32"),
            "h2d",
        )

        def fn2(q, s, z):
            return QD.dequantize_per_channel(q, s, z, 1, -128, 127, torch.int8).permute(1, 0, 2).contiguous()

        gm = capture(fn2, torch.zeros(2, 3, 4, dtype=torch.int8), torch.ones(3), torch.zeros(3, dtype=torch.int32))
        members = [n.name for n in gm.graph.nodes if n.op == "call_function"]
        backend._importer = cast_candidate(gm, fn2, recipe, (), members, parameters=("s_1", "z_1"))
        compiled_fn = backend(gm, None)
        node = next(n for n in compiled_fn.rewritten.graph.nodes if n.target is torch.ops.reloc_torch.typed_transfer.default)
        assert [p.name for p in node.args[1]] == ["s_1", "z_1"] and node.args[7] == torch.float32
        q = torch.randint(-128, 128, (2, 3, 4), dtype=torch.int8)
        s = torch.tensor([0.5, 1.0, 2.0])
        z = torch.tensor([-128, 0, 127], dtype=torch.int32)
        with torch.no_grad():
            actual = compiled_fn(q, s, z)
        expected = fn2(q, s, z)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), "0 ulp against the audited PyTorch dequantize"
        z2 = torch.tensor([5, 5, 5], dtype=torch.int32)
        with torch.no_grad():
            again = compiled_fn(q, s, z2)
        assert torch.equal(again.view(torch.int32), fn2(q, s, z2).view(torch.int32))
        assert not torch.equal(again, actual)
        assert runtime.executions == 3
        # An invalid runtime value fails preflight and falls back to the
        # original region with the reason, without a launch.
        with torch.no_grad():
            bad = compiled_fn(q, torch.tensor([0.5, 0.0, 2.0]), z)
        assert torch.equal(bad.view(torch.int32), fn2(q, torch.tensor([0.5, 0.0, 2.0]), z).view(torch.int32))
        assert runtime.executions == 3 and backend.stats()["fallbacks"] == {"bind_error": 1}
    finally:
        backend.close()


def test_typed_op_schema_is_explicit_and_registered_once():
    from reloc_torch import ops

    assert str(torch.ops.reloc_torch.typed_transfer.default._schema) == (
        "reloc_torch::typed_transfer(Tensor src, Tensor[] parameters, str handle, SymInt[] symbols, "
        "SymInt[] out_shape, SymInt[] out_strides, Device device, ScalarType dtype) -> Tensor"
    )
    assert ops.TYPED_OP is torch.ops.reloc_torch.typed_transfer.default
    import importlib

    importlib.reload(ops)
    assert torch.ops.reloc_torch.typed_transfer.default is ops.TYPED_OP


#===----------------------------------------------------------------------===#
# Real GPU rows through torch.compile and the production adapter.
#===----------------------------------------------------------------------===#

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compiled_layout_plus_cast_and_dequantize_execute_on_cuda(compiler):
    from reloc_torch.backend import RelocBackend
    from reloc_torch.runtime import TransportAdapter

    backend = RelocBackend(compiler=compiler, runtime=TransportAdapter())
    try:
        def cast_fn(x):
            return x.to("cuda", torch.float16).t().contiguous()

        compiled_fn = torch.compile(cast_fn, backend=backend, dynamic=True)
        with torch.no_grad():
            x = torch.randn(6, 8) * 100
            actual = compiled_fn(x)
            expected = cast_fn(x)
        assert actual.dtype == torch.float16 and actual.device.type == "cuda"
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        stats = backend.stats()
        assert stats["typed_executions"] >= 1 and stats["replaced_regions"] >= 1

        def dequant_fn(q, s, z):
            return QD.dequantize_per_channel(q, s, z, 1, -128, 127, torch.int8).permute(1, 0, 2).contiguous().to("cuda")

        compiled_fn = torch.compile(dequant_fn, backend=backend, dynamic=False)
        q = torch.randint(-128, 128, (2, 3, 4), dtype=torch.int8)
        s = torch.tensor([0.5, 1.0, 2.0])
        z = torch.tensor([-128, 0, 127], dtype=torch.int32)
        with torch.no_grad():
            actual = compiled_fn(q, s, z)
            expected = dequant_fn(q, s, z)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        assert backend.stats()["typed_executions"] >= 2
        assert set(backend.stats()["dispatches"]) <= {"cpu_reference"}
    finally:
        backend.close()


def test_dynamo_style_operator_packets_resolve_to_the_audited_overloads():
    """Dynamo records ``OpOverloadPacket`` targets (``make_fx`` records
    overloads); both forms must import identically."""
    from reloc_torch.recipe import BindingParam, InlineParam

    def per_channel(q, s, z):
        return QD.dequantize_per_channel(q, s, z, 1, -128, 127, torch.int8).permute(1, 0, 2).contiguous().to("cuda")

    gm = torch.fx.symbolic_trace(per_channel)
    assert any(n.target is QD.dequantize_per_channel for n in gm.graph.nodes)
    inputs = (torch.zeros(2, 3, 4, dtype=torch.int8), torch.ones(3), torch.zeros(3, dtype=torch.int32))
    report = importer().import_graph(gm, inputs)
    assert not report.exclusions, report.exclusions
    candidate, = report.candidates
    assert candidate.parameters == ("s", "z")
    assert isinstance(candidate.recipe.operations[0].scale, BindingParam)

    def per_tensor_scalar(q):
        return QD.dequantize_per_tensor(q, 0.3, 5, -128, 127, torch.int8).to("cuda")

    candidate, = importer().import_graph(torch.fx.symbolic_trace(per_tensor_scalar),
                                         (torch.zeros(5, dtype=torch.int8),)).candidates
    assert candidate.recipe.operations[0].scale == InlineParam("float32", (), (f32bits(0.3),))

    def per_tensor_tensor(q, s, z):
        return QD.dequantize_per_tensor(q, s, z, -128, 127, torch.int8).to("cuda")

    candidate, = importer().import_graph(torch.fx.symbolic_trace(per_tensor_tensor),
                                         (torch.zeros(5, dtype=torch.int8), torch.tensor(0.3), torch.tensor(5))).candidates
    assert candidate.parameters == ("s", "z")

    def quantize_packet(x):
        return QD.quantize_per_tensor(x, 0.5, 0, -128, 127, torch.int8).to("cuda")

    report = importer().import_graph(torch.fx.symbolic_trace(quantize_packet), (torch.ones(8),))
    assert not report.candidates
    assert "quantize_semantics_unproved" in {e.reason for e in report.exclusions}

"""Stacked host transfers on real CUDA (torch.stack support): exact values through the
production transport adapter, one artifact across shapes, the default gate."""
import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
]

DTYPES = [torch.float32, torch.float16, torch.int8]


def make_inputs(count, rows, columns, dtype, seed=0):
    generator = torch.Generator().manual_seed(seed)
    if dtype == torch.int8:
        return [torch.randint(-128, 128, (rows, columns), generator=generator, dtype=torch.int8)
                for _ in range(count)]
    return [torch.randn(rows, columns, generator=generator).to(dtype) for _ in range(count)]


@pytest.fixture
def stack_backend(compiler, real_runtime):
    from reloc_torch.backend import RelocBackend

    backend = RelocBackend(compiler=compiler, runtime=real_runtime, min_stack_bytes=0)
    yield backend
    backend.close()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("count", [2, 4, 16])
def test_compiled_stack_transfer_is_exact_and_reuses_one_artifact(stack_backend, dtype, dim, count):
    def fn(*xs):
        return torch.stack(xs, dim).to("cuda")

    compiled = torch.compile(fn, backend=stack_backend, dynamic=True)
    with torch.no_grad():
        for rows in (64, 128):
            xs = make_inputs(count, rows, 96, dtype, seed=rows)
            actual = compiled(*xs)
            expected = torch.stack(xs, dim).cuda()
            assert torch.equal(actual, expected) and actual.stride() == expected.stride()
    stats = stack_backend.stats()
    assert stats["stacked_executions"] == 2 and stats["plan_compiles"] == 1
    assert not stats["fallbacks"]


def test_large_stacks_cross_pipeline_chunks(stack_backend):
    def fn(a, b, c, d):
        return torch.stack([a, b, c, d], 1).to("cuda")

    xs = make_inputs(4, 1024, 1024, torch.float32)  # 16 MiB: several chunks
    with torch.no_grad():
        actual = torch.compile(fn, backend=stack_backend, dynamic=True)(*xs)
    expected = torch.stack(xs, 1).cuda()
    assert torch.equal(actual, expected) and actual.stride() == expected.stride()
    assert stack_backend.stats()["stacked_executions"] == 1


CHAINS = [
    lambda a, b: torch.stack([a, b], 0).permute(2, 0, 1).contiguous().to("cuda"),
    lambda a, b: torch.nn.functional.pad(torch.stack([a, b], 1), (1, 2)).to("cuda"),
    lambda a, b: torch.nn.functional.pad(torch.stack([a, b], -1), (1, 2)).to("cuda"),  # pads the stack axis
    lambda a, b: torch.stack([a, b], 1).to("cuda").transpose(0, 1).contiguous(),
]


@pytest.mark.parametrize("fn", CHAINS)
def test_layout_chains_around_the_stack_are_exact(stack_backend, fn):
    a, b = make_inputs(2, 48, 80, torch.float32)
    with torch.no_grad():
        actual = torch.compile(fn, backend=stack_backend, dynamic=True)(a, b)
        expected = fn(a, b)
    assert torch.equal(actual, expected) and actual.stride() == expected.stride()
    assert stack_backend.stats()["stacked_executions"] == 1


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_stack_to_a_non_current_device(stack_backend):
    compiled = torch.compile(lambda a, b: torch.stack([a, b], 1).to("cuda:1"), backend=stack_backend, dynamic=True)
    a, b = make_inputs(2, 64, 64, torch.float32)
    with torch.no_grad():
        actual = compiled(a, b)
    expected = torch.stack([a, b], 1)
    assert actual.device == torch.device("cuda", 1)
    assert torch.equal(actual.cpu(), expected) and actual.stride() == expected.stride()
    assert stack_backend.stats()["stacked_executions"] == 1


def test_result_is_ordered_on_a_non_default_caller_stream(stack_backend):
    compiled = torch.compile(lambda a, b: torch.stack([a, b], 1).to("cuda"), backend=stack_backend, dynamic=True)
    a, b = make_inputs(2, 256, 256, torch.float32)
    with torch.no_grad():
        # Warm up: Dynamo's first-call tracing/compiling alone takes tens of
        # milliseconds, which would dwarf the sleep below and make the
        # ordering assertion pass trivially regardless of correctness.
        compiled(a, b)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    sleep_retired = torch.cuda.Event()
    with torch.no_grad(), torch.cuda.stream(stream):
        torch.cuda._sleep(50_000_000)  # queued work ahead of the transfer
        sleep_retired.record(stream)
        result = compiled(a, b)
        # The transfer's own dispatch blocks the host until its copy
        # completes; if it is correctly ordered after whatever the caller
        # already queued on this stream (same-stream FIFO order), the sleep
        # recorded ahead of it on the same stream must have retired too by
        # the time the call returns. A transfer issued on an unrelated idle
        # stream would return in microseconds, long before the 50 ms sleep,
        # and this assertion is what would catch that (values alone cannot:
        # the sleep has no data dependency on the transfer).
        assert sleep_retired.query()
        doubled = result * 2
    stream.synchronize()
    assert torch.equal(doubled.cpu(), torch.stack([a, b], 1) * 2)
    assert stack_backend.stats()["stacked_executions"] == 2  # the warmup call and the timed one


def test_default_gate_keeps_small_stacks_in_pytorch(compiler, real_runtime):
    from reloc_torch.backend import RelocBackend

    backend = RelocBackend(compiler=compiler, runtime=real_runtime)
    compiled = torch.compile(lambda a, b: torch.stack([a, b], 1).to("cuda"), backend=backend, dynamic=True)
    a, b = make_inputs(2, 64, 64, torch.float32)
    with torch.no_grad():
        assert torch.equal(compiled(a, b), torch.stack([a, b], 1).cuda())
    stats = backend.stats()
    assert stats["stacked_executions"] == 0
    gated = stats["fallbacks"].get("below_stack_threshold", 0) + stats["exclusions"].get("below_stack_threshold", 0)
    assert gated == 1
    backend.close()


def test_copy_then_stack_is_unchanged(stack_backend):
    compiled = torch.compile(lambda a, b: torch.stack([a.to("cuda"), b.to("cuda")], 1),
                             backend=stack_backend, dynamic=True)
    a, b = make_inputs(2, 64, 64, torch.float32)
    with torch.no_grad():
        assert torch.equal(compiled(a, b), torch.stack([a, b], 1).cuda())
    stats = stack_backend.stats()
    assert stats["stacked_executions"] == 0 and stats["runtime_executions"] == 2


def test_opcheck_with_a_cuda_destination(compiler, real_runtime):
    from conftest import make_entry, stacked_recipe
    from reloc_torch import ops
    from reloc_torch.cache import REGISTRY

    entry = make_entry(compiler.compile(stacked_recipe(count=3, dim=1)), real_runtime,
                       lambda *xs: torch.stack(xs, 1).cuda())
    registration = REGISTRY.register(entry)
    try:
        with torch.inference_mode():
            xs = [torch.randn(4, 5) for _ in range(3)]
        values = {"s0": 4, "s1": 5}
        args = (xs, registration.handle, [values[n] for n in entry.compiled.symbols], [4, 3, 5], [15, 5, 1],
                torch.device("cuda", torch.cuda.current_device()))
        assert set(torch.library.opcheck(ops.STACKED_OP, args).values()) == {"SUCCESS"}
        assert torch.equal(ops.STACKED_OP(*args).cpu(), torch.stack(xs, 1))
    finally:
        registration.release()

"""CUDA FP32 -> FP16 -> FP16 consumer, with one explicit rounding per source op."""
import ctypes
import time

import triton
import triton.language as tl


@triton.jit(do_not_specialize=['n', 'increment'])
def _cast_add(src, dst, n, increment, PASSES: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = i < n
    raw = tl.load(src + i, mask=active, other=0)
    value = raw.to(tl.float16, fp_downcast_rounding='rtne')
    step = tl.full((BLOCK,), increment, tl.float16)
    for _ in tl.static_range(PASSES):
        # Each eager FP16 add has a distinct round-to-nearest-even boundary.
        value = tl.inline_asm_elementwise('add.rn.f16 $0, $1, $2;', constraints='=h,h,h',
                                          args=[value, step], dtype=tl.float16, is_pure=True, pack=1)
    tl.store(dst + i, value, mask=active)


def cast_add(src, dst, passes, increment):
    return _cast_add[(triton.cdiv(src.numel(), 256),)](src, dst, src.numel(), increment, passes, 256)


def launch_nodes(t, source, out, passes):
    """Count actual CUDA graph nodes for one invocation; replay the captured graph."""
    graph = t.cuda.CUDAGraph(keep_graph=True)
    with t.cuda.graph(graph):
        cast_add(source, out, passes, 1)
    driver = ctypes.CDLL('libcuda.so.1')
    graph_handle = ctypes.c_void_p(graph.raw_cuda_graph())
    count = ctypes.c_size_t()
    driver.cuGraphGetNodes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.POINTER(ctypes.c_size_t)]
    driver.cuGraphNodeGetType.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    assert driver.cuGraphGetNodes(graph_handle, None, ctypes.byref(count)) == 0
    nodes = (ctypes.c_void_p * count.value)()
    assert driver.cuGraphGetNodes(graph_handle, nodes, ctypes.byref(count)) == 0
    node_types = []
    for node in nodes:
        kind = ctypes.c_int()
        assert driver.cuGraphNodeGetType(node, ctypes.byref(kind)) == 0
        node_types.append(kind.value)
    if node_types != [0]:  # CU_GRAPH_NODE_TYPE_KERNEL == 0
        raise AssertionError(f'expected one CUDA kernel node, found {node_types}')
    graph.instantiate()
    graph.replay()
    t.cuda.synchronize()
    return dict(kernel_nodes=1, total_nodes=len(node_types), replayed=True)


def qualify(t):
    """Check all adjacent finite binary16 round-to-nearest boundaries against eager."""
    start = time.perf_counter()
    fp16 = t.arange(65536, dtype=t.int32).to(t.uint16).view(t.float16).float()
    finite = t.sort(fp16[t.isfinite(fp16)]).values
    mid = (finite[:-1] + finite[1:]) * 0.5
    edge = t.tensor([-0.0, 0.0, 65519., 65520., -65519., -65520., 1e35, -1e35, 2.0**-149])
    values = t.cat((finite, t.nextafter(mid, t.full_like(mid, -float('inf'))), mid,
                    t.nextafter(mid, t.full_like(mid, float('inf'))), edge))
    source = values.cuda()
    out = t.empty_like(source, dtype=t.float16)
    rows = []
    for passes in (1, 3):
        for step in (1, 2):
            device_ref, cpu_ref = source.half(), values.half()
            for _ in range(passes):
                device_ref.add_(step)
                cpu_ref.add_(step)
            t.cuda.synchronize()
            begin = time.perf_counter()
            compiled = cast_add(source, out, passes, step)
            t.cuda.synchronize()
            first_launch_ms = (time.perf_counter() - begin) * 1000
            for ref in (cpu_ref, device_ref.cpu()):
                if not t.equal(out.cpu().view(t.int16), ref.view(t.int16)):
                    raise AssertionError(f'fused bit mismatch for {passes} additions of {step}')
            collapsed = (values + step * passes).half().view(t.int16)
            witnesses = int((collapsed != cpu_ref.view(t.int16)).sum())
            if witnesses == 0:
                raise AssertionError('qualification must distinguish omitted round boundaries')
            row = dict(passes=passes, increment=step, samples=len(values), eager_cpu_and_cuda_bits=True,
                       collapsed_rounding_mismatches=witnesses, compile_and_first_launch_ms=first_launch_ms,
                       kernel=compiled.name)
            if step == 1:
                row['graph'] = launch_nodes(t, source, out, passes)
                if not t.equal(out.cpu().view(t.int16), cpu_ref.view(t.int16)):
                    raise AssertionError('graph replay changed the result')
            rows.append(row)
    return dict(triton=triton.__version__, cache='fresh per-process temporary directory',
                qualification_ms=(time.perf_counter()-start)*1000, cases=rows)

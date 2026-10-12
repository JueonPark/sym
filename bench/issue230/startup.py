#!/usr/bin/env python3
"""One fresh process, one path/case; complete startup and 100 live transfers.

Predeclared cases: small cast (32 KiB wire), DLRM transpose/cast (6.5 MiB),
KV restore transpose (8 MiB). All paths get eight CPU threads, <=64 MiB
native/host staging, fresh outputs, and one explicit warm call when prepared.
No whole-model performance claim is made by this completed-transfer benchmark.
"""
import time
SCRIPT_START = time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys


def stamp():
    return time.perf_counter()


def milliseconds(start):
    return (stamp() - start) * 1000


def make_recipe(case):
    from reloc_torch.recipe import Recipe, TensorSpec, Cast, Transpose
    from reloc_torch.symbolic import Const, Symbol, dense_strides
    spec = lambda shape, dtype: TensorSpec(shape, dense_strides(shape), Const(0), dtype)
    n = Symbol('N')
    if case == 'small':
        src = dst = (n, Const(128))
        ops = (Cast('float16', 'ieee_rne'),)
    elif case == 'dlrm':
        src, dst = (Const(26), n, Const(64)), (n, Const(26), Const(64))
        ops = (Transpose((1, 0, 2)), Cast('float16', 'ieee_rne'))
    else:
        src, dst = (n, Const(16), Const(64)), (Const(16), n, Const(64))
        ops = (Transpose((1, 0, 2)),)
    return Recipe(spec(src, 'float32'), ops,
                  spec(dst, 'float32' if case == 'kv' else 'float16'), 'h2d')


def main(args):
    if args.prime:
        start = stamp()
        from reloc_torch import CompilerClient
        recipe = make_recipe(args.case)
        compiler = CompilerClient.from_environment(cache_dir=args.cache)
        compiler.compile(recipe)
        assert 'torch' not in sys.modules
        args.output.write_text(json.dumps(dict(prime_ms=milliseconds(start), cache=compiler.cache_stats(),
            script_start_perf=SCRIPT_START, ready_from_script_ms=milliseconds(SCRIPT_START))))
        return
    start = stamp()
    import torch
    if args.path.startswith('sym'):
        from reloc_torch import CompilerClient, PreparedTransfers, TransferResources
        from reloc_torch import dispatch, transport
        recipe = make_recipe(args.case)
    imports_ms = milliseconds(start)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(230)
    shape = {'small': (128, 128), 'dlrm': (26, 2048, 64), 'kv': (2048, 16, 64)}[args.case]
    dtype = torch.float32 if args.case == 'kv' else torch.float16
    transform = (lambda x: x) if args.case == 'small' else (lambda x: x.transpose(0, 1))
    # CPU-only input/expected setup is common and excluded from serving samples.
    start = stamp()
    source = torch.randn(shape)
    expected = transform(source).to(dtype).contiguous()
    fixture_ms = milliseconds(start)
    initialized_ms = 0.
    if args.initialized:
        start = stamp()
        torch.cuda.init()
        # Identical unrelated context/allocator initialization for every path.
        torch.empty(1, device='cuda:0')
        torch.cuda.synchronize()
        initialized_ms = milliseconds(start)
    rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    owner = session = compiled = wire = host = None
    options = dict(n_buffers=2, n_streams=1, gather_threads=8, pinning='pinned')
    limits = dict(max_retained_bytes=64 << 20, max_live_staging_bytes=64 << 20,
                  max_background_workers=7, max_typed_retained_bytes=64 << 20,
                  max_typed_live_bytes=64 << 20, max_typed_background_workers=7)
    phases = dict(compiler_ms=0., cuda_init_ms=0., resource_warm_ms=0.)

    def setup():
        nonlocal session, owner, compiled, wire, host
        if args.path.startswith('sym'):
            if args.path == 'sym_lazy':
                start = stamp()
                compiled = CompilerClient.from_environment().compile(recipe)
                phases['compiler_ms'] = milliseconds(start)
                owner = TransferResources(**limits)
            else:
                session = PreparedTransfers({'case': recipe},
                    compiler=CompilerClient.from_environment(
                        cache_dir=args.cache if args.path == 'sym_cached' else None),
                    resource_options=limits, transfer_options=options,
                    max_output_bytes=64 << 20)
        else:
            def host_into(x, out):
                out.copy_(transform(x))
            host = torch.compile(host_into, fullgraph=True) if args.path.startswith('inductor') else host_into

    def transfer():
        nonlocal wire
        if session is not None:
            return session.transfer('case', source, 'cuda:0')
        if args.path == 'sym_lazy':
            if recipe.typed:
                req = dispatch.prepare_typed_transfer(compiled, source, 'cuda:0', threads=8)
                return dispatch.execute_typed_transfer(req, resources=owner, **options).tensor
            req = transport.prepare_transfer(compiled, source, 'cuda:0')
            return transport.execute_transfer(req, resources=owner, **options)
        if wire is None:
            wire = torch.empty(expected.shape, dtype=dtype, pin_memory=True)
        host(source, wire)
        return wire.to('cuda:0', non_blocking=True)

    def initialize_cuda():
        start = stamp()
        torch.cuda.init()
        torch.cuda.current_stream('cuda:0').synchronize()
        phases['cuda_init_ms'] += milliseconds(start)

    def completed(call):
        start = stamp()
        output = call()
        torch.cuda.synchronize()
        return output, milliseconds(start)

    def memory():
        return dict(rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    torch_gpu_allocated=torch.cuda.memory_allocated(),
                    torch_gpu_reserved=torch.cuda.memory_reserved(),
                    torch_gpu_peak_allocated=torch.cuda.max_memory_allocated(),
                    native=session.stats()['resources'] if session else owner.stats() if owner else None,
                    torch_wire_bytes=0 if wire is None else wire.numel() * wire.element_size())

    prepared = args.path.endswith('prepared') or args.path == 'sym_cached'
    prep_start = stamp()
    warm_output = None
    if prepared:
        setup()
        if session:
            session.initialize_cuda('cuda:0')
            session.warm('case', source, 'cuda:0')
        else:
            initialize_cuda()
            warm_output, phases['resource_warm_ms'] = completed(transfer)
            del warm_output
    preparation_ms = milliseconds(prep_start) if prepared else 0.
    prep_memory = memory() if prepared else None
    readiness_ms = milliseconds(SCRIPT_START)
    samples = []
    first_output = None
    for i in range(args.samples):
        source.neg_()  # fresh live data; expected alternates bit-exactly
        def call():
            if i == 0 and not prepared:
                setup()
                initialize_cuda()
            return transfer()
        output, elapsed = completed(call)
        samples.append(elapsed)
        oracle = -expected if i % 2 == 0 else expected
        assert torch.equal(output.cpu(), oracle)
        if i == 0:
            first_output = output
        else:
            assert output.data_ptr() != first_output.data_ptr()
            assert torch.equal(first_output.cpu(), -expected)
        del output
    pre_close_memory = memory()
    diagnostics = session.stats() if session else None
    del first_output
    start = stamp()
    if session:
        session.close()
    if owner:
        owner.close()
    wire = host = compiled = None
    torch.cuda.synchronize()
    retirement_ms = milliseconds(start)
    after_close = memory()
    files = [Path(os.environ['SYM_RELOC_EXPORT'])]
    files += [Path(line.split()[-1]) for line in Path('/proc/self/maps').read_text().splitlines()
              if 'libreloc_runtime.so' in line]
    import reloc_torch
    files += list(Path(reloc_torch.__file__).parent.glob('*.py'))
    result = dict(path=args.path, case=args.case, initialized=args.initialized,
        script_start_perf=SCRIPT_START, imports_ms=imports_ms, fixture_ms=fixture_ms, initialized_process_setup_ms=initialized_ms,
        readiness_from_script_ms=readiness_ms, preparation_ms=preparation_ms,
        phases=phases, samples_ms=samples, retirement_ms=retirement_ms,
        total_horizons_ms={str(n):preparation_ms + sum(samples[:n]) + retirement_ms
                           for n in (1, 10, 100) if n <= len(samples)},
        diagnostics=diagnostics, preparation_memory=prep_memory, pre_close_memory=pre_close_memory,
        after_close_memory=after_close, rss_start=rss_start,
        checks=2 * args.samples - 1, independent_outputs=True,
        metadata=dict(torch=torch.__version__, cuda=torch.version.cuda, python=sys.version,
                      affinity=sorted(os.sched_getaffinity(0)), threads=torch.get_num_threads(),
                      interop_threads=torch.get_num_interop_threads(),
                      file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(files)}))
    args.output.write_text(json.dumps(result, separators=(',', ':'))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', required=True)
    parser.add_argument('--case', choices=['small','dlrm','kv'], required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--prime', action='store_true')
    parser.add_argument('--initialized', action='store_true')
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

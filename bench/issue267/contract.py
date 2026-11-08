"""Frozen tuning, held-out and consumer grids; no Torch dependency."""

FAMILIES = ('roundtrip1', 'roundtrip2', 'roundtrip4', 'rank3_chain',
            'dequant_transpose', 'padded_roundtrip')
ARMS = ('sym_unfused', 'sym_folded', 'manual_fused', 'torch_eager', 'torch_inductor')
AFFINITY = (4, 5, 6, 7, 20, 21, 22, 23)
SAMPLES = 30
WARMUP = 5
ROUNDS = 3


def case_key(case):
    return f"{case['family']}_n{case['n']}_m{case['m']}_o{case['outer']}"


def cases(suite):
    if suite == 'main':
        values = [(family, n, 128, 8) for family in FAMILIES for n in (33, 512, 8192)]
    elif suite == 'heldout':
        # Frozen before kernel tuning. Different chain lengths, dimensions,
        # outer extents, and unsupported stage families are retained.
        values = [('roundtrip7', 257, 67, 8),
                  ('dequant_transpose', 769, 97, 8),
                  ('padded_roundtrip', 257, 67, 8),
                  ('rank3_chain', 257, 67, 5),
                  ('rank3_chain', 1, 67, 3),
                  ('bf16_roundtrip', 257, 67, 8),
                  ('dequant_roundtrip', 769, 97, 8),
                  ('quant_roundtrip', 257, 67, 8)]
    elif suite == 'consumer':
        # One transferred weight matrix consumed by linear + GELU. This is
        # a scoped consumer benchmark, not model/server throughput.
        values = [('dequant_transpose', 1024, 512, 8),
                  ('padded_roundtrip', 1024, 256, 8)]
    else:
        raise ValueError('unknown suite')
    return [dict(family=f, n=n, m=m, outer=o) for f, n, m, o in values]


def arms(case):
    manual = case['family'] in FAMILIES or case['family'].startswith('roundtrip')
    # Preserve the original manual control's fixed rank3 extent, rather than
    # inventing a manual implementation for each new chain/shape.
    manual &= case['family'] != 'rank3_chain' or case['outer'] == 8
    return list(ARMS if manual else tuple(a for a in ARMS if a != 'manual_fused'))


def modes(suite):
    return ['completed_consumer'] if suite == 'consumer' else ['producer', 'completed_h2d']


def jobs(rounds=ROUNDS):
    result = []
    for round_number in range(1, rounds + 1):
        for suite in ('main', 'heldout', 'consumer'):
            for threads in (1, 8):
                # Alternate old/new process order to reduce monotonic drift.
                runtimes = ('baseline', 'candidate') if round_number % 2 else ('candidate', 'baseline')
                for runtime in runtimes:
                    result.append(dict(id=f'{suite}-r{round_number}-t{threads}-{runtime}',
                                       suite=suite, round=round_number, threads=threads,
                                       runtime=runtime))
    return result

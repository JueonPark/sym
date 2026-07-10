"""The randomized oracle harness (issue #46 acceptance): >=100 seeded
cases per run; every case byte-compares pyreloc.relocate against the numpy
replay of the corpus recipe, then round-trips through relocate_inverse.
Strategies rotate across cases so all CPU executors are exercised."""
import zlib

import numpy as np
import pytest

import oracle
from conftest import SEED, corpus_entries

pyreloc = pytest.importorskip("pyreloc")
from pyreloc.torch_interop import as_ptr

ENTRIES = corpus_entries()
# >=100 randomized cases per run regardless of corpus size.
REPEATS = max(7, -(-100 // max(1, len(ENTRIES))))
STRATEGIES = ("auto", "single_thread_simd", "multi_thread_tiled",
              "chunked_pipeline")

CASES = [(name, blob, meta, k)
         for name, blob, meta in ENTRIES for k in range(REPEATS)]


def _case_rng(name, k):
    return np.random.default_rng(zlib.crc32(f"{SEED}:{name}:{k}".encode()))


@pytest.mark.parametrize("name,blob,meta,k", CASES,
                         ids=[f"{c[0]}-b{c[3]}" for c in CASES])
def test_relocate_matches_numpy_and_round_trips(name, blob, meta, k):
    rng = _case_rng(name, k)
    binding = oracle.sample_binding(meta["symbols"], rng)
    strategy = STRATEGIES[int(rng.integers(0, len(STRATEGIES)))]
    bound = pyreloc.bind(pyreloc.load_plan(blob), binding, strategy=strategy)

    src = oracle.random_src(meta, binding, rng)
    expected = oracle.reference(meta, binding, src)
    assert expected.nbytes == bound.total_bytes, (
        "oracle/plan disagree on dst size -- recipe or fold bug")
    # Corpus sources are dense tensors: the strided span equals the array.
    assert bound.min_src_bytes == src.nbytes

    dst = np.zeros(bound.total_bytes, dtype=np.uint8)
    pyreloc.relocate(bound, *as_ptr(src), *as_ptr(dst))
    got, want = dst.tobytes(), expected.tobytes()
    if got != want:
        pytest.fail(oracle.mismatch_report(meta, binding, strategy, got,
                                           want), pytrace=False)

    back = np.zeros(bound.min_src_bytes, dtype=np.uint8)
    pyreloc.relocate_inverse(bound, *as_ptr(dst), *as_ptr(back))
    if back.tobytes() != src.tobytes():
        pytest.fail(oracle.mismatch_report(
            meta, binding, strategy + " (inverse round-trip)",
            back.tobytes(), src.tobytes()), pytrace=False)


def test_case_count_meets_acceptance():
    assert len(CASES) >= 100, f"only {len(CASES)} randomized cases"

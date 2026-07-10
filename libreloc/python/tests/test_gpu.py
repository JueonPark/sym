"""GPU leg (issue #46 acceptance): torch pinned-host -> GPU h2d matches
the CPU reference, and d2h round-trips back. Marked gpu; auto-skips in CI
(no CUDA build / no GPU / no torch)."""
import numpy as np
import pytest

pyreloc = pytest.importorskip("pyreloc")
torch = pytest.importorskip("torch")

from conftest import corpus_entries
from pyreloc.torch_interop import as_ptr

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not pyreloc.cuda_enabled,
                       reason="pyreloc built without RELOC_ENABLE_CUDA"),
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="no CUDA device"),
]


def _entry(name):
    for n, blob, meta in corpus_entries():
        if n == name:
            return blob, meta
    pytest.fail(f"corpus entry {name!r} missing")


@pytest.mark.parametrize("n_buffers", [1, 2, 4])
def test_pinned_h2d_matches_cpu_and_round_trips(n_buffers):
    blob, _ = _entry("transpose_2d_f32")
    bound = pyreloc.bind(pyreloc.load_plan(blob), {})

    src = torch.randn(32, 64, dtype=torch.float32).pin_memory()
    dev = torch.zeros(bound.total_bytes, dtype=torch.uint8, device="cuda")
    pyreloc.h2d(bound, *as_ptr(src), *as_ptr(dev), n_buffers=n_buffers)

    ref = np.zeros(bound.total_bytes, dtype=np.uint8)
    pyreloc.relocate(bound, src.data_ptr(),
                     src.numel() * src.element_size(), *as_ptr(ref))
    assert dev.cpu().numpy().tobytes() == ref.tobytes()

    back = torch.zeros(32 * 64, dtype=torch.float32).pin_memory()
    pyreloc.d2h(bound, *as_ptr(dev), *as_ptr(back), n_buffers=n_buffers)
    assert torch.equal(back.view(32, 64), src)

"""V3 (issue #97): cost-model bindings — the prediction test's foundation.
The Python surface is a thin veneer; all arithmetic is the C++ component."""
import pytest

pyreloc = pytest.importorskip("pyreloc")

CAL = """# costmodel calibration v0
# machine: testbox
pcie.h2d_gbps 10
cpu.t8.contiguous.contig_read_gbps 20
cpu.t8.contiguous.convert_f32_f16_gbps 20
cpu.t8.contiguous.quantize_pack_gbps 20
cpu.t8.contiguous.pack_s8_s4_gbps 20
hbm.bw_gbps 100
hbm.m.contiguous 1
overhead.a_ms 0.5
overhead.b_ms 0.1
"""


@pytest.fixture()
def cal(tmp_path):
    p = tmp_path / "test.cal"
    p.write_text(CAL)
    return pyreloc.load_calibration(str(p))


def test_load_calibration_machine(cal):
    assert cal.machine == "testbox"


def test_load_calibration_rejects_bad_file(tmp_path):
    p = tmp_path / "bad.cal"
    p.write_text("no header\n")
    with pytest.raises(ValueError):
        pyreloc.load_calibration(str(p))


def test_predict_matches_cpp_arithmetic(cal):
    # Mirrors CostModelPathCosts: S=1e9, r=0.25 -> A 50.5ms, B 100.1ms.
    d = pyreloc.predict(cal, pattern="contiguous", src_bytes=10**9, r=0.25)
    assert d["method"] == "a"
    assert d["t_a_ms"] == pytest.approx(50.5)
    assert d["t_b_ms"] == pytest.approx(100.1)
    assert d["threshold_bytes"] == pytest.approx(8e6, rel=1e-6)
    assert d["pattern"] == "contiguous"


def test_predict_prefold_arm(cal):
    d = pyreloc.predict(cal, pattern="contiguous", src_bytes=10**9, r=0.25,
                        n_reuse=16)
    assert d["method"] == "a_prefold"


def test_predict_missing_keys_raise(cal):
    with pytest.raises(ValueError):
        pyreloc.predict(cal, pattern="blocked", src_bytes=10**9, r=0.25)

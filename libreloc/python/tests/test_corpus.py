"""Every committed corpus entry must decode, match its JSON recipe, and
bind with a constraint-satisfying binding (fixed factor: deterministic)."""
import pytest

from conftest import corpus_entries

pyreloc = pytest.importorskip("pyreloc")

ENTRIES = corpus_entries()


def test_corpus_is_present():
    assert len(ENTRIES) >= 10, (
        "committed corpus missing/too small -- run "
        "libreloc/test/corpus/generate_corpus.py and commit its output")


@pytest.mark.parametrize("name,blob,meta", ENTRIES,
                         ids=[e[0] for e in ENTRIES])
def test_corpus_entry_decodes_and_binds(name, blob, meta):
    plan = pyreloc.load_plan(blob)
    assert sorted(plan.symbols) == sorted(meta["symbols"].keys())
    assert plan.element_size == meta["element_size"]
    binding = {s: c["multiple_of"] * c["min_factor"]
               for s, c in meta["symbols"].items()}
    bound = pyreloc.bind(plan, binding)
    assert bound.total_bytes > 0
    assert set(meta.keys()) >= {"name", "dtype", "element_size", "src_shape",
                                "ops", "symbols", "mlir"}

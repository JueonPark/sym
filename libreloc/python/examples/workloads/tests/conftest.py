"""pytest setup for the workload example tests.

CI does not collect this directory; run it explicitly with the CUDA build:

    export PYTHONPATH="$SYM_BUILD/python:$PWD/libreloc/python"
    export SYM_RELOC_EXPORT="$SYM_BUILD/sym/tools/sym-reloc-export"
    /tmp/sym-torch-cuda/bin/python -m pytest libreloc/python/examples/workloads/tests -q

CPU tests cover the helpers and the runner's accounting; tests marked ``gpu``
run the examples end to end and skip without CUDA.
"""
import pathlib
import sys

import pytest

WORKLOADS = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKLOADS))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA-enabled pyreloc build + GPU + torch")


@pytest.fixture(autouse=True)
def reset_dynamo():
    """Start every test from an empty Dynamo cache."""
    try:
        import torch
    except ImportError:
        yield
        return
    torch._dynamo.reset()
    yield

import dataclasses
import json
import os
import pathlib
import subprocess
import sys

import pytest
import pyreloc

from reloc_torch import check_version
from reloc_torch.compat import CompatibilityError, _validate_version
from reloc_torch.eligibility import classify
from reloc_torch.records import Eligibility, GraphRecord, TensorMetadata


def test_version_accepts_running_qualified_cpu_environment():
    check_version()


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"python_version": (3, 14, 6)}, "unsupported_python_version"),
        ({"python_releaselevel": "candidate"}, "unsupported_python_version"),
        ({"implementation": "PyPy"}, "unsupported_python_build"),
        ({"gil_enabled": False}, "unsupported_python_build"),
        ({"soabi": "cpython-314t-x86_64-linux-gnu"}, "unsupported_python_build"),
        ({"soabi": "cpython-314-aarch64-linux-gnu"}, "unsupported_python_build"),
        ({"torch_version": "2.14.0rc1+cpu"}, "unsupported_torch_version"),
        ({"torch_version": "2.14.1+cpu"}, "unsupported_torch_version"),
        ({"torch_version": "2.14.0+cu130", "torch_cuda": "13.0"}, "unsupported_torch_build"),
        ({"torch_version": "2.14.0+cu126", "torch_cuda": None}, "unsupported_torch_build"),
        ({"torch_version": "2.14.0+cpu", "torch_cuda": "12.6"}, "unsupported_torch_build"),
    ],
)
def test_version_rejects_unqualified_runtime_metadata(change, reason):
    qualified = {
        "python_version": (3, 14, 7),
        "python_releaselevel": "final",
        "implementation": "CPython",
        "gil_enabled": True,
        "soabi": "cpython-314-x86_64-linux-gnu",
        "torch_version": "2.14.0+cpu",
        "torch_cuda": None,
    }
    qualified.update(change)
    with pytest.raises(CompatibilityError) as caught:
        _validate_version(**qualified)
    assert caught.value.reason == reason


def test_reloc_torch_records_import_without_importing_torch():
    source = __file__.rsplit("/tests/", 1)[0]
    code = (
        f"import sys;sys.path.insert(0,{source!r});"
        "import reloc_torch;import json;"
        "print(json.dumps('torch' in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) is False


def test_core_pyreloc_import_does_not_import_torch():
    build_python = pathlib.Path(pyreloc.__file__).resolve().parents[1]
    source_python = pathlib.Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(build_python), str(source_python)))
    code = "import pyreloc,json,sys;print(json.dumps('torch' in sys.modules))"
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) is False


def test_records_are_frozen_plain_value_snapshots(h2d_record):
    with pytest.raises(dataclasses.FrozenInstanceError):
        h2d_record.phase = "output"
    symbolic = TensorMetadata(
        shape=("s0", "4*s1"), strides=("s1", 1), storage_offset="offset",
        dtype="float16", device_type="cpu", device_index=None,
        requires_grad=False, layout="strided", pinned=True,
        is_subclass=False, storage_capacity_bytes=None,
    )
    graph = GraphRecord(
        node_name="to", node_kind="call_function", target="aten._to_copy.default",
        input_nodes=("x",), user_nodes=("output",), tensor_metadata=symbolic,
        metadata_reason=None,
    )
    assert graph.tensor_metadata.shape == ("s0", "4*s1")
    assert dataclasses.asdict(graph)["tensor_metadata"]["dtype"] == "float16"


def test_classify_valid_layout_transfer(h2d_record):
    assert classify(h2d_record) == Eligibility(
        category="transfer", candidate=True,
        reason="needs_compile_and_runtime_check",
    )


def test_mutation_cannot_be_a_functional_candidate(h2d_record):
    changed = dataclasses.replace(h2d_record, mutates=True)
    assert classify(changed) == Eligibility("mutation", False, "mutation")


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("requires_grad", True, "requires_grad"),
        ("device_type", "mps", "unsupported_device"),
        ("dtype", "float64", "unsupported_dtype"),
        ("layout", "sparse_coo", "unsupported_layout"),
        ("shape", (), "unsupported_rank"),
        ("shape", (0, 6), "empty_tensor"),
        ("storage_offset", 1, "storage_offset"),
        ("is_subclass", True, "tensor_subclass"),
    ],
)
def test_classify_rejects_unsupported_source_metadata(
    h2d_record, field, value, reason
):
    source = dataclasses.replace(h2d_record.source, **{field: value})
    changed = dataclasses.replace(h2d_record, source=source)
    assert classify(changed) == Eligibility("transfer", False, reason)


def test_classify_rejects_nonblocking_transfer(h2d_record):
    changed = dataclasses.replace(h2d_record, non_blocking=True)
    assert classify(changed) == Eligibility(
        "transfer", False, "nonblocking_unavailable"
    )


@pytest.mark.parametrize(
    ("source_change", "destination_change", "reason"),
    [
        ({"strides": (1, 4)}, {}, "unsupported_layout"),
        ({}, {"strides": (1, 4)}, "unsupported_layout"),
        ({"strides": ("s1", 1)}, {}, "unsupported_layout"),
        ({"storage_capacity_bytes": 0}, {}, "empty_tensor"),
        ({}, {"storage_offset": 2}, "storage_offset"),
    ],
)
def test_classify_requires_qualified_source_and_destination_roots(
    h2d_record, source_change, destination_change, reason
):
    changed = dataclasses.replace(
        h2d_record,
        source=dataclasses.replace(h2d_record.source, **source_change),
        destination=dataclasses.replace(h2d_record.destination, **destination_change),
    )
    assert classify(changed) == Eligibility("transfer", False, reason)


def test_classify_rejects_cuda_to_cuda_as_unsupported_direction(h2d_record):
    changed = dataclasses.replace(
        h2d_record,
        source=dataclasses.replace(
            h2d_record.source, device_type="cuda", device_index=1
        ),
    )
    assert classify(changed) == Eligibility(
        "transfer", False, "unsupported_device"
    )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"source": {"device_index": 0}}, "unsupported_device"),
        ({"destination": {"device_index": None}}, "unsupported_device"),
        ({"aliases_source": True}, "unsupported_layout"),
    ],
)
def test_classify_requires_well_formed_devices_and_fresh_output(
    h2d_record, change, reason
):
    record_changes = {}
    for field, value in change.items():
        if field in {"source", "destination"}:
            record_changes[field] = dataclasses.replace(
                getattr(h2d_record, field), **value
            )
        else:
            record_changes[field] = value
    changed = dataclasses.replace(h2d_record, **record_changes)
    assert classify(changed) == Eligibility("transfer", False, reason)


def test_classify_does_not_enable_unknown_operator_from_metadata(h2d_record):
    changed = dataclasses.replace(h2d_record, operator="unknown.transfer")
    assert classify(changed) == Eligibility("transfer", False, "unsupported_operator")


def test_classify_distinguishes_copy_cast_layout_and_other(h2d_record):
    same_device = dataclasses.replace(
        h2d_record, destination=h2d_record.source
    )
    forced = classify(same_device)
    cast = classify(dataclasses.replace(
        same_device,
        destination=dataclasses.replace(same_device.destination, dtype="float16"),
    ))
    layout = classify(dataclasses.replace(
        same_device, operator="aten.view.default", aliases_source=True
    ))
    other = classify(dataclasses.replace(same_device, operator="aten.sin.default"))
    assert forced == Eligibility("same_device_copy", False, "same_device_copy")
    assert cast == Eligibility("cast", False, "typed_transform_unavailable")
    assert layout == Eligibility("layout", False, "layout_only")
    assert other == Eligibility("other", False, "unsupported_operator")

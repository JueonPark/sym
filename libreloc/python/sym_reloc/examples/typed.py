"""Analytical cast/quantization witnesses, independent of the source test oracle."""


def require_gpu_report(report):
    if (
        report.get("implementation") != "cpu_stages_cuda_stages@0"
        or report.get("wire_bytes", 0) <= 0
    ):
        raise RuntimeError("typed GPU witness did not execute the requested GPU stages")


def run(device):
    import numpy as np
    import pyreloc
    from reloc_torch import CompilerClient
    from reloc_torch.recipe import (
        Recipe,
        TensorSpec,
        Cast,
        Quantize,
        Dequantize,
        BindingParam,
        Transpose,
    )
    from reloc_torch.symbolic import Const, dense_strides

    def spec(shape, dtype):
        shape = tuple(Const(n) for n in shape)
        return TensorSpec(shape, dense_strides(shape), Const(0), dtype)

    def view(x):
        return pyreloc.BufferView(
            x.ctypes.data,
            x.nbytes,
            0,
            list(x.shape),
            [s // x.itemsize for s in x.strides],
            x.itemsize,
            "host",
        )

    scale = BindingParam("scale", "float32", ())
    source = np.array([-100, -0.75, -0.25, 0.25, 0.75, 100], dtype=np.float32)
    quantized = np.array([-128, -2, 0, 0, 2, 127], dtype=np.int8)
    cases = [
        (
            "cast",
            source.reshape(2, 3),
            (Transpose((1, 0)), Cast("float16", "ieee_rne")),
            source.reshape(2, 3).T.astype(np.float16),
            {},
        ),
        (
            "quantize",
            source,
            (Quantize("int8", scale, None, None, "symmetric_rne"),),
            quantized,
            {"scale": np.array(0.5, dtype=np.float32)},
        ),
        (
            "dequantize",
            quantized,
            (
                Dequantize(
                    "float32",
                    scale,
                    BindingParam("zero_point", "int32", ()),
                    None,
                    "affine",
                ),
            ),
            np.array([-64, -1, 0, 0, 1, 63.5], dtype=np.float32),
            {
                "scale": np.array(0.5, dtype=np.float32),
                "zero_point": np.array(0, dtype=np.int32),
            },
        ),
    ]
    reports = {}
    compiler = CompilerClient.from_environment()
    for name, source, ops, expected, params in cases:
        recipe = Recipe(
            spec(source.shape, source.dtype.name),
            ops,
            spec(expected.shape, expected.dtype.name),
            "h2d",
        )
        compiled = compiler.compile(recipe)
        if device == "cuda":
            import torch
            from reloc_torch.dispatch import (
                prepare_typed_transfer,
                execute_typed_transfer,
            )

            reports[name] = {}
            for implementation in ("", "cpu_stages_cuda_stages@0"):
                request = prepare_typed_transfer(
                    compiled,
                    torch.from_numpy(source.copy()),
                    "cuda",
                    parameters={
                        k: torch.from_numpy(v.copy()) for k, v in params.items()
                    },
                    policy="original_cpu" if not implementation else "auto",
                    implementation=implementation,
                )
                result = execute_typed_transfer(request)
                actual = result.tensor.cpu().numpy()
                np.testing.assert_array_equal(actual, expected)
                if implementation:
                    require_gpu_report(result.report)
                reports[name][implementation or "cpu_reference"] = dict(result.report)

        else:
            plan = pyreloc.load_typed_plan(compiled.plan_bytes)
            bound = pyreloc.bind_typed(
                plan,
                {},
                {
                    k: (v.dtype.name, list(v.shape), v.tobytes())
                    for k, v in params.items()
                },
            )
            actual = np.empty(expected.shape, dtype=expected.dtype)
            reports[name] = pyreloc.execute_dispatch(
                pyreloc.prepare_dispatch(
                    bound, view(source), view(actual), "h2d", policy="original_cpu"
                )
            )
        np.testing.assert_array_equal(actual, expected)
    return reports

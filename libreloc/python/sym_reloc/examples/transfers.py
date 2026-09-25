"""Prove compiled H2D and forward D2H execute on default and caller streams."""


def run(device):
    import torch
    from reloc_torch import RelocBackend

    if device != "cuda":
        raise ValueError("transfer witness requires CUDA")
    reports = []
    for nondefault in (False, True):
        for direction in ("h2d", "d2h"):
            backend = RelocBackend()
            stream = torch.cuda.Stream() if nondefault else torch.cuda.default_stream()
            target = "cuda" if direction == "h2d" else "cpu"

            def prepare(x):
                return x.reshape(x.shape[0] // 64, 64).t().contiguous().to(target)

            try:
                compiled = torch.compile(prepare, backend=backend, dynamic=True)
                with torch.no_grad(), torch.cuda.stream(stream):
                    for n in (128, 192, 256):
                        source = torch.arange(
                            n,
                            dtype=torch.float32,
                            device="cpu" if direction == "h2d" else "cuda",
                        )
                        actual = compiled(source)
                        torch.testing.assert_close(actual, prepare(source))
                        # Immediate consumer must observe the delivered values.
                        if actual.sum().item() != n * (n - 1) / 2:
                            raise RuntimeError("transfer stream ordering failed")
                stats = backend.stats()
                if stats["runtime_executions"] < 3:
                    raise RuntimeError(
                        f"{direction} silently fell back instead of executing Sym"
                    )
                reports.append(
                    dict(direction=direction, nondefault_stream=nondefault, stats=stats)
                )
            finally:
                backend.close()
    return reports

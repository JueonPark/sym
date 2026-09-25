"""Validate stable weight reuse and mutation invalidation through public APIs."""


def run(device):
    import torch
    from reloc_torch import RelocBackend, prepare_weights
    from reloc_torch.recipe import Recipe, TensorSpec, Transpose
    from reloc_torch.symbolic import Const, dense_strides
    from reloc_torch.prefold import typed_prefold_capability

    def spec(shape):
        shape = tuple(Const(n) for n in shape)
        return TensorSpec(shape, dense_strides(shape), Const(0), "float32")

    recipe = Recipe(spec((2, 3)), (Transpose((1, 0)),), spec((3, 2)), "h2d")
    module = torch.nn.Module()
    module.register_parameter(
        "weight", torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
    )
    module.register_buffer("buffer", torch.arange(6.0).reshape(2, 3) + 10)
    backend = RelocBackend()
    try:
        with (
            torch.no_grad(),
            prepare_weights(
                module,
                {"weight": recipe, "buffer": recipe},
                backend=backend,
                stable=True,
            ) as prepared,
        ):
            for name in ("weight", "buffer"):
                first = prepared.get(name, device=device)
                second = prepared.get(name, device=device)
                torch.testing.assert_close(
                    first.cpu(), getattr(module, name).t().contiguous()
                )
                torch.testing.assert_close(second, first)
                getattr(module, name).add_(1)
                torch.testing.assert_close(
                    prepared.get(name, device=device).cpu(),
                    getattr(module, name).t().contiguous(),
                )
        stats = backend.stats()
        if stats["weight_invalidations"] < 2:
            raise RuntimeError("weight mutation failed to invalidate preparation")
        if typed_prefold_capability() is None:
            raise RuntimeError("typed weight exclusion unexpectedly absent")
        return dict(
            preparations=stats["weight_preparations"],
            invalidations=stats["weight_invalidations"],
            typed_weight_exclusion=typed_prefold_capability(),
        )
    finally:
        backend.close()

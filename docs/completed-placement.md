# Completed-call placement

`RelocBackend(placement=PlacementPolicy(profile, context=...))` opts into
completed-cost selection. The default backend behavior remains unchanged.
Profiles are application-owned JSON measurements; execution never benchmarks
alternatives, reads source values to make a decision, or polls GPU utilization.

```python
from reloc_torch import PlacementPolicy, PlacementProfile, RelocBackend

policy = PlacementPolicy(
    PlacementProfile.load("placement.json"),
    context=lambda: {"cpu_load": "idle", "gpu_load": "idle", "overlap": "none"},
)
backend = RelocBackend(placement=policy, compute_backend="inductor")
compiled = torch.compile(model, backend=backend)
# Use no_grad/inference_mode for Inductor composition.
print(backend.stats()["placement"])
```

The context must describe the application's scheduling state. Use `unknown`
when the application cannot establish idle/busy status. An idle profile must
not be applied to known contention. `overlap` distinguishes `none`, `cpu_h2d`
and `consumer`; these are calibration contexts, not a promise that a blocking
call overlaps work. Async queues, grouped transfers and prepared weight
snapshots retain their own policies and are outside this blocking-call policy.

## Qualification and failures

The current qualified set is parameter-free dense permutations and C1
FP16/FP32 casts. Guards run before selection. Other recipes retain the existing
runtime, including quantization, padding, indexed gathers and reshape chains.
No alternate quantization or reduced arithmetic precision is introduced.

| Path | Meaning |
| --- | --- |
| `native` | Saved Torch region, dtype-preserving recipes only |
| `torch_cpu` | Torch layout/cast on CPU, transfer before/after as required |
| `torch_gpu` | Torch layout/cast on GPU, transfer before/after as required |
| `sym_cpu` | Existing Sym layout runtime / typed CPU reference |
| `sym_gpu` | Qualified typed CUDA stage row (rank at most eight) |

For typed graphs, the opaque native region can hide the wire dtype of a
combined device/dtype copy. Its controlled `torch_cpu` replay is the fallback
instead. CPU H2D transforms send destination bytes; GPU H2D transforms send
source bytes. D2H reverses these roles. Reports record actual wire bytes.
Torch alternatives use Torch's allocator; the profile execution fingerprint
records the Sym scratch/pinning configuration, but is not a hard limit on
Torch's allocator. Applications calibrating a memory budget must bound all
candidate intermediates, as the benchmark does.

FP32→FP16 CUDA narrowing implements round-to-nearest-even, including signed
zero, subnormals and overflow to infinity. Non-NaN values are bit exact against
the C1 CPU reference. C1 requires NaN-ness, not a particular payload/sign.
Outputs are fresh and contiguous; execution honors the caller's current CUDA
stream and completes before returning. A failure after selecting a path is
terminal, with no retry after a possible launch. Forced paths remain subject
to recipe and runtime capability checks.

## Profiles and conservative decisions

Profiles use `sym.completed-placement/1` and contain no executable code.
Every observation records family, shape, dense strides, direction (in family),
path, wire bytes, resource state, context and completed milliseconds.
`with policy.force("sym_gpu"):` supports explicit calibration and diagnosis.
Store the median of repeated **whole calls**, including allocations, frontend
binding and completion. Compile first; record compile/setup separately.

The hardware fingerprint covers GPU UUID/architecture, driver, CUDA/Torch,
PCIe capability, GPU NUMA locality, CPU model/ISA, live affinity and thread
settings, and native/extension/frontend hashes. Resource limits, pinning,
streams/buffers and completion semantics form a separate execution fingerprint.
This keeps placement separate from [pinning allocation](pinning-cost-model.md).
No hardware profile is silently transferred to a different device or build.

Interpolation uses up to four nearby shapes in log2 extent space, bounded by
measured per-axis minima/maxima and a maximum nearest distance of two. No
extrapolation is allowed. Only compatible wire representations are compared.
Selection requires coverage for the conservative Torch alternative; missing
costs for other candidates exclude them. Profiles may contain at most 4096
observations. Shape/path warm history is bounded (default 128), stores no
tensors and is invalidated when the resource owner is cleared. Warm means a
successful matching call in the current owner generation; it does not promise
that PyTorch's caching allocator will retain every allocation under external
memory pressure. Use cold/load-specific calibration when reuse is uncertain.

Diagnostics expose choices, predicted costs, wire bytes, unprofiled paths and
fallback reasons: `no_profile`, `profile_hardware_mismatch`,
`profile_hardware_unavailable`, `profile_execution_mismatch`,
`profile_missing_native_coverage`, or `legacy:unsupported_recipe`.
Profiles are process-local; policies reject use after fork.

See [the reproducible qualification](../bench/issue227/README.md) for selection
overhead, wrong-choice regret and limitations measured on held-out workloads.

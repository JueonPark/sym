# SqueezeNet 1.1 application protocol

## Question and eligibility

Measure completed single-image inference from a normalized dense CPU HWC
float32 input to GPU ImageNet logits. The inference function is
HWC → CHW permutation and contiguous materialization → blocking H2D →
official pretrained SqueezeNet 1.1. Model and weights are identical across
arms. The model stays resident and inference runs without gradients or TF32.
The photo and input variation exercise a real application boundary on one
RTX 4070 Ti SUPER. Results cover a single model, photo, batch size, precision,
hardware configuration and fixed shape. The campaign does not estimate
accuracy on a dataset, JPEG-to-logits latency, a serving SLO, or tail latency
under concurrency. [Pinned upstream inputs](application-assets.json)

The qualifying path is a dense 224 × 224 × 3, offset-zero tensor. Input
size is 602,112 bytes. The automatic arm uses the published #265 defaults
(64 KiB minimum, one prior use, device row policy, 32 entries and 256 KiB
parameter-key budget). It should fall back on the initial request, prepare
one native layout specialization on the next request, and reuse it across
remaining dense requests using current tensor storage. A non-dense strided
view with identical logical values exercises fallback after all timings.
The app supplies no Sym recipe, custom kernel, or special transfer API.
[#265 policy](../issue265/POLICY.json)

## Arms and controls

- **Torch eager** runs the ordinary full application function.
- **Torch Inductor** compiles that function with the standard Inductor backend.
- **Sym + Inductor** uses a prepared-enabled RelocBackend as the Dynamo
  backend with Inductor lowering the rewritten full graph. This gives both
  compiled arms the same consumer, built-in kernel choices, and Inductor
  settings, subject to the effect of the actual relocation rewrite.

Both compiled arms use fullgraph capture, dynamic=False, default Inductor
optimizations, and CUDA Graph replay disabled. They have independent, empty
local compiler cache directories per process. No arm gets ahead-of-time
compiled app code. All use identical weights, model tensors/operations,
float32, disabled TF32, a serial request stream and explicit device completion.
The automatic adapter uses pinned staging with one transfer stream, eight
gather threads and two buffers under the supplied runtime defaults. Eager and
Inductor use the original synchronous Torch transfer API. Device completion
and the same final tensor contract apply to all arms; internal CPU/device
placement and scratch choices are part of the executed method.

Use one GPU and CPU affinity 0–7, eight Torch intra-op workers, one interop
worker, eight OMP/MKL workers, and one Inductor compile worker. Retain default
system clocks. Charge peak Torch allocator bytes and boundary snapshots of
physical device memory separately. Apply the same 4 GiB PyTorch allocator
limit to all arms; native runtime memory is outside that limit and is reported
through resource counters and boundary physical snapshots. Those snapshots
can miss transient device peaks. Keep at most the logits of all 112 measured
and warmup requests, and three GPU input tensors during the untimed Sym
inspection. Record process peak RSS independently. No concurrent GPU campaign.

TorchVision's original Fire and SqueezeNet class definitions from the verified
source are loaded by selecting those two AST class nodes, with torch and nn in
scope. This avoids an incompatible or absent TorchVision binary extension.
Load the verified official checkpoint with strict state-dict matching and
weights_only=True. Pillow 12.2.0 decodes the pinned public dog photo to RGB.
Resize bilinearly to a 256-pixel short side (floor the long side). Produce
fresh ordered 224 × 224 crops for step i with top
\`(7*i) % (height-223)\` and left \`(17*i) % (width-223)\`. Horizontally flip
when i is odd. Scale by 255 and apply ImageNet channel mean (0.485, 0.456,
0.406) and standard deviation (0.229, 0.224, 0.225) as float32. Freeze steps
0–111 identically across arms and rounds. Hash every input and require distinct
values. Decoding, crop creation, normalization, oracles and diagnostic
interposition sit outside timed requests. [TorchVision preprocessing contract](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.squeezenet1_1.html)

## Samples, hypotheses and stopping

Run six independent paired rounds with the six permutations of eager,
Inductor and Sym in that order from itertools.permutations. Within each fresh
process run one cold request (step 0), 15 warmup requests (steps 1–15), then
96 timed warm requests (steps 16–111). Synchronize before each timed interval
and measure submission through post-call CUDA synchronization with
perf_counter_ns. Separate startup/configuration, input and model setup,
CUDA initialization and model residency, callable setup, cold first
completion, warmup, warm time, and owner retirement. Report total through
the measured trace as setup plus all 112 completed calls plus owner
retirement, excluding the oracles and diagnostic work. Record a separate
inspection after timing for Sym, including native dispatch and cache counters,
an exact intermediate GPU transfer oracle and retained-output checks. Save
per-request timings, output error and digest, residency and environment.

The mechanistic hypothesis succeeds if exactly one dense request falls back,
111 complete native prepared execution with one construction and 110 rebinds,
the post-timing excluded view adds no native work, and the output checks pass.
The latency hypothesis is lower warm completed p95 for Sym + Inductor versus
Torch Inductor. Support it only if the upper paired 95% bootstrap bound on the
p95 latency ratio is below 1. Completed p50 and eager comparisons are
secondary. Require bit-exact transferred tensors on all three instrumented
dense calls, correct rejection, independent held outputs and finite logits
within atol=1e-4 and rtol=2e-4 of eager Torch, with identical top-1.
Verify the retained initial model result again after later calls and after
owner retirement. Begin acceptance timing only after a verified untimed smoke
of all arms. Stop after six scheduled rounds; do not extend based on outcomes.
Record crashes, OOMs, failed checks, invalid or incomplete pairs and all
regressions. Eligibility/correctness failure invalidates a performance claim;
it remains evidence in the campaign record. Repeat only an externally
interrupted or invalid cell, documenting and retaining the failed run.

Compute nearest-rank p50 and p95 from each process's 96 warm completions,
then report the median of those process percentiles. Pair by round and report
the geometric mean of the process latency ratios (Sym divided by each
baseline) and percentile bootstrap bounds across rounds, using 20,000
paired draws and seed 274. Keep all per-round ratios. Report the corresponding
ratios for cold completed latency and total lifecycle descriptively.
Uncertainty covers process pairs in this campaign on this machine.

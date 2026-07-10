# Nsight Systems overlap-trace evidence (issue #47, C7)

## Capture command

```bash
/usr/local/cuda/bin/nsys profile -t cuda,osrt --force-overwrite true \
  -o bench/results/poc_overlap \
  build/cuda/bench/bench-poc-transpose --n 16384 --warmup 2 --iters 5 --reruns 1 --json -
```

`nsys` attached and captured cleanly under WSL2 (no repeat of C5's racecheck
attach failure) — CUDA 13.2, driver 595.79, `NVIDIA Nsight Systems version
2025.6.3.541-256337736014v0`. A smaller `--n 16384` (1 GiB total, 4× 256 MiB
chunks/iteration) was used per the brief, so the trace shows the overlap
mechanism, not the full-size headline numbers (those are in
`docs/poc-reproduction.md` / `poc_transpose_n32768.json`).

Driver JSON (same invocation, captured from stdout):

```json
"ours_pipeline_2buf_2stream": {"wall_ms": {"median": 173.494, "iqr": 0.423757, "n": 5}}
"baseline_pinned_memcpy_naive_kernel": {"wall_ms": {"median": 45.6441}, "gpu_ms": {"median": 42.8823}}
"speedup_wall_median": 0.263087
```

CSV extraction:

```bash
/usr/local/cuda/bin/nsys stats --report cuda_gpu_trace --format csv \
  --output bench/results/poc_overlap bench/results/poc_overlap.nsys-rep
```

The CSV header on this nsys version is `Start (ns), Duration (ns), ...,
Strm, Name`, with H2D rows named `[CUDA memcpy Host-to-Device]` (not
`HtoD` as in the brief's snippet — filter adapted to
`"host-to-device" in name.lower()`).

## Computed numbers

The trace contains two phases distinguished by CUDA stream: the pipeline
(`ours`) uses streams 13/14 (`n_streams=2`), the baseline uses a single
stream 15. Analysis below is the 5 **timed** iterations only (2 warmup
iterations excluded).

**Ours-pipeline phase** (20 H2D copies = 5 iters × 4 chunks):

- H2D memcpys: 20, busy 212.28 ms over span 830.12 ms → **copy-engine
  utilization 25.6%**.
- Per-iteration copy-busy: ~42.45 ms (4 × ~10.6 ms/chunk), consistent
  across all 5 iterations (42.44–42.47 ms).
- Per-iteration wall, computed from the trace as first-chunk-start to
  next-iteration's first-chunk-start: **173.70 ms mean** (174.91, 171.75,
  177.23, 170.90 ms) — matches the driver's reported wall median
  (**173.494 ms**) to within 0.1%, confirming the trace and the driver
  are measuring the same steady-state loop.
- Chunk-issue period (start-to-start gap between consecutive chunk
  copies, a proxy for the single-threaded CPU gather time per 256 MiB
  chunk, since the ~10.6 ms copy is much shorter and fully contained
  inside it): mean 43.13 ms/chunk (range 37.5–51.6 ms).
- **Overlap check**: last-copy-end(iteration i) → first-copy-start
  (iteration i+1) gap is 38–41 ms — i.e. even the *last* chunk's H2D
  copy of one iteration finishes well before the *first* chunk's gather
  of the next iteration completes. No copy in the trace is ever the
  long pole; every copy (~10.6 ms) is fully swallowed by a surrounding
  gather (~40+ ms).
- **Wall vs. gather+copy**: per-iteration wall (173.70 ms) ≈ sum of the
  4 chunk-issue periods (~172.5 ms) and is **well below** the
  non-overlapped serial estimate of gather + copy
  (~172.5 ms + 42.45 ms ≈ 215 ms) — a ~41 ms (~19%) saving per
  iteration from the double-buffered overlap, and consistent with
  copy time being close to *fully* absorbed (wall tracks gather alone,
  not gather+copy).

**Baseline phase** (7 H2D copies, one per iteration, single stream):
utilization 94.4% (span≈busy — expected, since it's one contiguous
whole-tensor copy, not a multi-chunk pipeline; not itself overlap
evidence, included for contrast).

## Interpretation

The trace shows the double-buffered pipeline is doing exactly what it is
designed to do: each ~10.6 ms H2D copy runs *while* the CPU is still
gathering later chunks, and is fully hidden — no copy ever exposes the
GPU stream engine as the long pole, and the last copy of each iteration
finishes tens of milliseconds before the next iteration's first gather
does. **Overlap is present and working.** The reason whole-run copy-engine
utilization is nevertheless low (25.6%, well under the "≳80% ⇒
overlapped" rule of thumb) is *not* an overlap failure — it is the
gather-bound regime documented in `docs/poc-reproduction.md`: the
single-threaded CPU gather (~40 ms/256 MiB chunk here, ~8.6 GB/s at the
full N=32768 size) is so much slower than the H2D copy (~10.6 ms/chunk,
~25 GB/s) that the copy engine spends most of its time idle waiting for
the next chunk, by design of what's being overlapped rather than a
pipelining defect. This is consistent with, and mechanistically explains,
the 0.36× wall-clock result at full size: the pipeline's throughput is
capped by the CPU gather, and the H2D copies it overlaps are already free.

Local artifacts (not committed — GitHub CLI cannot attach binary files to
an issue; a true attachment needs a manual web-UI drag-and-drop):
`bench/results/poc_overlap.nsys-rep`, `bench/results/poc_overlap.sqlite`,
`bench/results/poc_overlap_cuda_gpu_trace.csv`.

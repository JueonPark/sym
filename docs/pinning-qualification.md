# Qualification of the initial pinning gate (#189)

The first-stage implementation is an explicit size gate, with a conservative
pageable default when no threshold is supplied. #200 introduced the policy;
#206 adds selection reports and removes a universal implicit threshold; #207
qualifies mode changes and ownership. The proposed adaptive model is documented
in [pinning-cost-model.md](pinning-cost-model.md) and tracked separately in #208.

[Standalone visual report (download HTML)](https://raw.githubusercontent.com/JueonPark/sym/72f9d89fcc06090bddb0f4620ec59b805d574b97/bench/results/issue-189-qualification/report.html)
· [Raw calls, plots, traces and validation](https://github.com/JueonPark/sym/tree/72f9d89fcc06090bddb0f4620ec59b805d574b97/bench/results/issue-189-qualification)

## Calibration scope

The qualification machine is an AMD EPYC 7351 and RTX 2080 Ti (GPU 0), CUDA 12.6,
Torch 2.14.0+cu126, CPython 3.14.7, Release runtime with GPU architectures 75/89.
The size sweep uses a dense FP32 transpose with 512 input columns, eight CPU
threads on CPUs 4–7,20–23, one Torch interop thread, four staging slots and two
copy streams. CPU/NUMA binding is not generalized beyond this affinity. GPU
clocks are unlocked. All source, binary and workload hashes accompany the raw
results. The ordinary timing runtime has NVTX disabled; a separate enabled build
is used only for overlap traces.

An explicitly configured 8 MiB gate has a measured profitable **warm retained**
region on this configuration. It is not a first-call win or the exact optimal
crossover. The H2D 8, 9 and 16 MiB round ranges are separated between forced
pinned and pageable. D2H below the gate includes overlapping round ranges at
7 MiB. Preserve the full ranges when assessing a proposed new threshold.

| Direction / wire bytes | First pinned | First pageable | Warm pinned | Warm pageable |
|---|---:|---:|---:|---:|
| H2D / 8 MiB | 11.070 ms | 2.830 ms | 2.261 ms | 2.600 ms |
| H2D / 16 MiB | 21.449 ms | 4.904 ms | 3.700 ms | 4.596 ms |
| H2D / 32 MiB | 24.479 ms | 9.484 ms | 6.818 ms | 9.053 ms |
| D2H / 8 MiB | 10.833 ms | 2.780 ms | 2.010 ms | 2.496 ms |
| D2H / 16 MiB | 20.675 ms | 4.555 ms | 3.533 ms | 3.986 ms |
| D2H / 32 MiB | 40.284 ms | 10.874 ms | 9.224 ms | 10.467 ms |

These are medians of three rounds, not confidence bounds. Each warm round has
ten samples after one preserved first sample and two warmups. First use means a
new owner in an initialized process, not cold CUDA initialization or a fresh
Torch allocator. Compilation precedes the size sweep; prepare/bind, fresh
output allocation, CPU transform, copy and completion remain inside each timer.
Inputs, result checks and result destruction are outside it. Every output is
checked. Separate runs preserve repeated ephemeral allocation costs and the
matching Torch CPU-transpose/transfer baseline.

The first-call loss matters. At 16 MiB the forced-mode medians imply roughly
20 H2D or 37 D2H calls to repay allocation, using
`first + (calls - 1) * warm`. This is an illustrative calculation, excludes final
owner retirement, and assumes stable warm timings; it is not an implemented
reuse prediction. A short-lived owner or a different shape may never repay it.
Retaining buffers alone does not demonstrate enough reuse to justify a new pin.
The configured gate leaves that decision with the caller; unconfigured auto
remains pageable. For an existing pinned warm cache, the allocation cost is
already paid, but the configured policy must still select the compatible kind.

Use `pinning="auto", min_pinned_bytes=8 << 20` only when this measured scope and
reuse regime match the application. Omit the threshold elsewhere until a profile
is qualified. Forced policies are measurement controls, and all modes retain the
same stream, completion, memory-budget and ownership guarantees. A dense typed
input can bypass staging; forced pinning does not register or recopy that input.

## Overlap and workload results

The configured four-buffer H2D trace contains 0.893, 0.896 and 0.906 ms of
CPU gather of chunk `n+1` overlapping actual DMA of chunk `n` across three
requests. The otherwise matched one-buffer control has zero overlap in all
three requests. Unconfigured pageable staging has 0.151, 0.156 and 0.174 ms:
pageable retains some overlap on this driver, but substantially less in this
trace. This is direct evidence for the selected pinned layout path; it does not
extend to every typed execution path or guarantee overlap on another machine.

All 48 unchanged-workload runs passed correctness checks. Median completed
transfer totals excluding the first call of each kind are:

| Workload | Auto, unconfigured | Auto, 8 MiB | Forced pinned | Forced pageable | Torch¹ |
|---|---:|---:|---:|---:|---:|
| DLRM | 8.087 ms | 8.014 ms | 7.709 ms | 8.079 ms | 5.607 ms |
| GNN | 13.906 ms | 12.925 ms | 12.930 ms | 13.816 ms | 9.494 ms |
| LLM | 365.798 ms | 362.070 ms | 362.725 ms | 364.005 ms | 182.967 ms |
| MoE | 125.431 ms | 126.583 ms | 126.306 ms | 125.423 ms | 75.658 ms |

¹ Paired auto-unconfigured processes; all policies' paired Torch timings and
three-round ranges are published. These small differences between some modes
are not established statistical wins. Torch remains faster on all four complete
workloads. The report also plots all-call totals including first-call setup;
model compute is excluded in both summaries. Pinning selection does not close
the remaining Sym/Torch gap. With retention already enabled, pinning changes
matter less than in the original per-call-allocation experiment.

## Reproduction

Use the compiler and runtime from the candidate stack, with the same Release
flags for the normal and NVTX builds. Set `PYTHONPATH`, `SYM_RELOC_EXPORT` and
`SYM_OPT` to those builds; pin the benchmark process to the recorded CPUs.
Do not run timing jobs concurrently with one another or a GPU test suite.

```sh
python bench/issue189/pinning_sweep.py --direction h2d --output sweep-h2d.json
python bench/issue189/pinning_sweep.py --direction d2h --output sweep-d2h.json
python bench/issue189/run_matrix.py --repo /path/to/unmerged-pr162-worktree \
  --output matrix --rounds 3 --variants auto_default,auto_configured,pinned,pageable
```

The unchanged #162 workload sources are commit
`cdc44e278eac08d51abc6e1fd13123a5a46a26b8`. The wrapper passes an explicit retained
owner to direct typed calls and records this option. It does not alter shapes,
GPU computation or correctness checks. All processes run Torch before Sym as in
the original workload; the 48 policy/workload process jobs are shuffled across
three rounds. Report both totals excluding the first call of each kind and
all-call totals including first-call compilation/setup. Model compute is outside
these transfer timers. PR #162 stays unmerged.

For each of `configured --buffers 4`, `configured --buffers 1`, and
`unconfigured --buffers 4`, select the NVTX-enabled runtime and run:

```sh
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true --output traces/configured-4 \
  python bench/issue189/profile_pinning.py --policy configured --buffers 4 \
    --output traces/configured-4.run.json
nsys export --type sqlite --force-overwrite=true \
  --output traces/configured-4.sqlite traces/configured-4.nsys-rep
python bench/issue189/profile_pinning.py --policy configured --buffers 4 \
  --sqlite traces/configured-4.sqlite --output traces/configured-4.analysis.json
python bench/issue189/qualification_report.py --results /path/to/results
```

The fixed trace shape is 4096×1024 FP32 (16 MiB). Both slot-count controls use
four equal 4 MiB chunks; the analyzer verifies those sizes, completed-request
boundaries, native work-before-submit ordering and API-to-GPU correlation.
Three captured requests follow five warmups. An actual GPU copy must overlap
native gather of the next chunk for the four-slot pinned case; none may do so
in the one-slot control. Pageable behavior is characterized without requiring
either outcome. CPU work spans are wall intervals and can include preemption.

The report writes standalone HTML with embedded images plus PNG, SVG and PDF.
Raw calls, round order, traces, analysis and checksums remain available separately.

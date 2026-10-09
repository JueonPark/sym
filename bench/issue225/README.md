# Sym + Inductor: end-to-end qualification (#225)

## Protocol declared before acceptance measurements

Four paths: native Torch; Inductor; current Sym with eager FX compute; Sym with
Inductor compute. All start from identical data, precision, layouts and output
ownership. This is a new model-inclusive comparison, not a rerun of #217's
transfer-only totals. The older transfer totals cannot be divided by these
end-to-end latencies to claim a speedup.

| Regime | Timed work | Target / rationale |
| --- | --- | --- |
| DLRM, 512 / 2048 samples | 26 CPU EmbeddingBag lookups + stack, FP32→FP16 layout/H2D, bottom/top MLP and dot interaction | Large case: improve over eager-compute Sym by ≥10%; small case exposes fixed overhead |
| GraphSAGE, 256 / 1024 seeds | CPU feature gather + FP16 H2D, two GPU SAGE-mean layers | Large case: improve over eager-compute Sym by ≥10%; compare against fused Inductor host gather |
| GPT, 4 decode steps / 256-token prefill | 4 layers, INT8 weight prefetch, attention/MLP, logits, greedy token selection; decode includes growing KV eviction/restore | Prefill target ≥10% improvement over eager-compute Sym; decode may remain dispatch-bound |
| MoE, 1 / 512 tokens | 2 blocks, 8 experts, top-2 routing, selected weight prefetch, expert FFNs and combine | Dense target ≥10% improvement over eager-compute Sym; sparse case exposes queue/dispatch overhead |

The project-level target is ≥5% lower median completed latency than **both**
matched Torch controls in at least one useful regime, consistently in all three
rounds. Report misses, regressions and uncertainty; feature completion alone does
not establish this performance target. No sum of individual optimization gains.

DLRM and GraphSAGE model mathematics come from PR #162 at `cdc44e2`; their source
has not merged, so `models.py` contains the relevant benchmark definitions. GPT
and MoE use main's existing model weights and independently check the factored
compute against the original model forward. Model arithmetic is FP32, TF32 off.
Transfer results are byte-exact in the frontend tests; Inductor model results use
`rtol=2e-4, atol=2e-5`, with exact generated tokens / selected experts.

### Boundaries and fairness

- DLRM/GraphSAGE Sym paths capture transfer and GPU model compute in one graph.
  Torch controls get retained pinned FP16 staging and, for Inductor, a compiled
  host gather/layout/cast into that staging plus compiled GPU model compute.
  This explicit host boundary ensures the same FP16 wire representation.
  All paths include the CPU feature lookup; DLRM embedding-bag lookup/stack is
  common eager code. Graph sampling and local-index uploads, plus DLRM dense
  input upload, are common setup outside this inference unit.
- GPT/MoE all use an explicit immutable INT8 `[out,in]` snapshot with FP32 scales,
  and the same two-slot lookahead schedule: 128 MiB output and 64 MiB scratch
  budgets. Torch gets identical offline prepacking and GPU dequantization;
  Inductor compiles conversion **and** model tensor regions. Sym uses #224's
  prepared snapshots, #223's prefetch queue and #220's grouped submission.
  Snapshot and artifact/owner preparation costs are separate from warm reuse.
- Async queue iteration, MoE's dynamic active-expert discovery (`unique`, host
  list, `nonzero`), greedy token scalar reads and Python scheduling stay outside
  compute graphs and inside end-to-end timing. All compute regions use
  `fullgraph=True`; these deliberate boundaries are not implicit graph breaks.
  This does not claim a single captured graph for an entire offloaded model.
- Completed wall time includes allocations, transfer preparation/submission,
  prefetch startup/drain and consumer completion. Transfer diagnostics are
  separate serialized invocations, not subtractable phases or proof of overlap.
  GPT/MoE diagnostics cover all request weight groups but exclude KV/routing.
- First calls include Dynamo/AOT/Inductor and lazy runtime setup, with backend
  compile time and generated-kernel counts recorded separately. An empty cache
  per process does not make every later path's first call independently cold;
  rotated path order and compile counts expose shared-cache effects.
- Three independent processes, 5 warmups, 30 observations/path/case, path order
  rotated both between rounds and samples. Preserve every accepted raw sample
  in compact arrays. Record p50/p95, recompiles, graph breaks, resource counters,
  hardware, source and binary hashes. No tests, builds or unrelated GPU jobs
  during headline timing. Collect any profiler trace in a separate invocation.
- Shapes within each regime are known (including the four decode steps), so
  acceptance uses `dynamic=False` for all compiled paths, warming every shape
  before sampling. `--dynamic` separately exercises forced symbolic codegen;
  dynamic correctness/rebinding is also covered by the integration tests.

```sh
export PYTHONPATH=/tmp/sym-225-build/python
export SYM_RELOC_EXPORT=/tmp/sym-225-build/sym/tools/sym-reloc-export
export SYM_OPT=/tmp/sym-225-build/sym/tools/sym-opt
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
export TORCHINDUCTOR_COMPILE_THREADS=1
TORCHINDUCTOR_CACHE_DIR=/tmp/sym225-round1 taskset -c 4-7,20-23 \
  /tmp/sym-210-venv/bin/python bench/issue225/compare.py --output /tmp/sym225-round1.json
```

Acceptance results will be added after the implementation and harness pass
correctness checks. Verbose compiler logs and profiler databases are not checked in.

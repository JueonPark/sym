# Completed-cost placement (#227)

## Protocol fixed before acceptance runs

The selector ranks qualified paths using whole completed calls. Calibration
and held-out evaluation use separate sizes. Matrix calibration uses rows
64/256/1024/4096 and columns 256/1024; held-out shapes are 128×512, 512×512,
2048×512 and 2048×1024. KV calibration uses sequence lengths 4/8/32/128/512;
held-out lengths are 12/24/64/256, with 16 heads of width 64. Both directions
are measured, with matrix layout-only and FP32→FP16 variants.

Each of three fresh processes calibrates independently. Each forced path gets
three cold-owner samples and 30 warm samples, after three warmups. Cold-owner
means `TransferResources.clear()` before the call, excluding clear itself;
compiler and global Torch allocator are already initialized. This is not a
fresh-process allocation measurement. Warm automatic selection first warms
all qualified alternatives at the held-out shape. The first automatic call
after owner clear is also retained separately. Model inference exercises
natural repeated-call reuse rather than warming unchosen alternatives.

Controls include the original eager Torch region, explicit CPU-transform
Inductor and explicit GPU-transform Inductor, and the existing unprofiled Sym
backend. Each has identical input/output precision and fresh output ownership.
CPU/GPU cuts can send different wire bytes; each has a matching Inductor cut.
The largest source is 16 MiB and each candidate's transform/transfer temporary
storage is bounded within the common 64 MiB scratch allowance. Output size is
at most 16 MiB. All Sym owners use 64 MiB retained/live scratch limits. Torch
uses its caching allocator; the bound follows from these fixed shapes.

All timings synchronize the request's current stream and include allocation
and frontend costs. Correctness checks happen outside timing. Inductor
compilation happens before timing and CUDA graphs/TF32 are disabled. Controls
rotate within samples; forced-path order rotates between rounds. Selection
alone is timed separately without binding, copying or GPU work. Held-out forced
paths and automatic selection also run in the same rotated interleaved loop,
so regret compares equal cache/allocator pressure. Regret is automatic median /
best qualified interleaved forced warm median − 1, including selector overhead;
path regret excludes that overhead. Report all misses; exploratory runs are
excluded from acceptance. An initial run using consecutive forced samples for
regret was discarded because its scheduling differed from automatic samples.

Hypotheses: reduce small KV latency by at least 30% versus the current Sym
backend; keep median held-out regret within 10% and p95 within 20%. These are
targets, not prerequisites for claiming an implementation or guaranteed gains.
Also evaluate a held-out GPU/thread profile and explicitly tagged load/context
changes. No builds, tests or unrelated GPU jobs run during headline timing.
Three GPT decode comparisons reuse #225's model and matched weight delivery.

```sh
export PYTHONPATH=/tmp/sym-227-build/python
export SYM_RELOC_EXPORT=/tmp/sym-227-build/sym/tools/sym-reloc-export
export SYM_OPT=/tmp/sym-227-build/sym/tools/sym-opt
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
export TORCHINDUCTOR_COMPILE_THREADS=1
TORCHINDUCTOR_CACHE_DIR=/tmp/sym227-round1 taskset -c 4-7,20-23 \
  /tmp/sym-210-venv/bin/python bench/issue227/placement.py \
  --round 1 --output /tmp/sym227-round1.json
```

Raw timing arrays and scalar metadata are retained compactly. Build/test logs,
compiler caches and profiler databases are excluded from the PR.

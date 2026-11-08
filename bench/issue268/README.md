# CUDA Graph replay qualification (#268)

This campaign compares original E2/E5 workloads on the merged #267 native runtime
with explicit GPU consumer regions. Read [PROTOCOL.md](PROTOCOL.md) before running.
See [the public API contract](../../docs/gpu-replay.md).

The main campaign contains 48 cells / 288 fresh-process arms over three rounds.
Each cell compares Torch/Sym × unwrapped/slots-off/replay-on. Six supplemental
processes cover stable inputs, shape variation, limited graph budgets and launch
profiling. Tests and pilots are separate from qualification.

## Dependencies and reproduction

The qualified environment is Python 3.14.7, Torch 2.14.0+cu126 on an RTX 2080 Ti,
CUDA toolkit 12.5 for original native control dependencies, and the existing #267
native runtime. Compiler and original E1–E5 dependencies are public through the
[Wiki artifact catalog](https://github.com/JueonPark/sym/wiki/Artifacts).
No download requires access to the author's local filesystem.

A runtime root contains `python/reloc_torch`, `python/pyreloc` and
`libreloc/libreloc_runtime.so`. The original artifact root contains the unmodified
`00_protocol`, `02_fp16_attribution` and `05_lifecycle_concurrency` directories
and its `SHA256-MANIFEST.json`. Compiler path selects the #265 `sym-reloc-export`,
with its sibling `sym-opt`. Use the archived versions and hashes in the final
artifact manifest; current checkout sources are not interchangeable.

```sh
python bench/issue268/run.py --runtime RUNTIME --artifact ORIGINAL_ARTIFACT \
  --compiler COMPILER/sym-reloc-export --output OUTPUT --wait-for-idle
```

The runner pins all sources/dependencies before launch, observes 60 quiet seconds
before each six-arm cell and audits every process. It rejects stale or partial
unrecorded outputs; an unchanged manifest can resume completed jobs. `--pilot`
with `--jobs JOB_ID...` is explicitly excluded from acceptance.

Run supplements for eager/Inductor and rounds 1–3 with the same runtime/compiler
in `PYTHONPATH`, `LD_LIBRARY_PATH`, `SYM_RELOC_EXPORT`, `CUDA_VISIBLE_DEVICES=0`,
`TORCHINDUCTOR_COMPILE_THREADS=1`, and affinity 4–7,20–23. Give each a fresh
`TORCHINDUCTOR_CACHE_DIR`. Output paths are
`OUTPUT/supplements/{eager,inductor}-r{1,2,3}`. Add `--profile` in round 1:

```sh
python bench/issue268/micro.py --compute eager --round 1 --profile \
  --runtime RUNTIME --artifact ORIGINAL_ARTIFACT \
  --output OUTPUT/supplements/eager-r1
```

After all processes complete, saved-data analysis requires only the standard
library and verifies membership, checksums, environment, quiet/audit evidence,
correctness/retention checks and physical memory traces:

```sh
python3 bench/issue268/analyze.py OUTPUT --output analysis/measurements.json
```

It writes per-round measurements, comparison ratios/break-even estimates and
supplemental summaries. `Total(N)` includes recorded preparation, all first calls,
and all warmup costs, followed by N warm completed requests. E5 warm latency is
completed burst time divided by eight; service/sojourn p95 remain separate.
Qualification reports use unprofiled times; diagnostic traces establish launch
attribution only. Negative comparisons and physical/admission failures remain.

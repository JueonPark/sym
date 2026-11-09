# Consumer layout elimination qualification (#269)

The opt-in rule removes private inverse KV permutation copies before fresh
concatenation. Read the [API contract](../../docs/consumer-layout.md) and
[frozen protocol](PROTOCOL.md). Both Torch and Sym receive the same opportunity;
Inductor may already remove the materialization without this rule.

The completed three-round campaign establishes **no qualified end-to-end Sym
performance win**: zero of 24 parent targets pass. All 162 model arms completed;
the 90 supplementary rows retain six actual native-path memory-pressure OOMs.
See [results](RESULTS.md), [complete tables](TABLES.md), [review disposition](REVIEW.md)
and the [public artifact](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-269).

| Variant | Restore boundary | Consumer region | Compiler rule |
|---|---|---|---|
| `native` | H2D plus batched inverse permutation | Ordinary cat and decoder | Off |
| `materialized` | Shared raw H2D copy | Explicit inverse copies, cat and decoder | Off; Inductor has its ordinary fusion opportunity |
| `lowered` | Same raw H2D copy | Guarded permutation views feed fresh cat | On |
| `resident` | Weights/caches already on GPU | Original decoder | Off; separate deployment policy |

The transparent materialized path is a compiler-opportunity control. Its eager
inverse copies are separate kernels; the deployed native control already batches
them. Compare the candidate with both, and with the fastest observed alternative.

The declared campaign has 162 model arms and six supplemental processes over
three rounds. Each model arm starts in a fresh process with a fresh compiler
cache. Each supplement process runs all five shapes and three variants; its
compiler/library cache is shared across those rows, so supplemental preparation
costs are not independent cold-start comparisons. Variant order rotates by round.
Accepted timings never run under the profiler. Separate diagnostic pilots retain
traces, generated FX graphs and compiler caches.

## Reproduce

Use Python 3.14.7, Torch 2.14.0+cu126, an RTX 2080 Ti, eight CPU threads with
mask 4–7,20–23, the unchanged #267 native runtime, and the public #265 compiler.
The [Wiki catalog](https://github.com/JueonPark/sym/wiki/Artifacts) supplies the
compiler and original E1–E5 controls. A runtime root contains
`python/reloc_torch`, `python/pyreloc` and `libreloc/libreloc_runtime.so`.

```sh
python bench/issue269/run.py --runtime RUNTIME --artifact ORIGINAL_CONTROLS \
  --compiler COMPILER/sym-reloc-export --output OUTPUT --wait-for-idle
python3 -B -S bench/issue269/analyze.py OUTPUT --output ANALYSIS
```

The serial runner observes 60 quiet seconds before each cell, audits each process,
and freezes all executed sources/dependencies. An unchanged manifest can resume
completed jobs. Failed/unrecorded attempts require investigation and remain
preserved. `--pilot --jobs JOB_ID...` is explicitly excluded from qualification;
`--profile` is available only for diagnostic pilots.

The saved-data analyzer uses only the standard library and checks membership,
hashes, resource/environment contracts, numerical/retention counts, actual
eligibility, stable compilation/capture, isolation, and physical memory traces.
The final Wiki package includes raw data and portable verification scripts.

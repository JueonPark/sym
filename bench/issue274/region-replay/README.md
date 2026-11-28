# Replaying the #270 region comparison

These records are a fresh correctness run for issue #274. The original #270
report gives the row counts, byte example and runner. Its original execution
log was unavailable. The implementation, test fixtures and runner at the
#274 protocol freeze have the same bytes as in Sym revision
`cb25e615a31c387e5db95b0558f0c0813eb5e1f0`.
[Original report](../../../docs/relocation-regions.md#same-engine-ablation) · [Fixture hashes](../historical-manifest.json)

All 19 host and 44 CUDA rows passed on this build. Each row uses the same bound artifact, source, parameters, device and native
implementation for both FX graphs. The original graph clones its host source
and materializes a transfer result before concatenation. The lowered graph
reads the private source and writes directly to the output suffix. For every
eligible row, the harness checks declared dispatch order, output bits, layout,
payload and the temporary sizes in both graphs. The transposed FP16 fixture
has a 96-byte source clone and a 48-byte destination temporary. It retains the
output allocation and the specified transfer. The records contain logical
byte counts with no latency or physical-memory sampling.

Use the specialized `on` CUDA build described in the [build instructions](../../issue272/README.md#reproduce).
Commands below assume that `python` is in the Torch environment and `BUILDS`
points to the directory containing the build. Run from the checkout root.

```sh
export PYTHONPATH="$BUILDS/on/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
taskset -c 0-7 python -m bench.issue270.ablate --device cpu > /tmp/region-host.jsonl
taskset -c 0-7 python -m bench.issue270.ablate --device cuda > /tmp/region-cuda.jsonl
diff -u bench/issue274/region-replay/host.jsonl /tmp/region-host.jsonl
diff -u bench/issue274/region-replay/cuda.jsonl /tmp/region-cuda.jsonl
```

The run used the Ryzen 7 7800X3D, RTX 4070 Ti SUPER, WSL2,
Python 3.14.7, Torch 2.14.0+cu126 and NumPy 2.5.3. It exercised the typed
corpus in its declared dtypes with the runner's single-call, synchronous
completion. No physical VRAM reservation or deliberate pressure was applied.
The native shared library SHA-256 was
`07ac69da5b3cfa218879d08c7c14936efb5aeb4a0be20d4b0f2b916d1a27ce55`.
The SqueezeNet archive's `accepted/manifest.json` records the hashes of the same
runtime, Python extension and exporter. The [catalog](../CATALOG.md) identifies
the original fixture and source files.

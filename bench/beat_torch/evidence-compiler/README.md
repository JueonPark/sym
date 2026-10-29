# Scope and correction of the historical compiler evidence

The committed round files and `plans.json` are historical measurements from
the revision in `manifest.json`; they have not been rewritten after review.

The `identity_view_2048x128` canonical entry in `plans.json` is **invalid as a
transfer-latency measurement**. The old harness caught `no_copy_layout` and
recorded 0 without performing H2D, while the raw arm actually transferred the
tensor (about 276 µs). Do not compare these values or count them as a speedup.
The current `plans.py` measures completed direct H2D for the canonical arm.
A fresh qualified measurement is needed before publishing a replacement number.

`--no-canonicalize` retains `foldChainOp()` and plan generation. These rounds
measure the additional effect of canonicalization for these workloads, not
the contribution of all AST folding. Most cases do not stress composed layout
and value chains: weights are prepacked offline, KV uses a simple permutation,
indexed gather bypasses this canonicalizer, and pooled DLRM bypasses the compiler.

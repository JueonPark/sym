# Automatic guarded preparation: issue 265

This harness connects evaluation to the public `RelocBackend` configuration.
Production code contains no workload recipes or names. See
[the runtime contract](../../docs/automatic-prepared-execution.md),
[the frozen protocol](PROTOCOL.md), [results](RESULTS.md), and [evidence index](INDEX.json).

Inputs are the immutable `sym_experiments_261104` directory from the
[E1–E5 release](https://github.com/JueonPark/sym/releases/tag/experiments-2026-11-04-v1).
Use the baseline and candidate runtime trees packaged with this issue's evidence
release, or build both revisions with the same native/compiler configuration.
Both trees must provide `python/`, `libreloc/libreloc_runtime.so`, the native Torch
bridge and `sym/tools/{sym-opt,sym-reloc-export}`. Use their matching Python/Torch
ABI. Paths are explicit; no author's home directory is required.

```bash
# From the source checkout; set these to the extracted artifact/runtime paths.
python bench/issue265/run.py \
  --artifact "$E3_ARTIFACT" \
  --baseline-build "$BASELINE_BUILD" --candidate-build "$CANDIDATE_BUILD" \
  --python "$PYTHON" --suite selection --rounds 2 --epochs 5 \
  --output "$OUTPUT/selection"
python3 bench/issue265/analyze.py "$OUTPUT/selection" \
  --output "$OUTPUT/selection-analysis.json"

# Choose/freeze SELECTED_POLICY using only selection evidence, then qualify.
python bench/issue265/run.py \
  --artifact "$E3_ARTIFACT" \
  --baseline-build "$BASELINE_BUILD" --candidate-build "$CANDIDATE_BUILD" \
  --python "$PYTHON" --suite e3 --rounds 3 --epochs 5 \
  --selected-policy "$SELECTED_POLICY" --output "$OUTPUT/e3"
python bench/issue265/run.py \
  --artifact "$E3_ARTIFACT" \
  --baseline-build "$BASELINE_BUILD" --candidate-build "$CANDIDATE_BUILD" \
  --python "$PYTHON" --suite heldout --rounds 3 --epochs 5 \
  --selected-policy "$SELECTED_POLICY" --output "$OUTPUT/heldout"
python3 bench/issue265/analyze.py "$OUTPUT/e3" --output "$OUTPUT/e3-analysis.json"
python3 bench/issue265/analyze.py "$OUTPUT/heldout" --output "$OUTPUT/heldout-analysis.json"
```

The runner freezes the complete matrix, code/input/runtime digests and environment
in `contract.json`. Completed records are validated on resume; unfinished caches
require a separate output directory. `--only` runs a subset for a pilot or resume,
while final analysis still requires the complete matrix. Audits reject concurrent
GPU/build/packaging jobs. Use `--affinity` and `--device` for an explicitly different
machine; it is a new campaign, not an extension of an existing frozen contract.

Analysis uses only Python's standard library. Validation tests additionally need
pytest: `PYTHONPATH=. "$PYTHON" -m pytest -q bench/issue265/test_analysis.py`.
Negative results, capture exclusions and fallback-selection costs are part of
the result. Do not interpret a reduction in compilation count as a speedup.

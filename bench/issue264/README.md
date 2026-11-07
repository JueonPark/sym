# Issue 264: reusable group qualification

This harness qualifies the production prepared-group implementation against the original runtime and a new-build legacy control. It reuses the published #263 E2 runner and the immutable E5 programs; model mathematics, kernels, grouping, prefetch, precision, and budgets stay fixed.

Inputs are explicit: `--artifact` is the extracted November E1–E5 artifact; `--harness` is a Git checkout of the #263 source Git tag (its provenance helper reads Git metadata); `--baseline-build` and `--candidate-build` identify separate runtime trees. See [protocol](PROTOCOL.md) for measurement scopes and limitations. See [final results](RESULTS.md), [grouped ablation](GROUPS.md), [result index](INDEX.json), and the [versioned Wiki artifact](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-264).

```sh
python bench/issue264/run.py \
  --artifact /path/to/sym_experiments_261104 \
  --harness /path/to/issue263/source \
  --baseline-build /path/to/baseline/build \
  --candidate-build /path/to/candidate/build \
  --python /path/to/python3.14 --suite groups --output /path/to/new/group-results
```

Use `--suite acceptance --candidate prepared-native` for the registered E2/E5 matrix after selecting a policy from grouped measurements. A new output directory freezes the matrix, runtime and source hashes, workload inputs, and environment. Resuming verifies these contracts. `--only PREFIX` runs a subset while preserving the complete matrix. Incomplete or contaminated jobs must be moved aside explicitly; they are never silently accepted or overwritten.

The optional native Torch bridge is required for prepared groups. The frozen baseline retains its historical Python files; a clean candidate build may omit obsolete baseline modules. Candidate provenance records missing files and hashes its actual runtime. No stale modules are copied into the candidate.

CPU-only analysis and its regression tests require no GPU or local runtime. Raw timing, memory samples, audits, and exact input/build provenance belong in a versioned Wiki artifact, not the source tree. No SVG files are tracked.

Analyze saved data without a GPU:

```sh
python bench/issue264/analyze.py /path/to/evidence/groups-v1 --output /tmp/groups-analysis.json
python bench/issue264/analyze.py /path/to/evidence/acceptance-v1 --output /tmp/acceptance-analysis.json
python -m pytest -q bench/issue264/test_analysis.py
```

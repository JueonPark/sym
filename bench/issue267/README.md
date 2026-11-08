# Issue 267: automatic fused-chain kernel quality

See [protocol](PROTOCOL.md) and `contract.py` for the fixed matrix. This directory
adapts the original E1 cases/manual controls from the
[November Wiki artifact](https://github.com/JueonPark/sym/wiki/Artifacts-Experiments-2026-11-04).
The source compiler is unchanged; the work specializes checked CPU execution.

Build `native.cpp` against each selected runtime using `build_native.py`. Each
runtime directory contains `libreloc/`, `python/` and the helper in `helper/`.
Run `run.py --baseline ... --candidate ... --compiler ... --output ...
--wait-for-idle` using the qualified Python/Torch environment. Use a fresh output
directory; `--pilot --jobs ...` labels a subset as excluded diagnostic evidence.
Saved-data analysis uses only the standard library:

```sh
python3 -X utf8 -B -S bench/issue267/analyze.py /path/to/complete-campaign
```

No accepted results or public artifact have been published yet.

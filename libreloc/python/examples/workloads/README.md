# Offloaded weights with WeightFetcher

`WeightFetcher` in `common.py` compiles one symbolic typed recipe:

```text
CPU int8[in, out] + float32 scale[out]
  -> dequantize(scale, axis=1) -> transpose
  -> CUDA float32[out, in]
```

Each `fetch(q, scale, device)` binds the current shape, payload and scales to
that artifact and returns a fresh tensor in `nn.Linear`'s layout. Calibration
lets `policy="auto"` choose the transformation placement; without calibration
the runtime uses `cpu_reference`. The main runtime selects and restores the
CUDA launch device, including when experts alternate between devices.

This extraction preserves direct calls' **per-call resource ownership**.
`RelocBackend` already manages frontend AUTO ownership, while a direct
`execute_typed_transfer` needs an explicit `TransferResources` to retain
scratch, streams and workers. The separate optimization in #231 adds that
ownership to `WeightFetcher`; it is not part of this baseline.

## Examples

| Example | PyTorch compute | Sym transfers |
| --- | --- | --- |
| `llm_offload.py` | GPT-style decoder, attention, MLP and greedy decoding | Fetch each layer's int8 weights as float32 `[out, in]`; move a 3D KV cache between CPU and GPU |
| `moe_experts.py` | Expert routing, token dispatch, expert compute and combine | Fetch only the selected experts' int8 weights to their assigned GPU |

Both synthetic examples run plain PyTorch and Sym on the same inputs and
check transferred tensors and model outputs bit for bit. Offline symmetric
int8 quantization uses the same values and scales for both paths. Several
GPUs demonstrate placement and correctness; execution is sequential and
blocking. These examples do not establish a speed advantage over PyTorch.

This is the WeightFetcher-related subset extracted from
[PR #162](https://github.com/JueonPark/sym/pull/162) at
`cdc44e278eac08d51abc6e1fd13123a5a46a26b8`. It includes the common report,
calibration, environment checks, runner and tests needed to execute these
callers independently on main. DLRM and GNN remain in #162.

## Run

Use a CUDA-enabled Sym build and the qualified Python/PyTorch environment
from [the getting-started guide](../../../../docs/getting-started.md#use-with-pytorch):

```sh
export SYM_BUILD="$PWD/build/torch-cuda"
export SYM_PYTHON=/path/to/qualified/python
python3 libreloc/python/examples/workloads/run_workloads.py --quick
python3 libreloc/python/examples/workloads/run_workloads.py --only llm
python3 libreloc/python/examples/workloads/run_workloads.py \
  --only moe --devices cuda:0,cuda:1
```

The standard-library runner sets the build's Python and compiler paths for
each child process. `--quick` selects small models, `--calibration` accepts
`auto`, `none` or a `.cal` path, and `--output-dir` selects the report directory.
`--build` and `--python` override the environment. `--timeout` bounds each
child (default 900 seconds). Exit status is 0 for success, 1 for failed
checks/crashes/timeouts, or 2 for missing prerequisites.

Reports contain per-check observations, compilation/execution counters,
dispatch rows, transfer bytes, model outputs, and transfer timings. The
first call of every transfer kind is reported separately from subsequent
calls. KV first calls include graph compilation; the weight recipe is
compiled before timing begins. Timings include transfer completion but
exclude model compute and correctness checks. Wire bytes follow from the
chosen precision; equivalent PyTorch code moves the same wire payload.

## Tests

```sh
export PYTHONPATH="$SYM_BUILD/python:$PWD/libreloc/python"
export SYM_RELOC_EXPORT="$SYM_BUILD/sym/tools/sym-reloc-export"
export SYM_OPT="$SYM_BUILD/sym/tools/sym-opt"
"$SYM_PYTHON" -m pytest -q libreloc/python/examples/workloads/tests
```

CPU tests cover reporting, calibration, quantization, prerequisite handling
and runner accounting. GPU tests cover shape rebinding with one artifact,
non-current destinations, both examples, and multi-GPU expert placement.
GPU cases skip without the required hardware; run the complete suite on a
CUDA host. The existing CI test collection does not include this directory.

# Sym

Sym is a compiler and runtime for **changing tensor layouts and values while
moving data between the CPU and GPU**. For example, it can turn a sequence
of reshapes, transposes, padding, and supported data-type conversions into a
reusable execution plan. Tensor dimensions can remain symbolic, so the same
plan can run on different input sizes after checking their constraints.

Use Sym through an opt-in **PyTorch backend**, or compile plans explicitly
and execute them from **Python or C++**. The compiler uses MLIR; the
standalone `libreloc` runtime does not depend on MLIR, LLVM, or PyTorch.
The project also provides symbolic tensor types and shape inference for
MLIR developers.

## Quick start

Start with the [installation guide](docs/installation.md). A complete Sym wheel
installs the compiler, runtime, and Python APIs together and discovers its tools
automatically. The guide includes a source-wheel route available now and the
binary installer interface for qualified releases; release binaries are not yet
published. After installation, run `sym-doctor --require-torch` and
`sym-demo --device cpu`.

For PyTorch CPU–GPU transfers, install a qualified CUDA variant using the
[CUDA source setup](docs/torch-integration.md#2-build-the-compiler-tools-and-the-cp314-extension), then try:

```python
import torch
from reloc_torch import RelocBackend

# Reshape and transpose a CPU tensor, then transfer the result to the GPU.
def prepare_input(x):
    return x.reshape(x.shape[0] // 64, 64).t().contiguous().to("cuda")

backend = RelocBackend()
compiled = torch.compile(prepare_input, backend=backend, dynamic=True)

with torch.no_grad():
    for n in (128, 192, 256):
        result = compiled(torch.arange(n, dtype=torch.float32))
        print(result.shape, result.device)  # [64, n // 64] on cuda:0

print(backend.stats())  # Executed plans, shape bindings, and fallback reasons.
backend.close()
```

Sym compiles supported regions and binds their symbolic dimensions for each
call. Unsupported regions retain their original PyTorch behavior. The
integration targets inference and blocking transfers; nonblocking requests
and unsupported tensor layouts use PyTorch. The qualified frontend baseline
is Linux x86_64, regular-GIL **Python 3.14.7**, and **PyTorch 2.14.0**
(`+cpu` / `+cu126`); GPU execution also requires a CUDA-enabled Sym build.

For explicit recipes, Sym supports float32/float16 casts and signed-int8
quantization/dequantization with supplied parameters. Automatic PyTorch
capture supports a narrower set; see the
[typed support matrix](docs/typed-relocation-support.md). Precision changes
are always explicit. Performance depends on the workload; the current
[integration measurements](docs/runtime-integration.md#5-evidence) do not
establish a speed advantage over PyTorch.

## Learn more

| I want to… | Guide |
| --- | --- |
| Install the complete compiler/runtime/Python package | [Installation](docs/installation.md) |
| Build Sym and execute my first plan | [Getting started](docs/getting-started.md) |
| Use compiled/eager PyTorch transfers or prepare inference weights | [PyTorch integration](docs/torch-integration.md) |
| Write recipes, save artifacts, and use the Python/C++ runtime | [Plan export](docs/reloc-export.md) · [Runtime APIs](libreloc/README.md) |
| Choose typed execution paths and inspect transferred bytes | [Runtime dispatch](docs/runtime-dispatch.md) |
| Work with symbolic shapes in MLIR | [Symbolic shapes](docs/symbolic-shapes.md) |
| Reproduce examples, tests, and CPU/CUDA results | [Integration guide](docs/runtime-integration.md) |
| Read the research results and limitations | [Claim ledger](docs/claim-ledger.md) |

## License

See [LICENSE](LICENSE).

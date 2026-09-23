"""Optional Torch frontend contracts.

Importing this package is intentionally Torch-free. Torch is loaded only when
``check_version`` or a later explicit observer/inventory/backend entry point is
used. ``RelocBackend`` (a ``torch.compile`` backend) and ``eager_transfers``
(a scoped dispatch mode) are the opt-in execution entry points.
"""

from .compat import CompatibilityError, check_version
from .eligibility import classify
from .records import Eligibility, GraphRecord, TensorMetadata, TransferRecord

__all__ = (
    "CompatibilityError",
    "Eligibility",
    "GraphRecord",
    "TensorMetadata",
    "TransferRecord",
    "check_version",
    "classify",
    "observe_transfers",
    "TransferObserver",
    "graph_inventory",
    "normalize_graph",
    "import_graph",
    "CompilerClient",
    "CompiledRecipe",
    "UnsupportedRecipe",
    "RelocBackend",
    "eager_transfers",
    "prepare_weights",
    "PreparedWeights",
)


def __getattr__(name):
    if name in {"CompilerClient", "CompiledRecipe", "UnsupportedRecipe"}:
        from .compiler import CompilerClient, CompiledRecipe, UnsupportedRecipe

        globals().update(
            CompilerClient=CompilerClient,
            CompiledRecipe=CompiledRecipe,
            UnsupportedRecipe=UnsupportedRecipe,
        )
        return globals()[name]
    if name == "RelocBackend":
        from .backend import RelocBackend

        globals()[name] = RelocBackend
        return RelocBackend
    if name == "eager_transfers":
        from .eager import eager_transfers

        globals()[name] = eager_transfers
        return eager_transfers
    if name in {"prepare_weights", "PreparedWeights"}:
        from .weights import PreparedWeights, prepare_weights

        globals().update(prepare_weights=prepare_weights, PreparedWeights=PreparedWeights)
        return globals()[name]
    if name in {"normalize_graph", "import_graph"}:
        from .fx_import import import_graph, normalize_graph

        globals().update(import_graph=import_graph, normalize_graph=normalize_graph)
        return globals()[name]
    if name == "graph_inventory":
        from .graph_inventory import graph_inventory
        globals()[name] = graph_inventory
        return graph_inventory
    if name in {"observe_transfers", "TransferObserver"}:
        from .observer import TransferObserver, observe_transfers

        return {"observe_transfers": observe_transfers, "TransferObserver": TransferObserver}[name]
    raise AttributeError(name)

"""Optional Torch frontend contracts.

Importing this package is intentionally Torch-free. Torch is loaded only when
``check_version`` or a later explicit observer/inventory entry point is used.
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

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
)


def __getattr__(name):
    if name in {"observe_transfers", "TransferObserver"}:
        from .observer import TransferObserver, observe_transfers

        return {"observe_transfers": observe_transfers, "TransferObserver": TransferObserver}[name]
    raise AttributeError(name)

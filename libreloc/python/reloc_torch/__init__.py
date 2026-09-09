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
)

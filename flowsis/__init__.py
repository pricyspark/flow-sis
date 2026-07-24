from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import FlowSISBase
    from .mask_head import MaskHead
    from .model import FlowSIS


def __getattr__(name: str) -> Any:
    if name == "FlowSISBase":
        from .base import FlowSISBase

        return FlowSISBase
    if name == "FlowSIS":
        from .model import FlowSIS

        return FlowSIS
    if name == "MaskHead":
        from .mask_head import MaskHead

        return MaskHead
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FlowSISBase",
    "FlowSIS",
    "MaskHead",
]

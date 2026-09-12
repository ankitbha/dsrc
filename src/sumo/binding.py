"""The SUMO Python binding, in one place so importing it pulls in nothing else.

`libsumo` runs in-process and is roughly an order of magnitude faster than TraCI's
socket, which would otherwise dominate a long episode. It is process-global, so only one
environment may be live at a time, and whoever opened it is responsible for closing it.

This lived at the top of the topology environment, which meant importing the binding
pulled in the road builder, the topology view, the sensing model and the metrics stack.
Those belong to the earlier local-sensing formulation and are gone; the binding is not.
"""
from __future__ import annotations

try:  # pragma: no cover - exercised by whichever binding is installed
    import libsumo as _sumo

    _BINDING = "libsumo"
except ImportError:  # pragma: no cover
    import traci as _sumo

    _BINDING = "traci"

__all__ = ["_sumo", "_BINDING"]

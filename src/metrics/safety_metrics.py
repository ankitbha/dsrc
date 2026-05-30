from __future__ import annotations

from typing import Any, Mapping, Sequence


def diagnostic_count(
    diagnostics: Mapping[str, Sequence[Mapping[str, Any]]],
    key: str,
) -> int:
    return len(diagnostics.get(key, ()))


def rear_ttc(
    *,
    rear_gap_m: float,
    rear_relative_speed_mps: float,
) -> float:
    if rear_relative_speed_mps <= 0:
        return float("inf")
    return max(0.0, rear_gap_m / rear_relative_speed_mps)


def hard_brake_count(accelerations_mps2: Sequence[float], threshold_mps2: float) -> int:
    return sum(1 for acceleration in accelerations_mps2 if acceleration <= threshold_mps2)

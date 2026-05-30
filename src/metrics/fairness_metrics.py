from __future__ import annotations

from typing import Mapping, Sequence


def jain_fairness(values: Sequence[float] | object) -> float:
    value_list = [float(value) for value in values]
    if not value_list or all(value == 0.0 for value in value_list):
        return 1.0
    numerator = float(sum(value_list) ** 2)
    denominator = float(len(value_list) * sum(value * value for value in value_list))
    return numerator / denominator if denominator > 0 else 1.0


def branch_metric_maps(
    branch_completed: Mapping[str, int],
    branch_travel_times: Mapping[str, Sequence[float]],
) -> tuple[dict[str, int], dict[str, float]]:
    branch_ids = sorted(set(branch_completed) | set(branch_travel_times))
    throughput = {branch_id: int(branch_completed.get(branch_id, 0)) for branch_id in branch_ids}
    travel_time_mean = {
        branch_id: _mean(branch_travel_times.get(branch_id, ())) if branch_travel_times.get(branch_id) else 0.0
        for branch_id in branch_ids
    }
    return throughput, travel_time_mean


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0

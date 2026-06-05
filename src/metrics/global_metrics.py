from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.metrics.fairness_metrics import branch_metric_maps, jain_fairness
from src.metrics.safety_metrics import diagnostic_count


@dataclass(frozen=True)
class MetricThresholds:
    queue_speed_mps: float = 5.0
    hard_braking_mps2: float = -3.0
    throughput_window_s: float = 60.0
    low_speed_free_flow_delta_mps: float = 8.0
    uncongested_density_threshold_veh_per_km: float = 12.0


def metric_thresholds_from_config(config: Mapping[str, Any] | None) -> MetricThresholds:
    cfg = dict(config or {})
    metrics_cfg = cfg.get("metrics", {})
    if not isinstance(metrics_cfg, Mapping):
        metrics_cfg = {}
    thresholds = metrics_cfg.get("thresholds", {})
    if not isinstance(thresholds, Mapping):
        thresholds = {}
    return MetricThresholds(
        queue_speed_mps=float(thresholds.get("queue_speed_mps", 5.0)),
        hard_braking_mps2=float(thresholds.get("hard_braking_mps2", -3.0)),
        throughput_window_s=float(thresholds.get("throughput_window_s", 60.0)),
        low_speed_free_flow_delta_mps=float(thresholds.get("low_speed_free_flow_delta_mps", 8.0)),
        uncongested_density_threshold_veh_per_km=float(thresholds.get("uncongested_density_threshold_veh_per_km", 12.0)),
    )


def compute_step_metrics(
    *,
    time_s: float,
    active_vehicle_records: Sequence[Mapping[str, Any]],
    segment_metrics: Mapping[str, Mapping[str, Any]],
    diagnostics: Mapping[str, Sequence[Mapping[str, Any]]],
    completed_vehicle_count: int,
    recent_completion_times: Sequence[float],
    completed_travel_times: Sequence[float],
    branch_completed: Mapping[str, int],
    branch_travel_times: Mapping[str, Sequence[float]],
    lane_change_dwell_times: Sequence[float],
    hard_braking_count: int,
    hard_brakes_caused_by_av: int,
    follower_delay_imposed_by_av: float,
    rear_ttc_after_av_lane_change_min: float,
    thresholds: MetricThresholds,
) -> dict[str, Any]:
    speeds = [float(record.get("speed", 0.0)) for record in active_vehicle_records]
    active_av_count = sum(1 for record in active_vehicle_records if record.get("role") == "av")
    collision_count = sum(1 for record in active_vehicle_records if bool(record.get("crashed", False)))
    new_collision_count = sum(1 for record in active_vehicle_records if bool(record.get("newly_crashed", False)))
    queue_length_total = int(sum(int(metrics.get("queue_length", 0)) for metrics in segment_metrics.values()))
    jam_values = [float(metrics.get("jam_fraction", 0.0)) for metrics in segment_metrics.values()]
    lane_change_count = sum(1 for record in active_vehicle_records if bool(record.get("lane_changed_this_step", False)))
    av_distance_km = sum(float(record.get("distance_traveled_m", 0.0)) for record in active_vehicle_records if record.get("role") == "av") / 1000.0
    branch_throughput, branch_travel_time_mean = branch_metric_maps(branch_completed, branch_travel_times)
    throughput_recent = sum(
        1
        for completion_time in recent_completion_times
        if time_s - float(completion_time) <= thresholds.throughput_window_s
    )
    segment_count = max(len(segment_metrics), 1)
    rolling_roadblock_score = sum(float(metrics.get("rolling_roadblock_score", 0.0)) for metrics in segment_metrics.values()) / segment_count
    all_lane_low_speed = sum(float(metrics.get("all_lane_av_low_speed_occupancy", 0.0)) for metrics in segment_metrics.values()) / segment_count
    av_low_speed_uncongested = _av_low_speed_uncongested_fraction(active_vehicle_records, thresholds)
    return {
        "time": float(time_s),
        "active_vehicle_count": len(active_vehicle_records),
        "active_av_count": active_av_count,
        "completed_vehicle_count": int(completed_vehicle_count),
        "mean_speed": _mean(speeds),
        "speed_std": _std(speeds),
        "jam_fraction": _mean(jam_values),
        "hard_braking_count": int(hard_braking_count),
        "collision_count": int(collision_count),
        "new_collision_count": int(new_collision_count),
        "lane_change_count": int(lane_change_count),
        "lane_changes_per_av_km": float(lane_change_count / max(av_distance_km, 1e-9)) if active_av_count else 0.0,
        "min_lane_change_dwell_time": float(min(lane_change_dwell_times)) if lane_change_dwell_times else float("inf"),
        "queue_length_total": queue_length_total,
        "throughput_recent": int(throughput_recent),
        "travel_time_mean": _mean(completed_travel_times),
        "travel_time_count": len(completed_travel_times),
        "hard_brakes_caused_by_av": int(hard_brakes_caused_by_av),
        "rear_ttc_after_av_lane_change_min": float(rear_ttc_after_av_lane_change_min),
        "follower_delay_imposed_by_av": float(follower_delay_imposed_by_av),
        "av_low_speed_uncongested_fraction": av_low_speed_uncongested,
        "all_lane_av_low_speed_occupancy": all_lane_low_speed,
        "rolling_roadblock_score": rolling_roadblock_score,
        "safety_masked_action_count": diagnostic_count(diagnostics, "safety_masked_action"),
        "etiquette_blocked_action_count": diagnostic_count(diagnostics, "etiquette_blocked_action"),
        "follower_disruption_blocked_count": diagnostic_count(diagnostics, "follower_disruption_blocked"),
        "external_safety_override_count": diagnostic_count(diagnostics, "external_safety_override"),
        "simulator_blocked_action_count": diagnostic_count(diagnostics, "simulator_blocked_action"),
        "branch_throughput": branch_throughput,
        "branch_travel_time_mean": branch_travel_time_mean,
        "fairness_jain": jain_fairness(branch_throughput.values()),
    }


def _av_low_speed_uncongested_fraction(
    active_vehicle_records: Sequence[Mapping[str, Any]],
    thresholds: MetricThresholds,
) -> float:
    av_records = [record for record in active_vehicle_records if record.get("role") == "av"]
    if not av_records:
        return 0.0
    low_speed = 0
    for record in av_records:
        density = float(record.get("segment_density", 0.0))
        speed = float(record.get("speed", 0.0))
        free_flow = float(record.get("free_flow_speed_mps", 30.0))
        if density < thresholds.uncongested_density_threshold_veh_per_km and speed < free_flow - thresholds.low_speed_free_flow_delta_mps:
            low_speed += 1
    return low_speed / len(av_records)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)

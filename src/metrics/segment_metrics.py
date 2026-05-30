from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.metrics.global_metrics import MetricThresholds


def compute_segment_metrics(
    *,
    segment_ids: Sequence[str],
    segment_lengths_m: Mapping[str, float],
    lane_counts: Mapping[str, int],
    active_vehicle_records: Sequence[Mapping[str, Any]],
    step_inflow: Mapping[str, int],
    step_outflow: Mapping[str, int],
    thresholds: MetricThresholds,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {
        segment_id: {
            "vehicle_count": 0,
            "av_count": 0,
            "mean_speed": 0.0,
            "speed_std": 0.0,
            "density": 0.0,
            "queue_length": 0,
            "jam_fraction": 0.0,
            "inflow": int(step_inflow.get(segment_id, 0)),
            "outflow": int(step_outflow.get(segment_id, 0)),
            "lane_counts": {str(lane_id): 0 for lane_id in range(int(lane_counts.get(segment_id, 1)))},
            "lane_fractions": {str(lane_id): 0.0 for lane_id in range(int(lane_counts.get(segment_id, 1)))},
            "lane_av_counts": {str(lane_id): 0 for lane_id in range(int(lane_counts.get(segment_id, 1)))},
            "all_lane_av_low_speed_occupancy": 0.0,
            "rolling_roadblock_score": 0.0,
            "branch_queue_length": {},
            "spillback_depth": 0.0,
        }
        for segment_id in segment_ids
    }
    speeds: dict[str, list[float]] = {segment_id: [] for segment_id in segment_ids}
    av_speeds: dict[str, list[float]] = {segment_id: [] for segment_id in segment_ids}
    branch_queues: dict[str, dict[str, int]] = {segment_id: {} for segment_id in segment_ids}
    for vehicle in active_vehicle_records:
        segment_id = vehicle.get("segment_id")
        if segment_id not in records:
            continue
        speed = float(vehicle.get("speed", 0.0))
        role = vehicle.get("role")
        lane_id = str(int(vehicle.get("lane_id", 0)))
        branch_id = str(vehicle.get("branch_id", "unknown"))
        records[segment_id]["vehicle_count"] += 1
        records[segment_id]["lane_counts"][lane_id] = int(records[segment_id]["lane_counts"].get(lane_id, 0)) + 1
        speeds[segment_id].append(speed)
        if speed < thresholds.queue_speed_mps:
            records[segment_id]["queue_length"] += 1
            branch_queues[segment_id][branch_id] = branch_queues[segment_id].get(branch_id, 0) + 1
        if role == "av":
            records[segment_id]["av_count"] += 1
            records[segment_id]["lane_av_counts"][lane_id] = int(records[segment_id]["lane_av_counts"].get(lane_id, 0)) + 1
            av_speeds[segment_id].append(speed)

    for segment_id in segment_ids:
        segment_speeds = speeds[segment_id]
        lane_total = max(int(records[segment_id]["vehicle_count"]), 1)
        lane_count = max(int(lane_counts.get(segment_id, 1)), 1)
        length_km = float(segment_lengths_m[segment_id]) / 1000.0
        records[segment_id]["density"] = int(records[segment_id]["vehicle_count"]) / max(length_km, 1e-9)
        records[segment_id]["branch_queue_length"] = branch_queues[segment_id]
        records[segment_id]["spillback_depth"] = (
            float(records[segment_id]["queue_length"]) / max(float(records[segment_id]["vehicle_count"]), 1.0)
        )
        for lane_id, count in records[segment_id]["lane_counts"].items():
            records[segment_id]["lane_fractions"][lane_id] = int(count) / lane_total
        if segment_speeds:
            records[segment_id]["mean_speed"] = _mean(segment_speeds)
            records[segment_id]["speed_std"] = _std(segment_speeds)
            records[segment_id]["jam_fraction"] = _mean([float(speed < thresholds.queue_speed_mps) for speed in segment_speeds])
        records[segment_id]["all_lane_av_low_speed_occupancy"] = _all_lane_av_low_speed_occupancy(
            lane_av_counts=records[segment_id]["lane_av_counts"],
            av_speeds=av_speeds[segment_id],
            lane_count=lane_count,
            free_flow_speed_mps=_mean_free_flow(active_vehicle_records, segment_id),
            thresholds=thresholds,
        )
        records[segment_id]["rolling_roadblock_score"] = _rolling_roadblock_score(records[segment_id])
    return records


def _all_lane_av_low_speed_occupancy(
    *,
    lane_av_counts: Mapping[str, int],
    av_speeds: Sequence[float],
    lane_count: int,
    free_flow_speed_mps: float,
    thresholds: MetricThresholds,
) -> float:
    if lane_count <= 0 or not av_speeds:
        return 0.0
    all_lanes_have_av = all(int(lane_av_counts.get(str(lane_id), 0)) > 0 for lane_id in range(lane_count))
    if not all_lanes_have_av:
        return 0.0
    return 1.0 if _mean(av_speeds) < free_flow_speed_mps - thresholds.low_speed_free_flow_delta_mps else 0.0


def _rolling_roadblock_score(segment_record: Mapping[str, Any]) -> float:
    if not float(segment_record.get("all_lane_av_low_speed_occupancy", 0.0)):
        return 0.0
    if float(segment_record.get("jam_fraction", 0.0)) > 0.25:
        return 0.0
    if int(segment_record.get("queue_length", 0)) > 0:
        return 0.0
    return 1.0


def _mean_free_flow(active_vehicle_records: Sequence[Mapping[str, Any]], segment_id: str) -> float:
    speeds = [
        float(record.get("free_flow_speed_mps", 30.0))
        for record in active_vehicle_records
        if record.get("segment_id") == segment_id
    ]
    return _mean(speeds) if speeds else 30.0


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)

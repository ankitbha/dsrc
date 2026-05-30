# Metrics and Outputs

This file standardizes experiment output names and metric identifiers that later analysis code will depend on.

## Standard Output Files

Every experiment should use these canonical artifact names:

- `episode_summary.json`
- `step_metrics.csv`
- `segment_metrics.csv`

Expected output root:

- `outputs/metrics/<experiment_id>/`

Parquet can be added later as an optional compatibility artifact, but JSON/CSV are the canonical task-6 outputs.

## Metric Thresholds

Experiment configs may define optional metric thresholds:

- `metrics.thresholds.queue_speed_mps`
- `metrics.thresholds.hard_braking_mps2`
- `metrics.thresholds.throughput_window_s`
- `metrics.thresholds.low_speed_free_flow_delta_mps`
- `metrics.thresholds.uncongested_density_threshold_veh_per_km`

## Step Metrics

Standard step-level metric names:

- `time`
- `active_vehicle_count`
- `active_av_count`
- `completed_vehicle_count`
- `mean_speed`
- `speed_std`
- `jam_fraction`
- `hard_braking_count`
- `collision_count`
- `lane_change_count`
- `lane_changes_per_av_km`
- `min_lane_change_dwell_time`
- `queue_length_total`
- `throughput_recent`
- `hard_brakes_caused_by_av`
- `rear_ttc_after_av_lane_change_min`
- `follower_delay_imposed_by_av`
- `av_low_speed_uncongested_fraction`
- `all_lane_av_low_speed_occupancy`
- `rolling_roadblock_score`
- `safety_masked_action_count`
- `etiquette_blocked_action_count`
- `follower_disruption_blocked_count`
- `external_safety_override_count`
- `simulator_blocked_action_count`

## Segment Metrics

Standard segment-level metric names:

- `vehicle_count`
- `av_count`
- `mean_speed`
- `speed_std`
- `density`
- `queue_length`
- `jam_fraction`
- `inflow`
- `outflow`

Segment records may also include additive lane-use and obstruction-analysis fields:

- `lane_counts`
- `lane_fractions`
- `lane_av_counts`
- `all_lane_av_low_speed_occupancy`
- `rolling_roadblock_score`
- `branch_queue_length`
- `spillback_depth`

Active counts and segment counts should exclude vehicles that have exited the topology. Exited vehicles should contribute to completed vehicle counts, throughput, and episode summaries, but not to active AV/RV computation.

Safety metrics should distinguish integrated safety masking, etiquette blocks, follower-disruption blocks, external safety overrides, and simulator-level blocked actions.

The `rolling_roadblock_score` is the fraction of time where AVs occupy all lanes in a segment, AV mean speed is substantially below free-flow speed, and downstream congestion does not justify the slowdown. This score should remain near zero.

## Tree and Merge Metrics

Additional canonical metrics for merge and tree experiments:

- `merge_delay`
- `spillback_depth`
- `branch_throughput`
- `branch_queue_length`
- `branch_travel_time_mean`
- `fairness_jain`

Branch fairness is safety-critical for merge and tree experiments: a controller should not improve trunk throughput by starving one branch or origin.

## Repo Ownership

Metric collection code should later live under:

- `src/metrics/`

Output-writing logic should later be shared by:

- `scripts/run_baseline.py`
- `scripts/evaluate_policy.py`
- `src/analysis/`

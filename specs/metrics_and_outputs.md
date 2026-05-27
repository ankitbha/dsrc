# Metrics and Outputs

This file standardizes experiment output names and metric identifiers that later analysis code will depend on.

## Standard Output Files

Every experiment should use these canonical artifact names:

- `episode_summary.json`
- `step_metrics.csv`
- `segment_metrics.parquet`

Expected output root:

- `outputs/metrics/<experiment_id>/`

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
- `queue_length_total`
- `throughput_recent`

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

## Tree and Merge Metrics

Additional canonical metrics for merge and tree experiments:

- `merge_delay`
- `spillback_depth`
- `branch_throughput`
- `branch_queue_length`
- `branch_travel_time_mean`
- `fairness_jain`

## Repo Ownership

Metric collection code should later live under:

- `src/metrics/`

Output-writing logic should later be shared by:

- `scripts/run_baseline.py`
- `scripts/evaluate_policy.py`
- `src/analysis/`


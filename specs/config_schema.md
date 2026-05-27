# Config Schema

This file standardizes config families, file locations, and the composed experiment bundle.

## Config Families

All configs should use YAML and live under `configs/`:

- `configs/topology/`
- `configs/demand/`
- `configs/human_models/`
- `configs/experiments/`
- `configs/training/`

## Family Rules

Each family file should include:

- `id`
- `kind`
- family-specific content

Expected `kind` values:

- `topology`
- `demand`
- `human_model`
- `training`
- `experiment`

## Experiment Config Contract

Each experiment config should reference the family entries it composes:

```yaml
id: exp_ring_wave_damping
kind: experiment
refs:
  topology: ring
  demand: medium
  human_model: heterogeneous
  training: mappo
experiment:
  id: exp_ring_wave_damping
  seed: 7
controller:
  family: rl
  name: mappo
metrics:
  primary:
    - mean_speed
outputs:
  episode_summary: outputs/metrics/exp_ring_wave_damping/episode_summary.json
overrides:
  demand:
    total_vehicles_per_hour: 1600
```

## Merge Precedence

The composed config bundle should load in this order:

1. referenced family configs from `refs`
2. experiment-local sections such as `experiment`, `controller`, `metrics`, and `outputs`
3. `overrides` applied last to the referenced family sections

## Composed Bundle Shape

The loader should return one dict with these top-level sections:

- `experiment`
- `topology`
- `demand`
- `human_model`
- `training`
- `controller`
- `metrics`
- `outputs`
- `resolved_refs`

## Repo Ownership

Executable config loading should live in:

- `src/config/loaders.py`

Human-readable config guidance should live in:

- `configs/README.md`


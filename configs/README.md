# Config Layout

All DSRC experiment configs should live under `configs/` and use YAML.

## Families

- `topology/`: road layout and detector placement
- `demand/`: inflow, AV penetration, burst behavior, branch splits
- `human_models/`: cautious, normal, aggressive, heterogeneous driver settings
- `experiments/`: experiment references, controller settings, outputs, and overrides
- `training/`: RL algorithm and optimizer defaults

## Naming Rules

Standard topology IDs:

- `ring`
- `straight_single_lane`
- `straight_multilane`
- `merge`
- `inverted_tree`

Standard vehicle roles:

- `av`
- `human`

## Composition Model

Experiment configs should reference family configs through a `refs` block. The config loader resolves those references and applies experiment overrides last.

Experiment or environment configs may include optional local AV sensing settings. If omitted, deterministic defaults are used:

```yaml
sensing:
  range_m: 150.0
  latency_s: 0.0
  position_noise_std: 0.0
  speed_noise_std: 0.0
  density_bin_edges_veh_per_km: [12.0, 30.0]
  mean_speed_bin_edges_mps: [8.0, 18.0]
  queue_speed_mps: 5.0
```

Controller configs may set `controller.safety_mode`. Use `integrated_rl` for the DSRC safety/physical-control layer and `simulator_default` for AV baselines that should use simulator/human-like safety behavior instead.

## Example Files

- `topology/ring.yaml`
- `demand/medium.yaml`
- `human_models/heterogeneous.yaml`
- `experiments/exp_ring_wave_damping.yaml`
- `training/mappo.yaml`
- `training/shared_ppo.yaml`
- `training/ippo.yaml`

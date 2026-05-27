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

## Example Files

- `topology/ring.yaml`
- `demand/medium.yaml`
- `human_models/heterogeneous.yaml`
- `experiments/exp_ring_wave_damping.yaml`
- `training/mappo.yaml`


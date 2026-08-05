# DSRC Simulation Repository

This repository studies centralized-training, decentralized-execution traffic control for self-regulating autonomous vehicles. The core question is whether a sparse fleet of AVs can realize network-control effects from inside the traffic stream, without infrastructure sensing, obstruction, lane hogging, or coordinated roadblocks.

The main hypothesis is:

> A small fraction of autonomous vehicles, trained with centralized traffic-level feedback but deployed with only local noisy sensing, can physically realize network-level congestion-control policies through smooth desired-speed and desired-headway targets with conservative lane preferences.

AVs act as mobile actuators. They use local observations to choose public v2 actions such as desired speed, desired headway, conservative lane preference, and merge behavior. Human vehicles are influenced only through ordinary traffic dynamics.

## Project Structure

- `src/envs/`: topology environment, local observations, vehicle lifecycle, and safety integration.
- `src/baselines/`: infrastructure-free non-learning AV baselines.
- `src/rl/`: shared PPO, IPPO, and MAPPO training components.
- `scripts/`: baseline runs, topology validation, RL training, learned-policy evaluation, and task validation.
- `configs/`: topology, demand, human model, and training configuration families.
- `specs/`: action, observation, controller, metrics, and environment contracts.
- `plans/`: project plans and task list.
- `tests/`: unit and smoke tests.

## Topologies

The simulator uses a topology ladder:

1. `ring`: closed road for stop-and-go wave damping and speed stabilization.
2. `straight_single_lane`: open highway with inflow, outflow, throughput, and travel-time metrics.
3. `straight_multilane`: open highway with conservative lane preferences and lane-change suppression.
4. `merge`: Y-merge bottleneck for cooperative gap creation and headway control.
5. `inverted_tree`: multi-branch-to-trunk network without a final lane-drop bottleneck for branch fairness.
6. `inverted_tree_bottleneck`: the same tree structure with a final one-lane bottleneck for spillback and stress tests.

## Controllers

Task-9 baselines are infrastructure-free and decentralized. Except for `no_av`, every baseline acts only from each AV's public local observation and must not consume global state.

- `no_av`: human-only reference case.
- `random_av`: seeded random valid public AV actions.
- `selfish_av`: ego-progress-oriented AV behavior.
- `density_lookup`: local density and queue estimate mapped to speed/headway bins.
- `dynamic_speed_limit`: local AV speed advisory from locally sensed congestion.
- `av_mediated_speed_harmonization`: local flow matching and speed-mismatch damping.
- `backpressure`: local pressure-inspired speed/headway metering near bottlenecks.
- `cooperative_smoothing`: hand-designed local smoothing using public local aggregate fields.

The RL stack trains a shared decentralized actor with:

- `shared_ppo`: shared actor and local critic.
- `ippo`: independent PPO-style local critic with shared actor parameters.
- `mappo`: CTDE training with a global critic and local-observation-only actor execution.

## Safety Model

AV control should improve traffic through smooth longitudinal damping and cooperative gap creation, not through obstruction. The safety and etiquette layer guards against:

- unsafe front or rear gaps
- excessive lane changes
- low-speed driving in uncongested traffic
- follower disruption
- rolling-roadblock behavior
- coordinated all-lane slowdown without downstream congestion

## Running Baselines

Run one baseline episode:

```bash
python scripts/run_baseline.py \
  --controller density_lookup \
  --topology ring \
  --demand medium \
  --seed 7 \
  --duration-steps 120 \
  --output-root outputs/metrics/baselines
```

Run topology-by-topology baseline validation:

```bash
python scripts/validate_topology_baselines.py --smoke
```

## Training And Evaluating RL

Train a tiny shared-PPO smoke run:

```bash
python scripts/train_policy.py \
  --training shared_ppo \
  --topology ring \
  --total-updates 1 \
  --rollout-steps 8 \
  --duration-steps 20 \
  --output-root outputs/checkpoints
```

Evaluate a learned actor:

```bash
python scripts/evaluate_policy.py \
  --actor outputs/checkpoints/shared_ppo_ring_speed_only_seed7/actor.pt \
  --topology ring \
  --duration-steps 120 \
  --output-root outputs/metrics/learned
```

Validate the training and evaluation entrypoints:

```bash
python scripts/validate_training_eval.py --matrix smoke
```

## HPC / Overlay Usage

This project is intended to run inside the project SIF and overlay environment on the HPC system. The key local files are:

- `cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif`
- `dsrc_gpu_env.ext3`

Use the existing environment wrapper for queued tests:

```bash
env_scripts/run_env_tests.sh
```

For direct Slurm/container runs, use the SIF plus overlay, source `/ext3/env.sh`, then run commands from the repository root.

## Outputs

Canonical run artifacts are written as:

- `episode_summary.json`
- `step_metrics.csv`
- `segment_metrics.csv`

Default output roots:

- checkpoints: `outputs/checkpoints/`
- baseline metrics: `outputs/metrics/baselines/`
- learned-policy metrics: `outputs/metrics/learned/`
- validation reports: `outputs/validation/`

Generated outputs are ignored by git.

## Current Project Story

The intended final story is that sparse AVs can approximate traffic-control policies from within the traffic stream. With centralized training but decentralized execution, AVs learn local speed and headway targets with conservative lane preferences. A hard safety and etiquette layer prevents obstruction, while experiments across ring, straight highway, merge, and inverted-tree topologies test wave damping, throughput, merge gap creation, spillback reduction, and branch fairness.

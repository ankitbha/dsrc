# DSRC Integrated Project Plan

This is the top-level paper-facing roadmap. The subsystem details remain in:

- `plans/plan_simulations.md`
- `plans/plan_world_model.md`
- `plans/plan_deployment.md`
- `plans/task_list.md`

## 1. Paper Claim

Sparse autonomous vehicles can regulate traffic from inside the flow using only local sensing, conservative public actions, and a common execution-time safety layer. The main technical contribution is a differentiable functional graph world model that represents each road segment as a spatial traffic field and uses learned graph dynamics for model-based policy improvement.

Final performance claims must be measured in the original highway-env simulator through the project environment wrapper. Learned-model rollouts are used for training, prediction evaluation, and ablation, not as the final evaluator.

## 2. System Stack

The project has three connected layers.

### Simulator and control foundation

The current repo provides:

```text
highway-env wrapper
topology ladder
demand generation
human-driver profiles
local AV sensing
public v2 action schema
common safety/etiquette/physical-control layer
metrics and logging
baseline controllers
model-free PPO/IPPO/MAPPO
```

This foundation establishes that sparse AVs can act as mobile actuators through smooth speed/headway targets and conservative lane preferences.

### Functional graph world model

The main method adds:

```text
TopologySpec -> GraphSpec
vehicle state -> traffic field Z_t
executed AV interventions -> action field U_t
(Z_t, U_t, G) -> differentiable world model -> Z_{t+1}
imagined rollout loss -> model-based policy improvement
final policy evaluation -> highway-env
```

Each graph edge is a road segment with a 1D traffic field, not just one scalar node feature. This is the key methodological distinction.

### Advisory-only deployment prototype

The deployment plan demonstrates feasibility of local observation and policy inference on edge hardware:

```text
camera/GPS/optional OBD
perception and tracking
observation builder
trained actor inference
dashboard and logs
no actuation
```

Deployment supports the practical story but is not the core control evaluation.

## 3. Implementation Roadmap

### Phase 1: Preserve and validate simulator foundation

Maintain the current environment, topology, demand, sensing, safety, metrics, baselines, and model-free RL stack. Continue using smoke validations before adding major world-model changes.

Primary scripts:

```text
scripts/run_baseline.py
scripts/train_policy.py
scripts/evaluate_policy.py
scripts/validate_project_interface.py
scripts/validate_topology_baselines.py
scripts/validate_training_eval.py
```

### Phase 2: Build world-model data path

Add `src/world_model/` with graph adapters, field extraction, executed-action aggregation, transition schemas, and dataset storage.

Deliverables:

```text
GraphSpec for all current topologies
TrafficFieldExtractor
ActionFieldAggregator
WorldModelTransition dataset
scripts/collect_world_model_data.py
scripts/validate_world_model_pipeline.py
```

### Phase 3: Train and validate world models

Implement the functional graph model and ablations:

```text
persistence baseline
scalar-GNN world model
no-graph Conv1D model
functional graph world model
functional graph + physics losses
```

Deliverables:

```text
scripts/train_world_model.py
scripts/evaluate_world_model.py
world-model prediction reports
multi-step rollout validation
```

Acceptance:

```text
functional graph model beats persistence on one-step and multi-step prediction
functional edge-field representation beats scalar graph ablation
graph message passing improves over no-graph Conv1D where topology matters
```

### Phase 4: Add model-based policy improvement

Freeze the trained world model and add an imagined rollout objective as an auxiliary policy loss.

Deliverables:

```text
src/world_model/policies/
scripts/train_model_based_policy.py
configs/training/model_based_policy.yaml
alpha sweep where alpha=0 reproduces model-free behavior
```

Final policies still execute in `HighwayTopologyEnv` and pass through the common runtime safety layer.

### Phase 5: Build experiment launch infrastructure

Before final experiments, add launch and dry-run support for the complete matrix.

Deliverables:

```text
scripts/run_experiment_matrix.py
experiment configs under configs/experiments/
plot/table scripts
artifact manifest
```

The launcher should cover model-free baselines, world-model prediction runs, model-based policy runs, ablations, and deployment metrics.

### Phase 6: Run final experiments and analysis

Only after all code paths and launchers are in place:

```text
run final training sweeps
run highway-env final evaluations
run world-model prediction evaluations
run ablations
run deployment prototype measurements
generate figures and tables
```

## 4. Experiment Matrix

### Topologies

```text
ring
straight_single_lane
straight_multilane
merge
inverted_tree
inverted_tree_bottleneck
```

### Demand

```text
low
medium
high
burst
```

### Human models

```text
normal
heterogeneous
aggressive
```

### AV penetration

```text
5%
10%
20%
```

### Controller and method families

```text
no_av
random_av
selfish_av
density_lookup
dynamic_speed_limit
av_mediated_speed_harmonization
backpressure
cooperative_smoothing
SharedPPO
IPPO
MAPPO
scalar-GNN model-based policy
no-graph Conv1D model-based policy
functional graph model-based policy
functional graph + physics losses
```

### Primary evaluation axes

```text
throughput
mean travel time
mean speed
speed variance
jam fraction
queue length
merge delay
spillback depth
branch fairness
hard braking
collisions
follower disruption
lane-change rate
rolling-roadblock score
model one-step prediction error
model multi-step prediction error
sample efficiency
deployment latency/FPS
```

## 5. Safety And Evaluation Principles

- All controllers propose public v2 AV actions.
- The common DSRC safety layer in `HighwayTopologyEnv.step()` is the runtime enforcement path.
- Imagined world-model rollouts may include differentiable soft safety penalties as training regularizers.
- The learned world model is never the final evaluator.
- Humans must remain passable when safe; performance must not come from obstruction.
- Branch fairness is required for merge/tree results.

## 6. Deployment Link

The Jetson prototype demonstrates that the local observation and actor inference pipeline can plausibly run on vehicle-edge hardware.

Deployment outputs:

```text
perception FPS
policy inference latency
end-to-end latency
observation quality
example dashboard
advisory-only safety statement
```

The prototype is non-actuating and should be presented as edge-feasibility evidence, not as an autonomous driving deployment.

## 7. Final Paper Story

The paper should read as:

1. Simulator foundation: sparse AVs interact with traffic using local observations and safe public actions.
2. Baselines and model-free RL establish the traffic-control setting.
3. Functional graph world model is introduced as the main contribution.
4. World-model prediction ablations prove the value of edge-field graph dynamics.
5. Model-based policy results show whether the learned surrogate improves sample efficiency, transfer, or final highway-env performance.
6. Deployment prototype shows the local observation and inference stack can run on edge hardware in advisory-only mode.

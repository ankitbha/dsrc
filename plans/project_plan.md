# DSRC Integrated Project Plan

This is the top-level paper-facing roadmap. The subsystem details remain in:

- `plans/plan_simulations.md`
- `plans/plan_deployment.md`
- `plans/task_list.md`

## 1. Paper Claim

Sparse autonomous vehicles can regulate traffic from inside the flow using only local sensing, conservative public actions, and a common execution-time safety layer. The contribution is a deployability argument backed by a working system: how little must change in an ordinary vehicle for it to self-regulate, and how few such vehicles a road network needs.

Two consequences shape everything below. Nothing central sits in the real-time loop: the cloud trains and ships model updates offline, and communication is an observability hint, not a control dependency. And performance claims are separated by kind — traffic-control effects are measured in the original highway-env simulator through the project environment wrapper, while edge-feasibility claims are measured on the prototype hardware. Neither substitutes for the other.

## 2. System Stack

The project has two connected layers.

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

This foundation establishes that sparse AVs can act as mobile actuators through smooth speed/headway targets and conservative lane preferences, and it is where every traffic-control number comes from.

### Advisory-only deployment prototype

The deployment plan demonstrates that local observation and policy inference run on vehicle-edge hardware:

```text
camera/GPS/optional OBD
perception and tracking
observation builder (sim-parity contract)
trained actor inference
safety/etiquette filter
dashboard and logs
no actuation
```

Sensing spans two devices, and the split is deliberate: the phone captures and forwards, while the Jetson owns every sensing setting as well as the policy. The phone holds none of the state that would justify a sensing decision.

The prototype is non-actuating. It is edge-feasibility evidence and the deployment argument's proof of existence, not an autonomous driving deployment.

## 3. Implementation Roadmap

### Phase 1: Preserve and validate the simulator foundation

Maintain the current environment, topology, demand, sensing, safety, metrics, baselines, and model-free RL stack. Keep running smoke validations before any behavioral change.

Primary scripts:

```text
scripts/run_baseline.py
scripts/train_policy.py
scripts/evaluate_policy.py
scripts/validate_project_interface.py
scripts/validate_topology_baselines.py
scripts/validate_training_eval.py
```

### Phase 2: Close the prototype hardware loop

The prototype is code-complete and validated on simulated drives. Remaining work is hardware, not software.

Deliverables:

```text
GPS device permissions resolved and rate confirmed
camera attached, selfcheck passing
mounted-camera calibration committed to config
recorded live run replayed for decision agreement
advisory hysteresis to damp leader-acquisition flicker
driver-facing readout showing the filtered cap, not the raw decode
in-vehicle advisory-only drive
```

### Phase 3: Extend prototype observation coverage

Roughly half the deployed observation is currently neutral fallback. Each item below converts placeholders into measured fields.

Deliverables:

```text
rear camera for follower and rear-gap fields
OBD-II speed as the primary ego-speed source
two-unit cooperative demo for nearby-AV and cooperation fields
map matching for distance-to-merge and downstream bottleneck
```

### Phase 4: Local-plus-aggregate simulation study

Replace global observation with the local-plus-aggregate contract and quantify what the aggregate buys, under one safety layer.

Deliverables:

```text
local-only, aggregate-assisted, and oracle controller comparison
message loss, delay, and staleness sweeps
sensing range and noise sweeps
compliance and penetration sweeps
bottleneck seeding vs uniform adoption at equal fleet size
```

### Phase 5: Build experiment launch infrastructure

Before final experiments, add launch and dry-run support for the complete matrix.

Deliverables:

```text
scripts/run_experiment_matrix.py
experiment configs under configs/experiments/
plot/table scripts
artifact manifest
```

The launcher should cover model-free baselines, learned-policy runs, robustness sweeps, and deployment metrics.

### Phase 6: Run final experiments and analysis

Only after all code paths and launchers are in place:

```text
run final training sweeps
run highway-env final evaluations
run robustness and partial-deployment sweeps
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
```

### Observation regimes

```text
local-only
local plus aggregate
oracle/global (reference upper bound, not a deployable condition)
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
sample efficiency
deployment latency/FPS
observation quality and field provenance
```

## 5. Safety And Evaluation Principles

- All controllers propose public v2 AV actions.
- The common DSRC safety layer in `HighwayTopologyEnv.step()` is the runtime enforcement path.
- The action to train on and measure is the one actually executed, not the raw proposal.
- Humans must remain passable when safe; performance must not come from obstruction.
- Branch fairness is required for merge/tree results.
- A network gain is never acceptable if bought by making human traffic less safe: report hard braking, collisions, and follower delay alongside every throughput result.
- The safety layer is specified, tested, and audited separately from the learned controller.

## 6. Deployment Link

The Jetson prototype is a first-class result, not an appendix. It establishes that the observation builder, actor inference, and safety filter fit inside a real-time budget on commodity hardware.

Deployment outputs:

```text
perception FPS
policy inference latency
end-to-end latency
ego-speed accuracy under dropout
observation quality and missingness
sim-to-prototype observation alignment
example dashboard
advisory-only safety statement
```

Two properties carry the deployment argument and should be reported explicitly: the observation contract is shared bit-for-bit with training, and every field is logged with its provenance, so a reader can see which values were measured and which were neutral fallbacks.

## 7. Final Paper Story

The paper should read as:

1. Highway congestion is a distributed stability problem, and the vehicle-side actuator has already been demonstrated; what remains is deployment.
2. Self-regulation needs no coordinator: each vehicle decides alone, and communication carries bounded aggregate state rather than commands.
3. The bill of changes that makes one ordinary car self-regulating, ending in a bounded advisory and a safety/etiquette filter.
4. A working edge prototype: sustained real-time operation and end-to-end latency well inside budget on commodity hardware, in advisory-only mode.
5. A minimal deployment model: because congestion is manufactured at bottlenecks, presence in a specific traffic stream beats market share, making a single fleet a sufficient launch vehicle.
6. The open networking questions this system raises: how little communication suffices, whether it can be private, what belongs in the cloud and how late it may be, robustness under partial deployment, and how safety should be verified.

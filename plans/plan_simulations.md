# DSRC Plan

**Centralized training, decentralized execution for self-regulating autonomous vehicles that use local sensing to realize network-level congestion control through smooth speed/headway damping and conservative lane preferences.**

This is a natural extension of the current self regulating cars papers. Earlier sensys draft already motivates vehicle-level local sensing, density estimation, speed regulation, partial adoption, and learning/adaptation beyond the current lookup-table controller. 

Below is the project structure and a step-by-step execution plan.

---

# 1. Core hypothesis

The main hypothesis should be:

> A small fraction of autonomous vehicles, trained with centralized traffic-level feedback but deployed with only local noisy sensing, can physically realize network-level congestion-control policies through smooth desired-speed and desired-headway targets with conservative lane preferences.

This connects several ideas:

1. **Local density and speed advisories** are the interpretable non-learning version.
2. **Self-regulating AVs** are the infrastructure-free physical realization.
3. **CTDE RL** learns when and where AVs should harmonize speed, increase headway, hold lanes, or create merge gaps.

> AVs act as mobile actuators that implement traffic-control policies from inside the flow.

The v2 deployed cooperation model should use only aggregate traffic-state context available in each AV's public local observation, not infrastructure sensing, identity-level V2V messages, runner-aggregated fleet state, or joint lane-occupation plans. Each AV may observe nearby AV count, density, mean speed, queue estimates, downstream congestion estimates, segment target speed, and merge pressure as local observation fields. If no AV is nearby, the policy must fall back to individual local operation.

---

# Safety-Constrained Physical Flow Control

AVs should control flow through smooth longitudinal damping and cooperative gap creation, not through obstruction. Good performance should not be achievable through lane hogging, oscillatory lane changes, rolling roadblocks, or trapping human drivers.

Primary mechanisms:

```text
speed harmonization
adaptive headway control
cooperative merge gap creation
lane-change suppression near bottlenecks
backpressure-inspired speed metering
disturbance absorption / jam wave cancellation
```

The actor should output:

```text
desired_speed_bin
desired_headway_bin
lane_preference: keep | prefer_left_if_safe | prefer_right_if_safe
merge_mode: normal | create_gap | hold_lane
```

The safety/etiquette layer should enforce:

```text
minimum lane-change dwell time
maximum lane changes per km
safe front and rear gaps
rear braking limit for target-lane followers
bounded acceleration and deceleration
minimum contextual speed
no low-speed driving in uncongested conditions
no coordinated all-lane slowdown unless downstream congestion justifies it
```

Allowed communication:

```text
aggregate density
mean speed
queue estimate
downstream congestion estimate
segment-level target speed
merge pressure
```

Disallowed communication:

```text
joint lane occupation plans
AV-to-lane blocking assignments
coordinated roadblock formations
```

---

# 2. Four topology levels

## Level 1: Ring road

Purpose: show emergent stop-and-go wave damping.

```text
closed circular road
```

What it tests:

```text
wave damping
speed stabilization
partial AV penetration
local sensing sufficiency
```

Main actions:

```text
desired speed only
```

Primary metrics:

```text
speed variance
jam duration
wave amplitude
mean speed
recovery time after perturbation
```

This is the cleanest proof of concept.

---

## Level 2: Open straight highway

Purpose: introduce demand, inflow, outflow, throughput, and virtual detectors.

```text
inflow  --->  straight highway  --->  outflow
```

Use two versions:

```text
2A. single-lane straight highway
    isolates speed control

2B. multi-lane straight highway
    introduces conservative lane preferences and lane-change suppression
```

What it tests:

```text
throughput
travel time
lane utilization
demand sensitivity
burst response
```

Primary metrics:

```text
downstream throughput
mean travel time
speed variance
jam fraction
lane utilization
hard braking
```

This is the right “middle” topology because it adds traffic-demand realism without merge complexity.

---

## Level 3: Merge / bottleneck

Purpose: isolate cooperative gap creation, headway control, and conservative lane preference behavior.

Use a Y-merge.

```text
mainline ----\
              ---> downstream trunk
ramp --------/
```

What it tests:

```text
merge coordination
gap creation
conservative lane preference
bottleneck smoothing
queue reduction
```

Primary metrics:

```text
merge delay
queue length upstream of merge
trunk throughput
hard braking near merge
speed drop near bottleneck
lane distribution before merge
```

This is the necessary bridge between the straight road and the inverted tree.

---

## Level 4: Inverted tree

Purpose: final network-level stress test.

```text
A1 ----\
A2 ----- B1 ----\
A3 ----/         \
                  C ---- D ---- exit
A4 ----\         /
A5 ----- B2 ----/
A6 ----/
```

What it tests:

```text
multi-branch congestion propagation
network-level regulation
fairness across branches
spillback control
multi-merge coordination
```

Primary metrics:

```text
downstream trunk throughput
queue length per branch
mean travel time per origin branch
merge delay at each merge node
segment occupancy over time
speed variance per segment
spillback depth
branch fairness
```

Add a fairness metric. Otherwise a controller could improve trunk throughput by starving one branch.

Useful fairness metrics:

```text
std(branch travel times)
max branch queue length
min_branch_throughput / avg_branch_throughput
Jain fairness over branch throughputs
```

---

# 3. Updated project structure

The current repository is organized like this, with world-model additions planned under the same conventions:

```text
dsrc/
  REDME.md

  configs/
    topology/
      ring.yaml
      straight_single_lane.yaml
      straight_multilane.yaml
      merge.yaml
      inverted_tree.yaml
      inverted_tree_bottleneck.yaml

    demand/
      low.yaml
      medium.yaml
      high.yaml
      burst.yaml

    human_models/
      cautious.yaml
      normal.yaml
      aggressive.yaml
      heterogeneous.yaml

    experiments/
      exp_ring_wave_damping.yaml
      ... planned final experiment configs ...

    training/
      shared_ppo.yaml
      ippo.yaml
      mappo.yaml
      model_based_policy.yaml        # planned

    world_model/                     # planned
      functional_graph.yaml
      scalar_gnn.yaml
      persistence.yaml

  src/
    envs/
      base_ctde_env.py
      topology_env.py
      wrappers.py

    road/
      topology_factory.py
      ring.py
      straight.py
      merge.py
      inverted_tree.py
      segment_graph.py

    demand/
      spawner.py
      demand_profiles.py
      route_sampler.py

    vehicles/
      behavior_profiles.py

    sensing/
      local.py

    safety/
      constraints.py
      etiquette.py
      safety_layer.py

    metrics/
      segment_metrics.py
      global_metrics.py
      safety_metrics.py
      fairness_metrics.py
      logger.py

    baselines/
      controllers.py
      registry.py

    rl/
      actions.py
      controller.py
      encoders.py
      models.py
      ppo.py
      rewards.py
      rollout_buffer.py
      trainers.py

    world_model/                    # planned
      graph/
      fields/
      data/
      models/
      policies/
      losses.py
      rewards.py
      evaluation.py

  scripts/
    run_baseline.py
    train_policy.py
    evaluate_policy.py
    validate_project_interface.py
    validate_topology_baselines.py
    validate_training_eval.py
    collect_world_model_data.py      # planned
    train_world_model.py             # planned
    evaluate_world_model.py          # planned
    train_model_based_policy.py      # planned
    validate_world_model_pipeline.py # planned
    run_experiment_matrix.py         # planned

  outputs/
    checkpoints/
    metrics/
    validation/
    world_model/                     # planned datasets/checkpoints/evaluation
```

---

# 4. Baselines

You should use the following baseline ladder.

## B1. No AVs / human-only traffic

All vehicles use the default human driving model.

Purpose:

```text
lower-bound traffic performance
natural congestion formation
```

Use this for every topology and every demand level.

---

## B2. Random AVs

AVs receive local observations but choose random desired speed/lane commands.

Purpose:

```text
sanity check
shows improvement is not just due to AV presence
```

---

## B3. Selfish AVs / non-cooperative AVs

Each AV optimizes only its own progress.

Reward:

```text
+ ego speed
+ ego progress
- collision
- hard braking
- excessive lane changes
```

No global throughput reward.

Purpose:

```text
tests whether selfish autonomy worsens or fails to improve network flow
```

This is an important baseline because it contrasts “autonomous driving for myself” with “autonomous driving as traffic regulation.”

---

## B4. Density lookup controller

This is the direct continuation of your current paper.

Policy:

```text
local density estimate -> target speed bin
```

Example:

```text
low density      -> high target speed
medium density   -> moderate target speed
high density     -> reduced target speed
jam density      -> strong damping target speed
```

Purpose:

```text
simple interpretable self-regulation
non-learning baseline
```

This should be strong on ring and straight highway, weaker on merge/tree where headway and merge-gap behavior matter.

---

## B5. Dynamic speed limit: local AV speed advisory

This is an infrastructure-free approximation of a dynamic speed limit.

Each AV uses only its own local observation to choose a temporary speed advisory for itself.

```text
local density / queue estimate -> local AV speed advisory
```

Only the AV acts on the advisory. Human vehicles are affected only through ordinary car-following dynamics.

Purpose:

```text
interpretable local congestion metering
```

This baseline answers:

> How fast should an AV go given locally sensed congestion?

---

## B6. AV-mediated speed harmonization

This is the flow-smoothing version of local AV speed control.

Each AV smooths relative to nearby traffic speed, leader speed, and local flow using only its public local observation.

```text
local traffic speed estimate exists
only the observing AV realizes the target smoothly
human vehicles are influenced physically through car-following
humans may pass if safe
AVs do not change lanes solely to cover all lanes
AVs do not slow below local traffic-control targets by more than a small tolerance
```

Purpose:

```text
tests whether sparse AVs can damp local speed mismatch and waves
```

This is probably one of your strongest baselines/contributions. You can say:

> Sparse AVs can approximate speed-harmonization effects by acting as moving compliant vehicles, not roadblocks.

Compare:

```text
local density advisory with 5%, 10%, 20% AV penetration
local flow harmonization with 5%, 10%, 20% AV penetration
learned CTDE AV policy
```

---

## B7. Local backpressure-style control

Classic backpressure idea:

```text
pressure(edge) = upstream queue - downstream queue
```

At a merge, if one branch has high pressure and downstream capacity exists, that branch should be released more aggressively. If downstream congestion is high, upstream branches should be slowed.

For highways, implement pressure-inspired behavior not as a traffic light or infrastructure controller but as **local AV speed/gap regulation**:

```text
high upstream pressure + low downstream congestion:
    allow faster target speed

high downstream pressure:
    reduce upstream speed to avoid spillback

merge imbalance:
    use AVs to create gaps or meter branch inflow
```

Purpose:

```text
local network-pressure baseline
tests queue-aware regulation from each AV's observation
```

This is especially important for the inverted tree.

---

## B8. Cooperative adaptive cruise control / smoothing controller

Simple rule:

```text
AV slows down when local density is high or leader speed variance is high
AV maintains larger headway near bottlenecks
AV avoids unnecessary lane changes
AV uses only local observation and local aggregate cooperation fields
```

Purpose:

```text
strong hand-designed decentralized baseline
```

This gives reviewers a non-RL decentralized controller to compare against.


---

## B9. CTDE learned policy: speed + headway + conservative lane

Main method.

Actor:

```text
local noisy AV observation -> speed/headway bins + conservative lane preference + merge mode
```

Critic:

```text
global segment state during training
```

Execution:

```text
decentralized, local sensing only
```

This is your main proposed method.

---

# 5. Human driving models

Test robustness to multiple regular-vehicle models.

## H1. Cautious humans

```text
lower desired speed
larger headway
less aggressive lane changing
higher politeness
```

Expected behavior:

```text
fewer collisions
lower throughput
less unstable but slower
```

---

## H2. Normal humans

Default setting.

Use as the main result.

---

## H3. Aggressive humans

```text
higher desired speed
shorter headway
more frequent lane changes
lower politeness
higher acceleration/deceleration
```

Expected behavior:

```text
more stop-and-go waves
more hard braking
more merge conflicts
```

This is the most important stress test.

---

## H4. Heterogeneous humans

Mixture:

```text
30% cautious
50% normal
20% aggressive
```

This should be the main “realistic” setting.

---

# 6. Core environment API to standardize

Before adding more learning, standardize the environment interface.

Every topology should support:

```python
env.reset(config)
env.step(av_actions)
env.get_local_observations()
env.get_global_state()
env.get_segment_metrics()
env.get_episode_summary()
```

Every AV action should use the same format:

```python
action = {
    "desired_speed_bin": "slow" | "nominal" | "fast",
    "desired_headway_bin": "normal" | "larger" | "largest",
    "lane_preference": "keep" | "prefer_left_if_safe" | "prefer_right_if_safe",
    "merge_mode": "normal" | "create_gap" | "hold_lane",
}
```

Lane commands are conservative preferences only. Do not expose direct `left`/`right`, lane-coverage plans, or roadblock formations. The safety/etiquette layer decides whether any lateral command is legal, comfortable, and non-obstructive.

Every baseline should implement:

```python
class Controller:
    def act(self, local_obs, global_state=None):
        return av_actions
```

This makes all baselines interchangeable.

---

# 7. Step-by-step task plan

## Phase 1: Stabilize the environment

Focus only on environment correctness.

Tasks:

```text
1. Confirm that reset/step works without RL.
2. Confirm AV and RV creation works.
3. Confirm RVs follow default IDM/MOBIL behavior.
4. Confirm AVs accept desired speed and desired lane commands.
5. Confirm safety layer blocks unsafe lane changes.
6. Confirm vehicles are removed after exit.
7. Confirm no memory leak as vehicles spawn/despawn.
8. Confirm inactive vehicles are absent from AV/RV computation, controller inputs, rewards, and segment metrics.
```

Deliverable:

```text
one script that runs each topology with random actions for 1 episode
outputs a metrics CSV
renders or saves a basic visualization
```

Suggested script:

```bash
python scripts/run_baseline.py --topology ring --controller random_av
python scripts/run_baseline.py --topology straight_multilane --controller random_av
python scripts/run_baseline.py --topology merge --controller random_av
python scripts/run_baseline.py --topology inverted_tree --controller random_av
```

Do not start RL before this is stable.

---

## Phase 2: Build topology factory

Implement all four topologies with the same interface.

Tasks:

```text
1. ring.py
2. straight.py
3. merge.py
4. inverted_tree.py
5. topology_factory.py
6. segment_graph.py
```

Each topology should return:

```python
road_network
segment_ids
segment_lengths
entry_segments
exit_segments
merge_nodes
detector_locations
```

Deliverable:

```text
plots/topology_ring.png
plots/topology_straight.png
plots/topology_merge.png
plots/topology_inverted_tree.png
```

For each topology, verify:

```text
vehicles spawn correctly
vehicles follow routes correctly
segment IDs are correct
exit counting works
detectors count throughput
```

---

## Phase 3: Demand and traffic generation

Implement flow-based demand.

Tasks:

```text
1. Poisson vehicle spawning.
2. Demand levels: low, medium, high, burst.
3. Branch split ratios for merge/tree.
4. AV penetration rate.
5. Vehicle desired-speed distribution.
6. Human-driver type distribution.
```

Config example:

```yaml
demand:
  total_veh_per_hour: 2400
  av_penetration: 0.1
  branch_split:
    A1: 0.2
    A2: 0.2
    A3: 0.2
    A4: 0.2
    A5: 0.2
  burst:
    enabled: true
    start_s: 300
    end_s: 600
    multiplier: 1.8
```

Deliverable:

```text
demand sanity plots:
  vehicles spawned over time
  vehicles exited over time
  active vehicles over time
  per-branch arrivals
```

---

## Phase 4: Metrics and logging

Do this before baselines. Otherwise you will not know what works.

Implement per-step metrics:

```text
time
active vehicles
active AVs
completed vehicles
mean speed
speed std
jam fraction
hard braking count
collision count
lane changes
total queue length
throughput over recent window
```

Implement segment-level metrics:

```text
segment vehicle count
segment mean speed
segment density
segment queue length
segment jam fraction
segment AV count
segment inflow/outflow
```

Implement tree-specific metrics:

```text
queue per branch
travel time per branch
throughput per branch
branch fairness
merge delay per merge node
spillback depth
```

Deliverable:

```text
outputs/metrics/<experiment>/step_metrics.csv
outputs/metrics/<experiment>/segment_metrics.csv
outputs/metrics/<experiment>/episode_summary.json
```

This phase is crucial because your paper depends on network-level evidence.

---

## Phase 5: Local sensing model

Implement the observation model for AVs.

Each AV observation should include:

```text
is active
ego speed
ego acceleration
ego lane
current segment
distance to next merge
distance to downstream bottleneck
leader gap and relative speed
follower gap and relative speed
left-lane front/rear gaps
right-lane front/rear gaps
local density bins
local mean speed bins
segment-level local queue estimate
local active vehicle and AV counts
nearby AV count, density, mean speed, and lane distribution
optional nearby AV intent summary
```

Cooperation fields should be local aggregates only. They should not expose neighboring AV identities or direct V2V messages in v1. When `nearby_av_count` is zero, emit neutral aggregate values and require the controller to operate as an individual local policy.

Then add realism:

```text
distance-dependent detection probability
distance-dependent position noise
speed noise
latency buffer
field-of-view limit
occlusion optional
```

Deliverable:

```text
unit test comparing true local state vs noisy observed state
plots showing error vs distance
```

This directly connects to your current paper’s sensing-range/latency/noise story.

---

## Phase 6: Safety, etiquette, and physical-control layer

Safety has one DSRC controller path:

```text
all controllers: propose public v2 AV actions
HighwayTopologyEnv.step(): applies the common safety, etiquette, and physical-control layer
human drivers: continue to use highway-env IDM/MOBIL behavior where appropriate
```

The RL policy should not directly set unsafe acceleration. Learned controllers and baselines request speed/headway/lane-preference/merge-mode actions, and the common execution-time safety layer converts those requests into bounded physical behavior while reporting diagnostics and penalties.

Implement:

```text
desired speed bin -> target speed with acceleration limits
desired headway bin -> target headway
lane preference -> target lane only if safe and courteous
merge mode -> gap creation or lane-hold behavior
unsafe lane change -> blocked
short headway -> override speed downward
low TTC -> emergency safety behavior
low speed in uncongested conditions -> blocked
all-lane low-speed AV occupancy -> blocked unless downstream congestion justifies it
```

Safety checks:

```text
target lane exists
front gap sufficient
rear gap sufficient
time-to-collision safe
acceleration/deceleration bounded
speed limit respected
lane-change dwell respected
lane changes per km bounded
target-lane rear vehicle does not need excessive braking
```

Directional weighting:

```text
leader gap, leader relative speed, forward TTC, and downstream bottleneck distance are primary safety constraints
rear/follower checks remain required for lane changes
rear/follower pressure should not dominate longitudinal safety decisions
density/control objectives may use both upstream and downstream aggregates
```

Safety diagnostics should distinguish:

```text
safety_masked_action
etiquette_blocked_action
follower_disruption_blocked
external_safety_override
simulator_blocked_action
```

Deliverable:

```text
stress test with random AV commands
collision count should remain near zero or much lower than without safety layer
```

This is essential for a robotics venue framing.

---

## Phase 7: Non-learning baselines

Implement baselines in this order.

### 7.1 No AV

```bash
python scripts/run_baseline.py --controller no_av --topology ring
```

### 7.2 Random AV

```bash
python scripts/run_baseline.py --controller random_av --topology ring --av_penetration 0.1
```

### 7.3 Selfish AV

Local ego reward only.

```bash
python scripts/run_baseline.py --controller selfish_av --topology straight_multilane
```

### 7.4 Density lookup

Local density to target speed.

```bash
python scripts/run_baseline.py --controller density_lookup --topology ring
```

### 7.5 Dynamic speed limit

Local density and queue estimate to per-AV speed advisory. This is not an infrastructure speed-limit controller.

```bash
python scripts/run_baseline.py --controller dynamic_speed_limit --topology straight_multilane
```

### 7.6 AV-mediated speed harmonization

Local flow-matching and speed-mismatch damping from each AV's own observation.

```bash
python scripts/run_baseline.py --controller av_mediated_speed_harmonization --topology straight_multilane
```

### 7.7 Backpressure

Local pressure-inspired speed/headway metering near merge/tree bottlenecks.

```bash
python scripts/run_baseline.py --controller backpressure --topology inverted_tree
```

### 7.8 Cooperative smoothing

Hand-designed local smoothing controller that uses local aggregate fields without global state.

```bash
python scripts/run_baseline.py --controller cooperative_smoothing --topology inverted_tree
```

Deliverable:

```text
baseline comparison table before any RL
```

This lets you know whether the environment is producing meaningful effects.

---

## Phase 8: Topology-by-topology validation

Before RL, run the baseline ladder across every topology and demand regime with hard invariants plus directional sanity checks.

Deliverable:

```bash
python scripts/validate_topology_baselines.py --smoke
python scripts/validate_topology_baselines.py
```

Expected output:

```text
outputs/validation/task10/
  run_summary.csv
  directional_checks.csv
  validation_summary.json
  validation_summary.md
  runs/<controller>_<topology>_<demand>_seed<seed>/
```

Hard failures include broken reset/step, bad action keys, active-count mismatches, nonmonotonic completed counts, negative segment metrics, missing branch metrics, ring exits, missing high-demand spawns, and fairness values outside `[0, 1]`.

Directional warnings include sanity expectations such as selfish AVs having higher early speed than density-based controllers, density/harmonization improving saturated throughput or queues over random AVs in stressed straight roads, backpressure/cooperative smoothing helping merge or tree queues/fairness, and rolling-roadblock scores remaining near zero.

Directional warnings are not final paper claims. They are smoke signals for implementation sanity and should be interpreted across multiple seeds.

---

## Phase 9: RL training, simple first

Start with plain-PyTorch shared PPO and IPPO before full MAPPO. All algorithms use the same decentralized actor; only MAPPO uses `global_state` through a centralized critic during training.

Training order:

```text
1. Ring road, speed only.
2. Straight single-lane, speed only.
3. Straight multi-lane, speed + lane.
4. Merge, speed + lane.
5. Inverted tree, speed + lane.
```

Actor input:

```text
local noisy observation
local aggregate AV cooperation fields
```

Actor output:

```text
desired speed
desired headway
conservative lane preference
merge mode
```

Reward:

```text
global traffic reward shared across AVs
```

Initial reward:

```text
reward =
  + throughput
  + mean_speed
  - speed_variance
  - jam_fraction
  - hard_braking
  - collisions
  - excessive_lane_changes
```

For tree:

```text
reward =
  + trunk_throughput
  - total_queue_length
  - speed_variance
  - jam_fraction
  - merge_delay
  - fairness_penalty
  - hard_braking
  - collisions
```

Deliverable:

```text
one trained speed-only policy that beats no-AV and random on ring
```

Do not move to the tree until this works.

---

## Phase 9: MAPPO / CTDE

Once simple shared PPO works, move to CTDE.

Actor:

```text
π(a_i | local_obs_i)
```

Critic:

```text
V(global_state)
```

Global state:

```text
segment counts
segment densities
segment mean speeds
segment queues
AV counts per segment
merge queues
demand level
```

Training should use the same public v2 action interface as baselines. Unsafe or inappropriate learned actions are handled by the common execution-time safety layer, which reports masked, overridden, and blocked actions separately and supplies safety penalties for learning.

If no AVs are in the local neighborhood, the actor must fall back to individual operation using neutral aggregate cooperation fields.

For inverted tree, this can initially be a flat vector. Later, you can replace the critic with a graph neural critic.

Deliverable:

```text
MAPPO beats independent/shared PPO on merge and inverted tree
```

This is one of the main technical results.

---

## Phase 10: Main experiment matrix

After the method works, run the full evaluation.

Topologies:

```text
ring
straight_single_lane
straight_multilane
merge
inverted_tree
```

Controllers:

```text
no_av
random_av
selfish_av
density_lookup
dynamic_speed_limit
av_mediated_speed_harmonization
backpressure
cooperative_smoothing
CTDE_speed_only
CTDE_lane_only
CTDE_speed_plus_lane
```

Demand:

```text
low
medium
high
burst
```

AV penetration:

```text
0%
2.5%
5%
10%
20%
40%
```

Human model:

```text
cautious
normal
aggressive
heterogeneous
```

Sensing:

```text
perfect
realistic noise
high noise
latency 0.15 s
latency 0.5 s
limited range
```

Run at least:

```text
5 seeds for development
10+ seeds for final paper results
```

---

# 8. Main paper figures

Target these figures.

## Figure 1: System overview

Local AV sensing → speed/headway targets + conservative lane preference → physical damping → network and safety metrics.

## Figure 2: Four topologies

Ring, straight, merge, inverted tree.

## Figure 3: Speed heatmaps

Compare:

```text
no AV
selfish AV
density lookup
CTDE speed+headway+conservative lane
```

on ring or straight road.

## Figure 4: Throughput vs AV penetration

For straight, merge, and inverted tree.

## Figure 5: Queue length over time

Especially for inverted tree.

## Figure 6: Baseline comparison

Bar chart/table:

```text
travel time
throughput
jam fraction
fairness
```

## Figure 7: Human-driver robustness

Normal vs aggressive vs heterogeneous.

## Figure 8: Sensing robustness

Perfect sensing vs noisy sensing vs noisy+latency.

## Figure 9: Ablation

```text
speed only
lane only
speed + headway + conservative lane
local reward
global reward
CTDE critic
safety/etiquette diagnostics
rolling-roadblock score
```


---

# 9. Strongest story for the project

The final story should be:

> We ask whether a sparse fleet of autonomous vehicles can realize network-control effects from within the traffic stream without infrastructure sensing or obstruction. Using centralized training but decentralized execution, AVs learn local desired-speed and desired-headway targets with conservative lane preferences. A hard safety and etiquette layer prevents lane hogging, oscillatory lane changes, follower disruption, and rolling-roadblock behavior. Experiments across ring, straight highway, merge, and inverted-tree topologies show when local AV control can damp disturbances, create merge gaps, reduce spillback, and improve fairness under varying demand, human driving behavior, and sensing noise.

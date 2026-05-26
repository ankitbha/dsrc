# DSRC Plan

**Centralized training, decentralized execution for self-regulating autonomous vehicles that use local sensing to realize network-level congestion control through desired speed and lane-placement commands.**

This is a natural extension of the current self regulating cars papers. Earlier sensys draft already motivates vehicle-level local sensing, density estimation, speed regulation, partial adoption, and learning/adaptation beyond the current lookup-table controller. 

Below is the project structure and a step-by-step execution plan.

---

# 1. Core hypothesis

The main hypothesis should be:

> A small fraction of autonomous vehicles, trained with centralized traffic-level feedback but deployed with only local noisy sensing, can physically realize network-level congestion-control policies through desired-speed and lane-placement actions.

This connects several ideas:

1. **Dynamic speed limits** are the infrastructure version.
2. **Self-regulating AVs** are the infrastructure-free physical realization.
3. **CTDE RL** learns when and where AVs should slow down, hold lanes, change lanes, or create gaps.

> AVs act as mobile actuators that implement traffic-control policies from inside the flow.

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
    introduces lane placement
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

Purpose: isolate lane-placement and gap-creation behavior.

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
lane placement
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

Organize the project like this:

```text
dsrc/
  README.md
  requirements.txt

  configs/
    topology/
      ring.yaml
      straight_single_lane.yaml
      straight_multilane.yaml
      merge.yaml
      inverted_tree.yaml

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
      exp_straight_throughput.yaml
      exp_merge_lane_placement.yaml
      exp_tree_network_control.yaml

    training/
      mappo.yaml
      ippo.yaml
      ppo_speed_only.yaml
      ppo_lane_only.yaml

  src/
    envs/
      base_ctde_env.py
      highway_tree_env.py
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
      av_vehicle.py
      human_vehicle.py
      behavior_profiles.py
      safety_layer.py

    sensing/
      local_sensor.py
      noise_models.py
      latency_buffer.py
      observation_encoder.py

    metrics/
      segment_metrics.py
      global_metrics.py
      safety_metrics.py
      fairness_metrics.py
      logger.py

    baselines/
      no_av.py
      random_av.py
      selfish_av.py
      density_lookup.py
      dynamic_speed_limit.py
      av_mediated_speed_limit.py
      backpressure.py
      cooperative_acc.py

    rl/
      models.py
      rollout_buffer.py
      mappo.py
      ippo.py
      train.py
      evaluate.py

    analysis/
      aggregate_results.py
      plot_timeseries.py
      plot_speed_heatmap.py
      plot_segment_occupancy.py
      plot_topology.py

  scripts/
    run_baseline.py
    train_policy.py
    evaluate_policy.py
    sweep_topologies.py
    sweep_demand.py
    sweep_human_models.py
    make_plots.py

  outputs/
    checkpoints/
    logs/
    metrics/
    plots/
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

This should be strong on ring and straight highway, weaker on merge/tree where lane placement matters.

---

## B5. Dynamic speed limit: infrastructure oracle

This is the infrastructure-based baseline.

A global controller observes segment density/speed and assigns each segment a target speed limit.

```text
segment density/speed -> segment speed limit
```

All vehicles obey it, or obey with a compliance probability.

Purpose:

```text
upper bound for infrastructure-based speed control
```

This baseline answers:

> What if we had perfect infrastructure control over segment speed limits?

---

## B6. AV-mediated dynamic speed limit

This is the more interesting version for your project.

The same dynamic speed limit controller computes desired segment speeds, but **only AVs implement them**.

```text
segment speed target exists
only AVs receive/realize the target
human vehicles are influenced physically through car-following
```

Purpose:

```text
tests whether sparse AVs can physically realize segment-level speed control
```

This is probably one of your strongest baselines/contributions. You can say:

> Dynamic speed limits require infrastructure; sparse AVs can approximate their effect by acting as mobile damping actuators.

Compare:

```text
infrastructure DSL with 100% compliance
AV-mediated DSL with 5%, 10%, 20% AV penetration
learned CTDE AV policy
```

---

## B7. Backpressure-style control

Classic backpressure idea:

```text
pressure(edge) = upstream queue - downstream queue
```

At a merge, if one branch has high pressure and downstream capacity exists, that branch should be released more aggressively. If downstream congestion is high, upstream branches should be slowed.

For highways, implement backpressure not as a traffic light but as **speed/gap regulation**:

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
network-control baseline
tests queue-aware regulation
```

Variants:

```text
B6a: infrastructure backpressure
     segment-level controller directly imposes speed/metering decisions

B6b: AV-mediated backpressure
     only AVs implement the recommended damping/gap creation
```

This is especially important for the inverted tree.

---

## B8. Cooperative adaptive cruise control / smoothing controller

Simple rule:

```text
AV slows down when local density is high or leader speed variance is high
AV maintains larger headway near bottlenecks
AV avoids unnecessary lane changes
```

Purpose:

```text
strong hand-designed decentralized baseline
```

This gives reviewers a non-RL decentralized controller to compare against.


---

## B9. CTDE learned policy: speed + lane

Main method.

Actor:

```text
local noisy AV observation -> desired speed + desired lane placement
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
    "desired_speed": float,
    "desired_lane": int or lane_preference,
}
```

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
outputs/metrics/<experiment>/segment_metrics.parquet
outputs/metrics/<experiment>/episode_summary.json
```

This phase is crucial because your paper depends on network-level evidence.

---

## Phase 5: Local sensing model

Implement the observation model for AVs.

Each AV observation should include:

```text
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
```

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

## Phase 6: Safety/control layer

The RL policy should not directly set unsafe acceleration.

Implement:

```text
desired speed -> target speed with acceleration limits
desired lane -> target lane if safe
unsafe lane change -> blocked
short headway -> override speed downward
low TTC -> emergency safety behavior
```

Safety checks:

```text
target lane exists
front gap sufficient
rear gap sufficient
time-to-collision safe
acceleration/deceleration bounded
speed limit respected
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

Global segment density to segment target speed.

```bash
python scripts/run_baseline.py --controller dynamic_speed_limit --topology straight_multilane
```

### 7.6 AV-mediated dynamic speed limit

Same segment targets, but only AVs implement them.

```bash
python scripts/run_baseline.py --controller av_mediated_speed_limit --topology straight_multilane
```

### 7.7 Backpressure

Queue-pressure control at merge/tree nodes.

```bash
python scripts/run_baseline.py --controller backpressure --topology inverted_tree
```

### 7.8 AV-mediated backpressure

Only AVs implement backpressure suggestions.

```bash
python scripts/run_baseline.py --controller av_mediated_backpressure --topology inverted_tree
```

Deliverable:

```text
baseline comparison table before any RL
```

This lets you know whether the environment is producing meaningful effects.

---

## Phase 8: RL training, simple first

Start with independent PPO or shared-policy PPO before full MAPPO.

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
```

Actor output:

```text
desired speed
desired lane
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
av_mediated_speed_limit
backpressure
av_mediated_backpressure
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

Local AV sensing → desired speed/lane → physical damping → network metrics.

## Figure 2: Four topologies

Ring, straight, merge, inverted tree.

## Figure 3: Speed heatmaps

Compare:

```text
no AV
selfish AV
density lookup
CTDE speed+lane
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
speed + lane
local reward
global reward
CTDE critic
```


---

# 9. Strongest story for the project

The final story should be:

> Classical traffic-control baselines such as dynamic speed limits and backpressure require infrastructure-level actuation. We ask whether a sparse fleet of autonomous vehicles can realize similar network-control effects from within the traffic stream. Using centralized training but decentralized execution, AVs learn local desired-speed and lane-placement policies that physically damp disturbances, create gaps near bottlenecks, and reduce spillback in branched networks. Experiments across ring, straight highway, merge, and inverted-tree topologies show when local AV control can approximate or outperform infrastructure-style regulation under varying demand, human driving behavior, and sensing noise.


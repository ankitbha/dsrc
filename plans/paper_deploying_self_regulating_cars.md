# Deploying self-regulating cars: the paper as Ankit has described it

Ankit's positions, recorded as stated, with the measurements that bear on each attached
underneath. The reasoning is his and the numbers are this repository's; the document
carries no recommendations or judgements of mine.

Per his standing instruction the earlier workshop paper is referred to but not drawn on.

## The contribution

**The deployed system.** In the task list's own words:

> The contribution is the deployed system: the phone-plus-Jetson advisory rig, the
> sampling controller, and the safety and etiquette filters. Those are what this project
> built and what the paper argues for.

Not the simulation, and not the observation result. A system was built and driven in a
car, and that is the paper.

## What was built

```text
  phone                    transport                  jetson
  -----                    ---------                  ------
  camera --+          network (dev) / USB (car)   +-- perception (TensorRT)
  GPS -----+-- sensor frames ------------------>  +-- fusion (camera + HERE)
  IMU -----+   (each at its own commanded rate)   +-- policy inference
  HERE ----+                                      +-- advisory decode
                                                  |
  display <---- advisory --------------------------+
  rate ctl <--- per-sensor rate commands ----------+-- sampling controller
```

Three pieces, and each is a claim:

**The rig.** An ordinary Android phone and a Jetson Orin Nano, nothing purpose-built.
The phone stays dumb -- it captures or queries at the rate it is told and forwards raw
data -- and every interpretation, association and control decision is on the Jetson. The
phone is tethered through a hotspot rather than carrying its own SIM, because the
alternative was buying a plan for one experiment. Two backends for the link: Tailscale
for development with the phone in hand, USB in the car.

**The sampling controller.** Each sensor's rate is set in real time by the Jetson. Its
binding costs are HERE API quota, thermal headroom and Jetson compute -- not energy,
since both devices are powered from the car.

**The safety and etiquette filters.** `src/safety/safety_layer.py`, `etiquette.py`,
`constraints.py`. The advisory is bounded before it reaches the driver.

**The cloud is observability, not control.** HERE supplies traffic state; nothing in the
loop waits on a server.

### Measured, on the device

| | |
|---|---|
| Jetson end-to-end | p50 **19.8 ms**, p95 20.2 ms at 48.5 FPS |
| detection | 17.7 ms, of which 3.9 ms is GPU |
| tracking and distance | 1.1 ms |
| observation and encode | 0.4 ms |
| actor and advisory | 0.5 ms |
| over USB, pooled across 2,684 ticks | p95 **116.19 ms** against a 200 ms target |
| the same link over Tailscale | p95 215.63 ms |
| the USB wire hop itself | about 2 ms; the run-to-run spread is phone-side queueing |

## What was run

**Eight drives, 2026-09-08, Westfield NJ**, on a OnePlus Nord N10 -- the Moto's cable
failed and the app was installed on the Nord in the car, with the swap recorded
automatically in `installed_apk.json`.

* **Six shadow runs**, 17,948 ticks, 45.9 to 47.7 km.
* **Two live runs**, 4,981 ticks, 40.7 to 42.4 km, the controller gating for real.
* **First working GPS and first working HERE in the project.** Every bench run before had
  `gps_hz 0.0` and `here=false`.
* On the shakedown: 1,632 ticks, 279.5 s, 6.24 km, mean 22.9 m/s, GPS valid on 1,299 of
  1,313 ticks, 4.69 Hz mean tick rate. Distance agrees to 0.01 km between integrating
  reported speed and the great-circle path through the fixes, which is two independent
  routes to one number.

**The shakedown did what a shakedown is for: it broke.** Three defects, each filed
separately -- a 90-degree frame rotation that explains every zero-detection drive in the
project, frames being discarded so the drive could not explain itself, and the `severe`
thermal tier ending a session rather than lowering rates.

## What the drives are, and are not, asked to establish

Ankit's boundary, and it is written into the task list rather than argued here:

> One instrumented vehicle can never demonstrate throughput or delay. That is why the
> flow-level half comes from simulation and the drives are never asked to support it. The
> drives support the deployment claims and calibrate the sensing model.

Explicitly out of scope for the drives: any traffic-flow effect, human compliance with the
advisory, and anything fleet-level.

**The drives feed the simulation back.** The sensing model's parameters were settled from
them -- `latency_s` and `queue_speed_mps` from the drives, `range_m` from optics. The
largest single correction: every training config carried `latency_s: 0.0` against the
**96.7 ms median measured on the road**. A policy trained before that was trained against
an observation model the paper would then have had to describe as wrong.

## Why there is a simulation half at all

Because one vehicle cannot show a flow effect, and the flow effect is the reason anyone
would deploy this. The simulation carries the scale claim; the deployment carries
feasibility.

## What the simulation shows

Mainz, EIDM fleet, three-into-two lane drop at the exit, sustained 4,500 veh/h. Seeds
1-10 train, 11-15 select, 16-30 evaluate. 80 episodes, 100% penetration, paired on seed.

| arm | flow veh/h | paired gain |
|---|---|---|
| the traffic-API observation | 3,765 | **+230 +/- 44** (+6.5%) |
| SRC's original six features | 3,759 | +224 +/- 61 (+6.3%) |
| no control | 3,535 | -- |

Two things, in the order they matter for this paper:

1. **SRC's controller reproduces in a second simulator.** It was shown in Vissim under
   Wiedemann-99; here it is shown again in SUMO under EIDM, on the same network.
2. **Restricting the observation to what a traffic API returns costs nothing.** The two
   arms differ by 6 veh/h against bars of 44 and 61, which bounds the cost at under about
   2% of baseline. This is what makes decentralized execution possible at all, and it is a
   supporting result rather than the paper's point.

### Why the observation question arises

SRC's policy reads six per-super-segment fields, four of which come from vehicle-level
ground truth no vehicle can obtain -- its `RL.py` computes the gap from each vehicle's
`FollowDistGr` and the rates from set differences over vehicle ids a minute apart. A
traffic API returns speed, free flow, jam factor, confidence and traversability. So a
vehicle **cannot evaluate the published policy**, and that is a fact about the input.

Aggregating from local sensing instead was ruled out: it needs either vehicle-to-vehicle
communication, untested in this deployment, or a server to aggregate at --

> "If we use a central server for aggregation, why not just deploy a central policy?"

> "Having each AV run a HERE api is much much simpler than having them talk to each other."

### Why decentralized execution needs no separate experiment

> "The shared model with shared map and shared clock ensures that the decentralized
> execution still implements the centrally trained policy."

Every vehicle holding the same weights, map and clock, querying the same API, computes the
same action for the same super-segment. It is an identity, not a result.

### What the local sensors are for

The safety gate, and nothing else:

> "Adding local metrics and objectives... is going to needlessly complicate the story."

This already holds in both artifacts rather than being asserted. In simulation the
advisory is written with `setSpeed`, which SUMO bounds by the car-following safe speed, so
a vehicle whose leader is slower follows its leader -- the same semantics as Vissim's
`DesSpeed`. On the device the perception stack and the safety filters are the same clamp.
**The two halves of the paper join at the gate**, in the same place, with the same rule.

## Where this sits

> "This paper is supposed to be the last chapter in the traffic story -- sudden traffic
> jams, self regulating cars, deploying self regulating cars."

Three chapters: the phenomenon, the controller, the deployment. The controller is the SRC
paper (arXiv:2506.11973, AAAI 2026), and it is not being changed:

> "There is no reason to specifically learn a new type of policy. We know a policy that
> works -- the SRC policy."

The reward, the training and the controller are SRC's. Only where the policy is evaluated
moves -- and the reason that is worth a paper is deployment, not control:

> "The only reason decentralization is attractive is because it eases the deployment...
> It is a shortest path to deployment story."

## Why the simulation is not a reproduction, and why that is the design

The result has been shown in two simulators with different car-following models, which is
stronger than matching one number in one of them. The comparison that carries the
observation claim is unaffected either way: both arms run the same vehicles on the same
network under the same demand, so the fleet cancels between them.

**On what the fleet was chosen for.** EIDM was chosen because it has a capacity drop. The
capacity drop is not a free parameter: real queue discharge is 5 to 20% below free-flow
capacity, which is a property of traffic rather than of a model. Measured here, W99
discharges **4.3% faster** than free flow -- the wrong sign -- and EIDM with a one-second
reaction time **21% slower**. In Ankit's words: the simulator must be consistent with
reality, and a simulation failing to replicate a measured phenomenon is a setting problem
rather than evidence that the simulation is the reference.

The same covers the exit lane drop and the right-of-way `netconvert --vissim-file` does
not import -- 2 junctions with conflicting movements against 10 active conflict areas in
the source `.inpx`, so the ported network has less constraint than the one the paper ran,
and the lane drop restores a bottleneck of the kind it is supposed to have. Both apply
identically to every arm.

**No number here is comparable to the published table.** Every claim is against this
port's own no-control baseline.

## The flow of the paper as described

1. The starting point: the SRC paper's controller and result.
2. Motivation and the bill of changes, from the earlier workshop paper.
3. The minimum viable deployment: what is the least that has to exist on a vehicle.
4. How the system was built, and our run of it.
5. The simulation result.

## Target venue

ICRA.

## Open

* **The deployed policy and the simulated policy are not yet the same object.** The rig's
  actor takes the 39-field local contract, vendored and test-locked on both sides, and the
  last recorded state of the policy bundle is random-init pending a checkpoint from the
  simulation side. The simulation result is on a 5-field traffic-API observation per
  super-segment. Joining them means splitting the deployed contract in two: the API
  observation for the policy, the local vector for the gate.
* **The shadow runs have not been scored.** `deployment/jetson/score_shadow.py` replays
  the logged per-tick inputs and scores candidate controllers against them, gating on the
  incumbent replaying byte-for-byte first. It has not been run against the six. Until it
  does, whether the shadow-mode predictions held in live mode is unanswered -- and that is
  the comparison the two live runs were collected for.
* **The gate's behaviour is unmeasured in simulation.** How often local safety clamps an
  advisory, and what it costs. Three runs would settle it: gated, ungated, no control.
* **One topology and one demand level** in the simulation half. `inverted_tree` is out of
  scope -- one link per super-segment, so there is no aggregation in it.
* **The critical path in `plans/task_list.md` is stale.** It names tasks 68 and 69,
  training and evaluating MAPPO on `inverted_tree`, as the flow-level half. That was
  superseded on 2026-09-11 by the SRC port on Mainz.

# Deploying self-regulating cars: the paper as Ankit has described it

Ankit's positions, recorded as stated, with the measurements that bear on each attached
underneath. Written so the argument can be picked up cold. The reasoning is his and the
numbers are this repository's; the document carries no recommendations or judgements of
mine. Open items are listed at the end as gaps in what has been measured, not as advice.

Per his standing instruction the earlier workshop paper is referred to but not drawn on;
nothing of its content is reproduced here.

## The one-line claim

A policy that is trained centrally can be executed by each vehicle independently, from
what a traffic API already returns, and it loses nothing by it.

## Where this sits

> "This paper is supposed to be the last chapter in the traffic story -- sudden traffic
> jams, self regulating cars, deploying self regulating cars."

Three chapters: the phenomenon, the controller, the deployment. The starting point is
the SRC paper (arXiv:2506.11973, AAAI 2026), which is chapter two. This is chapter three
and it is a deployment paper, not a control paper.

## What is not being changed, and why that is the point

> "There is no reason to specifically learn a new type of policy. We know a policy that
> works -- the SRC policy."

The controller is SRC's. The reward is SRC's, unchanged, on ground-truth density:
`-100 * 1[rho > rho*] + 0.2 * speed`. Training is as centralized as SRC's, in a
simulator, and may use privileged information because it happens offline.

> "Instead of a central system giving directions, we are doing decentralized execution.
> The training can be as centralized as the SRC paper."

**Only where the policy is evaluated moves.** That is the whole delta, and stating it
that narrowly is what keeps the paper a deployment paper.

## Why decentralized execution at all

Not for autonomy or robustness in the abstract:

> "The only reason decentralization is attractive is because it eases the deployment...
> It is a shortest path to deployment story."

A central controller needs a traffic authority to run it, a channel to every vehicle, and
the authority to command them. Decentralized execution needs a model file and a network
request. That is the argument, and it is an argument about what can actually be fielded.

## Why the observation had to change, and why nothing else did

SRC's policy reads six per-super-segment fields: density, lane count, mean speed, mean
gap, inflow, outflow. Four of the six come from vehicle-level ground truth that no
vehicle can obtain -- its `RL.py` computes the gap from each vehicle's `FollowDistGr` and
the rates from set differences over vehicle ids a minute apart. A traffic API returns
speed, free flow, jam factor, confidence and traversability, and nothing else.

So a vehicle **cannot evaluate the published policy**. That is the reason for retraining,
and it is a fact about the input, not a modelling preference.

### Why not aggregate from local sensing instead

Ankit ruled this out, and the reason is the deployment argument again. Aggregating local
observations into a super-segment state needs either vehicle-to-vehicle communication,
which is untested in the deployment, or a server to aggregate at:

> "If we use a central server for aggregation, why not just deploy a central policy?"

> "Having each AV run a HERE api is much much simpler than having them talk to each
> other."

A server would concede the thing the paper is arguing for.

## Why decentralized execution needs no separate experiment

> "The shared model with shared map and shared clock ensures that the decentralized
> execution still implements the centrally trained policy."

The policy is a deterministic function of the network's aggregate state. Every vehicle
holding the same weights, the same map and the same clock, and querying the same API,
computes the same action for the same super-segment. There is no consensus problem to
solve and therefore no consensus experiment to run. The claim is an identity, not a
result.

## What the local sensors are for

Not for the policy's input, and not for a second objective:

> "Adding local metrics and objectives... is going to needlessly complicate the story."

They are the **safety gate**. The aggregate advisory is a desired speed; local sensing
can only ever reduce it. The camera, IMU, GPS and fusion stack on the Jetson serves that
layer.

This already holds in both artifacts rather than being asserted. In simulation the
advisory is written with `setSpeed`, which SUMO bounds by the car-following safe speed,
so a vehicle whose leader is slower follows its leader -- the same semantics as Vissim's
`DesSpeed`. On the device the perception stack is the same clamp. **The two halves of the
paper join at the gate**, in the same place, with the same rule.

## The architecture, in three lines

1. A policy trained centrally in simulation, on SRC's reward, from privileged state.
2. Executed independently by each vehicle from a traffic API, a shared map and a shared
   clock -- five fields per super-segment, no vehicle-to-vehicle link, no server.
3. Clamped locally by on-board perception, which may slow the advisory and never raise it.

## The flow of the paper as described

1. The starting point: the SRC paper's controller and result.
2. Motivation and the bill of changes, from the earlier workshop paper.
3. The minimum viable deployment: what is the least that has to exist on a vehicle.
4. How the system was built, and our run of it.
5. The simulation result.

## Target venue

ICRA.

## What the simulation contributes

Mainz, EIDM fleet, three-into-two lane drop at the exit, sustained 4,500 veh/h. Seeds
1-10 train, 11-15 select, 16-30 evaluate. 80 episodes, 100% penetration, paired on seed.

| arm | flow veh/h | paired gain |
|---|---|---|
| DSRC, the traffic-API observation | 3,765 | **+230 +/- 44** (+6.5%) |
| SRC, its original six features | 3,759 | +224 +/- 61 (+6.3%) |
| no control | 3,535 | -- |

**The headline is the equivalence, not the gain.** The two arms differ by 6 veh/h with
bars of 44 and 61, which bounds the cost of restricting the observation at under about 2%
of baseline. That is the enabling claim for decentralized execution. Both arms beating no
control by about 6.4% is the supporting claim.

## Why this is not a reproduction, and why that is the design

Ankit's position, and it is a claim the paper should make rather than a limitation it
should concede: **the result has now been shown in two simulators with different
car-following models, which is stronger than matching one number in one of them.** SRC
showed the mechanism in Vissim under Wiedemann-99; this shows it again in SUMO under
EIDM. A mechanism that survives a change of driver model is robust to it; one that
appears only under a single calibration is fragile.

The comparison that carries the headline is unaffected either way. DSRC and SRC run the
same vehicles on the same network under the same demand, so the fleet cancels between
them entirely. The fleet only enters the supporting claim, that either arm beats no
control, and there it is doing the opposite of weakening it.

**On what the fleet was chosen for.** EIDM was chosen because it has a capacity drop. The
capacity drop is not a free parameter: real queue discharge is 5 to 20% below free-flow
capacity, which is a property of traffic rather than of a model. Measured here, W99
discharges **4.3% faster** than free flow -- the wrong sign -- and EIDM with a one-second
reaction time **21% slower**. In Ankit's words: the simulator must be consistent with
reality, and a simulation failing to replicate a measured phenomenon is a setting problem
rather than evidence that the simulation is the reference.

The same reasoning covers the exit lane drop and the dropped right-of-way. The port
carries 2 junctions with conflicting movements against 10 active conflict areas in the
source `.inpx`, because `netconvert --vissim-file` does not import them, so the ported
network has LESS constraint than the one the paper ran. The three-into-two lane drop at
the exit restores a bottleneck of the kind the network is supposed to have. Both are
stated modelling choices and both apply identically to every arm.

What follows from this, and should be written into the paper rather than left implicit:
**no number here is comparable to the published table.** Throughput, capacity and critical
density all belong to this network and this fleet. Every claim is against this port's own
no-control baseline.

## What is not established, stated flatly

* **The gate's behaviour is unmeasured.** How often local sensing clamps a DSRC advisory,
  and what it costs, is not known in this configuration. Three runs would settle it:
  advisory gated, advisory ungated, no control.
* **One topology and one demand level.** `inverted_tree` is out of scope -- one link per
  super-segment, so there is no aggregation in it -- which means Mainz is the only
  network, as it is for the AAAI paper. The demand level was chosen just past the point
  where the network breaks down; the gain has not been swept across demand.
* **The drives exist; their analysis does not.** Eight runs on 2026-09-08 in Westfield,
  NJ, on a OnePlus Nord N10: six shadow runs totalling 17,948 ticks over 45.9 to 47.7 km,
  and two live runs of 4,981 ticks over 40.7 to 42.4 km. First working GPS and first
  working HERE in the project -- every bench run before had `gps_hz 0.0` and `here=false`.
  Distance agrees to 0.01 km between integrating reported speed and the great-circle path
  through the fixes, which is two independent routes to one number. Recorded in
  `plans/task_list.md` section I, tasks 49 to 51, not in a results document.

  What has NOT been run is `deployment/jetson/score_shadow.py`, which replays the logged
  per-tick inputs and scores candidate sensing controllers against them. It gates on the
  incumbent replaying byte-for-byte and refuses otherwise, so that replay is the first
  step. Until it runs, whether the shadow-mode predictions held in live mode is unanswered.

  Two qualifications the drives carry: HERE was set to one query per minute in shadow
  mode, so they are not a full-rate HERE reference -- though that cadence is exactly
  `DECISION_INTERVAL_S`, the policy's own 60 s decision interval, so it is the rate the
  policy would query at anyway. And the shakedown found three defects, including a
  90-degree frame rotation that explains every zero-detection drive in the project.

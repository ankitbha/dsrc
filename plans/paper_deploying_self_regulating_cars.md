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

## What deployment cost, which is most of what there is to say

Ankit: this is extremely important to talk about. The failures below are not incidental
to the contribution; they are the part of it that cannot be obtained any other way.

### The car is a hostile environment and the faults are physical

Eight runs, **152.8 minutes of both devices powered from the car with no power
interruption**. The phone is USB-attached to the Jetson, which is both the sensor link and
the phone's charge source, and the Jetson reaches the network through the phone. Two
install faults, each of which cost drive time:

* **Cable type is not interchangeable.** The Moto's cable failed in the car and the app
  was moved to the OnePlus Nord N10 mid-session, with the swap recorded automatically in
  `installed_apk.json`.
* **Direct sunlight on the dashboard ended a drive.** It heated the phone enough that the
  tethering handset's hotspot shut off. The mount needs shade, not just a clamp.

### Thermal is the binding constraint, and it is not solved

**The backoff works, on hardware.** Two 900 s live runs: the thermal rule fired on 1,504
of 3,486 ticks and 1,561 of 3,451, cause `skin_warm`, and the phone applied the result --
delivered frame rate **3.000 and 2.998 Hz against a commanded 3.0**. The decision reads
device temperature and never ambient, so this clause does not depend on the environment.

**Steady state was never reached, and the extrapolation is the finding.** Both runs ended
because the clock expired, not because the handset equilibrated. Skin temperature slope
over the last two minutes was **+0.368 and +0.367 C/min**, still climbing close to
linearly after fifteen minutes. `SKIN_HOT_C` at 45 C is about **six minutes past where
both runs stopped**.

**And the soak has still not been run at load.** The camera held 5.0 Hz for roughly the
first 380 s before the loop cut it to 3.0 Hz, so maximum rate and an engaged backoff
cannot both hold -- the task's own wording is in tension. `gps_hz` and `here_hz` were 0.0
throughout, so two of the four modalities carried no load at all, and those two are
exactly the thermally scaled ones. "Stays within limits" currently holds only as a
statement about a desk.

### The instruments lied, repeatedly, and catching that was the work

**The two devices were in opposite states and finding that out was the first job.** The
phone's thermal chain was live end to end. The Jetson had **no thermal reading at all**,
and the one sampler adjacent to it degraded to a silent no-op when an optional import
failed, wrote records nothing collected, and had no test.

**The feature was inert on the one Jetson it was written for.** 150 s on the real Orin at
the deployed commit: **751 ticks, 0 sample records, 0 event records**, every tick reading
`absent`, reason `sampler_stopped`. Three of that machine's nine thermal zones answer
`EAGAIN`, which surfaces through the buffered text layer as a `TypeError` that the
reader's `except OSError` did not catch, so the sampler thread died on its first pass.
**The fixtures could not have caught it**: they make a zone unreadable by deleting the
file or denying permission, and both raise `OSError`. One sysfs quirk took the *phone's*
thermal record down with the Jetson's, on a drive where the phone was connected and
delivering telemetry throughout.

**What saved it was the reporting vocabulary.** Nothing read `quiet`, nothing read as a
zero, and the report said outright that the drive answered nothing. **The failure was
recorded as a failure.** After the fix, a confirmation drive on the same Orin: 1,200
ticks, 241 samples, `measured 241, absent 0`, and the `quiet` line now carries the
evidence for its own claim -- `241 of 241 passes fully readable` -- where before it
asserted "readable throughout" and printed the counters only on the branch where the
claim was not being made.

**Other things the field disagreed with the plan about.** Nine thermal zones, six usable,
and **13 cooling devices where the estimate assumed 3** -- the entire 39% overrun in
record size, against per-item figures that were right to a tenth of a byte. Thermal
headroom turned out to be **not a number on 456 of 456 and 297 of 297 reports**, on a
handset whose thermal HAL is connected and answering. A phone redial erased a drive's
throttle count and wrote a phantom event, because the sampler copied the phone's counter
instead of accumulating.

**Two method lessons, stated as such.** A fixture's failure mode has to be the field's:
a deleted file and a denied permission both raise one exception, the real device raised a
different one, and every test passed while the feature did nothing. And **five distinct
false readings were produced by measurement harnesses on that task alone**.

### One bug hid every perception result in the project

**The camera frames arrive rotated 90 degrees, and that is why no drive had ever seen a
vehicle.** Measured over a full day: **16 detection-bearing ticks out of 22,929**, or
0.070%, and the `vehicles` array was empty even on those 16, so **zero distance estimates
were recorded on the entire day**. Zero vehicle sightings across 4,151 ticks on the Moto
and 1,632 on the Nord, on two handsets.

Same frame, same detector, one rotation:

| frame | as delivered | rotated 90 clockwise |
|---|---|---|
| mid-drive | 0 | **2 vehicles, conf 0.90** |
| later | 0 | **5 vehicles, conf 0.87** |

Only clockwise; anticlockwise and 180 both give zero. The frame shows a van filling much
of the picture with a legible plate, so it was not a marginal detection. The fix moved the
intrinsics with the rotation -- `cx_px` 640 to 360, `cy_px` 360 to 640 -- and re-measured
the horizon at 717 px against the old landscape centre of 360.

**It was one bug, not the several structural limits it had been recorded as.** That is
the part worth writing down: the project had accumulated explanations for an absence that
had a single cause.

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

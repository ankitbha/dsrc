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

## What was built --- DESIGN

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

**The safety and etiquette filters.** `deployment/jetson/policy/safety_gate.py`, vendored from
`src/safety/safety_layer.py`, `etiquette.py` and `constraints.py`. The advisory passes the filter
before it reaches the driver, and the filter's twelve rules each record whether they could be
evaluated at all.

Stated carefully, because the obvious shorter sentence is false. When this was first written the
vendored filter had no caller outside its own test: nothing under `deployment/` imported it, and
the advisory's only bound was `max(12.0, base_speed + offset)`. It is wired now (task 144). But
**on the recorded corpus it binds nothing**, and not by accident of the drive: only three of its
twelve rules -- `low_speed_uncongested`, `target_lane_front_gap`, `forward_ttc` -- can ever be
evaluable on this rig, each requiring a tracked leader, and none of the 3,913 recorded ticks has
one. The other nine read at least one field the hardware cannot supply. The per-rule census is
committed at `results/safety/gate_census_corpus.json`.

So the claim the paper can make is that the filter is present, wired, and reports per rule what
it could and could not check -- not that it was observed to bound anything. What it would do in
traffic is not established here.

**The cloud is observability, not control.** HERE supplies traffic state; nothing in the
loop waits on a server.

### The device cannot drift from what was trained

The Jetson must not import the simulator stack, so `policy/sim_contract.py` vendors the
field lists, scales, a numpy twin of the encoder, the action heads, the decoders and the
neutral fallbacks, pinned to a named sim commit, and `export_policy.py` refuses dimension
mismatches.

**"Checkable rather than asserted" has to be earned, and for a period it was not.** The check
that made it checkable -- `test_sim_contract.py`, comparing the vendored copy against the
simulation -- opened with `importorskip("src.rl.encoders")`, and those modules were deleted in
`6b538f2`. From that commit until task 143 the test reported `1 skipped`: the guarantee was
being cited while nothing enforced it. The reference itself was never lost, which is the part
worth stating precisely -- the pinned commit is an ancestor of `main` and the dormant test passes
against it. It was unenforced, not unrecoverable.

What makes it checkable now is `specs/sim_contract_golden_vectors.json`: the encoded
observations, the action schema, both bin decoders and the neutral fallbacks, each derived twice
by `scripts/generate_sim_contract_golden_vectors.py` -- once from the pinned sim commit, once
from the vendored copy -- with the generator refusing to write on any disagreement. The
device-side test reads only the frozen file and imports no simulation module, so it runs on the
Jetson, where the simulation cannot be imported and where the original check therefore never
ran.

The boundary is worth stating rather than leaving implied: regenerating against the simulation
needs git and torch, so on the device that step is skipped and what runs is the comparison
against the frozen file.

### Two design stances

**Degrade, never die.** Missing GPS, no camera, no display and no trained checkpoint each
have a defined degraded mode, because in-car debugging time is expensive.

**Latest-value-wins on the hot path, no queues.** A frame never processed is dropped, a
slow dashboard skips ticks, a stalled SD card drops log records: stale data is worse than
missing data for a real-time advisory.

### Scale

| | |
|---|---|
| phone app | 17,837 lines of Kotlin |
| Jetson runtime | 25,100 lines of Python, with **34,798 lines of tests** |
| simulator | 11,012 lines, with 9,802 lines of tests |
| specifications | 8 documents, 1,548 lines, including an **854-line wire protocol** |
| suites | simulator 337 passing; Jetson 2,319 passing, 24 skipped |

The transport carries eight channels with per-channel sequence numbers, priorities,
overflow policies and depths, and `specs/transport_golden_frames.json` pins the wire
format.

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

## What was run --- EXPERIMENTS

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

### Shadow against live is verified through the wire, not in process

Logged shadow decisions are compared against live commands **decoded through the real
`rate_cmd` wire codec**, not through in-process objects. Three drives, 899/900/885 ticks:
**0 command-replay mismatches**, and the phone's own applier counters read `applied == 0`
with `shadowed == commands_sent` (37/38/38) -- the phone independently confirming it did
not act. The check itself carried the same defect class it was built to catch: when the
phone-side half *could not run*, it printed `ok=False` and **exited 0**.

## What deployment cost --- EXPERIENCE

Ankit: this is most of what there is to say, and it is extremely important to talk
about. The failures below are not incidental to the contribution; they are the part of it
that cannot be obtained any other way.

### A telemetry vocabulary, and the limit of it

Every instrumented quantity reports in a closed three-state vocabulary rather than a
number: **measured, or converted with a stated bound, or absent with a named reason --
never a zero.** Trigger attribution is **fired, quiet, or not evaluable with the missing
inputs named**, with a reconstruction identity `_clamp(base x scale) == rates[k]`. Field
provenance is a closed eleven-member vocabulary over all 39 encoder slots, and logs
recorded before that work **refuse by name** rather than being scored against defaults.
The session summary reports `answered of attempted` as two independently counted integers,
with **no percentage anywhere and no scalar health field** -- a scalar is what a dashboard
plots on its own, and once plotted the enumeration is gone.

**And then the limit, which is the part worth writing.** An axis of the form
`answered of attempted` is **blind by construction to an event that stops the attempting**.
A 54.58 s link outage destroyed numerator and denominator together -- about 273 of roughly
1,502 ticks, **18.2% of every denominator** -- and the summary reported `attempted = 1229`
and called every axis fully answered. The vocabulary protects the numerator; nothing was
watching the denominator.

### The two clocks, and a guarantee that states its own error

Two devices, two monotonic clocks. The discipline is that **no conversion returns a bare
number**: a converted instant carries its error bound and the id of the estimate that
produced it, so a cross-device timestamp cannot be mistaken for a same-device one, and
below the gate it **raises** rather than answering with a widened bound.

* **Loopback null case**, where one machine means the truth is zero: offset spread
  **12 microseconds over 145 s**, fitted slope −0.05 ppm. The estimator invents no
  structure where there is none.
* **Real link, 330 s**: 357 exchanges, every one matched, **zero refused**, and all nine
  outcome counters close against pings sent without subtracting anything. `rtt_min` p50
  **14.8 ms**, bound p50 **8.0 ms**. Under full sensor load -- 10 Hz camera at 40 KB,
  50 Hz IMU -- 218 exchanges with every channel on cadence.
* **The instrument built to validate the premise was degenerate, and algebra caught it,
  not a run.** Estimating the offset on each clock pair and differencing the slopes
  cancels the quantity of interest: it returns `s_local − s_remote`, not the skew.
  Confirmed rather than argued -- it reported **+12.06 ppm against an independently
  measured slew difference of +12.00 ppm**.
* What works needs no network: each device's own wall-minus-monotonic slew is exact.
  **True monotonic skew −12.00 ppm against the 50 ppm assumed, a 4.2x margin.** The
  estimator *fitted* −1.09 ppm while the truth was −12.00, so charging the fit's own
  magnitude would have under-bounded the error by ten times.

### What the failure inventory found before anything was built

**186 failure conditions were already detected across the two devices. Four record when
the failure happened. None records an episode** -- a second endpoint and an outcome.
Several had no reader at all. Two were detected nowhere: the tick loop's no-frame branch
counted nothing, so **a drive blind for 110 of 120 seconds wrote the artefact of one that
was never blind**, and the worker was `try/finally` with no `except`, so an exception ran
teardown and wrote a summary that read like a clean short run.

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

**What the drives do establish is that both layers of the proposed deployment ran on a
road at once.** The HERE query returned the advisory layer's actual inputs at one query
per minute, which is the policy's own decision interval, for the first time in the
project. The camera, GPS and IMU ran the safety layer's inputs over the same 45.9 to
47.7 km. Neither had been exercised on a road before, and they were exercised together.

A separate measurement from the drives, recorded because it is real even though the paper
does not use it: `latency_s` and `queue_speed_mps` for the simulator's local-sensing
model, against configs that carried `latency_s: 0.0` where the road measured a **96.7 ms
median**. That model belongs to the earlier local-sensing formulation, which is not this
paper's, and the SRC port models no observation latency because it queries HERE on a 60 s
interval.

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

### Sim-to-real parity holds on the observation this paper uses

The simulation's observation is HERE-shaped -- speed, free flow and jam factor per
super-segment, with lane count and length from the map -- and the deployment queries HERE
for exactly those. **The parity is by construction, not by approximation**, and the drives
collected it: `here=true` on the road for the first time in the project, at one query per
minute, which is `DECISION_INTERVAL_S`, the policy's own decision interval.

`src/analysis/observation_parity.py` audits a different thing: a 39-field local-sensing
contract, on which it reports 8 slots identical, 18 approximated, 7 substituted and 6
structurally absent. **That contract is not part of this paper.** It describes an
observation for which no data was collected, beyond the minimum viable deployment, and
carrying it would mean claiming a sensing model the drives never exercised.

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

**The split is a match of sensor to job, and the audit says so.** HERE's traffic API
reports aggregate speed, free-flow speed and jam factor **per road segment**. There are no
individual vehicles in it, so it cannot give a leader distance, a follower gap or a lane
distribution -- and it does not need to, because the advisory is a super-segment decision
and those are exactly the quantities it does not read. The camera is the opposite: it
measures the vehicle ahead and nothing about the network. So:

| layer | sensor | what it produces | what it decides |
|---|---|---|---|
| advisory | HERE, per super-segment | aggregate speed, free flow, jam factor | the desired speed |
| safety | camera, GPS, IMU | leader distance, closing speed, ego state | whether to allow it |

Neither sensor can do the other's job, and neither is asked to. An earlier formulation put
per-vehicle quantities into the *policy's* input and then needed HERE to restore them,
which it cannot: 13 of 39 fields were ones the deployed vehicle either cannot sense or
replaces with a constant, all six of the absent ones rear-facing because the vehicle list
is forward-camera derived. That formulation is not this paper's. Here the per-vehicle
quantities stay where they are measurable and do the one job they are good for.

In the Mainz simulation the advisory is written with `setSpeed`, which SUMO bounds by the
car-following safe speed, so a vehicle whose leader is slower follows its leader -- the
same semantics as Vissim's `DesSpeed`. On the device the perception stack and the safety
filters are the clamp.

**Those are not the same mechanism, and the difference has to be stated.** What the
simulation has is SUMO's implicit car-following bound. The project's explicit safety and
etiquette layer, `src/safety/safety_layer.py` with `etiquette.py` and `constraints.py`, is
**unexercised in simulation**: in the SUMO topology environment three of the four action
heads are inert -- `lane_preference` and `merge_mode` are discarded, and
`desired_headway_bin` feeds back only into the agent's own observation because SUMO's
`tau` is never set -- so of four heads one acts on the world, one acts on the observation
and two act on nothing, and the safety layer never runs.

**The gate is wired on the device and implicit in the simulation, and the paper cannot claim
the two are the same rule.** Two qualifications, because "real on the device" was the original
wording and it claimed more than the repository supported. First, it was not wired at all until
task 144: the vendored filter's only caller was its own test. Second, now that it is wired, it
is *present and recorded* rather than *observed to act* -- only three of its twelve rules can
ever be evaluable on this rig, each needing a tracked leader, and no recorded tick has one. So
the asymmetry the paper describes is real and larger than first stated: in simulation the layer
cannot run, and on the device it runs and has never had the evidence to bind.

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

### What the two papers cover between them

The simulation half is not asked to establish generality on its own, and a topology or
demand sweep would be re-establishing what the SRC paper already did. Across the pair:

| | SRC (AAAI) | this paper |
|---|---|---|
| simulator | PTV Vissim | SUMO |
| car following | Wiedemann-99 | EIDM with a 1 s reaction |
| demand | the published surge, 18,000 veh/h for 1,200 s | a sustained 4,500 veh/h |
| road network | Mainz | Mainz |

Two simulators, two driver models, two demand profiles. The road network is nominally the
same, but the pair is further apart than that row suggests: this port carries 2 junctions
with conflicting movements against 10 active conflict areas in the source, a fleet whose
capacity differs, and an exit lane drop the published layout does not have. What is
common between the two runs is the controller and the topology's shape; almost everything
that decides how it behaves differs.

`inverted_tree` is out of scope for an unrelated reason -- one link per super-segment, so
there is no aggregation for the observation to summarise.

## The flow of the paper as described

1. The starting point: the SRC paper's controller and result.
2. Motivation and the bill of changes, from the earlier workshop paper.
3. The minimum viable deployment: what is the least that has to exist on a vehicle.
4. How the system was built, and our run of it.
5. The simulation result.

## Target venue

ICRA.

## Where to read, by section

Paths verified against the tree on 2026-09-12. An agent writing a section should read
what is listed under it; the repository is small enough now that everything named here is
live code or a live record.

### Read first, whatever you are writing

| | |
|---|---|
| `README.md` | the map, and a closing note on what was removed |
| `plans/implementation_records.md` | the working record. Sections D to I are the system and the evidence; K is findings. **The paper's own framing is in its "The paper" section** |
| `deployment/jetson/ARCHITECTURE.md` | dataflow, module map, the measured latency budget, contract vendoring, degraded modes |

### DESIGN

**The rig.** The phone captures and forwards; the Jetson interprets and decides.

* Phone, under `phone/app/src/main/kotlin/com/dsrc/phone/`: `SensingService.kt` and
  `SensingStateMachine.kt`, with the four sources in `sensors/` -- `CameraPipeline.kt`,
  `GpsPipeline.kt`, `ImuPipeline.kt`, `HerePipeline.kt` and `HereClient.kt`. Status in
  `plans/section_e_status.md`.
* Jetson tick loop: `deployment/jetson/pipeline.py`.
* Perception, under `deployment/jetson/perception/`: `detector.py`, `tracker.py`,
  `distance.py`, `observation_builder.py`, and `feed_fusion.py`, which decides *which
  observation fields the traffic feed may own, and on what terms*.
* Policy and advisory, under `deployment/jetson/policy/`: `actor_runtime.py`,
  `advisory.py`.
* Display: `deployment/jetson/ui/dashboard.py`.

**The sampling controller**, which is a contribution and not plumbing:

* `deployment/jetson/policy/sensing_controller.py` -- "four rates, decided here and
  commanded to the phone".
* `deployment/jetson/policy/sensing_loop.py` (decide, mark the mode, send) and
  `deployment/jetson/policy/shadow_mode.py` (gated for real,
  or only recorded).
* `plans/plan_task29_sensing_controller.md`; `plans/implementation_records.md` task 34 for the attribution
  record, which is the composition chain the code actually walks.

**The transport**, two devices and one wire:

* `specs/transport_protocol.md` -- 854 lines, the contract: frame layout, header fields,
  the eight channels with priorities, overflow policies and depths, and the shared
  timebase.
* `specs/transport_golden_frames.json` pins the wire format; both sides test against it.
* Jetson side `deployment/jetson/transport/` (18 modules); phone side
  `phone/transport/src/main/kotlin/com/dsrc/transport/` (17).
* `plans/plan_task19_gps_and_transport.md`, `plans/plan_task32_network_end_to_end.md`.

**Safety and etiquette**, the gate:

* `src/safety/safety_layer.py`, `src/safety/etiquette.py`, `src/safety/constraints.py`, and
  `deployment/jetson/policy/safety_gate.py`, which vendors all three so the device runs the
  filter without importing the simulation.
* `specs/action_schema.md` for the action format the layer decodes.

**Contract vendoring**, which is how the device cannot drift from what was trained:

* `deployment/jetson/policy/sim_contract.py`, `deployment/jetson/policy/export_policy.py`,
  `specs/observation_schema.md`, and ARCHITECTURE section 6.

### EXPERIENCE

**The telemetry vocabulary** -- measured, converted with a bound, or absent with a named
reason:

* `deployment/jetson/perception/provenance.py` -- the closed vocabulary itself.
* `deployment/jetson/logio/failure_log.py` -- "a time axis, an episode, and a reader for
  the failures this repository already had".
* `plans/plan_task33_per_stage_timestamps.md` through `plans/plan_task39_session_summary.md`,
  and `plans/implementation_records.md` tasks 33 to 39. **Task 39 carries the denominator-blindness
  finding**, which is the limit of the whole vocabulary.

**The two clocks:**

* `deployment/jetson/transport/timebase.py` -- "the one sanctioned way to compare the two
  devices' clocks" -- and `deployment/jetson/transport/clock.py`, which forbids the unsanctioned way.
* `specs/transport_protocol.md`, Shared Timebase section; `scripts/run_timebase_probe.py`.
* `plans/implementation_records.md` task 15 for the measurements and the degenerate validator.

**Thermal:**

* `deployment/jetson/sensors/thermal.py`; phone side `ThermalReader`,
  `ThermalStatusWatcher`, `ThermalZones` under `phone/app/.../sensors/`.
* `plans/plan_task37_thermal.md`; `plans/implementation_records.md` task 37 for the inert-sampler finding
  and task 45 for the soak that was never reached.

**The install faults and the rotation:**

* `plans/implementation_records.md` task 48 (cable, sunlight, 152.8 minutes of car power) and task 63 (the
  90-degree rotation, 16 detection-bearing ticks in 22,929).
* `deployment/jetson/calibration/auto_horizon.py`, `deployment/jetson/calibration/camera_calibration.py`.

### EXPERIMENTS

**The drives:** `plans/implementation_records.md` section I, tasks 49 to 52. Eight runs, 2026-09-08.

**Latency:** ARCHITECTURE section 4 for the on-device budget;
`plans/results_task42_43_47_usb_campaign.md` for the USB campaign -- p95 116.19 ms pooled
over 2,684 ticks against a 200 ms target, and the tailnet baseline at 215.63 ms;
`deployment/jetson/bench_latency.py` for how it is produced.

**Shadow against live:** `deployment/jetson/score_shadow.py` (scores candidates against
one drive's logged decisions) and `deployment/jetson/check_shadow_commands.py` (compares through the
real wire codec); `plans/plan_task35_shadow_decision_log.md`; `plans/implementation_records.md` tasks 43 and 44.

**Scoring a run:** `deployment/jetson/eval_run.py` for the gated PASS/FAIL report,
`deployment/jetson/drive_health.py` for health while a drive is happening.

### THE SIMULATION HALF

* `plans/mainz_src_port.md` -- the full account, and it leads with the result.
* `src/sumo/mainz.py` -- the environment, SRC's reward, both observations, the entry
  gate, and `DENSITY_CRITICAL` with the measurement that set it.
* `src/rl/src_q.py` -- SRC's own Q-learning, ported.
* `scripts/build_mainz_scenario.py` -- every conversion decision is in its docstrings:
  why `--flatten`, why the speed factor, why the routes need both end links, and why the
  exit is a lane drop rather than a signal.
* `scripts/train_mainz_src.py` -- train on 1-10, select on 11-15, read 16-20 once. Its
  `TEST_SEEDS` is `range(16, 21)`; the fifteen-seed figures in this document come from
  `scripts/evaluate_mainz_checkpoints.py`, whose artefact is `results/evaluation/`.
* `scripts/measure_mainz_fundamental_diagram.py` -- **the gate**. Served flow must fall as
  demand rises, by more than the seed spread.
* `scripts/measure_fd_straight.py`, `scripts/measure_fd_open_road.py`, `scripts/measure_fd_exit.py`.
* `configs/human_models/eidm_reaction.yaml` -- why the fleet is not the paper's, with the
  measurements, in its header.
* `results/` with its own README, and `data/mainz/` for the scenario itself.

### Assets

`paper/images/` holds `system_architecture.pdf`, `system_vision.png`,
`on-vehicle-perception.jpg`, `bus_depth.jpg` and `owl_predict.jpg`; the two `*_old.*`
files beside them are superseded. `paper/refs.bib` has 211 entries, of which the current
draft cites 37. `paper/related_materials/` holds two reference PDFs.

The draft's sections are `paper/abstract.tex`, `intro.tex`, `vision.tex`,
`architecture.tex`, `eval.tex`, `related.tex` and `discussion.tex` -- 259 lines in total,
carried over from the earlier submission and not yet rewritten for this story.
`paper/main.tex` is the IEEE/ICRA shell and carries a table of what the port from acmart
changed.

## Do not read these: superseded

Everything superseded is now in one file, `plans/detours.md`, with a warning at its top.
It describes the project's earlier formulation -- a MAPPO policy over a 39-field
local-sensing observation, trained on a ladder of synthetic topologies with
`inverted_tree` as the road. **None of it is this paper**, and an agent that reads it
without the warning will write the wrong one. The code is deleted; the records remain
because they are how the current shape was arrived at.

`plans/detours.md` holds: the MAPPO replication section, findings 77 to 140, the ordering
correction and items 72 to 76, the paused state of 2026-09-09, the original
`plan_simulations.md` and `project_plan.md`, and that leg's own closing record and raw
results.

Two things about it are worth knowing rather than avoiding. Its index names four findings
the current work still cites -- the unintended permanent yield that set a network's
capacity, what a gradient-norm instrument can resolve, a throughput gain retracted as a
step-size artefact, and the calibrated model giving up the collision-free guarantee. And
`plans/implementation_records.md` still holds two items from that era on purpose: task 87,
which is why the simulated gate and the deployed gate are not the same rule, and task 141,
the supersession itself.

One more, and it is live rather than superseded: the 39-field local-sensing observation
contract. Its parity ledger was deleted, and the contract itself still describes what the
safety gate reads. Saying it "is not the policy's input" is true of the simulation and false
of the device: `ObservationBuilder` -> `ActorRuntime.act` -> `AdvisoryDecoder` is exactly that
39-field path, and it is the one that has driven. **The rig now carries two policies.** Task
145 added a second runtime rather than replacing the first: the DSRC runtime, which reads the
whole network as 12 super-segments x 5 features. The paper's controller is the second one, and
it is the one that has not driven.

## Open

* **The HERE-observation policy has not yet been run on the device.** "What remains is
  loading that policy onto the rig" was wrong, and is restated as task 145 in the
  implementation records. After 145 the device-side runtime exists: it executes a DSRC
  checkpoint from super-segment features, refuses a bundle whose network identity does not
  match its own, and reproduces `src.rl.src_q.greedy_actions` exactly on the 185 recorded
  decisions across the checkpoint's five held-out test seeds. What remains is three named
  pieces, none of them loading: a super-segment partition of a network the vehicle can
  actually drive on, a HERE query bounded by that network's extent, and a training run on it.
  `SrcQNetwork`'s weights are trained for Mainz and its input is the whole network, so the
  policy is network-specific, and the 123 HERE bodies collected on 2026-09-08 are New Jersey
  roads -- there is no Mainz observation in the corpus and no Westfield policy, so the gap
  cannot be closed by replaying the drives.

  Separately, and it is a smaller thing said plainly: the DSRC path is **unreachable from
  `run_demo.py` today**. `run_demo.py:183-186` constructs the pipeline with six positional
  arguments and one keyword, and passes none of `dsrc_runtime`, `dsrc_segment_builder` or
  `dsrc_advisory_decoder`; `here_feed_source` is never passed either, and `config.yaml`'s
  `policy` block has no `dsrc_bundle` entry. The three stay `None` and the path never runs on
  a real drive.
* **The shadow runs have not been scored.** `deployment/jetson/score_shadow.py` replays
  the logged per-tick inputs and scores candidate controllers against them, gating on the
  incumbent replaying byte-for-byte first. It has not been run against the six. Until it
  does, whether the shadow-mode predictions held in live mode is unanswered -- and that is
  the comparison the two live runs were collected for.
* **The gate's behaviour question, as previously posed, cannot be answered on either side.**
  It asked how often local safety clamps an advisory and what it costs, and said three runs
  would settle it: gated, ungated, no control. Both halves are now measured, and neither is a
  matter of spending more compute.

  *In simulation, there is nothing to gate.* `setSpeedMode` appears nowhere in the repository,
  and `AVAction` is not used on the SUMO path at all; `src/sumo/mainz.py` calls only `setSpeed`,
  at `:330` to command the fraction and `:338` to release it. `setSpeed` is already bounded by
  SUMO's own car-following safe speed -- the file's own docstring at `:320` says so -- and there
  is no discrete action for `apply_safety_layer` to intercept. So a "gated" arm has nothing to do
  differently from an "ungated" one. Making the comparison constructible is a change to the
  action contract, which is task 87's undecided question, not an experiment.

  *On the device, the filter runs and binds nothing, for a structural reason rather than a
  contingent one.* Over the 3,913 recorded ticks none of the twelve rules has evaluable inputs.
  More usefully, and this is the claim rather than the observation: **only three of the twelve
  rules can ever be evaluable on this rig** -- `low_speed_uncongested`, `target_lane_front_gap`
  and `forward_ttc`, each of which becomes evaluable once a leader is tracked. The other nine
  cannot be evaluable whatever data is collected, because every verdict they depend on reads at
  least one field the rig has no instrument for: no rear sensor (three rules), no lane detection,
  no lane-change detector (two), no cooperating peers, no passing-lane index, and no non-leader
  relative speed. Measured over 20,000 draws constructed only through the observation builder,
  those three are the only rules ever evaluable.
  The census is recorded per rule and committed at `results/safety/gate_census_corpus.json`.
  That is also why the lane advisory is withheld on every tick: three of its eight guards read
  fields this rig has no instrument for.
* **The critical path in `plans/implementation_records.md` is stale.** It names tasks 68 and 69,
  training and evaluating MAPPO on `inverted_tree`, as the flow-level half. That was
  superseded on 2026-09-11 by the SRC port on Mainz.

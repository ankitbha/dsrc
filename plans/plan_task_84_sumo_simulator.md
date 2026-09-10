# Task 84 — Replace highway_env with SUMO

**Command:** `plan_dsrc_rec`. Decisions are taken by recommendation, not put to the
user, and marked as such.

## Short version

**Move the physics, road network and demand layers onto SUMO. Keep everything
above the sensing seam unchanged.**

Three attempts to make `highway_env` collision-free produced diminishing returns:
the safe-velocity bound took human-only collisions from 98 to 6 at 120 steps, the
node-geometry fix took the worst lateral discontinuity from 10 m to 0 m, and
closest-point-of-approach took the residual from 23 to 14 over ten 600-step runs
while making four seeds worse. Each helped; none finished.

**SUMO does not need constraining.** Measured before writing this plan, on the same
3-into-1 merge shape that broke `highway_env`, at 6000 veh/h into a two-lane exit —
three times past capacity — for 1800 steps with 134 vehicles on the network:
**zero collisions.** Its car-following model cannot produce one, because the safe
speed is the model rather than a cap applied over it.

It is also what the replication target uses. Vinitsky et al. and the Flow benchmarks
run on SUMO, so this removes a difference between our setup and theirs that was
never deliberate.

## The seam, and why the change is smaller than it looks

`VehicleSnapshot` is the boundary between the simulator and everything this project
built. Above it sit the sensing model, the safety layer, the etiquette filters, the
observation encoders, the policy and the trainers — none of which know what produces
the snapshots. Below it sits vehicle physics, the road network and demand.

All eleven `VehicleSnapshot` fields are available from TraCI:

| field | TraCI source |
|---|---|
| `vehicle_id` | the SUMO id |
| `role` | our own bookkeeping, unchanged |
| `segment_id` | edge id, mapped through the topology spec |
| `lane_index` | edge id plus lane index, translated |
| `lane_id` | `getLaneIndex` |
| `position` | `getPosition` |
| `longitudinal_m` | `getLanePosition` |
| `speed_mps` | `getSpeed` |
| `acceleration_mps2` | `getAcceleration` |
| `free_flow_speed_mps` | `getAllowedSpeed` |
| `crashed` | always `False`; SUMO does not produce collisions |

The environment's whole public surface is six methods — `reset`, `step`,
`get_segment_metrics`, `get_global_state`, `get_episode_summary`,
`get_local_observations` — so a SUMO-backed class implementing those is a drop-in.

## Scope boundary

- **In:** a new `src/envs/sumo_env.py`; SUMO network generation for `inverted_tree`
  and `inverted_tree_bottleneck`; demand as SUMO flows; AV actuation through TraCI;
  metric collection from TraCI.
- **Preserved unchanged:** `src/sensing/`, `src/safety/`, `src/metrics/` (its inputs
  are re-populated, not rewritten), `src/rl/`, `src/baselines/`, the observation
  encoders and `sim_contract`.
- **Becomes dead code:** `CollisionFreeMixin`, `MergeAwareIDMVehicle`,
  `SafeControlledVehicle`, and the merge-projection work in `src/sensing/local.py`
  that exists only because `highway_env` vehicles could not see across an arc.
- **Out:** the other four topologies. Only `inverted_tree` is needed, per the
  section C scope decision. `merge` and `straight_multilane` congested in 0 of 12
  cells and are not used.
- **Out:** the Jetson tree. Nothing under `deployment/` changes.

## Decisions, taken by recommendation

| # | decision | taken | rejected, and why |
|---|---|---|---|
| D1 | Where to cut | At `VehicleSnapshot` | Cutting higher would mean rewriting the sensing model, which is the project's own contribution and is already validated |
| D2 | TraCI or libsumo | **libsumo** where available, TraCI as fallback | libsumo runs in-process and is roughly an order of magnitude faster; TraCI's socket per step would dominate a 3600-step episode |
| D3 | Network definition | Generate `.nod.xml`/`.edg.xml` from the existing `TopologySpec` and run `netconvert` | Hand-written `.net.xml` files would duplicate the topology definition and drift from it |
| D4 | Demand | SUMO `<flow>` elements, rate taken from the existing demand configs | Keeping our own spawner would reintroduce the spawn-side defects SUMO handles natively |
| D5 | AV actuation | `traci.vehicle.setSpeed` on the commanded speed, with the safety layer still producing the command | `setAcceleration` exists but SUMO applies its own safety bound afterwards either way; speed is the more direct expression of what the controller decides |
| D6 | Keep `highway_env` installed | Yes, and leave the old env in place | The parity ledger, the observation audit and 341 passing tests reference it; removing it in the same change would make a regression unattributable |
| D7 | Collision reporting | Record `getCollidingVehiclesNumber` every step and assert it is zero | A silent assumption that SUMO cannot collide is exactly the kind of unmeasured premise this project has been caught by three times |

## Open items for the user

1. **Whether the old `highway_env` path is eventually deleted.** Keeping both is the
   safe choice now, but two simulators is a maintenance cost and a source of
   divergent results. Recommendation if asked: delete it once the SUMO path has
   reproduced a full evaluation, not before.
2. **Whether the sensing model's route-aware machinery stays.** In SUMO a vehicle's
   route is explicit, so the graph walk added under task 67 becomes unnecessary for
   the simulator — but the parity argument for it was about the *deployed* system,
   which is unchanged. Recommendation: keep it, and feed it SUMO's route.

## Steps

| # | step | done when |
|---|---|---|
| 1 | `SumoNetwork`: generate `.nod.xml`/`.edg.xml` from `TopologySpec`, run `netconvert`, expose the edge↔segment mapping | `inverted_tree` builds and `netconvert` reports no errors; a test asserts the node and edge counts against the spec |
| 2 | `SumoTopologyEnv.reset/step` with libsumo, producing `VehicleSnapshot`s | A 600-step run completes and the snapshots pass the same field checks the old env's do |
| 3 | Demand as flows, rate from the demand config | Measured throughput at 1800 veh/h is within 10% of the configured rate |
| 4 | AV actuation and the safety layer through TraCI | An AV commanded to a lower speed measurably slows, verified per step |
| 5 | Metrics from TraCI into the existing `global_metrics` shape | `mean_speed`, `throughput_recent`, `jam_fraction` and `collision_count` all populated; collision count asserted zero |
| 6 | Re-measure the capacity sweep on SUMO | The demand that congests without gridlocking is identified, as task 79 did for the old simulator |
| 7 | Retrain MAPPO on the SUMO env | A checkpoint exists, trained on SUMO, with its config recorded |
| 8 | Evaluate against `no_av` over an hour | Throughput compared with every run completing, since collisions can no longer truncate |

## Risks

- **Speed.** libsumo is fast but a 3600-step episode with 130 vehicles has real cost.
  Step 2 measures it before step 7 depends on it.
- **The observation may change meaning.** SUMO lane positions and our
  `longitudinal_m` are both distance along a lane, but SUMO's lanes are per-edge
  while `highway_env`'s were per-arc. Step 1's mapping is where that is settled, and
  a wrong mapping would silently move every gap the sensing model computes.
- **Any trained checkpoint is invalidated.** The actor's input distribution changes,
  which is the same dependency that put task 9 ahead of task 68. The existing
  checkpoint is a `highway_env` policy and cannot be evaluated on SUMO.
- **`SUMO_HOME` is unset**, so XML validation is disabled. Harmless for generated
  files, but it means a malformed network is caught by `netconvert` rather than by
  a schema.

## Sign-off

- [ ] Steps 1–5 complete, tests passing, collision count asserted zero across a full run
- [ ] Capacity re-measured on SUMO
- [ ] MAPPO trained and evaluated over an hour, with every run completing
- [ ] `validate_dsrc_3` clean


---

# Steps 1-6 measured 2026-09-09

**Steps 1 to 5 are implemented and tested.** 22 new tests across network
generation, the environment, the road view and the control wiring; the simulator
suite is at 369 passed.

## What SUMO changed, measured

| | highway_env | SUMO |
|---|---|---|
| collisions, 600 steps at every demand | 14 to 23 over ten runs | **0** |
| wall time, 600 steps | ~40 s with CPA | **0.2 s** |
| runs completing | 6 or 7 of 10 | **all** |
| congestion reachable | only via crash queueing | **yes, by demand** |

An hour-long episode costs about 1.2 s, so the evaluation that was unmeasurable is
now nearly free.

## Step 6: capacity is junction-limited and much lower than it appeared

1800-step runs, no AVs, insertion keeping up throughout:

| veh/h | mean speed | arrived | departed | ratio |
|---|---|---|---|---|
| 450 | 26.26 | 222 | 228 | 1.01 |
| 600 | 18.11 | 289 | 300 | 1.00 |
| 750 | 20.25 | 361 | 384 | 1.02 |
| **900** | 7.35 | **378** | 456 | 1.01 |
| 1050 | 2.11 | 349 | 528 | 1.01 |
| 1800 | 1.18 | 425 | — | — |
| 3000 | 1.39 | 538 | — | — |

**Capacity is about 900 veh/h**: arrivals are maximal there and fall at 1050 while
insertion still keeps up. `configs/demand/sumo_saturating.yaml` is 1050, past the
knee, with 29 of 378 arrivals lost to congestion as the headroom a controller has.

**It is junction-limited, not lane-limited.** The three-into-two merges at b1 and b2
are unsignalised, so vehicles yield and throughput is throttled well below what the
two-lane trunk could carry. Realistic for an uncontrolled merge, and the reason
SUMO's capacity here is far below the ~1800 the previous simulator appeared to
manage — that figure was measured on a road where vehicles could drive through each
other.

**One instrument caveat.** `jam_fraction` as computed here is the share of vehicles
below `queue_speed_mps` of 5 m/s, and it reads 0.167 even at 450 veh/h where mean
speed is 26 m/s, because vehicles just inserted and vehicles crossing junctions are
momentarily slow. Mean speed and arrivals are the reliable congestion indicators at
this demand range; `jam_fraction` is not, and should not be used as the operating
point criterion.


---

# Steps 7 and 8 complete 2026-09-09: an hour of simulated driving, measured

**The evaluation task 74 asked for has run.** Five seeds per arm, 3600 steps each,
at `sumo_saturating` with 20% AV penetration, on the checkpoint from a 400-update
SUMO training run:

| arm | arrivals | mean speed | completed | collisions |
|---|---|---|---|---|
| `no_av` | 679.0 ± 8.5 | 1.01 ± 0.11 | 5/5 | 0 |
| `mappo` | 685.4 ± 3.0 | 1.06 ± 0.07 | 5/5 | 0 |
| difference | **+6.4 (+0.9%)** | +0.05 | | |

**Every run completed and no run produced a collision.** That has not happened
before in this project: the same comparison on the previous simulator completed 7
of 18 conditions and every figure came from runs selected for not having crashed.
Ten hour-long runs took 2 minutes 18 seconds in total.

**The instrument is now sensitive enough to trust a null.** Arrivals have a standard
deviation of 3.0 and 8.5 on a mean of 680, so the resolution is about 1%. The 52%
effect measured under task 86 would be unmistakable at this resolution. The +0.9%
reported here is therefore a real null and not a measurement failure.

**And the null has a known cause.** Task 86 measured that every action the policy
can express decodes to 20-30 m/s at a 30 m/s limit, while the effect lives at
10 m/s, and that arrivals were identical across `slow`, `nominal`, `fast` and no
action. A policy choosing among equivalent actions cannot produce a difference. So
this evaluation confirms the instrument rather than the policy, which is the useful
thing it can do until the action space is rescaled.

**One caveat on the operating point.** Mean speed is about 1.0 m/s across both
arms, so at 1050 veh/h the network is heavily congested by the end of an hour --
more so than the 900-step measurements that chose that rate showed. An hour-long
evaluation wants a demand chosen against hour-long runs, which is a shorter version
of the capacity sweep at 3600 steps.


## The operating point depends on the run length

`sumo_saturating` at 1050 veh/h was chosen against 600- and 1800-step runs. Over an
hour it is heavily congested: mean speed 4.04 m/s with no AVs over three seeds, and
a 23.9% throughput deficit. It remains the right level for 600-step runs, where a
controller recovers a measurable part of that deficit; the hour-long level is a
separate config for that reason.

Hour-long capacity, 3600 steps after a 300-step warm-up, no AVs, three seeds
(7, 17, 27), with the fleet the demand config declares:

| veh/h | mean speed | arrived | deficit | deficit % |
|---|---|---|---|---|
| 750 | 20.10 | 748.3 +/- 2.1 | 1.7 | 0.2% |
| 780 | 19.76 | 777.7 +/- 1.2 | 2.3 | 0.3% |
| 810 | 19.07 | 807.7 +/- 3.2 | 2.3 | 0.3% |
| 840 | 17.64 | 830.0 +/- 10.6 | 10.0 | 1.2% |
| 870 | 15.45 | 856.0 +/- 12.5 | 14.0 | 1.6% |
| 900 | 12.37 | **858.0 +/- 24.3** | 42.0 | 4.7% — arrivals peak here |
| 930 | 8.44 | 840.3 +/- 15.5 | 89.7 | 9.6% |
| 960 | 6.86 | 833.0 +/- 18.3 | 127.0 | 13.2% |
| 990 | 5.35 | 813.7 +/- 21.1 | 176.3 | 17.8% |
| 1020 | 4.35 | 797.7 +/- 10.3 | 222.3 | 21.8% |
| 1050 | 4.04 | 799.0 +/- 15.7 | 251.0 | 23.9% |
| 1100 | 3.43 | 789.3 +/- 13.6 | 310.7 | 28.2% |
| 1200 | 2.90 | 810.3 +/- 12.7 | 389.7 | 32.5% |

**This table replaces two earlier ones, and the reason is a defect rather than
noise.** `_write_routes` read only `max_mps` from the demand config's speed
distribution and hardcoded the desired-speed spread as a factor on the lane limit,
so every earlier measurement was taken on a fleet desiring about 30 m/s where the
config declares 24. Hour-long capacity was measured at 858 veh/h rather than the
802 recorded before, and the arrival peak moved from 810 to 900 veh/h.

Attributed, at 1050 veh/h over an hour with three seeds:

| route writer | arrived | mean speed |
|---|---|---|
| lane-limit fleet, inserted at rest (the previous behaviour) | 677.0 +/- 7.0 | 2.74 |
| declared fleet, inserted at rest | 798.0 +/- 18.2 | 3.98 |
| declared fleet, inserted at its desired speed (now) | 799.0 +/- 15.7 | 4.04 |

The whole difference is the desired-speed distribution. Adding
`departSpeed="desired"` changes arrivals by 1.0 against a standard deviation of 16,
which is nothing; it is in for fidelity with the other simulator's spawner, not for
an effect.

The direction is worth stating plainly: **a fleet that wants to drive more slowly
gets more vehicles through this junction** — 798 against 677 per hour at the same
demand. That is the same speed-harmonisation effect the controller is meant to
exploit, appearing here in the fleet's own desired speed rather than in a control
action.

840 veh/h remains the choice: it sits past the arrival peak, which is what gives a
controller a deficit to recover, and it is congested while still moving.


## One demand level serves both horizons, now that the fleet is the declared one

Over an hour, with AVs at 20% penetration held at 10 m/s against an uncommanded
fleet, three seeds:

| veh/h | uncommanded | AVs at 10 m/s | gain | deficit recovered |
|---|---|---|---|---|
| 930 | 840.3 +/- 15.5 | 901.3 +/- 29.8 | +7.3% | 61.0 of 89.7 |
| 960 | 833.0 +/- 18.3 | 931.0 +/- 30.1 | +11.8% | 98.0 of 127.0 |
| 1050 | 799.0 +/- 15.7 | **966.3 +/- 31.8** | **+20.9%** | 167.3 of 251.0 |

The effect grows with demand and is larger over an hour than over 600 steps: 20.9%
against 17.6% at the same 1050 veh/h. At that level a controller recovers 67% of the
throughput deficit.

**`configs/demand/sumo_hour.yaml` is therefore removed.** It existed because
`sumo_saturating` at 1050 veh/h appeared to gridlock over an hour — 0.89 m/s in the
first measurement — leaving no deficit a controller could recover. That gridlock was
a property of the undeclared 30 m/s fleet: with the fleet the config actually
declares, 1050 veh/h is congested and moving at both horizons (4.04 m/s uncommanded,
6.11 m/s under control) and carries the largest recoverable deficit of any level
measured. The second level was an artifact of the defect, not a property of the
road. Nothing referenced the file.

## The measured effect depends on the episode length, and 120 steps hides it

Measured 2026-09-09 and re-measured after the demand fix below; these numbers
supersede every earlier figure for a commanded-speed arm. Two defects sat under the
earlier ones: the fairness term's denominator excluded branches that had completed
nothing, and the fleet's desired speed was the lane limit rather than the 24 m/s the
demand config declares.

Every AV commanded to hold one speed for the whole episode, at `sumo_saturating`
(1050 veh/h, 20% penetration), five seeds:

| arm | 120 steps: arrivals | reward | 600 steps: arrivals | reward |
|---|---|---|---|---|
| uncommanded | 27.8 +/- 3.1 | +1.339 | 132.0 +/- 6.4 | +0.823 |
| 5 m/s | 23.2 +/- 3.0 | +0.670 | 123.4 +/- 19.9 | +0.520 |
| 8 m/s | 24.4 +/- 1.9 | +0.773 | 148.8 +/- 9.0 | +0.915 |
| 10 m/s | 26.2 +/- 4.3 | +0.848 | **155.2 +/- 10.5** | +0.930 |
| 12 m/s | 26.8 +/- 3.5 | +0.920 | 150.6 +/- 17.9 | +0.823 |
| 15 m/s | 27.4 +/- 4.0 | +1.132 | 148.0 +/- 17.1 | **+1.034** |
| 20 m/s | 28.4 +/- 4.2 | +1.359 | 138.8 +/- 10.6 | +0.987 |
| 24 m/s | 28.8 +/- 2.9 | +1.359 | 137.0 +/- 4.7 | +0.956 |

Three things follow.

**The effect does not exist at 120 steps.** No commanded arm beats an uncommanded
fleet there: the best is 28.8 +/- 2.9 against 27.8 +/- 3.1, which is noise. At 600
steps holding 10 m/s produces 155.2 +/- 10.5 against 132.0 +/- 6.4, an increase of
17.6% with a standard error of about 5 arrivals.
`configs/training/mappo_sumo.yaml` declared no `duration_steps` and therefore took
the trainer's default of 120, so training would have been run at the one horizon
where the effect it is meant to learn is absent. It now declares 600.

**The 52% figure is superseded twice over.** 167 arrivals against 110 was a single
seed, on the undeclared 30 m/s fleet, with the fairness defect present. The same
comparison is now 155.2 against 132.0. The direction has held through every
correction; the magnitude has fallen from 52% to 17.6%.

**The reward does not rank the arms the way arrivals do.** At 600 steps arrivals are
maximised at 10 m/s and the reward at 15 m/s, and at 120 steps the reward ranks the
fastest arms first — the reverse of the 600-step arrival order. The reward's
`throughput_recent` term is a 60-second window rather than the episode's total, and
`jam_fraction` at weight -2.0 grows as the network fills, so the reward is not a
monotone function of the quantity the experiment is about. Recorded, not fixed: it
bears on task 85 and on any claim that a trained policy maximising this reward
maximises throughput.


## Retraction: the throughput effect was an artefact of the simulation step

Measured 2026-09-09 by `scripts/measure_step_size_convergence.py`, after the step
size was changed to 0.1 s so the deployment's measured 96.7 ms sensing latency
could be represented at all. Five seeds, 600 s episodes after 300 s of warm-up, at
`sumo_saturating`:

| dt | steps | uncommanded | AVs at 10 m/s | gain |
|---|---|---|---|---|
| 1.0 | 600 | 132.0 +/- 6.4 | 155.2 +/- 10.5 | **+17.6%** |
| 0.5 | 1200 | 134.4 +/- 5.9 | 157.8 +/- 16.0 | +17.4% |
| 0.2 | 3000 | 157.6 +/- 12.3 | 160.4 +/- 7.2 | +1.8% |
| 0.1 | 6000 | 168.4 +/- 11.3 | 161.2 +/- 8.6 | **−4.3%** |
| 0.05 | 12000 | 173.0 +/- 8.4 | 161.6 +/- 8.6 | **−6.6%** |

The commanded arm moves by 6.4 arrivals across a twentyfold change in the step
size, which is inside its own standard deviation. The uncommanded arm rises 31%,
from 132.0 to 173.0. The treatment is invariant to the numerical parameter and the
control is not, so what separated them was never a property of the traffic.

**Mechanism.** SUMO's `--step-length` is the physics step, and this migration
passed `dt` straight into it. At dt 1.0 a vehicle travelling 24 m/s advances 24 m
per step, so junction gap acceptance and car following are resolved at 24 m
granularity and an uncommanded fleet loses throughput to the integration. A fleet
held at 10 m/s advances 10 m per step and loses much less of it.

**The other simulator already handled this.** `HighwayTopologyEnv` integrates at
`physics_substeps: 10` and says why at `src/envs/topology_env.py:233-236`: "a
single 1 s Euler step drives IDM vehicles through each other and to negative
speeds". Its decisions are taken once per second while its physics runs at 0.1 s.
The SUMO env had no equivalent, so from the first commit of this migration the
SUMO fleet integrated ten times more coarsely than the fleet it was being compared
against. Every capacity figure in this document above this section was measured on
the coarse integration, including the junction-limited capacity the demand levels
were chosen against.

Full arm table at dt 0.1, five seeds, from `scripts/measure_commanded_arms.py`:

| arm | arrivals | team reward per step | roadblock score |
|---|---|---|---|
| uncommanded | **168.4 +/- 11.3** | +2.159 | 0.002 |
| 8 m/s | 150.0 +/- 8.5 | +1.408 | 0.306 |
| 10 m/s | 161.2 +/- 8.6 | +1.534 | 0.257 |
| 12 m/s | 163.2 +/- 10.8 | +1.642 | 0.206 |
| 15 m/s | 168.0 +/- 11.7 | +2.090 | 0.122 |
| 20 m/s | 172.4 +/- 8.7 | +2.238 | 0.004 |
| 24 m/s | 172.6 +/- 10.8 | +2.323 | 0.002 |

No arm beats an uncommanded fleet by more than one standard error, and the reward
now ranks the arms in the same order as arrivals — which it did not at dt 1.0. The
disagreement that motivated the metering exemption was itself a symptom of the
coarse integration: the roadblock term was firing on AVs whose slowness was
recovering discretisation error.

**Two decision rates now differ between the simulators.** SUMO takes a decision
every 0.1 s and highway_env every 1.0 s. That is deliberate: the deployed loop runs
at about 30 Hz, so 10 Hz is the closer of the two, and it is what makes the
measured sensing latency representable. It does mean a per-step quantity is not
comparable across the two simulators without dividing by the step.
